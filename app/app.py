"""Native generation and one session-owned, evidence-bound review workspace."""
from copy import deepcopy
import json
from uuid import uuid4

import streamlit as st

import branding
import review_views as review_signals
import workspace as context
from generation_service import GenerationService
from molecule_catalog import MOLECULES
from presentation import reason_text, show_structure, stage_failures, term, theme_css
from review_export import export_csv, export_html, export_json
from toxicology import canonicalize_smiles
import ui_kit
import screening_judges
import tool_comparison_view
from tool_comparison_view import show_tool_comparison
from review_views import (
    activate_generation, capture_models, show_byproducts, show_excluded_routes,
    show_model_review, show_policy, show_route_risk,
)

TITLE = f"{branding.NAME}: {branding.SUBTITLE}"
# Two screens were removed rather than left half-right. The cost screen had no
# obtainable public reagent price source, so it only asked the reader to type
# prices it could not supply. The quantity screen back-calculated charges from
# coefficients and a yield the reader also had to supply, which made the
# software look as though it knew a recipe it had never been given.
TABS = ("위험 검토", "독성 예측 툴 비교")
CATALOG = {f"{item['name_ko']} ({item['name_en']}), {item['category']}": item for item in MOLECULES}
TARGETS = {label: item["smiles"] for label, item in CATALOG.items()}
TARGETS["직접 입력"] = ""
DEFAULT_TARGET = next(label for label, item in CATALOG.items() if item["name_en"].casefold() == "propranolol")
# SynthEx is not an alternative search: it runs AiZynthFinder for the search
# step and puts LLM agents around it (strategies, critique, analysis,
# stitching), which is why it checks the AiZynthFinder version before it
# starts. The labels used to read "원본 로컬 탐색" and "원본 전체 생성", which
# said neither that one of them needs a key nor that the other is inside it.
ENGINES = {"aizynthfinder": "AiZynthFinder: 이 컴퓨터에서 탐색",
           "synthex": "SynthEx: LLM이 전략을 세우고 AiZynthFinder로 탐색"}
# Each entry is a display name, an OpenAI-compatible base_url (None for the
# Gemini SDK) and a default model name. A default is filled in only where the
# provider's own documentation lists that exact ID for tool calling and JSON
# output; otherwise it is blank and the reader supplies it.
# "note" is the one line the consent caption adds about cost. It is written
# per provider because "유료"와 "무료" is not a property of the service: a
# Gemini key on the free tier is refused outright for a Pro model (the API
# answers `limit: 0`), while a Flash model on the same key costs nothing.
PROVIDERS = {
    "groq": {"label": "Groq: 무료 등급", "base_url": "https://api.groq.com/openai/v1",
             "model": "openai/gpt-oss-120b", "paid": False,
             "note": "무료 등급 한도가 적용됩니다."},
    "openrouter": {"label": "OpenRouter: 무료 모델", "base_url": "https://openrouter.ai/api/v1",
                   "model": "", "paid": False,
                   "note": "고른 모델에 따라 요금이 붙을 수 있습니다."},
    "custom": {"label": "직접 입력: OpenAI 호환", "base_url": "", "model": "", "paid": False,
               "note": "고른 모델에 따라 요금이 붙을 수 있습니다."},
    "gemini": {"label": "Gemini: Pro", "base_url": None,
               "model": "gemini-3.1-pro-preview", "paid": True,
               "note": "Pro 모델은 유료 등급에서만 돌아갑니다. 무료 키로는 한도가 0입니다."},
    # Flash models are free on a Gemini key, but Google does not publish one
    # model ID we can pin here and keep correct, so this entry asks for it.
    "gemini_model": {"label": "Gemini: 모델 직접 입력", "base_url": None, "model": "",
                     "paid": False,
                     "note": "Flash 계열은 무료 등급에서 돌아갑니다. 고른 모델에 따라 "
                             "요금이 붙을 수 있습니다."},
}
DEFAULT_PROVIDER = "groq"
# Stands in for a readiness answer while the provider choice is still blank, so
# no probe is started for an endpoint or model the reader has not supplied.
UNSET_PROVIDER = {"state": "not_configured", "ready": False, "live_verified": False,
                  "dependency_compatibility": "unverified", "configuration_id": None,
                  "configuration": None, "error": {"message": "API 주소와 모델 이름을 채우세요"}}
BUSY = {"queued", "running", "cancelling"}
STATUS = {"queued": "대기", "running": "실행 중", "cancelling": "취소 중", "succeeded": "전체 완료",
          "partial": "부분 완료", "failed": "실패", "cancelled": "취소됨",
          "timed_out": "시간 초과", "bounds_exceeded": "출력 상한 초과", "stale": "이전 요청"}


def archive_review(reason):
    workspace = capture_models(st.session_state.get("review_workspace"))
    if workspace and workspace["report"] is not None:
        st.session_state.setdefault("historical_reviews", []).append({
            "applicable": False, "reason": reason, "workspace": deepcopy(workspace),
        })
    st.session_state["review_workspace"] = None
    st.session_state.pop("_cost_snapshot", None)


def cancel_job():
    service = st.session_state.get("_native_service")
    for slot in ("_generation_job", "_readiness_job"):
        job = st.session_state.pop(slot, None)
        if job and service:
            try:
                service.cancel(job["request_id"])
            except Exception:
                pass
            st.session_state.setdefault("_retired_jobs", []).append(job)
            if slot == "_generation_job":
                attempt = next(item for item in st.session_state["generation_attempts"] if item["request"]["request_id"] == job["request_id"])
                attempt.update(status="cancelling", stale=True, applied=False)


def invalidate_generation(reason, *, clear_review=True):
    cancel_job()
    st.session_state["_generation_context"] = str(uuid4())
    st.session_state["generation_consent"] = False
    st.session_state.pop("_generation_pending", None)
    st.session_state.pop("_generation_consent_binding", None)
    st.session_state.pop("_readiness_marker", None)
    if clear_review:
        archive_review(reason)
    for key in ("_byproduct_context", "_byproduct_request", "_byproduct_binding", "_byproduct_consumed"):
        st.session_state.pop(key, None)
    st.session_state["byproduct_consent"] = False


def target_changed():
    invalidate_generation("목표 입력 변경 · 이전 결과")


def pick_target():
    st.session_state["target_smiles"] = TARGETS[st.session_state["target_catalog"]]
    target_changed()


def raw_target_changed():
    if st.session_state["target_smiles"] != TARGETS[st.session_state["target_catalog"]]:
        st.session_state["target_catalog"] = "직접 입력"
    target_changed()


def key_changed():
    st.session_state["_key_revision"] += 1
    invalidate_generation("개인 키 변경·삭제 · 이전 결과")


def clear_key():
    st.session_state["llm_api_key"] = ""
    key_changed()


def provider_changed():
    """A provider holds a different key and a different model name."""
    entry = PROVIDERS[st.session_state["llm_provider"]]
    st.session_state["llm_base_url"] = entry["base_url"] or ""
    st.session_state["llm_model"] = entry["model"]
    st.session_state["llm_api_key"] = ""
    key_changed()


def selection_changed():
    invalidate_generation("모델 설정 변경 · 이전 결과")


def engine_changed():
    st.session_state["llm_api_key"] = ""
    st.session_state["_key_revision"] += 1
    invalidate_generation("생성 엔진 변경 · 이전 결과")


def llm_selection(engine_id):
    """Return the nonsecret provider choice, or None when it is incomplete."""
    if engine_id != "synthex":
        return None
    entry = PROVIDERS[st.session_state["llm_provider"]]
    # A provider that documents its own address and model carries them here.
    # Only the ones left blank in PROVIDERS read what the reader typed.
    model = entry["model"] or st.session_state["llm_model"].strip()
    if not model:
        return None
    if entry["base_url"] is None:
        return {"provider": "gemini", "base_url": None, "model": model}
    base_url = (entry["base_url"] or st.session_state["llm_base_url"].strip()).rstrip("/")
    if not base_url:
        return None
    return {"provider": "openai_compatible", "base_url": base_url, "model": model}


def record_generation(snapshot, job):
    """Only full, current native success may become the current workspace."""
    attempt = next(item for item in st.session_state["generation_attempts"] if item["request"]["request_id"] == job["request_id"])
    attempt.update(status=snapshot["status"], result=deepcopy(snapshot.get("result")),
                   error=deepcopy(snapshot.get("error")), readiness=deepcopy(snapshot.get("readiness")))
    current = (job["context_id"] == st.session_state["_generation_context"]
               and job["engine"] == st.session_state["generation_engine"]
               and job["target"] == st.session_state["target_smiles"]
               and snapshot.get("request_id") == job["request_id"]
               and snapshot.get("context_id") == job["context_id"]
               and snapshot.get("configuration_id") == job["configuration_id"])
    if not current:
        attempt.update(stale=True, applied=False)
        return
    if snapshot["status"] != "succeeded":
        return
    if not (snapshot.get("readiness") or {}).get("live_verified"):
        attempt.update(status="failed", error={"message": "전체 실행 완료 근거가 없습니다."})
        return
    try:
        workspace = st.session_state.get("review_workspace") or context.new_workspace(job["target"])
        st.session_state["review_workspace"] = activate_generation(workspace, snapshot["result"], attempt["request"])
        attempt["applied"] = True
    except Exception:
        attempt.update(status="failed", error={"message": "생성 결과의 목표·스키마를 확인하지 못했습니다."})


def maybe_sweep_byproducts(workspace):
    """Run the whole-route ASKCOS sweep once per generated set of routes.

    Keyed on the generation attempt rather than the report, because applying
    candidates rewrites the report: keying on that would make every sweep
    look like a new set of routes and resend the same structures.
    """
    if not st.session_state.get("byproduct_sweep_consent"):
        return workspace
    applied = next((attempt for attempt in reversed(st.session_state["generation_attempts"])
                    if attempt.get("applied")), None)
    if applied is None:
        return workspace
    marker = applied["request"]["request_id"]
    if st.session_state.get("_byproduct_swept") == marker:
        return workspace
    st.session_state["_byproduct_swept"] = marker
    workspace, skipped = review_signals.sweep_byproducts(workspace)
    st.session_state["_byproduct_sweep_skipped"] = skipped
    return workspace


@st.fragment(run_every=0.5)
def poll_native_jobs():
    """Poll session-owned jobs without blocking the UI or starting an engine."""
    service = st.session_state["_native_service"]
    changed = False
    for slot in ("_readiness_job", "_generation_job"):
        job = st.session_state.get(slot)
        if job is None:
            continue
        try:
            snapshot = service.poll(job["request_id"], context_id=st.session_state["_generation_context"])
        except Exception:
            snapshot = {**job, "status": "failed", "result": None, "error": {"message": "실행 상태를 확인하지 못했습니다."}}
        if snapshot["status"] in BUSY:
            # A probe used to run behind a button that said "준비 중" and
            # nothing else for about five seconds, which reads as stuck
            # rather than busy.
            st.caption("실행 환경을 확인하는 중입니다" if slot == "_readiness_job"
                       else STATUS[snapshot["status"]])
        else:
            st.session_state.pop(slot, None)
            if slot == "_generation_job":
                record_generation(snapshot, job)
            else:
                st.session_state["_last_probe"] = deepcopy(snapshot)
            changed = True
    retired = []
    for job in st.session_state.get("_retired_jobs", []):
        try:
            snapshot = service.poll(job["request_id"])
        except Exception:
            snapshot = {"status": "failed", "result": None, "error": {"message": "취소 상태 확인 실패"}}
        if snapshot["status"] in BUSY:
            retired.append(job)
        elif job.get("operation") == "generate":
            attempt = next(item for item in st.session_state["generation_attempts"] if item["request"]["request_id"] == job["request_id"])
            attempt.update(status="stale", stale=True, result=deepcopy(snapshot.get("result")), error=deepcopy(snapshot.get("error")), applied=False)
            changed = True
    st.session_state["_retired_jobs"] = retired
    if changed:
        st.rerun(scope="app")


def provider_settings():
    """The API provider panel, open until a key and a model are set."""
    ready = bool(st.session_state.get("llm_api_key")
                 and st.session_state.get("llm_model"))
    with st.expander("모델 공급자", expanded=not ready):
        st.selectbox("모델 공급자", list(PROVIDERS),
                     format_func=lambda key: PROVIDERS[key]["label"],
                     key="llm_provider", on_change=provider_changed)
        entry = PROVIDERS[st.session_state["llm_provider"]]
        # Groq and Gemini document one address and one model that do tool
        # calling and JSON output, and those are written in PROVIDERS. Asking
        # for them again as empty boxes was asking the reader to retype a
        # constant. The two providers that name no model of their own are the
        # only ones that still get a box.
        if entry["base_url"] == "":
            st.text_input("API 주소", key="llm_base_url", on_change=selection_changed,
                          help="OpenAI 호환 /chat/completions 엔드포인트입니다.")
        if entry["model"] == "":
            st.text_input("모델 이름", key="llm_model", on_change=selection_changed,
                          help="공급자 문서에 적힌 모델 ID를 그대로 넣습니다.")
        st.text_input("API 키", type="password", key="llm_api_key",
                      on_change=key_changed,
                      help="이 세션과 SynthEx 실행에만 전달합니다.")
        st.button("키 지우기", key="clear_llm_key", on_click=clear_key)
    return entry


def check_readiness(service, engine_id, selection, workspace):
    """Probe the engine once per configuration and report what it found.

    A configuration change invalidates the current review, because routes
    produced by one engine setup are not evidence about another.
    """
    incomplete = engine_id == "synthex" and selection is None
    readiness = (deepcopy(UNSET_PROVIDER) if incomplete
                 else service.readiness(engine_id, selection))
    if incomplete:
        # Name only what this provider actually leaves blank. Groq and Gemini
        # never land here; the two that name no model of their own do.
        entry = PROVIDERS[st.session_state["llm_provider"]]
        missing = [name for name, blank in (("API 주소", entry["base_url"] == ""),
                                            ("모델 이름", entry["model"] == "")) if blank]
        readiness["error"] = {"message": "와 ".join(missing) + "을 채우세요"}
    configuration_id = readiness.get("configuration_id")
    previous = st.session_state.get("_configuration_ids", {}).get(engine_id)
    if previous is not None and previous != configuration_id:
        invalidate_generation("모델·실행 설정 변경 · 새 준비 확인 필요")
        workspace = None
    st.session_state.setdefault("_configuration_ids", {})[engine_id] = configuration_id

    context_id = st.session_state["_generation_context"]
    marker = (engine_id, context_id, json.dumps(selection, sort_keys=True))
    if (not incomplete and not readiness["ready"]
            and st.session_state.get("_readiness_marker") != marker):
        st.session_state["_readiness_marker"] = marker
        job = {"request_id": str(uuid4()), "context_id": context_id,
               "engine": engine_id, "operation": "probe"}
        try:
            service.start_readiness(engine_id, request_id=job["request_id"],
                                    context_id=context_id, llm_selection=selection)
            st.session_state["_readiness_job"] = job
        except Exception:
            st.session_state["_last_probe"] = {
                "status": "failed",
                "error": {"message": "로컬 실행 준비를 확인하지 못했습니다."}}
    if not readiness["ready"] and st.session_state.get("_readiness_job") is None:
        error = (readiness.get("error")
                 or (st.session_state.get("_last_probe") or {}).get("error") or {})
        st.warning(reason_text(error.get("message"))
                   or "엔진 환경과 필수 자산이 필요합니다.")
        if st.button("다시 확인", key="generation_probe_retry"):
            st.session_state.pop("_readiness_marker", None)
            st.rerun()
    return readiness, configuration_id, workspace


def bind_consent(engine_id, selection, readiness, configuration_id):
    """Tie consent to one exact request, so a changed setting clears it."""
    binding = {"session_id": st.session_state["_native_session_id"],
               "context_id": st.session_state["_generation_context"],
               "engine": engine_id, "target": st.session_state["target_smiles"],
               "configuration_id": configuration_id,
               "configuration": readiness.get("configuration"),
               "llm": deepcopy(selection),
               "key_revision": st.session_state["_key_revision"]}
    consumed = st.session_state.pop("_consume_generation_consent", False)
    if consumed or st.session_state.get("_generation_consent_binding") != binding:
        st.session_state["generation_consent"] = False
        st.session_state["_generation_consent_binding"] = deepcopy(binding)
        st.session_state["_generation_pending"] = {**deepcopy(binding),
                                                   "request_id": str(uuid4())}


def start_generation(service, engine_id, selection, configuration_id):
    """Archive what is on screen, queue the request and start the engine."""
    pending = st.session_state["_generation_pending"]
    archive_review("새 생성 요청 · 이전 결과")
    workspace = context.new_workspace(st.session_state["target_smiles"])
    st.session_state["review_workspace"] = workspace
    public = {name: deepcopy(pending[name])
              for name in ("request_id", "context_id", "engine", "target",
                           "configuration_id", "configuration")}
    st.session_state["generation_attempts"].append(
        {"request": public, "status": "queued", "result": None,
         "error": None, "applied": False, "stale": False})
    job = {**public, "operation": "generate"}
    st.session_state["_generation_job"] = job
    st.session_state["_consume_generation_consent"] = True
    arguments = {"request_id": job["request_id"], "context_id": job["context_id"],
                 "configuration_id": configuration_id}
    try:
        if engine_id == "synthex":
            service.start(engine_id, job["target"], **arguments,
                          llm_selection=selection,
                          api_key=st.session_state["llm_api_key"])
        else:
            service.start(engine_id, job["target"], **arguments)
    except Exception:
        record_generation({**job, "status": "failed", "result": None,
                           "error": {"message": "생성을 시작하지 못했습니다."}}, job)
        st.session_state.pop("_generation_job", None)
    return workspace


def show_last_attempt():
    attempts = st.session_state["generation_attempts"]
    if not attempts:
        return
    latest = attempts[-1]
    st.caption("생성 상태: " + STATUS.get(latest["status"], "미확인")
               + (", 이전 맥락" if latest["stale"] else ""))
    if latest["error"]:
        st.warning(reason_text(latest["error"].get("message")) or "생성 실패")
    if latest["status"] == "partial":
        st.warning("중간까지 나온 결과는 기록에만 남습니다")
    failures = stage_failures(latest)
    if failures:
        for stage, reason, raw in failures:
            st.caption(f"{stage}: {reason}" if reason else f"{stage}에서 멈췄습니다")
        with st.expander("제공자가 보낸 원문"):
            for stage, _, raw in failures:
                st.caption(stage)
                st.code(raw or "(내용 없음)", language=None)


def show_generation_controls(workspace, raw_target, target_error):
    """The one thing a reader does with a target: generate routes.

    The engine choice, the provider keys and the readiness line used to stand
    between the target and the button, four controls deep, although the
    default engine runs on this machine and needs none of them. They are one
    collapsed block now, and the button says when it is not ready.
    """
    service = st.session_state["_native_service"]
    with st.expander("생성 설정"):
        engine_id = st.selectbox(
            "엔진", list(ENGINES), format_func=ENGINES.get,
            key="generation_engine", on_change=engine_changed,
            help="AiZynthFinder는 이 컴퓨터에서만 돌고 키가 필요 없습니다. "
                 "SynthEx는 같은 AiZynthFinder로 탐색하되 전략 수립·비평·분석을 "
                 "LLM에 맡기므로, API 키가 있어야 하고 목표 구조가 외부 모델로 "
                 "나갑니다.")
        entry = (provider_settings() if engine_id == "synthex"
                 else PROVIDERS[st.session_state["llm_provider"]])
        selection = llm_selection(engine_id)
        readiness, configuration_id, workspace = check_readiness(
            service, engine_id, selection, workspace)
        bind_consent(engine_id, selection, readiness, configuration_id)
        if engine_id == "synthex":
            name = entry["label"].split(":")[0]
            model = selection["model"] if selection else ""
            st.caption(f"목표 구조를 {name}" + (f"의 {model}" if model else "")
                       + "로 보냅니다. "
                       + entry["note"] + " 900초까지 기다립니다.")
            st.checkbox("외부 전송과 유료 호출에 동의합니다" if entry["paid"]
                        else "외부 전송에 동의합니다",
                        key="generation_consent", disabled=not readiness["ready"])
            st.caption("키나 모델을 바꾸면 실행 중인 작업을 취소합니다")
        # Off until it is ticked. Ticking it is the consent: the line says
        # what leaves this computer, and the per-reaction button in the
        # byproduct tab stays where it is for anyone who would rather send
        # one reaction at a time.
        st.checkbox("부산물 후보도 같이 받기", key="byproduct_sweep_consent")
        st.caption("켜면 경로를 만든 뒤 남은 경로의 모든 반응을 공개 ASKCOS "
                   "서버로 보냅니다. 반응마다 반응물과 주생성물 구조가 나갑니다. "
                   "끄면 아무것도 나가지 않습니다.")

    consented = (engine_id != "synthex"
                 or (bool(st.session_state.get("llm_api_key"))
                     and st.session_state.get("generation_consent", False)))
    permitted = (workspace is not None and readiness["ready"] and consented
                 and st.session_state.get("_generation_job") is None)
    # Changing the target restarts the readiness probe, which locks the button
    # for a few seconds. The label carries that state so a disabled button is
    # never silent.
    if st.button("준비 중" if not readiness["ready"] else "경로 생성",
                 key="generation_run", type="primary", disabled=not permitted):
        workspace = start_generation(service, engine_id, selection, configuration_id)
        st.rerun()
    if st.session_state.get("_generation_job") and st.button("생성 취소",
                                                             key="generation_cancel"):
        invalidate_generation("사용자 취소 · 이전 요청")
        st.rerun()
    show_last_attempt()
    return workspace


def route_order(workspace, routes):
    """Cleanest first. Routes with no penalty keep their original order."""
    decisions = {item["id"]: item for item in (workspace["selection"] or {}).get("routes", [])}
    order = {route["id"]: index for index, route in enumerate(routes)}
    return sorted(routes, key=lambda route: (decisions.get(route["id"], {}).get("rank_penalty", 0),
                                             order[route["id"]]))


def route_label(workspace, route_id):
    decision = next((item for item in (workspace["selection"] or {}).get("routes", [])
                     if item["id"] == route_id), None)
    signals = sum(1 for reason in (decision or {}).get("reasons", [])
                  if reason["code"] in review_signals.DECIDING_CODES
                  and reason["code"] not in review_signals.NOTE_CODES)
    number = review_signals.route_number(workspace, route_id)
    return f"경로 {number}, 우려 {signals}건" if signals else f"경로 {number}"


def shared_selection(workspace):
    routes = route_order(workspace, context.eligible_routes(workspace))
    if not routes:
        return workspace
    owner = context.context_key(workspace)
    route_ids = [route["id"] for route in routes]
    labels = {route_id: route_label(workspace, route_id)
              for route_id in route_ids}
    if st.session_state.get("_selector_context") != owner or st.session_state.get("route_id") not in route_ids:
        st.session_state["route_id"] = workspace["route_id"]
        st.session_state["step_id"] = workspace["step_id"]
        st.session_state["_selector_route"] = workspace["route_id"]
    route_id = st.selectbox(
        "경로", route_ids, key="route_id",
        format_func=lambda value: labels.get(value, value),
        help="우려가 적은 순으로 놓았습니다. 제외된 경로는 위에 따로 있습니다.")
    # A route is the unit a reader chooses; a step is not. The risk tab lays
    # out every step of the route at once, so the only screen that still needs
    # one step singled out is the byproduct request, and it asks there. The
    # step kept here is just a valid default for that tab.
    route = next(item for item in routes if item["id"] == route_id)
    steps = [step["id"] for step in route["steps"]]
    if st.session_state.get("_selector_route") != route_id or st.session_state.get("step_id") not in steps:
        st.session_state["step_id"] = steps[0]
        st.session_state["_selector_route"] = route_id
    workspace = context.select_step(workspace, route_id, st.session_state["step_id"])
    st.session_state["_selector_context"] = context.context_key(workspace)
    return workspace


def freeze_snapshot(workspace, raw_target, target_error):
    active = workspace if workspace is not None else {}
    data = {"schema_version": "1.0", "mode": "route",
            "scope": "전체 생성 경로·기본 제외 결정·선택 단계 산술",
            "raw_target": raw_target, "target_error": target_error,
            "target": workspace["target"] if workspace else None,
            "original_input": active.get("original_input"),
            "active_input": active.get("active_input"),
            "generation_attempts": st.session_state["generation_attempts"],
            "historical_reviews": st.session_state.get("historical_reviews", []),
            "historical_byproduct_attempts": st.session_state.get("historical_byproduct_attempts", [])}
    for name in ("report", "report_sha256", "report_generation", "workspace_id", "context_revision", "route_id", "step_id",
                 "source_provenance", "selection", "policy", "model_evidence", "model_evidence_state"):
        data[name] = active.get(name)
    for name in ("model_evidence_history", "byproduct_attempts"):
        data[name] = active.get(name, [])
    return deepcopy(data)


@st.cache_data(show_spinner=False, max_entries=12)
def render_export(snapshot_text, file_format):
    snapshot = json.loads(snapshot_text)
    if file_format == "HTML":
        return export_html(snapshot)
    if file_format == "CSV":
        return "\ufeff" + export_csv(snapshot)
    return export_json(snapshot)


st.set_page_config(page_title=TITLE, layout="wide")
for name, value in (("target_catalog", DEFAULT_TARGET), ("target_smiles", TARGETS[DEFAULT_TARGET]),
                    ("generation_engine", "aizynthfinder"), ("light_mode", False), ("llm_api_key", ""),
                    ("llm_provider", DEFAULT_PROVIDER),
                    ("llm_base_url", PROVIDERS[DEFAULT_PROVIDER]["base_url"] or ""),
                    ("llm_model", PROVIDERS[DEFAULT_PROVIDER]["model"]),
                    ("_key_revision", 0), ("generation_consent", False),
                    ("byproduct_sweep_consent", False),
                    ("generation_attempts", [])):
    st.session_state.setdefault(name, value)
st.session_state.setdefault("_native_session_id", str(uuid4()))
st.session_state.setdefault("_generation_context", str(uuid4()))
if "_native_service" not in st.session_state:
    st.session_state["_native_service"] = GenerationService()
st.markdown(theme_css(st.session_state["light_mode"]), unsafe_allow_html=True)
ui_kit.use()
# The figures the screen opens with are the ones a reader needs before they
# trust anything below: what each in-house model scored on structures it had
# never seen, and the line it decides at. They come from the comparison
# screen's own table, so the header cannot drift away from it.
branding.show_header([
    # AUC rather than balanced accuracy, and nothing beside it. The chip used
    # to put the 0.50 decision line next to the score, which is a different
    # quantity entirely and read as if it were a baseline.
    (row.get("short_label") or row["label"], "AUC", f"{row['auc']:.3f}")
    for row in (tool_comparison_view.TOOL_PERFORMANCE["ames_xgb"],
                tool_comparison_view.TOOL_PERFORMANCE["ames_desc"])
] + [("도구", "", f"{len(tool_comparison_view.independent_tool_ids())}종")])
st.selectbox("목표 물질", list(TARGETS), key="target_catalog", on_change=pick_target)
with st.expander("목표 구조", expanded=st.session_state["target_catalog"] == "직접 입력"):
    raw_target = st.text_input("목표 물질 SMILES", key="target_smiles", on_change=raw_target_changed)
    if len(raw_target) <= 4096 and raw_target.strip():
        show_structure(raw_target)
    st.code(raw_target, language=None)
target_error = None
try:
    # A stored workspace whose report no longer validates makes this raise, and
    # an unguarded raise here blanks the whole page instead of one message.
    workspace = capture_models(st.session_state.get("review_workspace"))
except ValueError as exc:
    workspace = None
    st.error(f"저장된 검토를 읽지 못해 새로 시작합니다: {exc}")
try:
    if workspace is None or workspace["raw_target"] != raw_target:
        workspace = context.new_workspace(raw_target)
except ValueError as exc:
    workspace, target_error = None, str(exc)
    st.error(target_error)
st.session_state["review_workspace"] = workspace
workspace = show_generation_controls(workspace, raw_target, target_error)
st.session_state["review_workspace"] = workspace
poll_native_jobs()
if workspace and workspace["report"] is not None:
    workspace = maybe_sweep_byproducts(workspace)
    st.session_state["review_workspace"] = workspace
    for entry in st.session_state.get("_byproduct_sweep_skipped", []):
        st.caption(f"{entry['route_id']} {entry['step_id']}단계는 부산물 요청을 "
                   f"보내지 않았습니다: {entry['reason']}")
    # The rule and what it dropped are one subject, so they are one block.
    # Two expanders side by side made the reader open both to learn that the
    # second was the consequence of the first.
    with st.expander("제외 기준과 빠진 경로"):
        workspace = show_policy(workspace)
        workspace = capture_models(workspace)
        show_excluded_routes(workspace)
    workspace = shared_selection(workspace)
    st.session_state["review_workspace"] = workspace
    if context.eligible_routes(workspace):
        risk, comparison = st.tabs(TABS, key="review_tabs", on_change="rerun")
        with risk:
            # What produced these routes, and whether byproduct candidates
            # are in them, decides how to read everything below.
            byproducts_note = ("부산물 후보 포함"
                               if st.session_state.get("byproduct_sweep_consent")
                               else "부산물 미조회")
            st.caption(f"엔진: {ENGINES[st.session_state['generation_engine']]}"
                       f" · {byproducts_note}")
            show_route_risk(workspace)
            # The cards above carry the ICH M7 rule screen's call only. One
            # tool of the reader's choosing goes beside it here; every tool
            # at once is the next tab.
            tool_comparison_view.show_route_tool_table(workspace)
            # The candidates themselves are cards above, so what is left is
            # the request: what was sent where, what came back, what failed.
            # That is a record to reach for, not a tab to sit beside the
            # review, so it is folded shut here.
            previous = st.session_state.get("_byproduct_completion", 0)
            with st.expander("부산물 요청"):
                workspace = show_byproducts(workspace)
            st.session_state["review_workspace"] = workspace
            if st.session_state.get("_byproduct_completion", 0) != previous:
                st.rerun()
        with comparison:
            show_tool_comparison(workspace)
    else:
        st.warning("남은 경로가 없습니다. 위에서 제외 이유를 볼 수 있습니다.")
    previous_binding = context.context_key(workspace)
    workspace = show_model_review(workspace)
    st.session_state["review_workspace"] = workspace
    if context.context_key(workspace) != previous_binding:
        st.rerun()
st.session_state["review_workspace"] = workspace
snapshot = freeze_snapshot(workspace, raw_target, target_error)
st.session_state["review_export_snapshot"] = snapshot
if workspace and workspace["report"] or st.session_state["generation_attempts"] or st.session_state.get("historical_reviews"):
    st.divider()
    # One expander instead of a headed section with its own caption, format
    # radio and history block. Saving is the last thing a reader does and it
    # was taking up as much room as the review.
    with st.expander("저장"):
        file_format = st.radio("형식", ("JSON", "HTML", "CSV"), horizontal=True,
                               key="export_format")
        try:
            data = render_export(export_json(snapshot), file_format)
            mime = {"JSON": "application/json", "HTML": "text/html",
                    "CSV": "text/csv"}[file_format]
            st.download_button("내려받기", data,
                               "review_snapshot." + file_format.lower(), mime,
                               type="primary", key="review_download")
        except (ValueError, TypeError, KeyError, RecursionError):
            st.error("저장하지 못했습니다")
        st.caption("경로와 제외 사유, 생성 기록이 함께 들어갑니다")
        st.json({"generation_attempts": st.session_state["generation_attempts"],
                 "historical_reviews": st.session_state.get("historical_reviews", []),
                 "historical_byproduct_attempts": st.session_state.get("historical_byproduct_attempts", [])}, expanded=False)
branding.show_footer([f"{tool_comparison_view.MEASURED_ON} 기준"])
