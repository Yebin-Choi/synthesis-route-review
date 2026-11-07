"""Evidence-first route review and consent-bound byproduct presentation.

The exclusion section used to print every material of every excluded route as
a stack of six elements: a caption, a drawing, the SMILES, the verdict, the
origin and the source. Three routes of four materials each came to roughly
seventy stacked elements, and the reason the route was dropped was somewhere
inside them. It is now one table per step, where a column can be scanned, with
the drawings in a grid beneath it. Nothing was removed; the same structures,
verdicts, origins and sources are all still there.
"""
from copy import deepcopy

import streamlit as st

import engine
import ui_kit
import workspace as context
from presentation import (
    GLOSSARY,
    ALERTS, ROLES, STATUS, material_card, material_name, material_reason,
    route_dot, show_structure,
)
from route_ames_view import ensure_route_ames_evidence, get_route_ames_evidence, show_route_ames_review

ROUTE_STATUS = {"excluded": "우려 신호로 제외", "needs_review": "검토 필요",
                "insufficient_data": "정보 부족", "candidate": "후보"}
# A route kept for review and a route with a gap in its data are not
# clearances, so neither takes the chip a candidate route gets.
ROUTE_KINDS = {"excluded": ui_kit.CONCERN, "needs_review": ui_kit.REVIEW,
               "insufficient_data": ui_kit.UNKNOWN, "candidate": ui_kit.CLEAR}
# A material's own verdict. "해당 없음" means the list and the structure rules
# found nothing, which is not the same as the material being safe.
MATERIAL_KINDS = {"known_concern": ui_kit.CONCERN, "structural_alert": ui_kit.CONCERN,
                  "invalid": ui_kit.UNKNOWN, "no_alert_detected": ui_kit.CLEAR}
EVIDENCE = {"observed": "관측", "proposed": "제안", "unknown": "자료 없음"}
# How many structure drawings sit in one row of the grid.
DRAWINGS_PER_ROW = 4


def capture_models(workspace):
    if workspace is None or workspace["report"] is None:
        return workspace
    ensure_route_ames_evidence(workspace["report"], report_generation=workspace["report_generation"])
    envelope = get_route_ames_evidence(
        workspace["report"], report_generation=workspace["report_generation"],
        excluded_smiles=workspace["policy"]["excluded_smiles"],
    )
    return context.set_model_evidence(workspace, envelope) if envelope is not None else workspace


def activate_generation(workspace, result, request):
    """Publish only an already completed, validated native service result."""
    if not isinstance(result, dict) or not isinstance(result.get("payload"), dict) or result.get("item_failures"):
        raise ValueError("전체 완료된 경로 결과가 필요합니다. 부분 결과를 적용하지 않습니다.")
    state = context.load_routes(workspace, result["payload"])
    state["source_provenance"] = deepcopy({"kind": "native_generation", "request": request,
        "provenance": result["provenance"], "artifacts": result["artifacts"],
        "item_failures": result["item_failures"]})
    return state


def exclusion_choices(report):
    targets = set()
    for route in report["routes"]:
        target = context.validate_target(route["target"])
        targets.add(target)
        targets.update(engine.canonicalize_smiles(part) for part in target.split("."))
    choices = {}
    for route in report["routes"]:
        for step in route["steps"]:
            for material in step["materials"]:
                identity = material["canonical_smiles"]
                if not identity or identity in targets:
                    continue
                choice = choices.setdefault(identity, {"name": material_name(identity, material.get("name")), "roles": []})
                role = ROLES.get(material["role"], material["role"])
                if role not in choice["roles"]:
                    choice["roles"].append(role)
    return {identity: f"{item['name']} · {' / '.join(item['roles'])} · {identity}" for identity, item in choices.items()}


def show_policy(workspace):
    choices = exclusion_choices(workspace["report"])
    owner = (workspace["workspace_id"], workspace["report_generation"])
    if (st.session_state.get("_policy_owner") != owner
            or "strict_alerts" not in st.session_state or "excluded_smiles" not in st.session_state):
        st.session_state["strict_alerts"] = workspace["policy"]["strict_alerts"]
        st.session_state["excluded_smiles"] = [value for value in workspace["policy"]["excluded_smiles"] if value in choices]
        st.session_state["_policy_owner"] = owner
    strict = st.checkbox("규칙에 걸린 경로도 제외", key="strict_alerts")
    excluded = st.multiselect("직접 제외할 물질", list(choices), format_func=choices.get, key="excluded_smiles")
    st.caption("알려진 우려 물질, 경고구조, 조건부 경고, 모델 양성 중 하나라도 걸리면 제외합니다")
    if {"strict_alerts": strict, "excluded_smiles": excluded} != workspace["policy"]:
        try:
            workspace = context.set_policy(workspace, strict, excluded)
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"정책을 적용하지 못했습니다: {exc}")
    return workspace


def show_step_unknowns(step):
    # The warning itself, then its grounds folded away. Three stacked
    # paragraphs and a bare URL made the finding compete with its footnote.
    for warning in step["conditional_warnings"]:
        st.warning(warning["message"])
        # The grounds as a caption, not a fold. A fold here could not be drawn
        # at all where this step is listed inside another one, and two lines
        # are shorter than the control that would have hidden them.
        st.caption(" · ".join(part for part in
                              (warning["basis"], warning.get("source_url")) if part))
    condition, byproducts = step["conditions"], step["byproducts"]
    # One record instead of four stacked remarks. The wording of the first
    # line is unchanged because it is the line the saved file and the tests
    # both refer to.
    ui_kit.show_facts([
        ("단계 자료", "반응 조건: " + EVIDENCE[condition["status"]] + " · 부산물: " +
         {"not_run": "미실행", "failed": "평가 실패",
          "success": "제안 후보 평가"}[byproducts["status"]]),
        ("조건 설명", ui_kit.text(condition.get("description") or "")),
        ("조건 출처", ui_kit.text(condition.get("source") or "")),
    ])
    if byproducts.get("error"):
        st.warning("부산물 평가 오류: " + byproducts["error"])


# Codes that name an actual concern signal, versus routine gaps that repeat on
# every material and every step. Both stay in the saved evidence; only the
# screen collapses the routine ones into a single line.
DECIDING_CODES = {
    # Same wording as presentation.STATUS, so one material does not carry two
    # different names for the same finding on one screen.
    "known_concern": "ICH M7 목록 물질",
    "structural_alert": "위험 구조",
    "admet_positive": "모델 양성",
    "current_model_positive_exclusion": "모델 양성",
    "rule_admet_joint_positive": "규칙과 모델 모두 양성",
    "byproduct_concern": "부산물 우려 (제외 아님)",
    "prohibited_material": "제외하기로 한 물질",
    "prohibited_component": "제외하기로 한 성분",
    "invalid_material": "구조 해석 불가",
}
# The deciding list gained the two prohibition codes and the unreadable-
# structure code. All three were falling through to the routine line, where
# an unmapped code printed as itself: a route dropped for a prohibited
# material was reporting "prohibited_material" on screen.
ROUTINE_CODES = {
    "missing_information": "자료 없음",
    "conditions_not_observed": "조건 자료 없음",
    "byproducts_incomplete": "부산물 미평가",
    "admet_unassessed": "모델 미평가",
    "conditional_warning": "조건부 경고",
    "no_materials": "검사할 물질 없음",
}
# Shown when a code has no label here. The code itself stays in the saved
# file, so nothing is lost; it just does not go on the screen as itself.
UNKNOWN_CODE = "그 밖의 신호"


# Codes that annotate another reason rather than being a finding of their
# own. They are shown, because a reader needs to know a concern came from a
# predicted byproduct, but they are not counted as separate signals: the
# alert and the note beside it are one finding.
NOTE_CODES = {"byproduct_concern"}


# Codes whose full message belongs in the saved file rather than on screen.
# A negative model score is not a reason this route was excluded.
QUIET_CODES = {"admet_negative", "candidate_within_scope"}


def show_reasons(decision):
    deciding, routine = {}, set()
    for reason in decision["reasons"]:
        code = reason["code"]
        if code in DECIDING_CODES:
            # Two codes can carry the same verdict for one material, so group by
            # what the reader sees rather than by the internal code.
            key = (DECIDING_CODES[code], reason.get("smiles"))
            deciding.setdefault(key, set()).add(reason["step_id"])
        elif code in ROUTINE_CODES:
            routine.add(code)
        elif code not in QUIET_CODES:
            routine.add(code)
    for (label, smiles), steps in deciding.items():
        shown = sorted(step for step in steps if step)
        where = f"단계 {', '.join(shown)}" if shown else "경로 전체"
        name = material_name(smiles) if smiles else ""
        st.write(f"**{label}** · {name} · {where}" if name else f"**{label}** · {where}")
    if routine:
        st.caption(" · ".join(dict.fromkeys(
            ROUTINE_CODES.get(code, UNKNOWN_CODE) for code in sorted(routine))))


def route_number(workspace, route_id):
    """Position in the generated set. The engine's own id is long and carries
    no meaning for a reader, so it stays in the saved file only."""
    order = [route["id"] for route in (workspace["report"] or {}).get("routes", [])]
    return order.index(route_id) + 1 if route_id in order else None


def _material_rows(workspace, step, source_step):
    """One row per listed structure of this step, in the input's own order.

    The report omits exact target structures, so the source arrays are walked
    instead. A row whose structure is the target says so rather than implying
    the target was impurity-assessed, which it was not.
    """
    rows = []
    for role in (*engine.ROLES, *engine.ADDITIONAL_ROLES, "byproducts"):
        values = (source_step.get(role, []) if role != "byproducts"
                  else source_step[role]["candidates"])
        for raw in values:
            identity = engine.canonicalize_smiles(raw)
            assessment = next((item for item in step["materials"]
                               if item["role"] == role and item["input_smiles"] == raw), None)
            rows.append({
                "raw": raw, "role": role,
                "name": material_name(identity, (assessment or {}).get("name")),
                "is_target": identity == workspace["target"],
                "assessment": assessment,
            })
    return rows


def _verdict_cell(row):
    """The material's verdict, with the structures that triggered it beneath.

    The target is not given a verdict at all: it is excluded from impurity
    assessment, so a chip here would be a claim nobody made.
    """
    if row["is_target"]:
        return ui_kit.chip("목표물 · 검사 대상 아님", ui_kit.ABSENT)
    assessment = row["assessment"]
    if assessment is None:
        return ui_kit.verdict_chip("not_run")
    status = assessment["status"]
    alerts = ", ".join(ALERTS.get(value, value)
                       for value in assessment.get("alerts", []))
    return ui_kit.chip(STATUS[status], MATERIAL_KINDS[status],
                       note=ui_kit.text(alerts or material_reason(assessment)))


def _show_drawings(rows):
    """The structures as a grid, so a step is one block instead of a column."""
    drawable = [row for row in rows if len(row["raw"]) <= 4096]
    for start in range(0, len(drawable), DRAWINGS_PER_ROW):
        chunk = drawable[start:start + DRAWINGS_PER_ROW]
        columns = st.columns(DRAWINGS_PER_ROW)
        for column, row in zip(columns, chunk):
            with column:
                show_structure(row["raw"])
                st.caption(ROLES[row["role"]] + " · " + row["name"])


def show_excluded_routes(workspace):
    """Why each dropped route was dropped, under the rule that dropped it."""
    decisions = context.excluded_routes(workspace)
    if not decisions:
        st.caption("이 기준으로 빠진 경로가 없습니다")
        return
    st.divider()
    st.caption(f"이 기준으로 경로 {len(decisions)}개가 빠졌습니다")
    ui_kit.use()
    reports = {route["id"]: route for route in workspace["report"]["routes"]}
    sources = {route["id"]: route for route in workspace["active_input"]["routes"]}
    for decision in decisions:
        route, source = reports[decision["id"]], sources[decision["id"]]
        st.markdown(f"**경로 {route_number(workspace, route['id'])}**")
        show_reasons(decision)
        for order, step in enumerate(route["steps"], 1):
            st.caption(f"{order}단계")
            source_step = next(item for item in source["steps"]
                               if item["id"] == step["id"])
            rows = _material_rows(workspace, step, source_step)
            ui_kit.show_table(
                ["구조", "역할·이름", "판정", "출처"],
                [{"cells": [
                    ui_kit.text(row["raw"]),
                    ui_kit.text(ROLES[row["role"]] + " · " + row["name"]),
                    _verdict_cell(row),
                    ui_kit.text(" · ".join(part for part in (
                        (row["assessment"] or {}).get("origin"),
                        (row["assessment"] or {}).get("source_url")) if part)),
                ], "mark": not row["is_target"] and (row["assessment"] or {}).get(
                    "status") in ("known_concern", "structural_alert")}
                 for row in rows])
            _show_drawings(rows)
            show_step_unknowns(step)
        st.caption("판정 근거는 저장 파일에 함께 들어갑니다")


_CARD_ORDER = {"known_concern": 0, "structural_alert": 1,
               "invalid": 2, "no_alert_detected": 3}


def _step_block(step, order):
    """One reaction step: its materials, worst first, then what it left blank."""
    st.markdown(f"**{order}단계**")
    materials = step["materials"]
    if not materials:
        st.caption("기재된 비목표 물질이 없습니다")
    for material in sorted(materials, key=lambda item: _CARD_ORDER[item["status"]]):
        material_card(material, f"{order}단계")
    show_step_unknowns(step)


def show_route_risk(workspace):
    """The whole route on one screen, step by step.

    Each step used to need a selector of its own, which asked the reader to
    page through a route three reactions long to see whether any of it was a
    problem. The counts are the route's, so the answer is at the top.
    """
    route = next(item for item in workspace["report"]["routes"] if item["id"] == workspace["route_id"])
    decision = next(item for item in workspace["selection"]["routes"] if item["id"] == route["id"])
    number = route_number(workspace, route["id"])
    ui_kit.use()
    st.subheader(f"경로 {number}")
    verdict = ROUTE_STATUS[decision["status"]]
    st.markdown(ui_kit.chip(verdict, ROUTE_KINDS[decision["status"]],
                            hint=GLOSSARY.get(verdict)), unsafe_allow_html=True)
    materials = [item for step in route["steps"] for item in step["materials"]]
    concern = sum(item["status"] in ("known_concern", "structural_alert")
                  for item in materials)
    unreadable = sum(item["status"] == "invalid" for item in materials)
    ui_kit.metric_row([
        ("단계", len(route["steps"]), "이 경로를 이루는 반응 수입니다."),
        ("물질", len(materials), "이 경로에 기재된 비목표 물질 수입니다. "
                               "목표물 자체는 세지 않습니다."),
        ("우려", concern, "ICH M7 목록에 있거나 경고구조가 걸린 물질 수입니다."),
        ("못 읽음", unreadable, "SMILES를 분자로 읽지 못한 물질 수입니다. "
                              "신호가 없다는 뜻이 아니라 검사를 못 한 것입니다."),
    ])
    if unreadable:
        st.caption("구조를 읽지 못한 물질이 있습니다. 신호가 없다는 뜻이 아니라 "
                   "검사를 하지 못했다는 뜻입니다.")
    for order, step in enumerate(route["steps"], 1):
        _step_block(step, order)
    with st.expander("판정 사유와 원문"):
        show_reasons(decision)
        source = next(item for item in workspace["active_input"]["routes"] if item["id"] == route["id"])
        st.graphviz_chart(route_dot(source))
        st.caption("보고서에 적힌 반응물과 생성물만 잇습니다")
        st.json(source, expanded=False)


def show_model_review(workspace):
    with st.expander("변이원성 모델 검토"):
        state = context.model_evidence_status(workspace)
        st.caption("모델 상태: " + state["status"])
        if state["error"]:
            st.warning(state["error"])
        show_route_ames_review(workspace["report"], report_generation=workspace["report_generation"],
                               excluded_smiles=workspace["policy"]["excluded_smiles"])
    return capture_models(workspace)


def show_material_risk(raw_target, target_error):
    if st.button("독립 물질 검사", key="screen_material", disabled=target_error is not None):
        saved = {"raw_input": raw_target, "result": None, "error": None}
        try:
            context.validate_target(raw_target)
            saved["result"] = engine.assess_material(raw_target)
        except Exception:
            saved["error"] = "물질 검사에 실패했습니다"
        st.session_state["material_inspection"] = saved
        st.session_state["inspection_mode"] = "material"
    saved = st.session_state.get("material_inspection")
    if saved and saved["raw_input"] != raw_target:
        st.session_state.pop("material_inspection", None)
        saved = None
    if saved and st.session_state.get("inspection_mode") == "material":
        st.caption("목표물 하나만 검사한 결과입니다")
        if saved["error"]:
            st.error(saved["error"])
        elif saved["result"] is not None:
            # The card adds "독립 물질 검사" itself for a material with no
            # role in a route, so naming it here too read it out twice.
            material_card(saved["result"], "목표물 자체")
    return saved


def apply_current_byproduct_result(request_workspace, request, result):
    current = st.session_state.get("review_workspace")
    if current is None:
        detached = context.new_workspace(request_workspace["raw_target"])
        history = context.apply_byproduct_result(detached, request, result)
        st.session_state.setdefault("historical_byproduct_attempts", []).extend(history["byproduct_attempts"])
        return None
    return context.apply_byproduct_result(current, request, result)


def _pick_step(workspace):
    """Which reaction to ask about. Only this tab works on one step.

    A request goes to ASKCOS for one reaction, so the choice has to exist
    somewhere. It belongs here rather than beside the route selector, where
    it sat in front of every reader including the ones who never opened this
    tab. A route with one reaction asks nothing.
    """
    route = next(item for item in workspace["report"]["routes"]
                 if item["id"] == workspace["route_id"])
    steps = [item["id"] for item in route["steps"]]
    if len(steps) == 1:
        return workspace
    order = {value: index for index, value in enumerate(steps, 1)}
    # Keyed on step_id itself, so the selector and the workspace cannot hold
    # two different answers: a request that lands while the reader is on
    # another step must be judged against the step the app is actually on.
    chosen = st.radio("반응", steps, horizontal=True, key="step_id",
                      format_func=lambda value: f"{order[value]}단계")
    if chosen != workspace["step_id"]:
        return context.select_step(workspace, workspace["route_id"], chosen)
    return workspace


def show_byproducts(workspace):
    """One explicit click sends this step to the public ASKCOS server.

    Consent stays per request and informed: the button names the transmission,
    and the exact structures sent are shown above it. Replacing the former
    checkbox plus button with one button removes a step without removing the
    decision, because nothing leaves the machine until the button is pressed.
    """
    workspace = _pick_step(workspace)
    step = context.selected_step(workspace)
    owner = context.context_key(workspace)
    if st.session_state.get("_byproduct_context") != owner:
        st.session_state["byproduct_product"] = None
        st.session_state.pop("_byproduct_request", None)
        st.session_state.pop("_byproduct_binding", None)
        st.session_state["_byproduct_context"] = owner
    products = list(dict.fromkeys(step["products"]))
    if len(products) == 1:
        # A single listed product needs no choice; picking it by hand was a
        # step with exactly one possible answer.
        product = products[0]
        st.session_state["byproduct_product"] = product
    else:
        product = st.selectbox("주생성물", products, index=None, key="byproduct_product",
                               placeholder="생성물을 고르세요")
    binding = (owner, product)
    request = st.session_state.get("_byproduct_request")
    if (st.session_state.get("_byproduct_binding") != binding
            or request and request["request_id"] == st.session_state.get("_byproduct_consumed")):
        st.session_state["_byproduct_binding"] = binding
        st.session_state.pop("_byproduct_request", None)
        request = None
    if product is not None and request is None:
        try:
            if len(".".join(value.strip() for value in step["reactants"])) > 4096:
                raise ValueError("전송할 반응물 SMILES 합계가 4096자를 초과합니다. 잘라서 전송하지 않습니다.")
            request = context.make_byproduct_request(workspace, product)
            st.session_state["_byproduct_request"] = request
        except ValueError as exc:
            st.error(str(exc))
    if request is not None:
        # The exact structures that will leave this machine, above the button
        # that sends them. Consent means knowing what is sent, so this stays
        # on the surface and only the raw payload is collapsed.
        ui_kit.show_facts([
            ("보내는 곳", ui_kit.text(engine.ASKCOS_API)),
            ("반응물", ui_kit.text(request["reactants_smiles"])),
            ("주생성물", ui_kit.text(request["product_smiles"])),
        ])
        with st.expander("보낼 내용"):
            st.json({"rct_smi": request["reactants_smiles"],
                     "prd_smi": request["product_smiles"]})
    if st.button("ASKCOS로 보내 부산물 예측", key="byproduct_request", type="primary",
                 disabled=request is None):
        try:
            with st.spinner("예측 중"):
                result = engine.predict_byproducts(request["reactants_smiles"], request["product_smiles"], timeout=60)
        except Exception as exc:
            result = {"status": "failed", "candidates": [], "error": f"{type(exc).__name__}: {exc}", "source": engine.ASKCOS_API}
        workspace = apply_current_byproduct_result(workspace, request, result)
        st.session_state["_byproduct_consumed"] = request["request_id"]
        st.session_state["_byproduct_completion"] = st.session_state.get("_byproduct_completion", 0) + 1
    attempts = workspace["byproduct_attempts"] if workspace else []
    if attempts:
        latest = attempts[-1]
        if latest["error"]:
            st.error("후보 요청 실패: " + latest["error"])
        elif latest["applied"]:
            st.success("후보를 적용하고 경로를 다시 검토했습니다")
        if any(attempt["applied"] for attempt in attempts[:-1]):
            st.caption("이전 성공은 이력에 남습니다")
        show_byproduct_candidates(workspace)
        with st.expander("요청 기록"):
            st.json(attempts, expanded=False)
    return workspace


def show_byproduct_candidates(workspace):
    """What came back, with each candidate's own verdict beside it.

    The tab used to say only whether the request succeeded, so the reader had
    to go to another tab to find out what the request produced. These are
    proposed candidates for one step, not a complete list of what the reaction
    actually makes, and the caption says so.
    """
    if workspace is None or workspace["report"] is None:
        return
    # The source step holds the raw candidate list; the report step holds the
    # verdicts. Both are needed, and they are two different objects.
    source_step = context.selected_step(workspace)
    report_step = next(
        (item for route in workspace["report"]["routes"]
         if route["id"] == workspace["route_id"]
         for item in route["steps"] if item["id"] == workspace["step_id"]), None)
    if source_step is None or report_step is None:
        return
    candidates = source_step["byproducts"].get("candidates") or []
    assessed = {item["input_smiles"]: item for item in report_step["materials"]
                if item["role"] == "byproducts"}
    concerning = sum(
        (assessed.get(raw) or {}).get("status") in
        ("known_concern", "structural_alert") for raw in candidates)
    ui_kit.metric_row([
        ("후보", len(candidates), "ASKCOS가 이 단계에서 생길 수 있다고 내놓은 "
                                "구조 수입니다. 실제로 생긴다는 뜻이 아닙니다."),
        ("우려", concerning, "그 후보 중 ICH M7 목록에 있거나 위험 구조가 "
                            "걸린 것의 수입니다."),
        ("못 읽음", sum((assessed.get(raw) or {}).get("status") == "invalid"
                      for raw in candidates),
         "구조를 읽지 못해 검사하지 못한 후보 수입니다."),
    ])
    if not candidates:
        st.caption("이 단계에 적용된 부산물 후보가 없습니다.")
        return
    ui_kit.show_table(
        ["구조", "이름", "판정"],
        [{"cells": [
            ui_kit.text(raw),
            ui_kit.text(material_name(engine.canonicalize_smiles(raw),
                                      (assessed.get(raw) or {}).get("name"))),
            _verdict_cell({"raw": raw, "is_target": False,
                           "assessment": assessed.get(raw)}),
        ], "mark": (assessed.get(raw) or {}).get("status") in
            ("known_concern", "structural_alert")}
         for raw in candidates])
    if concerning:
        st.caption("우려가 걸린 후보가 있어 이 경로는 **순위가 내려갑니다.** "
                   "목록에서 빠지지는 않습니다.")
