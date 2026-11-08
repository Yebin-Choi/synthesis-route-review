"""Explicit, session-local AMES inference alongside the unchanged rule baseline.

The screen used to print the backend's own record: English status codes, an
`endpoint=ames · applicability=unknown` caption and the internal report path
`routes / steps / materials`. A reader had to know the code to read the
result. Verdicts are now chips with Korean words, the provenance is a
label/value grid in the order QMRF asks for, and the raw record moved into a
collapsed block where it is still reachable and no longer in the way.

Every scoring tool here runs at 0.5: ADMET-AI's own line, and now ours too,
because the in-house models carry the collection's imbalance in their training
weights rather than in the line (benchmark log 6-20). It is still not a
calibrated probability.
"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from uuid import UUID, uuid4

import streamlit as st

import ui_kit
from presentation import ROLES, limitation_text, reason_text
from admet_service import admet_readiness, assess_admet_materials
from ames_route_review import review_ames_routes
from route_selection import select_routes
from workspace import report_hash


_SESSION_KEY = "route_ames_review_evidence"
_SCHEMA_VERSION = 1
_THRESHOLD = 0.5
_SOURCE = "local_ADMET-AI_model_inference"
_READINESS_LABELS = {
    "interpreter": "Python 실행 파일",
    "worker": "AMES 작업 스크립트",
    "package": "ADMET-AI 패키지",
    "weights": "로컬 .pt 가중치",
}
STATUS_LABELS = {
    "excluded": "연구 정책으로 제외",
    "needs_review": "전문가 검토 필요",
    "insufficient_data": "정보 부족",
    "candidate": "후보",
}
# Which chip a route verdict gets. "정보 부족" and "전문가 검토 필요" are not
# clearances, so neither takes the colour a candidate route gets.
STATUS_KINDS = {
    "excluded": ui_kit.CONCERN,
    "needs_review": ui_kit.REVIEW,
    "insufficient_data": ui_kit.UNKNOWN,
    "candidate": ui_kit.CLEAR,
}
SCOPE_LABELS = {
    "materials": "경로 자료에 적힌 물질 가운데 목표물이 아닌 구조만 검사",
    "target": "목표물 자체는 불순물 평가에서 제외되며 이 교차 검사 대상이 아님",
    "unknown_byproducts": "구조가 알려지지 않은 부산물은 미평가",
    "coverage": "기재된 모든 비목표 구조가 범위이며 모든 가능한 생성물을 포괄하지 않음",
    "source": "로컬 실제 ADMET-AI 모델 추론 · benchmark CSV 조회가 아님 · 외부 API 호출 없음",
    "threshold": "0.5는 고정 기준선이며 최적 임계값이나 보정된 독성 확률이 아님",
    "policy": "이 절은 동일 물질의 규칙 우려 AND AMES 양성 비교입니다. 기본 경로 제시의 규칙 OR 현재 모델 양성 제외와 별개입니다.",
}


def _input_smiles(report):
    return list(dict.fromkeys(
        material["input_smiles"]
        for route in report["routes"] for step in route["steps"] for material in step["materials"]
    ))


def _evidence_metadata(report_sha256, report_generation, result):
    return {
        "evaluation_id": str(uuid4()),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_sha256": report_sha256,
        "report_generation": report_generation,
        "source": _SOURCE,
        "model": deepcopy(result["model"]),
        "threshold": result["threshold"],
    }


def ensure_route_ames_evidence(report, *, report_generation):
    """Run the local model once per report generation, without a button.

    Inference is local and offline, so it needs no transfer consent. A stored
    entry, including a stored failure, is never re-run for the same report.
    """
    try:
        UUID(report_generation)
    except (ValueError, TypeError, AttributeError):
        return
    report_sha256 = report_hash(report)
    entry = st.session_state.get(_SESSION_KEY)
    if (isinstance(entry, dict) and entry.get("schema_version") == _SCHEMA_VERSION
            and entry.get("report_generation") == report_generation
            and entry.get("report_sha256") == report_sha256):
        return
    if not admet_readiness()["ready"]:
        return
    smiles = _input_smiles(report)
    with st.spinner("AMES 모델 검사 중"):
        try:
            result = deepcopy(assess_admet_materials(smiles, threshold=_THRESHOLD))
            entry = {
                "schema_version": _SCHEMA_VERSION,
                "report_sha256": report_sha256,
                "report_generation": report_generation,
                "metadata": _evidence_metadata(report_sha256, report_generation, result),
                "model_result": result,
                "error": None,
            }
        except Exception as exc:
            entry = {
                "schema_version": _SCHEMA_VERSION,
                "report_sha256": report_sha256, "report_generation": report_generation,
                "metadata": None, "model_result": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    st.session_state[_SESSION_KEY] = entry


def get_route_ames_evidence(report, *, report_generation, excluded_smiles=()):
    """Copy only this exact report generation's envelope, including failures.

    Report hashing is shared with workspace. Matching material membership or
    an identical regenerated report alone never makes old model data current.
    combined_review is explicitly the separate same-material AND comparison.
    """
    try:
        UUID(report_generation)
    except (ValueError, TypeError, AttributeError):
        return None
    entry = st.session_state.get(_SESSION_KEY)
    if (not isinstance(entry, dict) or entry.get("schema_version") != _SCHEMA_VERSION
            or entry.get("report_generation") != report_generation
            or entry.get("report_sha256") != report_hash(report)):
        return None
    snapshot = deepcopy(entry)
    snapshot["combined_review"] = None
    snapshot["combined_review_policy"] = "same_material_AND"
    snapshot["review_error"] = None
    if snapshot.get("model_result") is not None:
        try:
            snapshot["combined_review"] = review_ames_routes(
                report, snapshot["model_result"], excluded_smiles=excluded_smiles,
            )
        except (ValueError, TypeError, KeyError) as exc:
            snapshot["review_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def _material_verdict(row):
    """One structure's call as a chip. An abstention carries why, never a word
    that could be read as a clearance."""
    if row["status"] != "success" or row["score"] is None:
        return ui_kit.verdict_chip("indeterminate",
                                   note=reason_text(row.get("error")))
    name = "positive" if row["prediction"] == "positive" else "negative"
    return ui_kit.verdict_chip(name, aside=f"{row['score']:.2f}")


def _material_places(report):
    """Where each listed structure shows up: which route, which step, what role.

    `_input_smiles` flattens the whole report, so the table printed every
    route's materials while the reader was looking at one route. This index
    walks the report the same way, so the two cannot drift apart.
    """
    places = {}
    for number, route in enumerate(report["routes"], 1):
        for order, step in enumerate(route["steps"], 1):
            for material in step["materials"]:
                entry = places.setdefault(material["input_smiles"],
                                          {"roles": [], "steps": []})
                role = ROLES.get(material["role"], material["role"])
                if role not in entry["roles"]:
                    entry["roles"].append(role)
                place = (route["id"], number, order)
                if place not in entry["steps"]:
                    entry["steps"].append(place)
    return places


def _row_places(row, places):
    """One assessment's roles and step positions, across its input aliases.

    A row can stand for several written inputs of the same structure, so
    every alias is looked up and the positions are merged.
    """
    aliases = row.get("input_smiles_list") or [row["input_smiles"]]
    roles, steps = [], []
    for smiles in aliases:
        entry = places.get(smiles)
        if entry is None:
            continue
        for role in entry["roles"]:
            if role not in roles:
                roles.append(role)
        for place in entry["steps"]:
            if place not in steps:
                steps.append(place)
    return roles, steps


def _step_text(steps, route_id):
    """Which step the structure appears at, as the reader counts steps.

    Inside one route only the step numbers are needed. Across the whole
    report the route has to be named too, or two routes' steps read as one.
    """
    if route_id is None:
        return " / ".join(f"경로 {number} {order}단계" for _, number, order in steps)
    orders = sorted({order for member, _, order in steps if member == route_id})
    return "·".join(str(order) for order in orders) + "단계" if orders else ""


def _show_provenance(result, metadata):
    """Where the numbers came from, in the field order QMRF asks for.

    QMRF v1.2 puts the endpoint, then the algorithm and software, then the
    applicability domain, then the measured performance. A field we cannot
    fill says so instead of being left out, which is what the form asks for
    too. https://www.qsardb.org/guidelines/qmrf/QMRF_v1.2-2008-05-26.pdf
    """
    model = result["model"]
    ui_kit.show_facts([
        ("판정 대상", "Ames 변이원성"),
        ("도구", ui_kit.text(f"{model['tool']} (이 컴퓨터에서 실행)")),
        ("버전", ui_kit.text(model["version"] or "미확인")),
        ("판정 기준선", f"{result['threshold']:g} · 이 도구의 측정된 균형점"),
        ("적용 범위 평가", "이 검사에서는 하지 않음"),
        ("가중치 해시", "있음" if model["weights_sha256"] else "없음"),
    ])
    if not model["weights_sha256"]:
        st.warning("모델 가중치 해시가 없습니다. 어떤 가중치로 낸 결과인지 "
                   "확인할 수 없습니다.")


def _show_result(result, review, metadata, *, report, route_id=None):
    _show_provenance(result, metadata)

    # The reader picked one route upstairs, so that route is what the table
    # shows. The whole report stays one click away.
    places = _material_places(report)
    rows = [(row, *_row_places(row, places)) for row in result["assessments"]]
    scope = route_id
    number = next((index for index, route in enumerate(report["routes"], 1)
                   if route["id"] == route_id), None)
    heading = "물질별 검사 결과" if number is None else f"경로 {number} · 물질별 검사 결과"
    st.markdown(f"**{heading}**")
    if route_id is not None and len(report["routes"]) > 1:
        if st.radio("보기 범위", ("선택한 경로", "전체 경로 보기"), horizontal=True,
                    key="route_ames_scope",
                    label_visibility="collapsed") == "전체 경로 보기":
            scope = None
    if scope is not None:
        rows = [item for item in rows
                if any(member == scope for member, _, _ in item[2])]
    assessments = [item[0] for item in rows]
    decided = [row for row in assessments
               if row["status"] == "success" and row["score"] is not None]
    positive = sum(row["prediction"] == "positive" for row in decided)
    ui_kit.metric_row([
        ("검사한 구조", len(assessments)),
        ("양성", positive),
        ("음성", len(decided) - positive),
        ("예측 결과 없음", len(assessments) - len(decided),
         "도구가 기권했거나 구조를 읽지 못한 경우입니다. 음성이 아닙니다."),
    ])
    if not decided:
        st.warning("점수가 나온 구조가 없습니다. 아래는 모두 예측 결과 없음이며, "
                   "음성으로 읽어서는 안 됩니다.")
    elif len(decided) != len(assessments):
        st.warning("일부 구조는 판정하지 못했습니다. 예측 결과 없음은 음성이 아닙니다.")

    if assessments:
        # The merged-alias column is gone. It repeated the structure column
        # for all but a few rows, and the full alias list is still in the
        # raw record below. What is here instead is where the structure sits
        # in the route: what it is used as, and at which step.
        ui_kit.show_table(
            ["구조(SMILES)", "물질 구분", "등장 단계", "모델 판정"],
            [{"cells": [ui_kit.text(row["canonical_smiles"] or row["input_smiles"]),
                        ui_kit.text(" / ".join(roles)),
                        ui_kit.text(_step_text(steps, scope)),
                        _material_verdict(row)],
              "mark": row["status"] != "success"}
             for row, roles, steps in rows])
    else:
        st.info("모델 결과가 없습니다")

    st.subheader("경로별 판정")
    ui_kit.metric_row([
        (STATUS_LABELS[status], review["counts"][status]) for status in STATUS_LABELS
    ])
    ui_kit.show_table(
        ["경로", "경로 판정", "사유"],
        [{"cells": [ui_kit.text(route["id"]),
                    ui_kit.chip(STATUS_LABELS[route["status"]],
                                STATUS_KINDS[route["status"]]),
                    f"{len(route['reasons'])}건"],
          "mark": route["status"] == "excluded"}
         for route in review["routes"]])
    reasons = [
        {"경로": route["id"], "경로 판정": STATUS_LABELS[route["status"]],
         "단계": reason["step_id"], "물질": reason["smiles"],
         "사유": reason["message"]}
        for route in review["routes"] for reason in route["reasons"]
    ]
    if reasons:
        # The reason is the only column worth reading here and it was the one
        # being cut off: stretching the table shares the width evenly, so a
        # long sentence lost its end while the route id kept room it did not
        # need. Fixed widths let the reason run long and the table scroll
        # sideways to reach it.
        st.dataframe(reasons, hide_index=True, width="stretch", column_config={
            "경로": st.column_config.TextColumn("경로", width="small"),
            "경로 판정": st.column_config.TextColumn("경로 판정", width="small"),
            "단계": st.column_config.TextColumn("단계", width="small"),
            "물질": st.column_config.TextColumn("물질", width="medium"),
            "사유": st.column_config.TextColumn("사유", width="large"),
        })
    # The internal codes behind those reasons, the evaluation identity and the
    # model hashes stay reachable here instead of being printed onto the page.
    ui_kit.raw_record({
        "evidence_metadata": metadata,
        "reason_codes": [
            {"route": route["id"], "status": route["status"],
             "step": reason["step_id"], "smiles": reason["smiles"],
             "code": reason["code"]}
            for route in review["routes"] for reason in route["reasons"]],
        "assessments": assessments,
    }, title="원본 기록 (평가 식별자·모델 해시·사유 코드)")


def show_route_ames_review(report, *, report_generation, excluded_smiles=(),
                           route_id=None):
    """Infer only on a button click, bound to the supplied workspace generation.

    Policy changes reuse frozen results. A new generation, even with identical
    report contents, cannot reuse a previous generation's model evidence.
    """
    ui_kit.use()
    st.subheader("ADMET-AI 교차 검사")
    st.caption("기재된 비목표 구조만 검사합니다")
    scope_details = st.expander("검사 범위와 해석 한계")
    with scope_details:
        # One bulleted block rather than ten stacked sentences: the same
        # statements, read as one record instead of ten remarks.
        st.markdown("\n".join(f"- {line}" for line in [
            *SCOPE_LABELS.values(),
            "결과는 현재 세션에만 한 번 보관합니다.",
            "같은 정규화 구조의 입력 별칭은 한 평가로 합쳐질 수 있습니다.",
            "모든 경로를 입력 순서대로 유지하며 모든 사유를 표시합니다.",
        ]))
    try:
        UUID(report_generation)
        select_routes(report)
        report_sha256 = report_hash(report)
        smiles = _input_smiles(report)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        st.error(f"ADMET 교차 검사 입력 보고서를 확인하지 못했습니다: {exc}")
        return

    entry = st.session_state.get(_SESSION_KEY)
    if entry is not None and (
        entry.get("schema_version") != _SCHEMA_VERSION
        or entry.get("report_generation") != report_generation
        or entry.get("report_sha256") != report_sha256
    ):
        st.session_state.pop(_SESSION_KEY, None)
        entry = None
    readiness = admet_readiness()
    if not readiness["ready"]:
        missing = " · ".join(_READINESS_LABELS.get(asset, asset) for asset in readiness["missing"])
        st.warning(f"검사 준비 필요: {missing}")
    if entry is None:
        # Before a run there is no result at all, and saying so plainly keeps
        # an empty screen from being read as an all-clear.
        st.caption(f"검사할 구조 {len(smiles)}개 · 파일 확인만 한 상태입니다. "
                   "아직 모델을 돌리지 않았습니다.")
    else:
        st.caption(f"검사할 구조 {len(smiles)}개")
    if st.button(
        "모델 검사 다시" if entry is not None else "모델 검사",
        key=f"route_ames_run_{report_generation}_{report_sha256}",
        type="primary",
        disabled=not readiness["ready"],
    ) and readiness["ready"]:
        previous = entry
        with st.spinner("모델 실행 중"):
            try:
                result = deepcopy(assess_admet_materials(smiles, threshold=_THRESHOLD))
                entry = {
                    "schema_version": _SCHEMA_VERSION,
                    "report_sha256": report_sha256,
                    "report_generation": report_generation,
                    "metadata": _evidence_metadata(report_sha256, report_generation, result),
                    "model_result": result,
                    "error": None,
                }
            except Exception as exc:
                entry = dict(previous) if previous is not None else {
                    "schema_version": _SCHEMA_VERSION,
                    "report_sha256": report_sha256, "report_generation": report_generation,
                    "metadata": None, "model_result": None,
                }
                entry["error"] = f"{type(exc).__name__}: {exc}"
        st.session_state[_SESSION_KEY] = entry
        st.rerun()

    if entry is None:
        st.info("아직 검사하지 않았습니다")
        return
    if entry["error"] is not None:
        st.error(f"모델 검사 실패: {entry['error']}")
        if entry["model_result"] is not None:
            st.warning("아래는 이전 실행 결과입니다")
    result = entry["model_result"]
    if result is None:
        return
    try:
        review = review_ames_routes(report, result, excluded_smiles=excluded_smiles)
        snapshot = {
            "schema_version": _SCHEMA_VERSION,
            "report_sha256": report_sha256,
            "report_generation": report_generation,
            "original_report": report,
            "model_result": result,
            "combined_review": review,
            "combined_review_policy": "same_material_AND",
            "evidence_metadata": entry["metadata"],
            "latest_attempt_error": entry["error"],
            "scope_labels": SCOPE_LABELS,
            "policy_status_labels": STATUS_LABELS,
        }
        json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    except Exception as exc:
        st.error(f"모델 근거를 확인하지 못했습니다: {exc}")
        return
    with scope_details:
        st.json(review["policy"], expanded=False)
        for note in review["limitations"]:
            st.write(limitation_text(note))
    _show_result(result, review, entry["metadata"], report=report, route_id=route_id)
    return snapshot
