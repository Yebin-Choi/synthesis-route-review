"""Read-only Streamlit inspection of independently recollected condition evidence."""

import json
from pathlib import Path

import streamlit as st

from condition_evidence import DEFAULT_PATH, get_combinations, list_sources, load_evidence
from toxicology import assess_material


SCOPE_NOTE = "공개 원자료 독립 재수집 · 팀 KB 원본 복원 아님 · 목표 반응에 적용 검증 안 됨"
REACTION_TYPES = ("Buchwald-Hartwig", "Suzuki-Miyaura")
BASIS_LABELS = {
    "UV_area_percentage": "UV면적%",
    "reported_Output_percent_basis_unverified": "Output (측정법 미확인)",
}
ROLE_LABELS = {
    "base": "염기", "ligand": "리간드", "additive": "첨가제",
    "catalyst": "촉매", "solvent": "용매", "reagent": "시약",
    "aryl_halide": "아릴 할라이드", "reactant_1": "반응물 1", "reactant_2": "반응물 2",
}
COMPONENT_STATUS = {
    "resolved": "구조 해석됨",
    "unresolved": "구조 미해결",
    "missing": "원자료 누락",
    "absent": "원자료에 없음",
}
SCREEN_STATUS = {
    "known_concern": "알려진 우려 물질",
    "structural_alert": "구조 경보",
    "no_alert_detected": "해당 없음",
    "invalid": "구조 해석 불가",
}


def _value(value):
    return "미확인" if value is None or value == "" else str(value)


def _basis_label(basis):
    return BASIS_LABELS.get(basis, f"측정 기준 미확인 ({_value(basis)})")


def _component_list(value):
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("구성물 목록이 올바르지 않습니다.")
    return value


def _all_components(group):
    yield from _component_list(group.get("components"))
    contexts = group.get("substrate_contexts", [])
    if not isinstance(contexts, list):
        raise ValueError("기질 문맥 목록이 올바르지 않습니다.")
    for context in contexts:
        if not isinstance(context, dict):
            raise ValueError("기질 문맥이 올바르지 않습니다.")
        yield from _component_list(context.get("components"))


def _matches(group, query):
    components = list(_all_components(group))
    if not query:
        return True
    for component in components:
        fields = [component.get(field) for field in ("name", "shorthand", "input_smiles", "canonical_smiles")]
        variants = component.get("input_smiles_variants", [])
        if isinstance(variants, list):
            fields.extend(variants)
        if any(query in text.casefold() for text in fields if isinstance(text, str)):
            return True
    return False


def _has_structure(component):
    smiles = component.get("canonical_smiles")
    return isinstance(smiles, str) and bool(smiles.strip())


def _component_status(component):
    if component.get("status") == "resolved" and not _has_structure(component):
        return "구조 미확인"
    return COMPONENT_STATUS.get(component.get("status"), "상태 미확인")


def _component_rows(components):
    rows = []
    for component in _component_list(components):
        role = component.get("role")
        rows.append({
            "역할": ROLE_LABELS.get(role, _value(role)),
            "원자료 이름": component.get("name") or "이름 미제공",
            "약칭": _value(component.get("shorthand")),
            "구조 상태": _component_status(component),
            "정규화 SMILES": _value(component.get("canonical_smiles")),
            "원문 SMILES": _value(component.get("input_smiles")),
            "당량": _value(component.get("equivalents")),
            "mmol": _value(component.get("mmol")),
        })
    return rows


def _show_audit_summary(kb, sources, evidence_path):
    st.caption(f"불러온 출처 {len(sources):,}개 · 조건 조합 {len(kb['combinations']):,}개")
    audit_path = evidence_path.with_name("audit.json")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (not isinstance(audit, dict) or audit.get("schema") != kb["schema"]
                or audit.get("provenance") != kb.get("provenance")
                or audit.get("sources") != kb["sources"]
                or audit.get("summary") != kb.get("summary")):
            raise ValueError("현재 근거 파일과 감사 요약이 일치하지 않습니다.")
        summary = audit.get("summary")
        if not isinstance(summary, dict):
            raise ValueError("감사 집계가 없습니다.")
        counts = [summary.get(field) for field in ("records_retained", "records_quarantined")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("감사 기록 수가 올바르지 않습니다.")
    except FileNotFoundError:
        st.caption("감사 요약 파일 없음")
        return
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        st.warning(f"감사 요약을 확인하지 못했습니다: {exc}")
        return
    st.caption(f"생성 감사 요약 · 원자료 기록 {counts[0]:,}행 · 격리 {counts[1]:,}행")


def _show_screening(components):
    st.caption("정규화 SMILES가 있는 조건 구성물만 검사합니다")
    if not st.button(
        "구성물 검사",
        key="condition_evidence_screen",
        disabled=not any(_has_structure(component) for component in components),
    ):
        return
    rows = []
    for component in components:
        role = component.get("role")
        row = {
            "역할": ROLE_LABELS.get(role, _value(role)),
            "이름 / SMILES": component.get("name") or _value(component.get("canonical_smiles")),
            "선별 결과": _component_status(component),
            "구조 경보": "미평가",
        }
        if _has_structure(component):
            try:
                result = assess_material(component["canonical_smiles"])
                row["선별 결과"] = SCREEN_STATUS.get(result["status"], "미확인")
                row["구조 경보"] = ", ".join(result["alerts"]) or (
                    "해당 없음" if result["status"] == "no_alert_detected" else "미확인"
                )
            except (ValueError, RuntimeError) as exc:
                row["선별 결과"] = "선별 실패"
                st.warning(f"{row['역할']} 검사 실패: {exc}")
        rows.append(row)
    st.dataframe(rows, width="stretch", hide_index=True)


def _show_combination(kb, group):
    payload = {
        "scope": "observed_in_original_only_not_proposed_for_target",
        "notice": SCOPE_NOTE,
        "limitations": kb.get("limitations", []),
        "combination": group,
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    components = _component_list(group.get("components"))
    measurement = group["measurement"]
    record_ids = group.get("record_ids")
    if not isinstance(record_ids, list) or any(not isinstance(item, str) for item in record_ids):
        raise ValueError("원자료 행 ID 목록이 올바르지 않습니다.")
    st.caption(
        f"출처 {group['source_id']} · 측정 기준 {_basis_label(group['measurement_basis'])} · "
        "n은 원자료 행 수입니다"
    )
    columns = st.columns(3)
    columns[0].metric("원자료 기록 n", _value(group.get("n")))
    columns[1].metric("유효 측정 n", _value(measurement.get("n_valid")))
    columns[2].metric(
        f"측정 중앙값 · {_basis_label(group['measurement_basis'])}",
        _value(measurement.get("median")),
    )
    st.caption(
        f"원문 측정 열: {_value(measurement.get('field'))} · 단위: {_value(measurement.get('unit'))} · "
        f"누락 측정: {_value(measurement.get('n_missing'))}행"
    )
    st.markdown("**원자료 조건 구성물**")
    if components:
        st.dataframe(_component_rows(components), width="stretch", hide_index=True)
    else:
        st.warning("조건 구성물 정보가 없습니다")
    temperature = group.get("temperature")
    if isinstance(temperature, dict) and temperature.get("value") is not None:
        st.write(f"원자료 온도: {temperature['value']} · 단위 {_value(temperature.get('unit'))}")
    else:
        st.write("온도: 자료 없음")
    _show_screening(components)
    with st.expander("기질·누락 정보"):
        st.caption("원자료의 기질 조합입니다")
        contexts = group.get("substrate_contexts", [])
        if not contexts:
            st.write("기질 정보 미확인")
        for index, context in enumerate(contexts, 1):
            st.write(f"기질 묶음 {index} · 원자료 {_value(context.get('n'))}행")
            rows = _component_rows(context["components"])
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
        st.write("누락 필드별 원자료 행 수")
        st.json(group.get("missing_fields", {}), expanded=False)
    with st.expander("출처·행 ID·해시"):
        source = group["source"]
        st.write("출처 ID: " + source["id"])
        st.write("조합 ID: " + group["id"])
        st.write("수집 시트: " + source["selected_sheet"])
        st.write("원자료 URL: " + source["url"])
        st.write("원자료 SHA-256:")
        st.code(source["sha256"], language=None)
        st.caption("수집 당시의 원자료 해시입니다")
        if record_ids:
            st.dataframe([{"원자료 행 ID": item} for item in record_ids], width="stretch", hide_index=True)
        else:
            st.write("원자료 행 ID 미확인")
        for note in kb.get("limitations", []):
            st.write(note)
    st.download_button(
        "내려받기",
        data=json_text,
        file_name="selected_condition_evidence.json",
        mime="application/json",
        key="condition_evidence_download",
    )


def show_condition_evidence(path=None):
    """Show local original observations, never infer or apply target conditions."""
    st.subheader("조건 근거 자료")
    st.caption("공개 원자료 독립 재수집")
    evidence_path = Path(path) if path is not None else DEFAULT_PATH
    try:
        kb = load_evidence(evidence_path)
        sources = list_sources(kb)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        st.error(f"조건 근거 자료를 불러오지 못했습니다. 파일을 확인하세요: {evidence_path}\n{exc}")
        return
    _show_audit_summary(kb, sources, evidence_path)
    reaction_type = st.selectbox("반응 유형", REACTION_TYPES, key="condition_evidence_type")
    st.caption("반응 유형은 자료 필터입니다")
    query = st.text_input(
        "구성물 이름 또는 SMILES 검색",
        key="condition_evidence_search",
        help="이름·약칭·SMILES를 글자로 찾습니다.",
    ).strip().casefold()
    try:
        groups = get_combinations(kb, reaction_type, limit=len(kb["combinations"]))
        filtered = [group for group in groups if _matches(group, query)]
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"조건 근거 조합을 읽지 못했습니다. 자료 형식을 확인하세요: {exc}")
        return
    st.caption(f"검색 일치 {len(filtered):,} / {len(groups):,}개 · 출처·측정 기준·조합 ID 순")
    if not filtered:
        st.info("일치하는 조합이 없습니다")
        return
    by_id = {group["id"]: group for group in filtered}
    labels = {
        group["id"]: f"{group['source_id']} · {_basis_label(group['measurement_basis'])} · {index}번 조합"
        for index, group in enumerate(filtered, 1)
    }
    if st.session_state.get("condition_evidence_combination") not in by_id:
        st.session_state["condition_evidence_combination"] = next(iter(by_id))
    selected_id = st.selectbox(
        "원자료 조합 선택", list(by_id), format_func=labels.__getitem__, key="condition_evidence_combination"
    )
    try:
        _show_combination(kb, by_id[selected_id])
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"선택한 원자료 근거를 표시하지 못했습니다. 자료 형식을 확인하세요: {exc}")
