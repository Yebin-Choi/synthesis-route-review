"""Pure, research-only post-selection of an existing engine.analyze_routes report."""

import engine
from toxicology import canonicalize_smiles


_STATUSES = ("excluded", "needs_review", "insufficient_data", "candidate")
_MATERIAL_STATUSES = (
    "known_concern", "structural_alert", "invalid", "no_alert_detected",
)
_CANDIDATE_MESSAGE = "입력된 평가 범위에서 선별 기준을 통과한 후보; 안전 보장 아님"
# Every sentence the screen can put on a route, in one place. They are the
# software's own vocabulary and are read by people deciding what to do next,
# so they live here rather than inline where only half of them are visible.
_MESSAGES = {
    "no_materials":
        "평가할 비목표 물질이 없는 단계입니다. 목표물 제외 후 빈 평가 범위를 "
        "통과 후보로 간주하지 않습니다.",
    "invalid_material":
        "기재 물질의 구조를 해석할 수 없어 평가가 불완전합니다.",
    "prohibited_material":
        "사용자가 명시적으로 금지한 전체 구조 {matched}과 정확히 일치합니다.",
    "prohibited_component":
        "사용자가 명시적으로 금지한 성분 {matched}이 비연결 물질 {whole}에 "
        "포함됩니다. 전체 물질과 성분의 동일성을 가정하지 않습니다.",
    "byproduct_alert":
        "부산물 후보에서 알려진 우려 또는 경고구조가 걸렸습니다. 독성이나 "
        "실제 생성의 입증이 아닙니다.",
    "byproduct_demotion":
        "위 신호는 부산물 후보에서 나왔습니다. 예측된 후보이고 실제로 얼마나 "
        "생기는지, 정제로 없어지는지는 확인하지 않았으므로 경로를 빼지 않고 "
        "순위만 내립니다.",
    "alert_excluded":
        "'규칙에 걸린 경로도 제외'를 켜 두어 우려 근거가 있는 경로를 뺐습니다. "
        "독성이 있다거나 그 물질이 실제로 생긴다는 증명이 아닙니다.",
    "alert_review":
        "기재 물질의 알려진 우려 또는 경고구조에 대한 전문가 검토가 필요합니다. "
        "독성이나 실제 생성의 입증이 아닙니다.",
    # These two are appended to the warning's own sentence, which already
    # ends with "생성 확정이나 생성량 예측은 아닙니다." Saying it again put
    # the same caveat twice in one bullet, so each says only what was done.
    "warning_excluded": "'규칙에 걸린 경로도 제외'를 켜 두어 뺐습니다.",
    "warning_review": "검토 대상으로 남깁니다.",
    "conditions_not_observed":
        "조건 상태 {state}: 실제 관측 조건이 확인되지 않았습니다. 추천·제안 "
        "조건은 관측 조건이 아닙니다.",
    "byproducts_incomplete":
        "부산물 평가 상태 {state}: 평가 시도가 완료되지 않았습니다.",
}
_LIMITATIONS = (
    "주어진 경로 집합만 입력된 순서대로 추립니다. 경로를 새로 만들거나 안전 순위를 매기거나 점수를 계산하지 않습니다.",
    "사용자가 직접 지정한 구조만 금지 목록에 들어갑니다. 구조를 비교할 때 원자 번호만 떼어내고 전하·입체화학·동위원소·호변이성질체는 다른 구조로 둡니다.",
    "알려진 우려 물질·경고구조·조건부 경고는 검토할 근거이지 결론이 아닙니다. 이 신호로 경로를 빼는 것은 안전 쪽으로 기운 연구용 기준이며, 독성이 있다거나 그 물질이 실제로 생긴다는 증명이 아닙니다.",
    "제안된 반응 조건은 실제로 관측된 조건이 아닙니다. 입력에 적힌 근거 상태를 관측 결과로 올려 적지 않습니다.",
    "부산물 평가가 성공했다는 것은 평가를 끝까지 돌렸다는 뜻일 뿐입니다. 그 결과가 정확하다거나 생길 수 있는 부산물을 다 찾았다는 뜻이 아닙니다.",
    "물질의 실제 생성 여부·생성량·잔류량·노출량·정제 효과를 예측하지 않으며, 경보 미검출 또는 후보 판정은 안전 보장이 아닙니다.",
    "목표물 자체는 불순물 평가 대상이 아닙니다. 목표물이나 그 금지 성분을 피하는 경로를 이 보고서로 고를 수는 없습니다.",
)


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label}: 객체가 필요합니다.")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: 비어 있지 않은 문자열이 필요합니다.")
    return value


def _list(value, label, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{label}: {'비어 있지 않은 ' if nonempty else ''}배열이 필요합니다.")
    return value


def _state(value, supported, label):
    if not isinstance(value, str) or value not in supported:
        raise ValueError(f"{label}: 지원되는 상태 {', '.join(supported)}가 필요합니다.")
    return value


def _validate_report(report):
    _object(report, "report")
    if report.get("schema_version") != "1.0":
        raise ValueError("schema_version이 '1.0'인 analyze_routes 보고서가 필요합니다.")
    route_ids = set()
    for route in _list(report.get("routes"), "routes", nonempty=True):
        _object(route, "route")
        rid = _text(route.get("id"), "route.id")
        if rid.strip() in route_ids:
            raise ValueError(f"중복 route.id: {rid}")
        route_ids.add(rid.strip())
        target = canonicalize_smiles(_text(route.get("target"), f"{rid}.target"))
        if target is None:
            raise ValueError(f"{rid}.target: 유효한 목표물 구조가 필요합니다.")
        step_ids = set()
        for step in _list(route.get("steps"), f"{rid}.steps", nonempty=True):
            _object(step, f"{rid}.step")
            sid = _text(step.get("id"), f"{rid}.step.id")
            label = f"{rid}/{sid}"
            if sid.strip() in step_ids:
                raise ValueError(f"{rid}: 중복 step.id {sid}")
            step_ids.add(sid.strip())
            for material in _list(step.get("materials"), f"{label}.materials"):
                _object(material, f"{label}.material")
                status = _state(material.get("status"), _MATERIAL_STATUSES,
                                f"{label}.material.status")
                smiles = _text(material.get("input_smiles"), f"{label}.input_smiles")
                canonical = canonicalize_smiles(smiles)
                if "canonical_smiles" not in material or material["canonical_smiles"] != canonical:
                    raise ValueError(f"{label}: canonical_smiles가 input_smiles 정규화 결과와 일치해야 합니다.")
                if (status == "invalid") != (canonical is None):
                    raise ValueError(f"{label}: 물질 구조와 평가 상태가 일치하지 않습니다.")
                if canonical == target:
                    raise ValueError(f"{label}: 목표물 target은 analyze_routes의 불순물 materials에 포함되지 않습니다.")
                alerts = _list(material.get("alerts"), f"{label}.material.alerts")
                for alert in alerts:
                    _text(alert, f"{label}.material.alerts 항목")
                special = material.get("special_concern")
                if type(special) is not bool or "name" not in material:
                    raise ValueError(f"{label}: 물질 평가의 special_concern(bool)과 name이 필요합니다.")
                if status == "known_concern":
                    _text(material["name"], f"{label}.material.name")
                elif material["name"] is not None:
                    raise ValueError(f"{label}: 알려진 우려 이외 상태에 알려진 물질명을 혼합할 수 없습니다.")
                if ((status == "structural_alert" and not alerts)
                        or (status in ("invalid", "no_alert_detected") and (alerts or special))
                        or (special and not alerts)):
                    raise ValueError(f"{label}: 물질 평가 상태와 경보 정보가 일치하지 않습니다.")
            for warning in _list(step.get("conditional_warnings"), f"{label}.conditional_warnings"):
                _object(warning, f"{label}.conditional_warning")
                _text(warning.get("code"), f"{label}.conditional_warning.code")
                _text(warning.get("message"), f"{label}.conditional_warning.message")
            for missing in _list(step.get("missing_information"), f"{label}.missing_information"):
                _text(missing, f"{label}.missing_information 항목")
            condition = _object(step.get("conditions"), f"{label}.conditions")
            state = _state(condition.get("status"), ("observed", "proposed", "unknown"),
                           f"{label}.conditions.status")
            if state != "unknown":
                _text(condition.get("source"), f"{label}.conditions.source")
                _text(condition.get("description"), f"{label}.conditions.description")
            byproducts = _object(step.get("byproducts"), f"{label}.byproducts")
            state = _state(byproducts.get("status"), ("success", "not_run", "failed"),
                           f"{label}.byproducts.status")
            candidates = _list(byproducts.get("candidates"), f"{label}.byproducts.candidates")
            for candidate in candidates:
                _text(candidate, f"{label}.byproducts.candidates 항목")
            if state == "success":
                _text(byproducts.get("source"), f"{label}.byproducts.source")
            elif candidates:
                raise ValueError(f"{label}: 미실행·실패 상태에 부산물 성공 후보를 혼합할 수 없습니다.")


def _normalize_exclusions(excluded_smiles):
    if not isinstance(excluded_smiles, (list, tuple, set, frozenset)):
        raise ValueError("excluded_smiles는 SMILES 문자열의 list/tuple/set/frozenset이어야 합니다.")
    normalized = []
    for smiles in excluded_smiles:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            raise ValueError(f"excluded_smiles: 비어 있지 않은 유효한 SMILES가 필요합니다: {smiles!r}")
        if canonical not in normalized:
            normalized.append(canonical)
    return sorted(normalized) if isinstance(excluded_smiles, (set, frozenset)) else normalized


def _prohibited_matches(canonical, exclusions):
    matches = [(canonical, False)] if canonical in exclusions else []
    if "." in canonical:
        components = {canonicalize_smiles(component) for component in canonical.split(".")}
        matches.extend((excluded, True) for excluded in exclusions
                       if "." not in excluded and excluded in components)
    return matches


def _review_material(material, exclusions, strict_alerts, reason):
    """What one material contributes: 'excluded', 'review' or 'incomplete'."""
    canonical = material["canonical_smiles"]
    status = material["status"]
    if status == "invalid":
        reason("invalid_material", _MESSAGES["invalid_material"], material["input_smiles"])
        return {"incomplete"}

    found = set()
    for matched, component in _prohibited_matches(canonical, exclusions):
        found.add("excluded")
        if component:
            reason("prohibited_component",
                   _MESSAGES["prohibited_component"].format(matched=matched,
                                                            whole=canonical), canonical)
        else:
            reason("prohibited_material",
                   _MESSAGES["prohibited_material"].format(matched=matched), canonical)
    if status not in ("known_concern", "structural_alert"):
        return found

    if material.get("role") == "byproducts":
        # A byproduct candidate demotes a route; it never removes one, even
        # under strict alerts. Measured 2026-11-04 on a live propranolol
        # search: one step's ASKCOS prediction returned 11 candidates, 7 of
        # them carrying a structural alert, and applying them as exclusions
        # took the surviving routes from 1 of 10 to 0 of 10. A reviewer left
        # with no routes has not been given a safety finding, only an empty
        # page. The asymmetry is in what the two kinds of material are: a
        # reagent is charged into the reaction, while a byproduct candidate
        # is a prediction whose amount and removal are both unknown here.
        #
        # Two rows, because they say different things: what the screen found,
        # and what was done about it. The finding keeps its own code so the
        # evidence reads as it would for any other material.
        found.add("review")
        reason(status, _MESSAGES["byproduct_alert"], canonical)
        reason("byproduct_concern", _MESSAGES["byproduct_demotion"], canonical)
    elif strict_alerts:
        found.add("excluded")
        reason(status, _MESSAGES["alert_excluded"], canonical)
    else:
        found.add("review")
        reason(status, _MESSAGES["alert_review"], canonical)
    return found


def _review_step(step, exclusions, strict_alerts, reason):
    """What one step contributes, as the set of findings it raised."""
    found = set()
    if not step["materials"]:
        found.add("incomplete")
        reason("no_materials", _MESSAGES["no_materials"])
    for material in step["materials"]:
        found |= _review_material(material, exclusions, strict_alerts, reason)
    for warning in step["conditional_warnings"]:
        found.add("excluded" if strict_alerts else "review")
        # The warning's own code stays in the report's conditional_warnings
        # entry, so it does not also sit at the front of the shown sentence.
        decision = _MESSAGES["warning_excluded" if strict_alerts else "warning_review"]
        reason("conditional_warning", f"{warning['message']} {decision}")
    # These three are recorded but do not decide the route's status, because
    # they are the same for every route in the report. Measured 2026-11-08 on
    # ten generated aspirin routes: conditions were unobserved on 10 steps of
    # 10 and byproducts unrun on 10 of 10, so "정보 부족" landed on 8 routes
    # for a reason that was equally true of the other 2. A label that every
    # route carries separates nothing, and it was hiding the two routes that
    # did have a structural alert. The gaps stay in the reasons, in the line
    # under the route chip and in the limitations; what they no longer do is
    # stand in for a finding about this route.
    for missing in step["missing_information"]:
        # The two lines about conditions and byproducts are reported again
        # below under their own codes, which word them better and carry the
        # state. Reporting both made the screen say the same thing twice,
        # once as "빠진 항목" and once by name.
        if (missing == engine.CONDITIONS_UNOBSERVED
                or missing.startswith(engine.BYPRODUCTS_INCOMPLETE)):
            continue
        reason("missing_information", missing)
    if step["conditions"]["status"] != "observed":
        reason("conditions_not_observed",
               _MESSAGES["conditions_not_observed"].format(state=step["conditions"]["status"]))
    if step["byproducts"]["status"] != "success":
        reason("byproducts_incomplete",
               _MESSAGES["byproducts_incomplete"].format(state=step["byproducts"]["status"]))
    return found


def select_routes(report, *, excluded_smiles=(), strict_alerts=False) -> dict:
    """Classify every supplied route, without regeneration, ranking or mutation.

    Consume the version 1.0 output of engine.analyze_routes, not route input JSON.
    All routes/steps and their material, warning, missing-information, condition
    and byproduct fields are required; malformed input raises ValueError.
    Policy structures are canonicalized and deduplicated. Only full identities
    or an explicitly prohibited single disconnected component match.

    Precedence is excluded > needs_review > insufficient_data > candidate;
    reasons are cumulative regardless of precedence. Every reason has code,
    message, step_id and smiles; route/step-wide reasons use None where needed.
    Returned routes preserve input IDs/order and are never replaced or dropped.
    """
    if type(strict_alerts) is not bool:
        raise ValueError("strict_alerts는 bool이어야 합니다. 문자열·숫자는 허용하지 않습니다.")
    _validate_report(report)
    exclusions = _normalize_exclusions(excluded_smiles)
    for route in report["routes"]:
        target = canonicalize_smiles(route["target"])
        matches = _prohibited_matches(target, exclusions)
        if matches:
            matched, component = matches[0]
            detail = f"성분 {matched}" if component else matched
            raise ValueError(
                f"{route['id']}: 명시적으로 금지한 {detail}이 목표물 target {target}과 일치합니다. "
                "목표물은 불순물 평가에서 제외되므로 동일 목표물을 피하는 경로 선별은 지원하지 않습니다."
            )

    selected = []
    counts = dict.fromkeys(_STATUSES, 0)
    for route in report["routes"]:
        reasons = []
        found = set()
        for step in route["steps"]:

            def reason(code, message, smiles=None, sid=step["id"]):
                reasons.append({"code": code, "message": message,
                                "step_id": sid, "smiles": smiles})

            found |= _review_step(step, exclusions, strict_alerts, reason)
        status = ("excluded" if "excluded" in found else
                  "needs_review" if "review" in found else
                  "insufficient_data" if "incomplete" in found else "candidate")
        if status == "candidate":
            reasons.append({"code": "candidate_within_scope", "message": _CANDIDATE_MESSAGE,
                            "step_id": None, "smiles": None})
        selected.append({"id": route["id"], "status": status, "reasons": reasons})
        counts[status] += 1
    return {"routes": selected, "counts": counts,
            "policy": {"excluded_smiles": exclusions, "strict_alerts": strict_alerts},
            "limitations": list(_LIMITATIONS)}
