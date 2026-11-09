"""Research-only Ames review of an existing, unchanged route-analysis report."""

from copy import deepcopy
import math
from numbers import Real

from admet_service import validate_admet_model
from route_selection import select_routes
from screening_judges import CALL_ABSTAIN, CALL_NEGATIVE, CALL_POSITIVE
from screening_policy import combine, evidence_weight, is_ranking_active, validate_policy
from toxicology import canonicalize_smiles


_STATUSES = ("excluded", "needs_review", "insufficient_data", "candidate")
_RULE_POSITIVE = ("known_concern", "structural_alert")
# Reason codes select_routes emits, grouped by the route status each implies.
# An unrecognized baseline code is an error, never a silently ignored one.
_PROHIBITED_CODES = ("prohibited_material", "prohibited_component")
_INCOMPLETE_CODES = ("no_materials", "invalid_material")
# Recorded, but they do not make a route "정보 부족". Every generated route
# has unobserved conditions and unrun byproducts, so a status built on them
# is the same on every route and separates none of them. route_selection
# stopped using them on 2026-11-08 (benchmark log 6-67); this screen was
# still using them, which is why the model-evidence table read 정보 부족 or
# 전문가 검토 필요 on every row.
_RECORDED_ONLY_CODES = ("missing_information", "conditions_not_observed",
                        "byproducts_incomplete")
# A concern raised by a byproduct candidate rather than by a material that is
# charged into the reaction. It carries the same penalty as any other concern,
# so the route sinks in the ranking, but it never removes a route: see the
# note in route_selection, and the 2026-11-04 measurement where applying
# byproducts as exclusions left zero routes of ten.
_DEMOTE_ONLY_CODES = ("byproduct_concern",)
_BASELINE_CODES = frozenset(
    _PROHIBITED_CODES + _INCOMPLETE_CODES + _RECORDED_ONLY_CODES + _RULE_POSITIVE
    + _DEMOTE_ONLY_CODES + ("conditional_warning", "candidate_within_scope")
)
_ASSESSMENT_FIELDS = {
    "canonical_smiles", "input_smiles", "status", "score", "prediction",
    "applicability", "error",
}
_LIMITATIONS = (
    "동일 비목표 물질의 규칙 우려 AND ADMET-AI Ames 양성일 때만 공동 연구 정책으로 제외합니다. "
    "명시적 금지 정책은 별도로 유지하며, 제외는 입증된 독성이나 실제 생성의 증거가 아닙니다.",
    "구조가 특정되지 않은 조건부 경고를 가상의 Ames 평가 구조로 만들지 않습니다. "
    "조건부 경고와 한쪽만 양성인 물질은 전문가 검토 대상입니다.",
    "ADMET-AI는 이 물질이 자기 적용 범위 안에 있는지 알려주지 않습니다. 적용 범위가 "
    "확인됐다거나 학습 자료와 겹치지 않는다고 보지 않습니다.",
    "Ames 모델 점수는 보정된 독성 확률이 아닙니다. 음성 예측이나 규칙 경보 미검출은 "
    "안전성, ICH M7 Class 5 또는 규제 적합성 판정이 아닙니다.",
    "누락·실패·유효하지 않은 Ames 평가는 미평가(abstain)이며 음성으로 대체하지 않습니다. "
    "후보는 입력된 정보와 제한적 평가 범위 내에서만 의미가 있습니다.",
)
_ELIGIBILITY_LIMITATIONS = (
    "기본 제시는 규칙 우려/조건부 경고 OR 유효한 현재 Ames 양성의 합집합으로 제외합니다. "
    "모델 음성은 규칙 우려를 덮지 않으며, 동일 물질 AND 비교와 별도 정책입니다.",
    *_LIMITATIONS[2:],
)


def _unit_number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number between zero and one")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number between zero and one") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label} must be a finite number between zero and one")
    return number


def _validated_assessments(report, admet_result):
    required = {"model", "threshold", "assessments", "limitations"}
    if not isinstance(admet_result, dict) or not required <= admet_result.keys():
        raise ValueError("admet_result requires model, threshold, assessments and limitations")
    model = admet_result["model"]
    if not isinstance(model, dict) or not {"tool", "version", "endpoint", "weights_sha256"} <= model.keys():
        raise ValueError("admet_result.model requires tool, version, endpoint and weights_sha256")
    model = validate_admet_model(model)
    threshold = _unit_number(admet_result["threshold"], "admet_result.threshold")
    limitations = admet_result["limitations"]
    if (not isinstance(limitations, list)
            or any(not isinstance(item, str) or not item.strip() for item in limitations)):
        raise ValueError("admet_result.limitations must be an array of non-empty strings")
    rows = admet_result["assessments"]
    if not isinstance(rows, list):
        raise ValueError("admet_result.assessments must be an array")

    identities = {
        (material["canonical_smiles"],
         None if material["canonical_smiles"] is not None else material["input_smiles"])
        for route in report["routes"] for step in route["steps"] for material in step["materials"]
    }
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not _ASSESSMENT_FIELDS <= row.keys():
            raise ValueError("Each ADMET assessment requires all assessment fields")
        smiles = row["input_smiles"]
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("ADMET assessment input_smiles must be a non-empty string")
        canonical = canonicalize_smiles(smiles)
        if row["canonical_smiles"] != canonical:
            raise ValueError("ADMET canonical_smiles must match canonicalized input_smiles")
        identity = (canonical, None if canonical is not None else smiles)
        if identity not in identities:
            raise ValueError("ADMET assessment structure is not present in report materials (stale evidence)")
        if identity in indexed:
            raise ValueError("Duplicate ADMET assessment identity is ambiguous")
        if "input_smiles_list" in row:
            aliases = row["input_smiles_list"]
            if (not isinstance(aliases, list) or not aliases
                    or any(not isinstance(alias, str) or canonicalize_smiles(alias) != canonical
                           or (canonical is None and alias != smiles) for alias in aliases)):
                raise ValueError("ADMET input_smiles_list must contain aliases of the same exact structure")
        status = row["status"]
        if status not in ("success", "failed", "invalid"):
            raise ValueError("ADMET status must be success, failed or invalid")
        if (status == "invalid") != (canonical is None):
            raise ValueError("ADMET status and structure validity do not match")
        if row["applicability"] != "unknown":
            raise ValueError("ADMET applicability must remain unknown")
        error = row["error"]
        if error is not None and not isinstance(error, str):
            raise ValueError("ADMET error must be a string or None")
        score = None
        if status == "success":
            if not model["weights_sha256"]:
                raise ValueError("Successful ADMET assessments require model weight provenance")
            score = _unit_number(row["score"], "ADMET score")
            expected = "positive" if score >= threshold else "negative"
            if row["prediction"] != expected:
                raise ValueError("ADMET prediction does not match score and threshold")
            if error is not None:
                raise ValueError("Successful ADMET assessments must not contain an error")
        elif row["score"] is not None or row["prediction"] != "abstain":
            raise ValueError("Failed or invalid ADMET assessments must have score=None and prediction=abstain")
        indexed[identity] = {"status": status, "score": score, "error": error}
    return indexed, model, threshold


def _rule_call(material):
    """The rule screen's own call for one material, from its report status."""
    if material["status"] in _RULE_POSITIVE:
        return CALL_POSITIVE
    if material["status"] == "no_alert_detected":
        return CALL_NEGATIVE
    return CALL_ABSTAIN


_BYPRODUCT_DEMOTION = (
    "위 판정은 부산물 후보에서 나왔습니다. 예측된 후보이고 실제 생성량과 정제 제거 "
    "여부를 확인하지 않았으므로 경로를 빼지 않고 순위만 내립니다.")
_CANDIDATE_REASON = {
    "code": "candidate_within_scope",
    "message": "입력된 비목표 물질에서 규칙 우려·Ames 양성이 없고 기재 정보가 갖춰진 "
               "제한적 평가 범위의 후보입니다. 안전 보장이나 규제 적합성 판정이 아닙니다.",
    "step_id": None, "smiles": None,
}


def _rule_participates(policy, severity, *, strict_alerts, union):
    """Whether the rule judge takes part, as a question about one status.

    The alert checkbox relaxes how strictly structural alerts are treated.
    Being on the ICH M7 list is a prohibition rather than an alert, so the
    checkbox does not drop it from the combination; unchecking the box used to
    turn a listed impurity into a demotion and let the route through. The AND
    comparison view runs without the rule judge on purpose.
    """
    always = "rules" in policy["tools"] and strict_alerts

    def participates(status):
        if always or not union or "rules" not in policy["tools"]:
            return always
        return severity.get(status) == "exclude"

    return participates


def _baseline_review(selected, route, policy, severity, *, strict_alerts, participates):
    """What the rule screen already found, as statuses, penalty and reasons."""
    statuses, penalty, reasons = set(), 0.0, []
    total = len(route["steps"])
    positions = {step["id"]: index for index, step in enumerate(route["steps"])}
    for baseline in selected["reasons"]:
        code = baseline["code"]
        if code == "candidate_within_scope":
            continue
        if code not in _BASELINE_CODES:
            raise ValueError(f"알 수 없는 규칙 선별 근거 코드: {code}")
        reasons.append(baseline)
        weight = evidence_weight(policy, positions.get(baseline["step_id"]), total)
        if code in _PROHIBITED_CODES:
            statuses.add("excluded")
        elif code in _INCOMPLETE_CODES:
            statuses.add("insufficient_data")
        elif code in _RECORDED_ONLY_CODES:
            # On the record, not on the verdict. See the note above.
            pass
        elif code in _DEMOTE_ONLY_CODES:
            # No penalty of its own: this code annotates the finding beside
            # it, and that finding already paid. Charging again would push a
            # byproduct concern below an identical concern on a reagent.
            statuses.add("needs_review")
        elif code == "conditional_warning":
            penalty += weight
            statuses.add("excluded" if strict_alerts
                         and severity["conditional_warning"] == "exclude"
                         else "needs_review")
        else:
            # known_concern / structural_alert. Where the rule judge takes
            # part the combination decides; without it the alert stays on
            # record for expert review.
            penalty += weight
            if not participates(code):
                statuses.add("needs_review")
    return statuses, penalty, reasons


def _model_calls(identity, entries, *, model_status, model_error, reason):
    """Each envelope's call for one material, recording why it said so."""
    calls = []
    if not entries:
        calls.append(CALL_ABSTAIN)
        detail = f" {model_error}" if model_error else ""
        reason("admet_unassessed",
               f"ADMET-AI Ames 미평가({model_status}). 음성으로 간주하지 않습니다.{detail}")
    for assessments, _model, threshold in entries:
        row = assessments.get(identity)
        if row is None or row["status"] != "success":
            calls.append(CALL_ABSTAIN)
            state = "missing" if row is None else row["status"]
            error = row["error"] if row is not None else None
            detail = f" {error}" if error else ""
            reason("admet_unassessed",
                   f"ADMET-AI Ames 미평가({state}). 음성으로 간주하지 않습니다.{detail}")
            continue
        positive = row["score"] >= threshold
        calls.append(CALL_POSITIVE if positive else CALL_NEGATIVE)
        reason("admet_positive" if positive else "admet_negative",
               f"ADMET-AI Ames 점수={row['score']:g}, 임계값={threshold:g}: "
               f"{'양성' if positive else '음성'} 예측. 독성 또는 안전성의 입증이 아닙니다.")
    return calls


def _union_statuses(material, calls, policy, severity, *, demote_only, participates, reason):
    """One material's statuses under the rule-OR-model policy."""
    counts = participates(material["status"])
    rule = _rule_call(material)
    decision = combine(([rule] if counts else []) + calls, policy["combination"])
    if decision == CALL_ABSTAIN:
        return {"insufficient_data"}
    if decision != CALL_POSITIVE:
        return set()
    model_positive = CALL_POSITIVE in calls
    kinds = [material["status"]] if counts and rule == CALL_POSITIVE else []
    if model_positive:
        kinds.append("model_positive")
    if demote_only:
        reason("byproduct_concern", _BYPRODUCT_DEMOTION)
        return {"needs_review"}
    if model_positive and severity["model_positive"] == "exclude":
        reason("current_model_positive_exclusion",
               "현재 보고서 생성에 명시적으로 결합된 Ames 양성 근거로 기본 제시에서 제외합니다. "
               "규칙과의 OR 연구 정책이며 입증된 독성·실제 생성 또는 규제 판정이 아닙니다.")
    excludes = [kind for kind in kinds if severity[kind] == "exclude"]
    return {"excluded" if excludes or not kinds else "needs_review"}


def _joint_statuses(material, calls, policy, severity, *, demote_only, reason):
    """One material's statuses under the same-material rule-AND-model policy."""
    decision = combine(calls, policy["combination"])
    if decision == CALL_ABSTAIN:
        return {"insufficient_data"}
    if decision != CALL_POSITIVE:
        return set()
    if material["status"] not in _RULE_POSITIVE:
        return {"needs_review"}
    if demote_only:
        reason("byproduct_concern",
               "위 판정은 부산물 후보에서 나왔습니다. 규칙 우려와 모델 양성이 함께 "
               "나왔지만 예측된 후보이므로 경로를 빼지 않고 순위만 내립니다.")
        return {"needs_review"}
    if (severity[material["status"]] == "exclude"
            and severity["model_positive"] == "exclude"):
        reason("rule_admet_joint_positive",
               "동일 물질의 규칙 우려 AND ADMET-AI Ames 양성에 따른 공동 연구 정책으로 제외합니다. "
               "입증된 독성·실제 생성·잔류 또는 규제 판정이 아닙니다.")
        return {"excluded"}
    return {"needs_review"}


def _review_route(route, selected, entries, policy, severity, *, strict_alerts, union,
                  participates, model_status, model_error, ranking):
    """Decide one route's status, reasons and ranking penalty."""
    statuses, penalty, reasons = _baseline_review(
        selected, route, policy, severity,
        strict_alerts=strict_alerts, participates=participates)
    total = len(route["steps"])
    for index, step in enumerate(route["steps"]):
        weight = evidence_weight(policy, index, total)
        for material in step["materials"]:
            canonical = material["canonical_smiles"]
            smiles = canonical if canonical is not None else material["input_smiles"]
            identity = (canonical, None if canonical is not None else smiles)
            # A byproduct candidate demotes a route and never removes one,
            # whichever judge raised the concern. The rule screen's side of
            # this lives in route_selection, so the two cannot disagree.
            demote_only = material.get("role") == "byproducts"

            def reason(code, message, step=step, smiles=smiles):
                reasons.append({"code": code, "message": message,
                                "step_id": step["id"], "smiles": smiles})

            calls = _model_calls(identity, entries, model_status=model_status,
                                 model_error=model_error, reason=reason)
            if CALL_POSITIVE in calls:
                penalty += weight
            if union:
                statuses |= _union_statuses(
                    material, calls, policy, severity, demote_only=demote_only,
                    participates=participates, reason=reason)
            else:
                statuses |= _joint_statuses(
                    material, calls, policy, severity,
                    demote_only=demote_only, reason=reason)
    statuses.add("candidate")
    selected["status"] = next(status for status in _STATUSES if status in statuses)
    if selected["status"] == "candidate":
        reasons.append(dict(_CANDIDATE_REASON))
    selected["reasons"] = reasons
    if ranking:
        selected["rank_penalty"] = round(penalty, 6)


def review_routes_multi(report, model_results=(), *, excluded_smiles, strict_alerts,
                        union, model_status="missing", model_error=None, policy=None) -> dict:
    """Review every supplied route against a list of model-evidence envelopes.

    Each envelope is validated exactly as a single one was: membership in this
    report, provenance, threshold and prediction consistency. Per material the
    enabled judges' calls are combined with the policy's combination rule; an
    abstention never becomes a negative. Severity decides whether a kind of
    evidence excludes a route or only demotes its ranking, and the ranking
    score field appears only when the policy actually ranks.
    """
    if isinstance(model_results, dict):
        raise ValueError("model_results must be a sequence of model-evidence envelopes")
    policy = validate_policy(policy)
    severity = policy["severity"]
    participates = _rule_participates(policy, severity, strict_alerts=strict_alerts,
                                      union=union)
    model_results = list(model_results)
    # The report is validated first, by the rule screen. An envelope is checked
    # for membership in that report, so there is nothing to check it against
    # until the report itself is known to be well formed.
    result = select_routes(report, excluded_smiles=excluded_smiles,
                           strict_alerts=strict_alerts)
    entries = [_validated_assessments(report, envelope) for envelope in model_results]
    for route, selected in zip(report["routes"], result["routes"]):
        _review_route(route, selected, entries, policy, severity,
                      strict_alerts=strict_alerts, union=union,
                      participates=participates, model_status=model_status,
                      model_error=model_error, ranking=is_ranking_active(policy))
    result["counts"] = {status: sum(route["status"] == status for route in result["routes"])
                        for status in _STATUSES}
    result["policy"].update({
        "rule_admet_exclusion": "rule_OR_current_model" if union else "same_material_AND",
        "rule_positive_statuses": list(_RULE_POSITIVE),
        "research_only": True,
        "admet_threshold": entries[0][2] if entries else None,
        "admet_model": entries[0][1] if entries else None,
        "screening_policy": deepcopy(policy),
    })
    if len(entries) > 1:
        result["policy"]["admet_models"] = [entry[1] for entry in entries]
        result["policy"]["admet_thresholds"] = [entry[2] for entry in entries]
    if union:
        result["policy"]["model_evidence_status"] = "available" if entries else model_status
    result["limitations"] = list(dict.fromkeys([
        *result["limitations"], *(_ELIGIBILITY_LIMITATIONS if union else _LIMITATIONS),
        *(limitation for envelope in model_results for limitation in envelope["limitations"]),
    ]))
    return result


def _review_routes(report, admet_result, *, excluded_smiles, strict_alerts,
                   union, model_status="missing", model_error=None, policy=None):
    """Thin wrapper: one optional envelope becomes a one-element model list."""
    return review_routes_multi(
        report, [] if admet_result is None else [admet_result],
        excluded_smiles=excluded_smiles, strict_alerts=strict_alerts, union=union,
        model_status=model_status, model_error=model_error, policy=policy,
    )


def review_ames_routes(report, admet_result, *, excluded_smiles=(), policy=None) -> dict:
    """Compare same-material rule AND Ames separately from default presentation.

    Preserve all routes and substantive baseline reasons. Missing/failed rows
    remain unassessed. Structure, provenance and threshold checks do not prove
    report-generation binding: workspace validates that envelope separately.
    This is a research comparison, not proof of toxicity or safety.
    """
    if admet_result is None:
        raise ValueError("admet_result requires model, threshold, assessments and limitations")
    return _review_routes(report, admet_result, excluded_smiles=excluded_smiles,
                          strict_alerts=False, union=False, policy=policy)


def review_route_eligibility(report, admet_result=None, *, excluded_smiles=(),
                             strict_alerts=True, model_status="missing", model_error=None,
                             policy=None) -> dict:
    """Preserve every route/reason under the rule OR current-model policy.

    The workspace must explicitly validate report-generation binding before
    supplying admet_result. This helper reuses assessment membership, provenance
    and threshold validation, without model inference. A model-only positive
    excludes; a negative never overrides a rule exclusion. Unavailable or
    incomplete model evidence adds unassessed reasons, never a negative.
    """
    return _review_routes(report, admet_result, excluded_smiles=excluded_smiles,
                          strict_alerts=strict_alerts, union=True,
                          model_status=model_status, model_error=model_error,
                          policy=policy)
