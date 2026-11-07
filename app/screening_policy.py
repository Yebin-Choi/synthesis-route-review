"""Replaceable genotoxicity screening policy: which judges, how to combine them.

The default policy runs the local rule screen plus the ADMET-AI AMES model,
combined as "any positive" at threshold 0.5. Only a named ICH M7 entry removes
a route; the other kinds of evidence rank it lower and leave it in the list.
Changing a tool, a combination rule or a severity is a configuration change,
not a code change, so the earlier exclude-everything behaviour is still
reachable by setting every severity to exclude.

Nothing here infers toxicity. A negative call is a screening result and an
abstention is never converted into a negative.
"""

from copy import deepcopy

from screening_judges import CALL_ABSTAIN, CALL_NEGATIVE, CALL_POSITIVE, get_judge


COMBINATION_RULES = ("any_positive", "unanimous", "majority")
# Evidence kinds a severity can be set for. The first three come from the rule
# screen and the route baseline; model_positive covers every enabled model.
SEVERITY_KINDS = ("known_concern", "structural_alert", "conditional_warning", "model_positive")
SEVERITY_ACTIONS = ("exclude", "demote")
POLICY_KEYS = ("tools", "combination", "thresholds", "severity", "position_weighting")

# Only a named entry on the ICH M7 list removes a route. A structural alert,
# a model call or a predicted byproduct is a difference of degree, so it ranks
# a route lower instead of deleting it. Excluding on all four left nothing to
# present: a live propranolol search kept 1 route of 10, and none once
# byproducts were added, which is not a screen but a refusal to answer.
DEFAULT_POLICY = {
    "tools": ["rules", "admet"],
    "combination": "any_positive",
    "thresholds": {"admet": 0.5},
    "severity": {
        "known_concern": "exclude",
        "structural_alert": "demote",
        "conditional_warning": "demote",
        "model_positive": "demote",
    },
    "position_weighting": False,
}


def default_policy() -> dict:
    """A fresh copy of the shipped default, safe to mutate."""
    return deepcopy(DEFAULT_POLICY)


def validate_policy(policy=None) -> dict:
    """Return a validated, complete copy, or raise. Never falls back silently.

    Unknown keys, unknown judge ids, judges without execution code, unknown
    combination rules, unknown severity kinds or actions and out-of-range
    thresholds are all rejected with an explicit error. Omitted keys take the
    default value; a present key is validated, never repaired.
    """
    if policy is None:
        return default_policy()
    if not isinstance(policy, dict):
        raise ValueError("정책은 딕셔너리여야 합니다.")
    unknown = [key for key in policy if key not in POLICY_KEYS]
    if unknown:
        raise ValueError(
            f"알 수 없는 정책 키: {', '.join(map(str, sorted(map(str, unknown))))}. "
            f"허용되는 키는 {', '.join(POLICY_KEYS)}입니다."
        )
    validated = default_policy()

    if "tools" in policy:
        tools = policy["tools"]
        if not isinstance(tools, (list, tuple)) or not tools:
            raise ValueError("tools는 비어 있지 않은 판정기 id 목록이어야 합니다.")
        chosen = []
        for judge_id in tools:
            entry = get_judge(judge_id)
            if not entry["implemented"]:
                raise ValueError(
                    f"판정기 '{judge_id}'({entry['tool']})는 아직 연동되지 않았습니다. "
                    "등록부에 실행 코드를 연결한 뒤에 정책에서 켜 주세요."
                )
            if judge_id in chosen:
                raise ValueError(f"중복된 판정기 id: {judge_id}")
            chosen.append(judge_id)
        validated["tools"] = chosen

    if "combination" in policy:
        rule = policy["combination"]
        if rule not in COMBINATION_RULES:
            raise ValueError(
                f"알 수 없는 조합 규칙: {rule!r}. "
                f"허용되는 규칙은 {', '.join(COMBINATION_RULES)}입니다."
            )
        validated["combination"] = rule

    if "thresholds" in policy:
        thresholds = policy["thresholds"]
        if not isinstance(thresholds, dict):
            raise ValueError("thresholds는 판정기 id별 임계값 딕셔너리여야 합니다.")
        merged = dict(validated["thresholds"])
        for judge_id, value in thresholds.items():
            get_judge(judge_id)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{judge_id} 임계값은 0과 1 사이의 수여야 합니다: {value!r}")
            number = float(value)
            if not 0 <= number <= 1:
                raise ValueError(f"{judge_id} 임계값은 0과 1 사이여야 합니다: {value!r}")
            merged[judge_id] = number
        validated["thresholds"] = merged

    if "severity" in policy:
        severity = policy["severity"]
        if not isinstance(severity, dict):
            raise ValueError("severity는 근거 종류별 처리 딕셔너리여야 합니다.")
        merged = dict(validated["severity"])
        for kind, action in severity.items():
            if kind not in SEVERITY_KINDS:
                raise ValueError(
                    f"알 수 없는 근거 종류: {kind!r}. "
                    f"허용되는 종류는 {', '.join(SEVERITY_KINDS)}입니다."
                )
            if action not in SEVERITY_ACTIONS:
                raise ValueError(
                    f"{kind}의 처리는 {' 또는 '.join(SEVERITY_ACTIONS)}여야 합니다: {action!r}"
                )
            merged[kind] = action
        validated["severity"] = merged

    if "position_weighting" in policy:
        weighting = policy["position_weighting"]
        if type(weighting) is not bool:
            raise ValueError("position_weighting은 bool이어야 합니다.")
        validated["position_weighting"] = weighting

    return validated


def combine(calls, rule: str) -> str:
    """Combine one structure's judge calls into a single call.

    Same rules as evaluation/compare_tools.combine, so an experiment scored
    there and a route reviewed here agree. A judge that abstained cannot vouch
    for a negative, so every rule needs every judge to have decided before it
    returns "0"; a tie under majority abstains rather than guessing.
    """
    if rule not in COMBINATION_RULES:
        raise ValueError(
            f"알 수 없는 조합 규칙: {rule!r}. "
            f"허용되는 규칙은 {', '.join(COMBINATION_RULES)}입니다."
        )
    calls = list(calls)
    for call in calls:
        if call not in (CALL_POSITIVE, CALL_NEGATIVE, CALL_ABSTAIN):
            raise ValueError(f"알 수 없는 판정 결과: {call!r}")
    decided = [call for call in calls if call in (CALL_NEGATIVE, CALL_POSITIVE)]
    if not decided:
        return CALL_ABSTAIN
    if rule == "any_positive":
        # A tool that abstained cannot vouch for a negative, so a negative
        # verdict requires every tool to have decided.
        if CALL_POSITIVE in decided:
            return CALL_POSITIVE
        return CALL_NEGATIVE if len(decided) == len(calls) else CALL_ABSTAIN
    if rule == "unanimous":
        if len(decided) != len(calls) or len(set(decided)) != 1:
            return CALL_ABSTAIN
        return decided[0]
    if len(decided) != len(calls):
        return CALL_ABSTAIN
    positive = decided.count(CALL_POSITIVE)
    if positive * 2 == len(decided):
        return CALL_ABSTAIN
    return CALL_POSITIVE if positive * 2 > len(decided) else CALL_NEGATIVE


def position_weight(index, total) -> float:
    """Weight one piece of evidence by where in the route it appears.

    Formula: weight = (index + 1) / total, with index the zero-based step
    position and total the number of steps. The last step weighs 1.0 and
    earlier steps less, in equal increments, because a material introduced
    nearer the final step is more likely to remain in the final product.
    Route-wide evidence with no step (index None) weighs 1.0.
    """
    if index is None or not isinstance(total, int) or total <= 0:
        return 1.0
    if not isinstance(index, int) or index < 0 or index >= total:
        raise ValueError("위치 가중의 단계 색인이 경로 범위를 벗어났습니다.")
    return (index + 1) / total


def evidence_weight(policy: dict, index, total) -> float:
    """Position weight when the policy asks for it, otherwise a flat 1.0."""
    if not policy["position_weighting"]:
        return 1.0
    return position_weight(index, total)


def is_ranking_active(policy: dict) -> bool:
    """True when the policy demotes any evidence or weights by position.

    The default policy demotes, so route results carry a ranking score. A
    policy that excludes on every kind of evidence has nothing to rank, and
    route results then keep only the fields they had before ranking existed.
    """
    return (policy["position_weighting"]
            or any(action == "demote" for action in policy["severity"].values()))


def model_threshold(policy: dict, judge_id: str = "admet") -> float:
    """The configured threshold for one model judge."""
    get_judge(judge_id)
    return policy["thresholds"][judge_id]
