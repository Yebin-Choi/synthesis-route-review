"""Registry of genotoxicity judges, looked up by id and returning one shape.

A judge turns supplied materials into one call per material: "1" (positive),
"0" (negative) or "abstain". An abstention is never a negative. Every judge
also reports its own provenance (tool, version, source) so a result can be
traced back to the software that produced it.

Toxtree and EPA TEST are registered as placeholders without execution code.
They are listed so a policy can name them and so the integration point is
visible, but running or enabling one raises instead of producing a result.
Nothing here fabricates a call for an unintegrated tool.
"""

CALL_POSITIVE = "1"
CALL_NEGATIVE = "0"
CALL_ABSTAIN = "abstain"
CALLS = (CALL_POSITIVE, CALL_NEGATIVE, CALL_ABSTAIN)

# Material assessment statuses the rule screen treats as a positive call.
RULE_POSITIVE_STATUSES = ("known_concern", "structural_alert")


def _call_row(input_smiles, canonical_smiles, call, *, score=None, kind=None, error=None):
    if call not in CALLS:
        raise ValueError(f"판정 결과는 {', '.join(CALLS)} 중 하나여야 합니다: {call!r}")
    return {"input_smiles": input_smiles, "canonical_smiles": canonical_smiles,
            "call": call, "score": score, "kind": kind, "error": error}


def _rules_judge(materials, *, threshold=None):
    """Run the local rule screen (ICH M7 list lookup plus custom alerts)."""
    from toxicology import ADDENDUM_URL, assess_material

    calls = []
    for smiles in materials:
        assessment = assess_material(smiles)
        status = assessment["status"]
        if status in RULE_POSITIVE_STATUSES:
            call, kind, error = CALL_POSITIVE, status, None
        elif status == "no_alert_detected":
            call, kind, error = CALL_NEGATIVE, None, None
        else:
            call, kind = CALL_ABSTAIN, None
            error = "구조를 해석할 수 없어 규칙 판정을 하지 않았습니다."
        calls.append(_call_row(smiles, assessment["canonical_smiles"], call,
                               kind=kind, error=error))
    return {
        "judge": "rules",
        "calls": calls,
        "evidence": {"tool": "local-rule-screen", "version": None, "endpoint": "ames",
                     "source": ADDENDUM_URL, "threshold": None},
    }


def _admet_judge(materials, *, threshold=0.5):
    """Run the installed ADMET-AI AMES model in its own interpreter."""
    from admet_service import assess_admet_materials

    result = assess_admet_materials(list(materials), threshold=threshold)
    calls = []
    for row in result["assessments"]:
        if row["status"] == "success":
            call = CALL_POSITIVE if row["score"] >= result["threshold"] else CALL_NEGATIVE
        else:
            call = CALL_ABSTAIN
        calls.append(_call_row(row["input_smiles"], row["canonical_smiles"], call,
                               score=row["score"],
                               kind="model_positive" if call == CALL_POSITIVE else None,
                               error=row["error"]))
    return {
        "judge": "admet",
        "calls": calls,
        "evidence": {"tool": result["model"]["tool"], "version": result["model"]["version"],
                     "endpoint": result["model"]["endpoint"],
                     "source": "local ADMET-AI model inference",
                     "threshold": result["threshold"],
                     "weights_sha256": result["model"]["weights_sha256"]},
        # The validated envelope ames_route_review consumes, unchanged.
        "model_result": result,
    }


def _ames_model_judge(materials, *, threshold=0.5):
    """Run the in-house Ames XGBoost model in its own interpreter."""
    from ames_model_service import assess_ames_model_materials

    result = assess_ames_model_materials(list(materials), threshold=threshold)
    calls = []
    for row in result["assessments"]:
        if row["status"] == "success":
            call = CALL_POSITIVE if row["score"] >= result["threshold"] else CALL_NEGATIVE
        else:
            call = CALL_ABSTAIN
        calls.append(_call_row(row["input_smiles"], row["canonical_smiles"], call,
                               score=row["score"],
                               kind="model_positive" if call == CALL_POSITIVE else None,
                               error=row["error"]))
    return {
        "judge": "ames_xgb",
        "calls": calls,
        "evidence": {"tool": result["model"]["tool"], "version": result["model"]["version"],
                     "endpoint": result["model"]["endpoint"],
                     "source": "local in-house Ames model inference",
                     "threshold": result["threshold"],
                     "weights_sha256": result["model"]["weights_sha256"]},
        "model_result": result,
    }


def _ames_descriptor_judge(materials, *, threshold=0.5):
    """Run the explainable descriptor-only in-house model.

    A second in-house model, not a replacement. On the independent structures
    it reads ROC-AUC 0.833 against the fingerprint model's 0.902, but answers
    every structure where that one abstains on 22%, and the two agree on 89.8%
    of calls (benchmark log 6-20). Nothing here reconciles them.
    """
    from ames_descriptor_service import assess_ames_descriptor_materials

    result = assess_ames_descriptor_materials(list(materials), threshold=threshold)
    calls = []
    for row in result["assessments"]:
        if row["status"] == "success":
            call = CALL_POSITIVE if row["score"] >= result["threshold"] else CALL_NEGATIVE
        else:
            call = CALL_ABSTAIN
        calls.append(_call_row(row["input_smiles"], row["canonical_smiles"], call,
                               score=row["score"],
                               kind="model_positive" if call == CALL_POSITIVE else None,
                               error=row["error"]))
    return {
        "judge": "ames_desc",
        "calls": calls,
        "evidence": {"tool": result["model"]["tool"], "version": result["model"]["version"],
                     "endpoint": result["model"]["endpoint"],
                     "source": "local in-house descriptor-only Ames model inference",
                     "threshold": result["threshold"],
                     "weights_sha256": result["model"]["weights_sha256"]},
        "model_result": result,
    }


def _java_tox_calls(judge_id, result, *, threshold, evidence):
    """Shape a Java tool's envelope into calls, keeping why it did not answer.

    These two tools decline often enough that the reason matters: Toxtree
    answered 80% and EPA TEST 77% of an independent 3,091-structure set. A
    declined row and a failed row are both abstentions, and the reason code
    travels with them so the screen can say which it was.
    """
    calls = []
    for row in result["assessments"]:
        if row["status"] != "success":
            call = CALL_ABSTAIN
        elif row.get("verdict") in ("positive", "negative"):
            # The tool published its own call. Report that, rather than
            # drawing our own line through its number and possibly
            # contradicting it. Measured 2026-11-06 on EPA TEST 5.1.2.0:
            # nitrobenzene reads 0.37 and the tool calls it negative, so a
            # 0.25 line would have reversed the tool on its own output. Its
            # figures in the performance table come from this call too.
            call = CALL_POSITIVE if row["verdict"] == "positive" else CALL_NEGATIVE
        else:
            call = CALL_POSITIVE if row["score"] >= threshold else CALL_NEGATIVE
        calls.append(_call_row(row["input_smiles"], row["canonical_smiles"], call,
                               score=row["score"],
                               kind="structural_alert" if call == CALL_POSITIVE
                               and evidence["kind"] == "rule" else
                               ("model_positive" if call == CALL_POSITIVE else None),
                               error=row.get("error")))
    return {"judge": judge_id, "calls": calls,
            "evidence": {"tool": result["model"]["tool"],
                         "version": result["model"]["version"],
                         "endpoint": result["model"]["endpoint"],
                         "source": result["model"]["source"],
                         "threshold": threshold if evidence["kind"] == "model" else None},
            "model_result": result}


def _toxtree_judge(materials, *, threshold=None):
    """Run the Toxtree Ames rulebase. It is a rule screen, so it has no cut-off."""
    from java_tox_service import assess_toxtree_materials

    result = assess_toxtree_materials(list(materials))
    # A rule screen reports alert or no alert, so any positive score is an
    # alert and the threshold argument is not used.
    return _java_tox_calls("toxtree", result, threshold=0.5,
                           evidence={"kind": "rule"})


def _epa_test_judge(materials, *, threshold=0.5):
    """Run the EPA TEST mutagenicity consensus model.

    The tool's own call is used where it gave one, so `threshold` only covers
    the case where a row carried a value without a verdict.
    """
    from java_tox_service import assess_epa_test_materials

    result = assess_epa_test_materials(list(materials))
    return _java_tox_calls("epa_test", result, threshold=threshold,
                           evidence={"kind": "model"})


def _unintegrated(judge_id, tool):
    def run(materials, *, threshold=None):
        raise NotImplementedError(
            f"판정기 '{judge_id}'({tool})는 아직 연동되지 않았습니다. "
            "실행 코드를 등록부에 연결하기 전에는 판정 결과를 만들지 않습니다."
        )
    return run


# id -> registry entry. "implemented" False means the slot exists but no
# execution code does; "run" then raises instead of returning a call.
JUDGES = {
    "rules": {"tool": "local-rule-screen", "kind": "rule", "implemented": True,
              "uses_threshold": False, "run": _rules_judge},
    "admet": {"tool": "ADMET-AI", "kind": "model", "implemented": True,
              "uses_threshold": True, "run": _admet_judge},
    "ames_xgb": {"tool": "in-house Ames XGBoost", "kind": "model", "implemented": True,
                 "uses_threshold": True, "run": _ames_model_judge},
    "ames_desc": {"tool": "in-house Ames descriptors", "kind": "model",
                  "implemented": True, "uses_threshold": True,
                  "run": _ames_descriptor_judge},
    "toxtree": {"tool": "Toxtree", "kind": "rule", "implemented": True,
                "uses_threshold": False, "run": _toxtree_judge},
    "epa_test": {"tool": "EPA TEST", "kind": "model", "implemented": True,
                 "uses_threshold": True, "run": _epa_test_judge},
}


def judge_ids() -> tuple:
    """Every registered id, integrated or not, in registration order."""
    return tuple(JUDGES)


def implemented_judge_ids() -> tuple:
    """Only the ids that have execution code behind them."""
    return tuple(key for key, entry in JUDGES.items() if entry["implemented"])


def get_judge(judge_id: str) -> dict:
    """Look up one registry entry; an unknown id is an error, not a default."""
    entry = JUDGES.get(judge_id) if isinstance(judge_id, str) else None
    if entry is None:
        raise ValueError(
            f"알 수 없는 판정기 id: {judge_id!r}. 등록된 id는 {', '.join(JUDGES)}입니다."
        )
    return entry


def run_judge(judge_id: str, materials, *, threshold=None) -> dict:
    """Run one registered judge over supplied materials, in input order.

    An unintegrated judge raises NotImplementedError. Invalid or failed
    materials stay abstentions and are never reported as negative.
    """
    entry = get_judge(judge_id)
    if entry["uses_threshold"] and threshold is not None:
        return entry["run"](materials, threshold=threshold)
    return entry["run"](materials)
