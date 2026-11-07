"""Pure shared route context; no UI, model inference or external requests."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from uuid import UUID, uuid4

import engine
from ames_route_review import review_ames_routes, review_route_eligibility


_NO_ABSENCE = (
    "부산물 후보는 제안된 구조이며 관측·생성량의 근거가 아닙니다. "
    "실패·빈 응답·후보 미반환은 부산물 부재를 뜻하지 않습니다."
)


def validate_target(smiles: str) -> str:
    """Return canonical target identity, or Korean ValueError for bad input.

    Reject more than 4096 raw characters before parsing. Only atom maps are
    removed by the existing canonicalizer; preserve raw text separately.
    """
    return engine.validate_target_smiles(smiles)


def new_workspace(target_smiles: str) -> dict:
    """Create an empty context for a validated target, without invented results.

    Use a new workspace for an explicit target switch; route loading never
    changes the target implicitly. The raw target is preserved exactly. A
    fresh UUID prevents restored target contexts from reviving old drafts.
    """
    target = validate_target(target_smiles)
    return {
        "workspace_id": str(uuid4()),
        "context_revision": 0,
        "raw_target": target_smiles,
        "target": target,
        "original_input": None,
        "active_input": None,
        "report": None,
        "report_sha256": None,
        "report_generation": None,
        "route_id": None,
        "step_id": None,
        "policy": {"strict_alerts": True, "excluded_smiles": []},
        "selection": None,
        "byproduct_attempts": [],
        "model_evidence": None,
        "model_evidence_state": {"status": "missing", "error": None},
        "model_evidence_history": [],
    }


def report_hash(report: dict) -> str:
    """Hash a strict finite JSON object deterministically, without mutation.

    Dictionary key order is irrelevant; array order and every value matter.
    Non-string keys, tuples, cycles, nonfinite numbers and non-JSON values
    raise ValueError instead of producing a lossy or nonstandard snapshot.
    """
    if not isinstance(report, dict):
        raise ValueError("보고서 해시에는 JSON 객체가 필요합니다.")
    try:
        # Depth is bounded explicitly rather than left to the serialiser.
        # On Python 3.14 json.dumps accepts 2,000-deep nesting that deepcopy
        # later refuses, so a value could hash here and crash downstream.
        if not engine.json_depth_within(report):
            raise ValueError(f"{engine.MAX_JSON_DEPTH}단계보다 깊게 중첩되었습니다.")
        text = json.dumps(report, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(",", ":"))
        if json.loads(text) != report:
            raise ValueError("JSON 변환으로 원래 값이 달라집니다.")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("보고서에는 문자열 키와 유한한 수를 가진 엄격한 JSON 값만 허용합니다.") from exc


def context_key(workspace: dict) -> str:
    """Identify a workspace generation and target/report/route/step transition.

    Consent belongs to the caller and must be cleared when this key changes.
    Revision distinguishes A→B→A stage transitions; workspace UUID separates
    freshly created contexts, even for the same target. Policy/UI changes do
    not change this key. No consent or pending request is persisted here.
    """
    return report_hash({
        "workspace_id": workspace["workspace_id"],
        "context_revision": workspace["context_revision"],
        "target": workspace["target"],
        "report_sha256": workspace["report_sha256"],
        "report_generation": workspace["report_generation"],
        "route_id": workspace["route_id"],
        "step_id": workspace["step_id"],
    })


def load_routes(workspace: dict, payload: dict) -> dict:
    """Atomically load matching routes and recompute all rules and selection.

    Every route must have the active canonical target; mixed or unrelated
    targets raise ValueError. Original input, active input and report are
    independent deep copies. Every accepted load is a fresh report generation;
    select its first eligible route/step or None when all are excluded. Derived
    calculations reset, while prior model/byproduct history remains historical.
    On failure the supplied workspace remains unchanged.
    """
    active_input = deepcopy(payload)
    engine.validate_routes(active_input)
    targets = {validate_target(route["target"]) for route in active_input["routes"]}
    if targets != {workspace["target"]}:
        raise ValueError("모든 경로의 목표물이 현재 목표물과 정확히 일치해야 합니다. 혼합 목표물은 적용하지 않으며 목표물 전환은 별도로 해야 합니다.")
    report = engine.analyze_routes(active_input)
    digest = report_hash(report)
    state = deepcopy(workspace)
    _archive_model_evidence(state, state["model_evidence"], "새 보고서 생성으로 이전 모델 근거는 이력입니다.")
    state.update({
        "original_input": deepcopy(active_input),
        "active_input": active_input,
        "report": deepcopy(report),
        "report_sha256": digest,
        "report_generation": str(uuid4()),
        "route_id": None,
        "step_id": None,
        "model_evidence": None,
        "model_evidence_state": {"status": "missing", "error": None},
    })
    return _reconcile_selection(state, workspace)


def _archive_model_evidence(state: dict, evidence, reason: str) -> None:
    if evidence is not None:
        retained, _ = _attempt_evidence(evidence)
        entry = {
            "applicable": False, "reason": reason,
            "report_sha256": retained.get("report_sha256") if isinstance(retained, dict) else None,
            "report_generation": retained.get("report_generation") if isinstance(retained, dict) else None,
            "evidence": retained,
        }
        if not state["model_evidence_history"] or state["model_evidence_history"][-1] != entry:
            state["model_evidence_history"].append(entry)


def _model_evidence_state(workspace: dict, evidence) -> dict:
    if evidence is None:
        return {"status": "missing", "error": None}
    if not isinstance(evidence, dict):
        return {"status": "invalid", "error": "모델 근거에는 보고서 생성에 결합된 객체가 필요합니다."}
    if (workspace["report"] is None or not workspace["report_generation"]
            or evidence.get("report_sha256") != workspace["report_sha256"]
            or evidence.get("report_generation") != workspace["report_generation"]):
        return {"status": "stale", "error": "모델 근거의 보고서 해시/생성 식별자가 현재 보고서와 일치하지 않습니다."}
    try:
        report_hash(evidence)
        if type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1:
            raise ValueError("모델 근거 schema_version=1이 필요합니다.")
        if report_hash(workspace["report"]) != workspace["report_sha256"]:
            raise ValueError("현재 보고서와 보고서 해시가 일치하지 않습니다.")
        if "error" not in evidence or evidence["error"] is not None:
            error = evidence.get("error")
            if not isinstance(error, str) or not error.strip():
                raise ValueError("모델 실행 오류는 비어 있지 않은 문자열이어야 합니다.")
            return {"status": "failed", "error": error}
        result = evidence.get("model_result")
        metadata = evidence.get("metadata")
        if not isinstance(result, dict) or not isinstance(metadata, dict):
            raise ValueError("모델 결과와 실행 출처 metadata가 필요합니다.")
        if (metadata.get("report_sha256") != workspace["report_sha256"]
                or metadata.get("report_generation") != workspace["report_generation"]):
            raise ValueError("모델 metadata의 보고서 해시/생성 식별자가 일치하지 않습니다.")
        UUID(metadata.get("evaluation_id", ""))
        if (metadata.get("source") != "local_ADMET-AI_model_inference"
                or not isinstance(metadata.get("completed_at_utc"), str)
                or not metadata["completed_at_utc"].strip()
                or metadata.get("model") != result.get("model")
                or isinstance(metadata.get("threshold"), bool)
                or metadata.get("threshold") != result.get("threshold")):
            raise ValueError("모델 실행 출처/임계값/metadata가 결과와 일치해야 합니다.")
        review_ames_routes(workspace["report"], result,
                           excluded_smiles=workspace["policy"]["excluded_smiles"])
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return {"status": "invalid", "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "available", "error": None}


def model_evidence_status(workspace: dict) -> dict:
    """Return available/missing/stale/failed/invalid without any inference.

    Available means a current explicitly bound, validated evidence envelope,
    not a successful row for every material or a scientifically proven domain.
    Failed/stale/unbound evidence stays historical and is never a negative.
    """
    if workspace["model_evidence"] is None:
        return deepcopy(workspace["model_evidence_state"])
    return _model_evidence_state(workspace, workspace["model_evidence"])


def current_model_evidence(workspace: dict) -> dict | None:
    """Copy only the valid current report-generation-bound model envelope."""
    return (deepcopy(workspace["model_evidence"])
            if model_evidence_status(workspace)["status"] == "available" else None)


def eligible_routes(workspace: dict) -> list[dict]:
    """Copy source-order nonexcluded report routes, including incomplete ones.

    Decision labels/reasons remain in selection; this accessor never drops
    excluded data from the report or changes its impurity-evaluation scope.
    """
    eligible = {route["id"] for route in (workspace["selection"] or {}).get("routes", [])
                if route["status"] != "excluded"}
    return [deepcopy(route) for route in (workspace["report"] or {}).get("routes", [])
            if route["id"] in eligible]


def excluded_routes(workspace: dict) -> list[dict]:
    """Copy every excluded decision with all reasons, in original route order."""
    return [deepcopy(route) for route in (workspace["selection"] or {}).get("routes", [])
            if route["status"] == "excluded"]


def _reconcile_selection(state: dict, previous: dict) -> dict:
    if state["report"] is None:
        state.update(selection=None, route_id=None, step_id=None)
        return state
    status = model_evidence_status(state)
    if state["model_evidence"] is not None and status["status"] != "available":
        _archive_model_evidence(state, state["model_evidence"], status["error"])
        state.update(model_evidence=None, model_evidence_state=status)
    evidence = state["model_evidence"]
    state["selection"] = review_route_eligibility(
        state["report"], evidence["model_result"] if evidence is not None else None,
        **state["policy"], model_status=status["status"], model_error=status["error"],
    )
    state["policy"] = {key: deepcopy(state["selection"]["policy"][key])
                       for key in ("strict_alerts", "excluded_smiles")}
    if evidence is not None:
        evidence["combined_review"] = review_ames_routes(
            state["report"], evidence["model_result"],
            excluded_smiles=state["policy"]["excluded_smiles"],
        )
        evidence["combined_review_policy"] = "same_material_AND"
        evidence["review_error"] = None
    eligible = {route["id"] for route in state["selection"]["routes"] if route["status"] != "excluded"}
    if (state["route_id"] not in eligible
            or _source_step(state["active_input"], state["route_id"], state["step_id"]) is None):
        route = next((route for route in state["active_input"]["routes"] if route["id"] in eligible), None)
        state["route_id"] = route["id"] if route is not None else None
        state["step_id"] = route["steps"][0]["id"] if route is not None else None
    if ((state["report_generation"], state["route_id"], state["step_id"])
            != (previous["report_generation"], previous["route_id"], previous["step_id"])):
        state.update(context_revision=previous["context_revision"] + 1)

    return state


def set_model_evidence(workspace: dict, evidence: dict | None) -> dict:
    """Freeze explicit model evidence, recompute eligibility, and reconcile stage.

    Require envelope schema 1 plus matching report_sha256/report_generation in
    both envelope and metadata, validated provenance, structures and threshold.
    Unbound/stale/failed/malformed evidence is preserved only as history with
    an explicit status; no automatic model/API call or invented negative.
    Excluding the active route selects the first eligible route or clears both
    IDs and invalidates stage calculations/consent via the context revision.
    """
    state = deepcopy(workspace)
    retained, malformed = _attempt_evidence(evidence)
    status = (_model_evidence_state(state, retained) if malformed is None else
              {"status": "invalid", "error": malformed})
    current = retained if status["status"] == "available" else None
    derived_fields = {"combined_review", "combined_review_policy", "review_error"}
    current_source = ({key: value for key, value in current.items() if key not in derived_fields}
                      if current is not None else None)
    previous_source = ({key: value for key, value in state["model_evidence"].items() if key not in derived_fields}
                       if isinstance(state["model_evidence"], dict) else state["model_evidence"])
    if current_source != previous_source:
        _archive_model_evidence(state, state["model_evidence"], "모델 근거가 교체되거나 무효화되어 이력으로 보존합니다.")
    if retained is not None and current is None:
        _archive_model_evidence(state, retained, status["error"])
    state.update(model_evidence=current, model_evidence_state=status)
    return _reconcile_selection(state, workspace)


def _source_step(payload: dict | None, route_id: str, step_id: str) -> dict | None:
    if payload is None:
        return None
    for route in payload["routes"]:
        if route["id"] == route_id:
            return next((step for step in route["steps"] if step["id"] == step_id), None)
    return None


def selected_step(workspace: dict) -> dict | None:
    """Return a copy of the active source step, or None without a selection.

    This is not an evaluated materials list: all raw products, including the
    target excluded from impurity assessment, and source metadata remain.
    """
    step = _source_step(workspace["active_input"], workspace["route_id"], workspace["step_id"])
    return deepcopy(step) if step is not None else None


def select_step(workspace: dict, route_id: str, step_id: str) -> dict:
    """Validate membership and select a source step without changing the report.

    Report-wide model
    evidence and attempt history remain; an identical selection is a no-op.
    The changed context key invalidates caller-owned request consent.
    """
    if not any(route["id"] == route_id and route["status"] != "excluded"
               for route in (workspace["selection"] or {}).get("routes", [])):
        raise ValueError("제외된 경로 또는 제시 가능한 경로가 아닌 선택은 허용하지 않습니다.")
    if _source_step(workspace["active_input"], route_id, step_id) is None:
        raise ValueError("선택한 경로에 해당 반응 단계가 없습니다.")
    state = deepcopy(workspace)
    if (route_id, step_id) != (state["route_id"], state["step_id"]):
        state.update(route_id=route_id, step_id=step_id,
                     context_revision=state["context_revision"] + 1)
    return state


def set_policy(workspace: dict, strict_alerts: bool, excluded_smiles: list) -> dict:
    """Recompute rule/model presentation using frozen evidence, without calls.

    Policy must be bool plus a list of valid structures. Canonicalize and
    deduplicate exclusions without merging salts, charge or stereochemistry.
    Same-stage calculations remain valid; active-route exclusion clears them
    and selects the first eligible route or None. No model/API call occurs.
    """
    if type(strict_alerts) is not bool:
        raise ValueError("strict_alerts는 bool이어야 합니다.")
    if not isinstance(excluded_smiles, list):
        raise ValueError("excluded_smiles는 SMILES 문자열 배열이어야 합니다.")
    exclusions = list(dict.fromkeys(validate_target(smiles) for smiles in excluded_smiles))
    policy = {"strict_alerts": strict_alerts, "excluded_smiles": exclusions}
    state = deepcopy(workspace)
    state["policy"] = policy
    return _reconcile_selection(state, workspace)


def _byproduct_inputs(workspace: dict, product_smiles: str) -> tuple[str, str]:
    step = selected_step(workspace)
    if step is None or workspace["report"] is None:
        raise ValueError("부산물 요청에는 선택한 경로와 반응 단계가 필요합니다.")
    product = validate_target(product_smiles)
    if not any(engine.canonicalize_smiles(value) == product for value in step["products"]):
        raise ValueError("확인한 주생성물은 선택 단계의 생성물 구조와 일치해야 합니다.")
    reactants = step["reactants"]
    if not reactants or any(engine.canonicalize_smiles(value) is None for value in reactants):
        raise ValueError("부산물 요청에는 비어 있지 않은 유효한 반응물 구조가 필요합니다.")
    joined = ".".join(value.strip() for value in reactants)
    if engine.canonicalize_smiles(joined) is None:
        raise ValueError("부산물 요청의 반응물 구조를 점으로 연결할 수 없습니다.")
    return joined, product_smiles.strip()


def make_byproduct_request(workspace: dict, product_smiles: str) -> dict:
    """Freeze a validated ASKCOS payload and its request/context identity.

    Product membership is confirmed by exact canonical identity in the source
    step. Join trimmed reactant entries without reconstructing components;
    retain the original text in active_input. This makes no HTTP call and
    grants no consent. The caller previews and sends these exact strings.
    """
    reactants, product = _byproduct_inputs(workspace, product_smiles)
    if report_hash(workspace["report"]) != workspace["report_sha256"]:
        raise ValueError("현재 보고서와 보고서 해시가 일치하지 않습니다.")
    return {
        "request_id": str(uuid4()),
        "report_sha256": workspace["report_sha256"],
        "route_id": workspace["route_id"],
        "step_id": workspace["step_id"],
        "reactants_smiles": reactants,
        "product_smiles": product,
        "context_key": context_key(workspace),
    }


def _attempt_evidence(value) -> tuple[object, str | None]:
    """Keep valid JSON verbatim; diagnose malformed evidence without losing exports."""
    try:
        report_hash({"evidence": value})
        return deepcopy(value), None
    except (ValueError, RecursionError) as exc:
        cause = exc.__cause__ or exc
        diagnostic = f"{type(cause).__name__}: {cause}"
        try:
            raw = repr(value)
        except Exception as repr_error:
            raw = f"raw_repr 실패: {type(repr_error).__name__}: {repr_error}"
        return {
            "status": "invalid_evidence",
            "diagnostic": diagnostic,
            "raw_repr": raw[:4096],
            "raw_repr_truncated": len(raw) > 4096,
        }, diagnostic


def apply_byproduct_result(workspace: dict, request: dict, result: dict) -> dict:
    """Record an attempt and atomically reanalyse a successful current response.

    Stale, failed, empty or malformed responses are history-only and never
    imply absence. Nonempty success requires candidate objects with nonempty
    smiles; invalid molecular strings remain visible as unevaluated material.
    Only the selected step in a cloned active input is replaced. Schema,
    whole-input analysis, policy and strict report hash must all succeed
    before publication. Any exception records application_error and preserves
    the prior consistent snapshot. A changed report clears
    and archives old model evidence in this attempt as explicitly historical.
    Non-JSON evidence is never applied: history contains invalid_evidence,
    its diagnostic and a bounded raw_repr with an explicit truncation flag.
    """
    state = deepcopy(workspace)
    request_evidence, request_error = _attempt_evidence(request)
    result_evidence, result_error = _attempt_evidence(result)
    attempt = {
        "request": request_evidence,
        "result": result_evidence,
        "applied": False,
        "stale": False,
        "error": None,
        "application_error": None,
        "limitations": [_NO_ABSENCE],
    }
    state["byproduct_attempts"].append(attempt)
    try:
        if request_error is not None:
            raise ValueError("부산물 요청 근거를 JSON으로 보존할 수 없습니다. " + request_error)
        fields = ("request_id", "report_sha256", "route_id", "step_id",
                  "reactants_smiles", "product_smiles", "context_key")
        if (not isinstance(request, dict)
                or any(not isinstance(request.get(field), str) or not request[field].strip()
                       for field in fields)):
            raise ValueError("부산물 요청 식별자와 구조 정보가 올바르지 않습니다.")
        UUID(request["request_id"])
        if (request["context_key"] != context_key(state)
                or any(request[field] != state[field]
                       for field in ("report_sha256", "route_id", "step_id"))):
            attempt.update(stale=True, error="요청 이후 목표물·보고서·경로·단계가 변경되어 이력으로만 보존합니다.")
            return state
        reactants, product = _byproduct_inputs(state, request["product_smiles"])
        if reactants != request["reactants_smiles"] or product != request["product_smiles"]:
            raise ValueError("요청의 반응물·주생성물이 현재 선택 단계와 일치하지 않습니다.")
        if report_hash(state["report"]) != state["report_sha256"]:
            raise ValueError("현재 보고서와 보고서 해시가 일치하지 않습니다.")
        if result_error is not None:
            raise ValueError("부산물 응답 근거를 JSON으로 보존할 수 없습니다. " + result_error)
        if not isinstance(result, dict):
            raise ValueError("부산물 응답은 JSON 객체여야 합니다.")
        candidates = result.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("부산물 응답에 후보 배열이 필요합니다.")
        if result.get("status") == "failed":
            if candidates:
                raise ValueError("실패 응답에 성공 후보를 혼합할 수 없습니다.")
            error = result.get("error")
            attempt["error"] = error if isinstance(error, str) and error.strip() else "부산물 요청이 실패했습니다. 미평가이며 부재를 뜻하지 않습니다."
            return state
        if result.get("status") != "success" or result.get("error") is not None:
            raise ValueError("부산물 응답의 성공 상태와 오류 정보가 올바르지 않습니다.")
        smiles = []
        for candidate in candidates:
            if (not isinstance(candidate, dict) or not isinstance(candidate.get("smiles"), str)
                    or not candidate["smiles"].strip()):
                raise ValueError("부산물 후보는 비어 있지 않은 smiles를 가진 객체여야 합니다.")
            smiles.append(candidate["smiles"])
        if not smiles:
            attempt["error"] = "성공 응답에 후보가 없어 기존 보고서를 유지합니다. 후보 미반환은 부산물 부재가 아닙니다."
            return state
        active_input = deepcopy(state["active_input"])
        step = _source_step(active_input, state["route_id"], state["step_id"])
        step["byproducts"] = {
            "status": "success", "candidates": smiles, "source": engine.ASKCOS_API,
            "error": None, "limitations": [_NO_ABSENCE],
        }
        step.setdefault("material_provenance", {})["byproducts"] = {
            "status": "proposed", "source": engine.ASKCOS_API,
        }
        engine.validate_routes(active_input)
        report = engine.analyze_routes(active_input)
        digest = report_hash(report)
        candidate_state = deepcopy(state)
        candidate_attempt = candidate_state["byproduct_attempts"][-1]
        updates = {"active_input": active_input, "report": deepcopy(report),
                   "report_sha256": digest}
        if digest != state["report_sha256"]:
            if state["model_evidence"] is not None:
                candidate_attempt["historical_model_evidence"] = {
                    "applicable": False,
                    "report_sha256": state["report_sha256"],
                    "report_generation": state["report_generation"],
                    "evidence": deepcopy(state["model_evidence"]),
                }
                _archive_model_evidence(candidate_state, state["model_evidence"],
                                        "부산물 재분석으로 보고서가 변경되어 이전 모델 근거는 이력입니다.")
            updates.update(model_evidence=None,
                           model_evidence_state={"status": "missing", "error": None},
                           report_generation=str(uuid4()))
        candidate_state.update(updates)
        _reconcile_selection(candidate_state, workspace)
        candidate_attempt["applied"] = True
        return candidate_state
    except Exception as exc:
        attempt["application_error"] = f"{type(exc).__name__}: {exc}"
        attempt["error"] = "부산물 결과를 적용하지 못했습니다. 이전 보고서를 유지하며 새 결과는 미평가입니다. " + attempt["application_error"]
    return state
