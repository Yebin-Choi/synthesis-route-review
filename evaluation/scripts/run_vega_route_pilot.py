"""Preregister, then run a bounded local VEGA pilot over saved route identities.

--prepare never imports Py4J or executes a model. --run requires the unchanged
manifest and source artifacts. This is not independent predictive validation.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))
from toxicology import canonicalize_smiles

SOURCE = ROOT / "evaluation" / "results" / "live_route_ames_experiment.json"
JAR = ROOT / "external_tools" / "VEGA GUI 1.2.6" / "Vega-GUI-1.2.6.jar"
MANIFEST = ROOT / "evaluation" / "results" / "vega_route_pilot_manifest.json"
OUTPUT = ROOT / "evaluation" / "results" / "vega_route_pilot_results.json"
PROVENANCE = {
    "vega_probe": ROOT / "evaluation" / "scripts" / "probe_vega_ames.py",
    "vega_smoke": ROOT / "evaluation" / "results" / "vega_ames_smoke.json",
    "canonicalization": ROOT / "app" / "toxicology.py",
    "runner": Path(__file__).resolve(),
}
SCHEMA = "vega-route-pilot-1"
MODEL_CLASS = "insilico.mutagenicity_sarpy.ismMutagenicitySarpy"
MODEL_IDENTITY = {"key": "MUTA_SARPY", "version": "1.0.8"}
EXPECTED_ADMET_IDENTITIES = 15
MAX_RAW_IDENTITIES = 20
SENTINELS = (("ethanol", "CCO"), ("epichlorohydrin", "ClCC1CO1"),
             ("NDMA", "CN(C)N=O"))
INVALID_CONTROL = "CC("
STATUS_CONSTANTS = {
    "OUTPUT_OK": 1,
    "OUTPUT_OK_AD_NOT_APPLICABLE": 3,
    "OUTPUT_OK_AD_MISSING": 2,
    "OUTPUT_NOT_CALCULATED": 0,
    "OUTPUT_ERROR": -1,
}
MAPPING = {
    "main_result_classes": {"1": "positive", "0": "negative", "-1": "abstain"},
    "minus_one_label": "Possible NON-Mutagenic",
    "successful_status_codes": [1, 2, 3],
    "identity_requirement": "Returned ID and map-free isomeric canonical SMILES must both match.",
    "domain_admission": {
        "status_code": 1,
        "assessment_class_exact": "GOOD reliability",
        "finite_adi_required": True,
        "numeric_threshold": None,
    },
    "other_classes_errors_missing_or_unknown": "abstain",
    "experimental_or_default_assessment_can_replace_prediction": False,
}
LIMITATIONS = [
    "Research-only bounded pilot; no independent validation, accuracy estimate, safety determination or regulatory compliance claim.",
    "The earlier three sentinel smokes were experimental-reference-backed API checks, not independent predictive validation. Model/training overlap for these inputs is not resolved.",
    "Experimental reference values and VEGA default assessments are retained separately and never replace the raw model prediction; Possible NON-Mutagenic (-1) is an abstention.",
    "Only the probe-observed exact ADI class GOOD reliability, a finite ADI and OUTPUT_OK admit a conservative assessment. No numerical applicability-domain cutoff is invented; missing or other domain evidence abstains.",
    "All 15 recorded ADMET canonical identities are retained regardless of scores or rules. OR-only evidence is descriptive disagreement (exactly one positive), not a sampling rule or validation label.",
    "Canonicalization preserves charge, stereochemistry and disconnected components; salts, protonation, tautomers and isotope forms are not merged.",
    "Known-status raw predictions can be retained when domain evidence is missing; only the separate conservative assessment is domain-admitted. A negative prediction does not establish safety.",
    "No network, installation, new route search, product UI or regulatory classification is part of this experiment. The parent must cap the complete run process at 300 seconds.",
    "Hashes detect artifact changes, not authenticated event time or an externally registered preregistration. Forced termination can leave a running row without a Java response; it remains an abstention.",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def manifest_sha256(manifest):
    return json_sha256({key: value for key, value in manifest.items()
                        if key != "manifest_sha256"})


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _number_raw(value):
    # JSON has no NaN/Infinity. Preserve their spelling instead of substituting 0.
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def _role_index(payload):
    roles = {}
    for route_index, route in enumerate(payload.get("analysis_report", {}).get("routes", [])):
        for step_index, step in enumerate(route.get("steps", [])):
            for material_index, material in enumerate(step.get("materials", [])):
                canonical = canonicalize_smiles(material.get("input_smiles"))
                if canonical is None:
                    continue
                roles.setdefault(canonical, []).append({
                    "kind": "route_material", "route_id": route.get("id"),
                    "step_id": step.get("id"), "role": material.get("role"),
                    "route_source": route.get("source"),
                    "source_pointer": (f"analysis_report.routes[{route_index}].steps"
                                       f"[{step_index}].materials[{material_index}]"),
                    "material": material,
                })
    return roles


def _score_rule_evidence(records, roles, threshold):
    scores = [record["record"].get("score") for record in records]
    score_available = (bool(records) and _finite(threshold) and 0 <= threshold <= 1
                       and all(record["record"].get("status") == "success"
                               and _finite(score) and 0 <= score <= 1
                               for record, score in zip(records, scores)))
    decisions = {score >= threshold for score in scores} if score_available else set()
    admet_positive = decisions.pop() if len(decisions) == 1 else None
    statuses = [role["material"].get("status") for role in roles]
    known_statuses = {"known_concern", "structural_alert", "no_alert_detected"}
    rule_positive = None
    if statuses and all(status in known_statuses for status in statuses):
        rule_positive = any(status != "no_alert_detected" for status in statuses)
    available = admet_positive is not None and rule_positive is not None
    return {
        "available": available, "recorded_scores": scores,
        "recorded_threshold": threshold, "recorded_rule_statuses": statuses,
        "admet_positive": admet_positive, "rule_positive": rule_positive,
        "or_positive": (admet_positive or rule_positive) if available else None,
        "joint_positive": (admet_positive and rule_positive) if available else None,
        "or_only": (admet_positive != rule_positive) if available else None,
        "basis": "Recorded successful score >= recorded threshold and recorded custom-rule statuses only; no new inference.",
        "missing_reason": None if available else "Missing, failed, conflicting or unusable recorded score/rule evidence; retain identity anyway.",
    }


def collect_inputs(payload):
    """Keep every recorded identity; deduplicate only valid canonical structures."""
    admet = payload.get("admet", {})
    assessments = admet.get("assessments")
    if not isinstance(assessments, list) or not assessments:
        raise ValueError("Source must contain admet.assessments")
    roles = _role_index(payload)
    by_canonical = {}
    for index, assessment in enumerate(assessments):
        if not isinstance(assessment, dict):
            raise ValueError("Recorded ADMET assessment must be an object")
        raw = assessment.get("input_smiles")
        canonical = canonicalize_smiles(raw)
        if canonical is None or assessment.get("canonical_smiles") != canonical:
            raise ValueError(f"Invalid or mismatched recorded ADMET identity at index {index}")
        if canonical not in by_canonical:
            by_canonical[canonical] = {
                "canonical_smiles": canonical, "raw_inputs": [],
                "planned_action": "execute_model", "source_roles": list(roles.get(canonical, [])),
                "recorded_admet": [],
            }
        row = by_canonical[canonical]
        if raw not in row["raw_inputs"]:
            row["raw_inputs"].append(raw)
        row["recorded_admet"].append({
            "source_index": index, "source_pointer": f"admet.assessments[{index}]",
            "record": assessment,
        })
    if len(by_canonical) != EXPECTED_ADMET_IDENTITIES:
        raise ValueError(f"Expected all {EXPECTED_ADMET_IDENTITIES} recorded ADMET canonical identities, got {len(by_canonical)}")
    for row in by_canonical.values():
        row["score_rule_evidence"] = _score_rule_evidence(
            row["recorded_admet"], row["source_roles"], admet.get("threshold"))
    for name, raw in SENTINELS:
        canonical = canonicalize_smiles(raw)
        if canonical not in by_canonical:
            by_canonical[canonical] = {
                "canonical_smiles": canonical, "raw_inputs": [],
                "planned_action": "execute_model", "source_roles": [], "recorded_admet": [],
                "score_rule_evidence": _score_rule_evidence([], [], admet.get("threshold")),
            }
        row = by_canonical[canonical]
        if raw not in row["raw_inputs"]:
            row["raw_inputs"].append(raw)
        row["source_roles"].append({"kind": "sentinel", "name": name, "raw_smiles": raw})
    inputs = list(by_canonical.values())
    inputs.append({
        "canonical_smiles": None, "raw_inputs": [INVALID_CONTROL],
        "planned_action": "skip_invalid_control",
        "source_roles": [{"kind": "invalid_control", "raw_smiles": INVALID_CONTROL}],
        "recorded_admet": [],
        "score_rule_evidence": _score_rule_evidence([], [], admet.get("threshold")),
    })
    for index, row in enumerate(inputs, 1):
        row["id"] = f"vega-route-{index:03d}"
    raw_count = len({raw for row in inputs for raw in row["raw_inputs"]})
    if raw_count > MAX_RAW_IDENTITIES or len(inputs) > MAX_RAW_IDENTITIES:
        raise ValueError(f"Pilot exceeds {MAX_RAW_IDENTITIES} distinct raw identities ({raw_count})")
    return inputs


def _planned_denominators(inputs):
    return {
        "planned_records": len(inputs),
        "distinct_raw_identities": len({raw for row in inputs for raw in row["raw_inputs"]}),
        "valid_canonical_identities": sum(row["canonical_smiles"] is not None for row in inputs),
        "recorded_admet_canonical_identities": sum(bool(row["recorded_admet"]) for row in inputs),
        "sentinel_roles": sum(role["kind"] == "sentinel" for row in inputs for role in row["source_roles"]),
        "invalid_control_records": sum(row["planned_action"] == "skip_invalid_control" for row in inputs),
        "note": "Sentinel and recorded ADMET roles can overlap; raw aliases are not repeated model calls.",
    }


def prepare_manifest(manifest_path=MANIFEST, output_path=OUTPUT, *, source_path=SOURCE,
                     jar_path=JAR):
    manifest_path, output_path = Path(manifest_path).resolve(), Path(output_path).resolve()
    if manifest_path == output_path:
        raise ValueError("Manifest and result paths must differ")
    if manifest_path.exists() or output_path.exists():
        raise FileExistsError("Refusing to overwrite an existing manifest or result")
    source = _load_json(source_path)
    inputs = collect_inputs(source)
    artifacts = {"route_experiment": _artifact(source_path),
                 **{name: _artifact(path) for name, path in PROVENANCE.items()}}
    manifest = {
        "schema_version": SCHEMA, "planned_status": "planned_not_executed",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All recorded route ADMET identities plus fixed sentinels and one invalid control; local VEGA SarPy pilot only.",
        "model_class": MODEL_CLASS, "expected_model_identity": dict(MODEL_IDENTITY),
        "jar": _artifact(jar_path), "source_artifacts": artifacts,
        "upstream_provenance": {
            "source_input_recorded": source.get("source_input"),
            "input_sha256_recorded_not_recomputed": source.get("input_sha256"),
            "scope": source.get("scope"), "limitations": source.get("limitations"),
            "admet_model": source["admet"].get("model"),
            "admet_limitations": source["admet"].get("limitations"),
        },
        "planned_output_path": str(output_path), "max_distinct_raw_identities": MAX_RAW_IDENTITIES,
        "output_status_constants": dict(STATUS_CONSTANTS), "raw_prediction_mapping": MAPPING,
        "limitations": list(LIMITATIONS), "inputs": inputs,
        "denominators": _planned_denominators(inputs), "input_sha256": json_sha256(inputs),
    }
    manifest["manifest_id"] = f"vega-route-pilot-{manifest['input_sha256'][:16]}"
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def verify_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("manifest_sha256") != manifest_sha256(manifest):
        raise ValueError("Manifest digest mismatch")
    expected = {
        "schema_version": SCHEMA, "planned_status": "planned_not_executed",
        "model_class": MODEL_CLASS, "expected_model_identity": MODEL_IDENTITY,
        "max_distinct_raw_identities": MAX_RAW_IDENTITIES,
        "output_status_constants": STATUS_CONSTANTS, "raw_prediction_mapping": MAPPING,
        "limitations": LIMITATIONS,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Unrecognized manifest policy or identity: {key}")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or manifest.get("input_sha256") != json_sha256(inputs):
        raise ValueError("Input digest mismatch")
    if manifest.get("manifest_id") != f"vega-route-pilot-{manifest['input_sha256'][:16]}":
        raise ValueError("Manifest ID mismatch")
    if not inputs or len(inputs) > MAX_RAW_IDENTITIES:
        raise ValueError("Manifest input cap exceeded or empty")
    raw_identities = {raw for row in inputs for raw in row.get("raw_inputs", [])}
    if len(raw_identities) > MAX_RAW_IDENTITIES:
        raise ValueError("Manifest distinct raw input cap exceeded")
    ids = [row.get("id") for row in inputs]
    canonical = [row.get("canonical_smiles") for row in inputs if row.get("canonical_smiles") is not None]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Manifest has missing or duplicate IDs")
    if len(set(canonical)) != len(canonical):
        raise ValueError("Manifest has duplicate canonical identities")
    artifacts = manifest.get("source_artifacts", {})
    if set(artifacts) != {"route_experiment", *PROVENANCE}:
        raise ValueError("Missing or unexpected source provenance")
    for artifact in [manifest["jar"], *artifacts.values()]:
        if file_sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Source/JAR digest mismatch: {artifact['path']}")
    expected_inputs = collect_inputs(_load_json(artifacts["route_experiment"]["path"]))
    if inputs != expected_inputs:
        raise ValueError("Manifest identities, IDs, roles or coverage differ from frozen source union")
    if manifest.get("denominators") != _planned_denominators(inputs):
        raise ValueError("Manifest denominator mismatch")


def open_gateway(jar_path):
    # Optional dependency is deliberately absent from module imports and prepare.
    from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

    return JavaGateway(gateway_parameters=GatewayParameters(
        port=launch_gateway(classpath=str(jar_path), die_on_exit=True), read_timeout=30))


def _error(stage, exc):
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}


def _capture(row, target, field, getter, transform=None):
    try:
        value = getter()
        target[field] = transform(value) if transform else value
    except Exception as exc:
        row["errors"].append(_error(field, exc))


def _blank_result(entry):
    invalid = entry["planned_action"] == "skip_invalid_control"
    return {
        **entry, "execution_status": "skipped_invalid_control" if invalid else "pending",
        "java_conversion_attempted": False, "java_model_called": False,
        "output_returned": False, "parsed_valid": None, "status_code": None,
        "results_raw": None, "main_value_raw": None, "raw_class_label": None,
        "default_assessment": None, "default_assessment_verbose": None,
        "error": None, "conversion_error_raw": None, "errors": [],
        "returned_smiles": None, "returned_id": None, "returned_canonical_smiles": None,
        "has_experimental": None, "experimental_value_raw": None,
        "experimental_numeric_raw": None, "adi": None, "ad_index_count": None,
        "identity_verified": False, "successful_known_status": False, "domain_admitted": False,
        "raw_prediction": "abstain", "raw_prediction_reason": "invalid_control" if invalid else "not_executed",
        "conservative_assessment": "abstain",
        "assessment_reasons": ["invalid_control_never_sent_to_java" if invalid else "not_executed"],
    }


def map_assessment(row):
    """Map only the numeric prediction, never default or experimental text."""
    row.update(raw_prediction="abstain", conservative_assessment="abstain",
               identity_verified=False, successful_known_status=False, domain_admitted=False)
    reasons = []
    if not row["output_returned"]:
        reason = ("invalid_control_never_sent_to_java" if row["planned_action"] == "skip_invalid_control"
                  else "no_java_output")
        row.update(raw_prediction_reason=reason, assessment_reasons=[reason])
        return
    canonical = canonicalize_smiles(row["returned_smiles"])
    row["returned_canonical_smiles"] = canonical
    if canonical != row["canonical_smiles"] or canonical is None:
        reasons.append("returned_smiles_identity_mismatch_or_missing")
    if row["returned_id"] != row["id"]:
        reasons.append("returned_id_mismatch_or_missing")
    row["identity_verified"] = not reasons
    status = row["status_code"]
    row["successful_known_status"] = type(status) is int and status in MAPPING["successful_status_codes"]
    if not row["successful_known_status"]:
        reasons.append("failed_missing_or_unknown_output_status")
    if row["parsed_valid"] is not True:
        reasons.append("converter_not_valid")
    if row["error"] not in (None, ""):
        reasons.append("reported_model_error")
    value = row["main_value_raw"]
    if not _finite(value):
        reasons.append("missing_or_nonfinite_raw_class")
    elif value not in (-1, 0, 1):
        reasons.append("unknown_raw_class")
    if reasons:
        row.update(raw_prediction_reason=";".join(reasons), assessment_reasons=reasons)
        return
    if value == -1:
        row.update(raw_prediction_reason="possible_non_mutagenic_is_abstention",
                   assessment_reasons=["possible_non_mutagenic_is_abstention"])
        return
    row["raw_prediction"] = "positive" if value == 1 else "negative"
    row["raw_prediction_reason"] = "successful_known_status_identity_verified_raw_class"
    adi = row["adi"]
    if status != STATUS_CONSTANTS["OUTPUT_OK"]:
        reasons.append("status_reports_missing_or_inapplicable_domain")
    if not isinstance(adi, dict):
        reasons.append("missing_applicability_domain")
    else:
        if not _finite(adi.get("value")):
            reasons.append("missing_or_nonfinite_domain_value")
        if adi.get("assessment_class") != "GOOD reliability":
            reasons.append("domain_not_explicit_good_reliability")
    if row["errors"]:
        reasons.append("incomplete_or_failed_output_extraction")
    row["domain_admitted"] = not reasons
    row["conservative_assessment"] = "abstain" if reasons else row["raw_prediction"]
    row["assessment_reasons"] = reasons or ["raw_class_with_probe_observed_good_domain"]


def extract_output(row, output):
    row["output_returned"] = True
    for field, getter, transform in (
        ("status_code", output.getStatus, None),
        ("results_raw", output.getResults, lambda values: list(values)),
        ("main_value_raw", output.getMainResultValue, _number_raw),
        ("default_assessment", output.getAssessment, None),
        ("default_assessment_verbose", output.getAssessmentVerbose, None),
        ("error", output.getErrMessage, None),
        ("returned_smiles", output.getMoleculeSMILES, None),
        ("returned_id", output.getMoleculeId, None),
        ("has_experimental", output.HasExperimental, None),
        ("ad_index_count", output.getADIndexSize, None),
    ):
        _capture(row, row, field, getter, transform)
    if row["results_raw"]:
        row["raw_class_label"] = row["results_raw"][0]
    if row["has_experimental"] is True:
        _capture(row, row, "experimental_value_raw", output.getExperimentalFormatted)
        _capture(row, row, "experimental_numeric_raw", output.getExperimental, _number_raw)
    domain = {}
    _capture(row, domain, "adi_object", output.getADI)
    if domain.get("adi_object") is not None:
        adi = domain["adi_object"]
        row["adi"] = {"name": None, "value": None, "assessment": None, "assessment_class": None}
        for field, getter, transform in (
            ("name", adi.GetIndexName, None), ("value", adi.GetIndexValue, _number_raw),
            ("assessment", adi.GetAssessment, None),
            ("assessment_class", adi.GetAssessmentClassAsString, None),
        ):
            _capture(row, row["adi"], field, getter, transform)
    row["execution_status"] = "output_received" if not row["errors"] else "output_extraction_failed"
    map_assessment(row)


def _execute_row(row, gateway, model):
    if row["planned_action"] != "execute_model" or row["canonical_smiles"] is None:
        map_assessment(row)
        return
    try:
        row["java_conversion_attempted"] = True
        molecule = gateway.jvm.insilico.core.molecule.conversion.SmilesMolecule.Convert(row["canonical_smiles"])
        molecule.SetId(row["id"])
        row["parsed_valid"] = molecule.IsValid()
        if row["parsed_valid"] is not True:
            row["execution_status"] = "conversion_failed"
            row["conversion_error_raw"] = molecule.GetErrors().GetMessages()
        else:
            row["java_model_called"] = True
            extract_output(row, model.Execute(molecule))
    except Exception as exc:
        row["execution_status"] = "row_failed"
        row["errors"].append(_error("execute_row", exc))
    map_assessment(row)


def _counts(rows):
    return {
        "java_conversion_attempts": sum(row["java_conversion_attempted"] for row in rows),
        "java_model_calls": sum(row["java_model_called"] for row in rows),
        "returned_output_records": sum(row["output_returned"] for row in rows),
        "identity_verified_records": sum(row["identity_verified"] for row in rows),
        "successful_known_status_records": sum(row["successful_known_status"] for row in rows),
        "domain_admitted_records": sum(row["domain_admitted"] for row in rows),
        "experimental_reference_records": sum(row["has_experimental"] is True for row in rows),
        "rows_with_errors": sum(bool(row["errors"]) or row["error"] not in (None, "") for row in rows),
        "raw_prediction_counts": {value: sum(row["raw_prediction"] == value for row in rows)
                                  for value in ("positive", "negative", "abstain")},
        "conservative_assessment_counts": {value: sum(row["conservative_assessment"] == value for row in rows)
                                           for value in ("positive", "negative", "abstain")},
        "recorded_admet_with_output": sum(bool(row["recorded_admet"]) and row["output_returned"] for row in rows),
        "recorded_admet_conservatively_assessed": sum(bool(row["recorded_admet"]) and row["conservative_assessment"] != "abstain" for row in rows),
    }


def _persist(destination, report):
    report["counts"] = _counts(report["results"])
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination = Path(destination)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                         dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp") as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        # Close the sibling file before replacement, including on Windows.
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass  # Cleanup must not hide the original checkpoint failure.
        raise


def run_pilot(manifest_path=MANIFEST, output_path=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = _load_json(manifest_path)
    verify_manifest(manifest)
    destination = Path(output_path or manifest["planned_output_path"]).resolve()
    if destination == manifest_path or destination.exists():
        raise FileExistsError("Refusing to overwrite an existing manifest or result")
    report = {
        "schema_version": SCHEMA, "run_status": "starting",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path), "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": file_sha256(manifest_path), "input_sha256": manifest["input_sha256"],
        "output_path": str(destination), "planned_output_path": manifest["planned_output_path"],
        "model_class": MODEL_CLASS, "jar": manifest["jar"],
        "source_artifacts": manifest["source_artifacts"],
        "upstream_provenance": manifest["upstream_provenance"],
        "raw_prediction_mapping": manifest["raw_prediction_mapping"],
        "limitations": manifest["limitations"], "denominators": manifest["denominators"],
        "errors": [], "results": [_blank_result(entry) for entry in manifest["inputs"]],
    }
    gateway = None
    # Admit only a new result, then close it before any atomic replacement.
    with destination.open("x", encoding="utf-8"):
        pass
    _persist(destination, report)
    try:
        gateway = open_gateway(manifest["jar"]["path"])
        model = gateway.jvm.insilico.mutagenicity_sarpy.ismMutagenicitySarpy()
        info = model.getInfo()
        report["model"] = {"name": info.getName(), "version": info.getVersion(),
                           "key": info.getKey(), "class_values_raw": str(info.getClassValues()),
                           "qmrf": info.getQMRF(), "summary": info.getSummary()}
        if any(report["model"][key] != value for key, value in MODEL_IDENTITY.items()):
            raise ValueError("Runtime model identity differs from the preregistered probe model")
        report["result_names"] = list(model.GetResultsName())
        output_class = gateway.jvm.insilico.core.model.InsilicoModelOutput
        report["output_status_constants"] = {name: getattr(output_class, name) for name in STATUS_CONSTANTS}
        if report["output_status_constants"] != STATUS_CONSTANTS:
            raise ValueError("Runtime output status constants differ from probe evidence")
        report["run_status"] = "running"
        _persist(destination, report)
        for row in report["results"]:
            if row["planned_action"] == "skip_invalid_control":
                continue
            row["execution_status"] = "running"
            row["raw_prediction_reason"] = "java_response_not_yet_available"
            row["assessment_reasons"] = ["java_response_not_yet_available"]
            _persist(destination, report)
            _execute_row(row, gateway, model)
            _persist(destination, report)
        report["run_status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        report["run_status"] = "failed"
        report["errors"].append(_error("run", exc))
        for row in report["results"]:
            if row["execution_status"] in ("pending", "running"):
                row["execution_status"] = "interrupted_or_not_executed"
                map_assessment(row)
    finally:
        # A failed or interrupted replacement leaves the previous JSON intact.
        try:
            _persist(destination, report)
        finally:
            try:
                if gateway is not None:
                    gateway.shutdown()
            except Exception as exc:
                report["errors"].append(_error("gateway_shutdown", exc))
                report["run_status"] = "failed"
            finally:
                report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                _persist(destination, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--prepare", action="store_true", help="Freeze the complete bounded input manifest; no Java")
    phases.add_argument("--run", action="store_true", help="Verify the frozen manifest, then execute the local model")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, help="Nonexisting result path; defaults to the prepared path")
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            manifest = prepare_manifest(args.manifest, args.output or OUTPUT)
            print(json.dumps({"manifest": str(args.manifest.resolve()),
                              "planned_status": manifest["planned_status"],
                              "manifest_sha256": manifest["manifest_sha256"],
                              "denominators": manifest["denominators"]}, ensure_ascii=False))
            return 0
        report = run_pilot(args.manifest, args.output)
        print(json.dumps({"output": report["output_path"], "run_status": report["run_status"],
                          "denominators": report["denominators"], "counts": report["counts"]}, ensure_ascii=False))
        return 0 if report["run_status"] == "completed" else 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"VEGA pilot refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
