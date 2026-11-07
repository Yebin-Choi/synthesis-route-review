"""Offline AMES benchmarking of fixed local rules and supplied predictions.

This module uses the standard library plus the existing toxicology module. It
never trains a model, chooses thresholds, invokes external tools, or establishes
that an experimental label or a model's training/test separation is authentic.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

if __package__:
    from . import toxicology
else:
    import toxicology


_DATA_COLUMNS = {"id", "smiles", "label", "source", "split"}
_PREDICTION_COLUMNS = (
    "id", "smiles", "tool", "model", "version", "endpoint", "prediction",
    "applicability", "source",
)
_SPLITS = ("development", "test")
_RULE_MAPPING = {
    "known_concern": 1,
    "structural_alert": 1,
    "no_alert_detected": 0,
    "invalid": None,
}
_ENSEMBLES = {
    "any_positive": "Any 1 gives 1; all sources returning 0 gives 0; otherwise abstain.",
    "strict_majority": (
        "All selected sources must return 0 or 1; more than half decides; ties abstain."
    ),
    "unanimity": "All selected sources must return the same non-abstaining value.",
}


def _csv_rows(path, required):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames
        if not headers or any(not name for name in headers):
            raise ValueError(f"{path}: a nonempty CSV header is required")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path}: duplicate CSV column names")
        missing = set(required) - set(headers)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            context = f"{path}:{reader.line_num}"
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{context}: row length does not match CSV header")
            yield context, row


def _text(row, column, context):
    value = row[column].strip()
    if not value or "\x00" in value:
        raise ValueError(f"{context}: {column} must be nonempty and contain no NUL")
    return value


def _load_dataset(path):
    materials = []
    by_structure = {}
    by_id = {}
    exclusions = []
    input_rows = 0
    for context, row in _csv_rows(path, _DATA_COLUMNS):
        input_rows += 1
        material_id = _text(row, "id", context)
        smiles = _text(row, "smiles", context)
        source = _text(row, "source", context)
        split = _text(row, "split", context)
        label_text = _text(row, "label", context)
        if label_text not in {"0", "1"}:
            raise ValueError(f"{context}: label must be 0 or 1")
        if split not in _SPLITS:
            raise ValueError(f"{context}: split must be development or test")
        label = int(label_text)
        canonical = toxicology.canonicalize_smiles(smiles)
        if canonical is None:
            raise ValueError(f"{context}: invalid SMILES for id {material_id!r}")
        previous = by_structure.get(canonical)
        if previous is not None:
            if previous["label"] != label:
                raise ValueError(
                    f"{context}: conflicting labels for normalized molecule "
                    f"{canonical!r} ({previous['id']!r}, {material_id!r})"
                )
            if previous["split"] != split:
                raise ValueError(
                    f"{context}: cross-split leakage for normalized molecule "
                    f"{canonical!r} ({previous['id']!r}, {material_id!r})"
                )
        previous_id = by_id.get(material_id)
        if previous_id is not None and previous_id["canonical_smiles"] != canonical:
            raise ValueError(f"{context}: id {material_id!r} refers to different structures")
        retained_id = previous["id"] if previous is not None else material_id
        by_id[material_id] = {
            "canonical_smiles": canonical,
            "retained_id": retained_id,
        }
        if previous is not None:
            exclusions.append({
                "row": context,
                "id": material_id,
                "smiles": smiles,
                "canonical_smiles": canonical,
                "label": label,
                "source": source,
                "split": split,
                "retained_id": retained_id,
                "reason": "duplicate_normalized_material_same_label_and_split",
            })
            continue
        material = {
            "id": material_id,
            "smiles": smiles,
            "canonical_smiles": canonical,
            "label": label,
            "source": source,
            "split": split,
            "predictions": {},
        }
        by_structure[canonical] = material
        materials.append(material)
    if not materials:
        raise ValueError(f"{path}: dataset must contain at least one valid material")
    return materials, by_id, {
        "path": str(Path(path)),
        "input_rows": input_rows,
        "retained_materials": len(materials),
        "exclusion_count": len(exclusions),
        "exclusions": exclusions,
        "deduplication": (
            "The first row for each normalized molecule is retained. Same-label, "
            "same-split duplicates are excluded, not additional observations. "
            "Predictions for excluded alternative ids are reported and excluded; "
            "they never substitute for a missing prediction for the retained id."
        ),
        "normalization": (
            "toxicology.canonicalize_smiles: atom-map-free isomeric canonical SMILES; "
            "no salt stripping, neutralization, tautomer merging or stereoisomer merging."
        ),
    }


def _load_external(path, by_id):
    records = {}
    seen = set()
    sources = {}
    row_counts = Counter()
    exclusions = []
    if path is not None:
        for context, row in _csv_rows(path, _PREDICTION_COLUMNS):
            fields = {name: _text(row, name, context) for name in _PREDICTION_COLUMNS}
            if fields["endpoint"] != "ames":
                raise ValueError(f"{context}: only endpoint='ames' is accepted")
            if fields["prediction"] not in {"0", "1", "abstain"}:
                raise ValueError(f"{context}: prediction must be 0, 1 or abstain")
            domain = fields["applicability"]
            if domain not in {"in_domain", "out_of_domain", "unknown"}:
                raise ValueError(f"{context}: invalid applicability {domain!r}")
            material_id = fields["id"]
            if material_id not in by_id:
                raise ValueError(f"{context}: unknown material id {material_id!r}")
            canonical = toxicology.canonicalize_smiles(fields["smiles"])
            if canonical is None:
                raise ValueError(f"{context}: invalid prediction SMILES for {material_id!r}")
            matched = by_id[material_id]
            if canonical != matched["canonical_smiles"]:
                raise ValueError(f"{context}: structure mismatch for id {material_id!r}")
            key = (fields["tool"], fields["model"], fields["version"])
            duplicate_key = (material_id, key)
            if duplicate_key in seen:
                raise ValueError(
                    f"{context}: duplicate tool-model-version prediction for {material_id!r}"
                )
            seen.add(duplicate_key)
            records.setdefault(key, {})
            sources.setdefault(key, Counter())[fields["source"]] += 1
            row_counts[key] += 1
            if matched["retained_id"] != material_id:
                exclusions.append({
                    "row": context,
                    "id": material_id,
                    "retained_id": matched["retained_id"],
                    "tool": key[0],
                    "model": key[1],
                    "version": key[2],
                    "source": fields["source"],
                    "reason": "prediction_for_excluded_duplicate_id",
                })
                continue
            reported = None if fields["prediction"] == "abstain" else int(fields["prediction"])
            effective = None if domain == "out_of_domain" else reported
            reason = None
            if domain == "out_of_domain":
                reason = "out_of_domain"
            elif reported is None:
                reason = "explicit_abstention"
            records[key][material_id] = {
                "prediction": effective,
                "reported_prediction": reported,
                "applicability": domain,
                "source": fields["source"],
                "abstention_reason": reason,
            }
    systems = []
    predictions = {}
    for index, key in enumerate(sorted(records), start=1):
        system_id = f"external_{index}"
        predictions[system_id] = records[key]
        systems.append({
            "id": system_id,
            "tool": key[0],
            "model": key[1],
            "version": key[2],
            "endpoint": "ames",
            "prediction_rows": row_counts[key],
            "retained_prediction_rows": len(records[key]),
            "source_counts": dict(sorted(sources[key].items())),
            "training_overlap": "unknown",
            "provenance_verification": "Supplied CSV claims; not independently verified.",
        })
    return systems, predictions, {
        "path": str(Path(path)) if path is not None else None,
        "input_rows": sum(row_counts.values()),
        "exclusion_count": len(exclusions),
        "exclusions": exclusions,
    }


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _metrics(labels, predictions):
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have equal lengths")
    counts = Counter((label, value) for label, value in zip(labels, predictions))
    tp, fn = counts[1, 1], counts[1, 0]
    tn, fp = counts[0, 0], counts[0, 1]
    assessed = tp + fn + tn + fp
    positives = sum(labels)
    negatives = len(labels) - positives
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        "total": len(labels),
        "positives": positives,
        "negatives": negatives,
        "assessed": assessed,
        "coverage": _ratio(assessed, len(labels)),
        "abstentions": len(labels) - assessed,
        "abstentions_by_label": {"0": counts[0, None], "1": counts[1, None]},
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None else None
        ),
        "precision": _ratio(tp, tp + fp),
        "accuracy": _ratio(tp + tn, assessed),
        "end_to_end_positive_detection_fraction": _ratio(tp, positives),
    }


def _system_metrics(materials, system_id):
    entries = [material["predictions"][system_id] for material in materials]
    result = _metrics(
        [material["label"] for material in materials],
        [entry["prediction"] for entry in entries],
    )
    domains = Counter(entry["applicability"] for entry in entries)
    result["applicability_counts"] = {
        domain: domains[domain]
        for domain in ("in_domain", "out_of_domain", "unknown", "missing")
    }
    result["unknown_applicability_assessed"] = sum(
        entry["applicability"] == "unknown" and entry["prediction"] is not None
        for entry in entries
    )
    result["abstention_reasons"] = dict(sorted(Counter(
        entry["abstention_reason"] for entry in entries if entry["prediction"] is None
    ).items()))
    return result


def _ensemble_predictions(values):
    complete = all(value is not None for value in values)
    positives = values.count(1)
    any_positive = 1 if positives else (0 if complete else None)
    majority = None
    if complete and positives * 2 != len(values):
        majority = int(positives * 2 > len(values))
    unanimous = values[0] if complete and len(set(values)) == 1 else None
    return {
        "any_positive": any_positive,
        "strict_majority": majority,
        "unanimity": unanimous,
    }


def _ensemble_metrics(materials):
    labels = [material["label"] for material in materials]
    return {
        name: _metrics(labels, [material["ensembles"][name] for material in materials])
        for name in _ENSEMBLES
    }


_RULES_SOURCE = "app/toxicology.py:assess_material"
_NOTES = (
    "Research benchmark only; no safety status, regulatory conclusion or accuracy guarantee.",
    "Local known_concern/structural_alert map to 1, no_alert_detected to 0, invalid to "
    "abstain ONLY for this AMES benchmark; no-alert is not evidence of safety.",
    "Labels, experimental provenance and external model/version/source claims are supplied "
    "by CSV and are not independently verified.",
    "No training, model calls, rule tuning or threshold selection is performed on either "
    "split. Development and test are reported separately; test must not be used for tuning.",
    "Overlap with pretraining/training data may be unknown. Known materials are part of "
    "the local lookup, not independent experimental verification.",
    "Sensitivity, specificity, balanced_accuracy, precision and accuracy condition on "
    "non-abstaining predictions. End-to-end positive detection divides TP by ALL labeled "
    "positives, including abstentions; undefined denominators are null.",
    "Unknown applicability is allowed and counted, not evidence of being in domain. "
    "Out-of-domain, missing and explicitly abstaining predictions are abstentions.",
    "The common intersection contains only materials assessed by EVERY selected source "
    "after applicability filtering; it can be empty and is not the full-dataset result.",
    "Every ensemble uses rules plus every distinct external tool/model/version in the "
    "prediction CSV. Different models from one tool are distinct, potentially correlated votes.",
    "Map-free canonicalization does not detect leakage across different salts, protonation "
    "states, tautomers or related stereoisomers.",
)


def _rules_system():
    """The local rule screen as one entry of the systems table."""
    digest = hashlib.sha256(Path(toxicology.__file__).read_bytes()).hexdigest()
    return {
        "id": "rules", "tool": "custom_rules", "model": "assess_material",
        "version": f"sha256:{digest}", "endpoint": "ames", "source": _RULES_SOURCE,
        "training_overlap": (
            "Known-material lookup is incorporated; it is not independent verification. "
            "Other overlap with the supplied labels is unknown."),
        "applicability": "unknown; no validated applicability domain is established",
    }


def _attach_predictions(materials, system_ids, external_predictions):
    """Record every system's call per material, abstentions included."""
    for material in materials:
        assessment = toxicology.assess_material(material["canonical_smiles"])
        status = assessment["status"]
        if status not in _RULE_MAPPING:
            raise ValueError(f"Unsupported rules assessment status: {status!r}")
        value = _RULE_MAPPING[status]
        material["rules_status"] = status
        material["rules_name"] = assessment.get("name")
        material["rules_alerts"] = assessment.get("alerts", [])
        material["predictions"]["rules"] = {
            "prediction": value, "reported_prediction": value,
            "applicability": "unknown", "source": _RULES_SOURCE,
            "abstention_reason": "invalid_assessment" if value is None else None,
        }
        # A system with no row for this material abstains. It is never read as
        # a negative, which is the whole reason the benchmark tracks coverage.
        for system_id, predictions in external_predictions.items():
            material["predictions"][system_id] = predictions.get(material["id"], {
                "prediction": None, "reported_prediction": None,
                "applicability": "missing", "source": None,
                "abstention_reason": "missing_prediction",
            })
        material["ensembles"] = _ensemble_predictions(
            [material["predictions"][system_id]["prediction"]
             for system_id in system_ids])


def _split_results(materials, system_ids, dataset):
    """Per-split metrics, each tool measured on the same assessed rows."""
    results = {}
    dataset["splits"] = {}
    for split in _SPLITS:
        subset = [material for material in materials if material["split"] == split]
        intersection = [
            material for material in subset
            if all(material["predictions"][system_id]["prediction"] is not None
                   for system_id in system_ids)]
        dataset["splits"][split] = {
            "total": len(subset),
            "positives": sum(material["label"] for material in subset),
            "negatives": sum(material["label"] == 0 for material in subset),
            "source_counts": dict(sorted(
                Counter(material["source"] for material in subset).items())),
            "known_material_count": sum(
                material["rules_status"] == "known_concern" for material in subset),
        }
        results[split] = {
            "systems": {system_id: _system_metrics(subset, system_id)
                        for system_id in system_ids},
            "common_intersection": {
                "total": len(intersection),
                "excluded_materials": len(subset) - len(intersection),
                "material_ids": [material["id"] for material in intersection],
                "systems": {system_id: _system_metrics(intersection, system_id)
                            for system_id in system_ids},
                "ensembles": _ensemble_metrics(intersection),
            },
            "ensembles": _ensemble_metrics(subset),
        }
    return results


def evaluate_dataset(data_path, predictions_path=None) -> dict:
    """Evaluate fixed AMES sources without training or tuning, returning JSON data.

    Invalid input raises ValueError (or a file/CSV decoding error). Each normalized
    molecule is one observation; the first same-split, same-label row wins. External
    rows must match both a dataset id and its normalized structure. Missing rows,
    explicit abstentions and out-of-domain rows never become negative predictions.
    Metrics are conditional on assessment except coverage, abstention counts and
    end_to_end_positive_detection_fraction. JSON null represents abstention or an
    undefined denominator. All tools share one assessed intersection per split.
    """
    materials, by_id, dataset = _load_dataset(data_path)
    external_systems, external_predictions, external_input = _load_external(
        predictions_path, by_id)
    systems = [_rules_system()] + external_systems
    system_ids = [system["id"] for system in systems]
    _attach_predictions(materials, system_ids, external_predictions)
    return {
        "schema_version": "1.0",
        "endpoint": "ames",
        "dataset": dataset,
        "external_input": external_input,
        "systems": systems,
        "rules_benchmark_mapping": dict(_RULE_MAPPING),
        "ensemble_definitions": dict(_ENSEMBLES),
        "ensemble_system_ids": system_ids,
        "notes": list(_NOTES),
        "results": _split_results(materials, system_ids, dataset),
        "materials": materials,
    }


def _export_rules(report, path):
    rules = report["systems"][0]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        columns = list(_PREDICTION_COLUMNS) + ["canonical_smiles", "label", "split", "rules_status"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for material in report["materials"]:
            value = material["predictions"]["rules"]["prediction"]
            writer.writerow({
                "id": material["id"],
                "smiles": material["smiles"],
                "tool": rules["tool"],
                "model": rules["model"],
                "version": rules["version"],
                "endpoint": "ames",
                "prediction": "abstain" if value is None else value,
                "applicability": "unknown",
                "source": rules["source"],
                "canonical_smiles": material["canonical_smiles"],
                "label": material["label"],
                "split": material["split"],
                "rules_status": material["rules_status"],
            })


def _number(value):
    return "n/a" if value is None else f"{value:.3f}"


def _print_summary(report):
    dataset = report["dataset"]
    print(f"AMES: {dataset['retained_materials']} materials; {dataset['exclusion_count']} duplicate rows excluded.")
    print("Research benchmark, not safety verification. Training overlap may be unknown.")
    for system in report["systems"]:
        print(f"{system['id']}: {system['tool']} / {system['model']} / {system['version']}")
    for split, result in report["results"].items():
        print(f"{split}: common assessed intersection={result['common_intersection']['total']}")
        for name, metrics in list(result["systems"].items()) + list(result["ensembles"].items()):
            domains = metrics.get("applicability_counts")
            domain_summary = (
                f" unknown_AD={domains['unknown']} out_of_domain={domains['out_of_domain']}"
                f" missing={domains['missing']}" if domains is not None else ""
            )
            print(
                f"  {name}: assessed={metrics['assessed']}/{metrics['total']} "
                f"coverage={_number(metrics['coverage'])} "
                f"sensitivity={_number(metrics['sensitivity'])} "
                f"specificity={_number(metrics['specificity'])} "
                f"balanced_accuracy={_number(metrics['balanced_accuracy'])} "
                f"end_to_end_positive_detection={_number(metrics['end_to_end_positive_detection_fraction'])}"
                f"{domain_summary}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Labeled AMES CSV")
    parser.add_argument("--predictions", type=Path, help="Optional external AMES predictions CSV")
    parser.add_argument("--output", required=True, type=Path, help="JSON report path")
    parser.add_argument("--export-rules", type=Path, help="Optional per-material rules predictions CSV")
    args = parser.parse_args(argv)
    try:
        inputs = {args.data.resolve()}
        if args.predictions is not None:
            inputs.add(args.predictions.resolve())
        outputs = [args.output.resolve()]
        if args.export_rules is not None:
            outputs.append(args.export_rules.resolve())
        if any(path in inputs for path in outputs) or len(set(outputs)) != len(outputs):
            raise ValueError("Output paths must differ from input paths and each other")
        report = evaluate_dataset(args.data, args.predictions)
        args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        if args.export_rules is not None:
            _export_rules(report, args.export_rules)
    except (ValueError, OSError, csv.Error, UnicodeError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 2
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
