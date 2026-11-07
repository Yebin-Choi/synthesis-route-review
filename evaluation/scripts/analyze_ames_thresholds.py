"""Describe saved AMES scores; never infer, select a threshold or claim validation.

The CLI joins ames_evaluation.csv to admet_predictions.csv, not the raw ADMET
matrix (which has no material IDs). Matplotlib is needed only to write the PNG.
"""

import argparse
from bisect import bisect_right
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'
sys.path.insert(0, str(PROJECT / "app"))
from toxicology import canonicalize_smiles


THRESHOLDS = tuple(index / 100 for index in range(101))
BASELINE_THRESHOLD = 0.5
HISTOGRAM_EDGES = tuple(index / 20 for index in range(21))
_DATA_COLUMNS = {"id", "smiles", "label", "source", "split"}
_SCORE_COLUMNS = {
    "id", "smiles", "tool", "model", "version", "endpoint", "prediction",
    "applicability", "source", "score",
}
_SCORE_STATUSES = (
    "valid", "missing_score", "non_numeric_score", "non_finite_score",
    "out_of_range_score",
)


def _csv_rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames
        if not headers or any(not name for name in headers):
            raise ValueError(f"{path}: a nonempty CSV header is required")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path}: duplicate CSV column names")
        missing = required - set(headers)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{reader.line_num}: row length does not match header")
            rows.append(row)
        return rows


def _text(row, column, context):
    value = row.get(column)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{context}: {column} must be nonempty text without NUL")
    # In particular, never trim or otherwise transform the identity join key.
    return value


def _label(value, context):
    text = str(value).strip()
    if isinstance(value, bool) or text not in {"0", "1"}:
        raise ValueError(f"{context}: observed label must be 0 or 1")
    return int(text)


def _materials(rows):
    materials = []
    by_id = {}
    by_structure = {}
    for index, row in enumerate(rows, start=1):
        context = f"labeled row {index}"
        material_id = _text(row, "id", context)
        smiles = _text(row, "smiles", context)
        label = _label(row.get("label"), context)
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            raise ValueError(f"{context}: invalid SMILES for id {material_id!r}")
        previous = by_id.get(material_id)
        if previous is not None:
            if previous["canonical_smiles"] != canonical or previous["label"] != label:
                raise ValueError(f"{context}: conflicting labeled rows for id {material_id!r}")
            raise ValueError(f"{context}: duplicate labeled id {material_id!r}")
        previous = by_structure.get(canonical)
        if previous is not None:
            if previous["label"] != label:
                raise ValueError(f"{context}: conflicting labels for canonical structure {canonical!r}")
            raise ValueError(f"{context}: duplicate canonical structure {canonical!r}")
        material = dict(row, id=material_id, label=label, canonical_smiles=canonical)
        materials.append(material)
        by_id[material_id] = material
        by_structure[canonical] = material
    if not materials:
        raise ValueError("The labeled dataset must contain at least one material")
    return materials, by_id


def join_rows(data_rows, prediction_rows):
    """Join one saved model's records by exact ID and verify canonical identity.

    Repeated IDs/structures, conflicting labels, unmatched IDs and mismatched
    structures are errors, never implicit deduplication or structure-only joins.
    A failed score must still have an identity-bearing prediction record.
    Prediction-file labels and saved binary prediction direction are not used
    as observed labels or threshold decisions.
    """
    materials, by_id = _materials(data_rows)
    for material in materials:
        _text(material, "source", f"labeled id {material['id']!r}")
        _text(material, "split", f"labeled id {material['id']!r}")
    joined = {}
    system = None
    for index, row in enumerate(prediction_rows, start=1):
        context = f"prediction row {index}"
        material_id = _text(row, "id", context)
        if material_id in joined:
            raise ValueError(f"{context}: duplicate prediction id {material_id!r}")
        if material_id not in by_id:
            raise ValueError(f"{context}: unmatched prediction id {material_id!r}")
        canonical = canonicalize_smiles(_text(row, "smiles", context))
        if canonical is None:
            raise ValueError(f"{context}: invalid prediction SMILES for id {material_id!r}")
        material = by_id[material_id]
        if canonical != material["canonical_smiles"]:
            raise ValueError(f"{context}: structure mismatch for id {material_id!r}")
        declared_system = tuple(_text(row, field, context) for field in ("tool", "model", "version"))
        if system is not None and system != declared_system:
            raise ValueError("Saved predictions must describe one tool/model/version, not pooled models")
        system = declared_system
        if _text(row, "endpoint", context) != "ames":
            raise ValueError(f"{context}: only endpoint='ames' is accepted")
        source = _text(row, "source", context)
        joined[material_id] = {
            "id": material_id,
            "smiles": material["smiles"],
            "canonical_smiles": canonical,
            "label": material["label"],
            "split": material["split"],
            "data_source": material["source"],
            "score": row.get("score"),
            "prediction": row.get("prediction"),
            "applicability": row.get("applicability"),
            "tool": system[0], "model": system[1], "version": system[2],
            "prediction_source": source,
        }
    missing = set(by_id) - set(joined)
    if missing:
        example = ", ".join(repr(value) for value in sorted(missing)[:5])
        raise ValueError(f"Missing prediction ids ({len(missing)}): {example}")
    return [joined[material["id"]] for material in materials]


def load_joined_rows(data_path, predictions_path):
    return join_rows(_csv_rows(data_path, _DATA_COLUMNS), _csv_rows(predictions_path, _SCORE_COLUMNS))


def _score(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "missing_score"
    if isinstance(value, bool):
        return None, "non_numeric_score"
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None, "non_numeric_score"
    if not math.isfinite(number):
        return None, "non_finite_score"
    if not 0 <= number <= 1:
        return None, "out_of_range_score"
    return number, "valid"


def _assessed_record(row):
    score, status = _score(row.get("score"))
    reasons = [] if status == "valid" else [status]
    if "prediction" in row:
        saved = str(row["prediction"]).strip()
        if row["prediction"] is None or saved.lower() in {"", "abstain", "error", "failed", "failure"}:
            reasons.append("saved_prediction_abstention")
        elif isinstance(row["prediction"], bool) or saved not in {"0", "1"}:
            reasons.append("invalid_saved_prediction")
    domain = str(row.get("applicability", "unknown")).strip()
    if domain == "out_of_domain":
        reasons.append("out_of_domain")
    elif domain not in {"in_domain", "unknown"}:
        reasons.append("invalid_applicability")
        domain = "invalid"
    return {
        "id": row["id"], "label": row["label"], "score": score,
        "score_text": "" if row.get("score") is None else str(row["score"]),
        "score_status": status, "applicability": domain, "reasons": reasons,
    }


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _threshold_metrics(records, threshold):
    counts = Counter(
        (row["label"], None if row["reasons"] else int(row["score"] >= threshold))
        for row in records
    )
    tp, fn, tn, fp = counts[1, 1], counts[1, 0], counts[0, 0], counts[0, 1]
    positives = sum(row["label"] for row in records)
    negatives = len(records) - positives
    assessed = tp + fn + tn + fp
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        "threshold": threshold,
        "total": len(records), "positives": positives, "negatives": negatives,
        "assessed": assessed, "assessed_positives": tp + fn, "assessed_negatives": tn + fp,
        "abstentions": len(records) - assessed,
        "abstentions_positive": counts[1, None], "abstentions_negative": counts[0, None],
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": sensitivity, "specificity": specificity,
        "ppv": _ratio(tp, tp + fp), "npv": _ratio(tn, tn + fn),
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None else None
        ),
        "coverage": _ratio(assessed, len(records)),
        "end_to_end_positive_detection_fraction": _ratio(tp, positives),
    }


def _distribution(records, label):
    group = [row for row in records if row["label"] == label]
    values = [row["score"] for row in group if not row["reasons"]]
    bins = [0] * (len(HISTOGRAM_EDGES) - 1)
    for value in values:
        index = min(bisect_right(HISTOGRAM_EDGES, value) - 1, len(bins) - 1)
        bins[index] += 1
    return {
        "observed_label": label,
        "observed_materials": len(group), "assessed_scores": len(values),
        "abstentions": len(group) - len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "histogram_counts": bins,
    }


def analyze(joined_rows):
    """Pure descriptive analysis of in-memory, identity-joined material records.

    Required identity fields are id, smiles and label; a missing score abstains.
    Optional prediction/applicability fields retain saved failures and domain
    exclusions. No file/model access, threshold fitting or label-based selection
    occurs. Inputs are not mutated; duplicate identities/structures are rejected.
    """
    materials, _ = _materials(joined_rows)
    records = [_assessed_record(row) for row in materials]
    sweep = [_threshold_metrics(records, threshold) for threshold in THRESHOLDS]
    baseline = next(metric for metric in sweep if metric["threshold"] == BASELINE_THRESHOLD)
    abstentions = [row for row in records if row["reasons"]]
    status_counts = Counter(row["score_status"] for row in records)
    domains = Counter(row["applicability"] for row in records)
    systems = {
        tuple(_text(row, field, f"joined id {row['id']!r}") for field in ("tool", "model", "version"))
        for row in materials if any(field in row for field in ("tool", "model", "version"))
    }
    if len(systems) > 1:
        raise ValueError("Joined rows must describe one tool/model/version, not pooled models")
    return {
        "analysis_type": "descriptive_exploratory_only",
        "baseline_threshold": BASELINE_THRESHOLD,
        "threshold_selection": "None; the prespecified 0.00..1.00 sweep does not choose or apply an optimum. The 0.5 baseline is retained, not established as optimal.",
        "prediction_rule": "An eligible score >= threshold is positive; an eligible score < threshold is negative. Ineligible records abstain at every threshold.",
        "statistical_unit": "One labeled material with a unique exact ID and unique map-free isomeric canonical structure; not assay replicates, atoms or ensemble members.",
        "identity_policy": "Exact, case-sensitive, untrimmed ID join plus toxicology.canonicalize_smiles equality. Duplicate IDs/structures, label conflicts, missing/extra IDs and structure mismatches fail; no rows are silently dropped.",
        "normalization": "Shared toxicology.canonicalize_smiles preserves charge, components and stereochemistry, clears atom maps and does not merge salts, protonation states or tautomers.",
        "declared_system": dict(zip(("tool", "model", "version"), next(iter(systems)))) if systems else None,
        "split_counts": dict(sorted(Counter(str(row.get("split", "unspecified")) for row in materials).items())),
        "data_source_counts": dict(sorted(Counter(str(row.get("data_source", "unspecified")) for row in materials).items())),
        "prediction_source_counts": dict(sorted(Counter(str(row.get("prediction_source", "unspecified")) for row in materials).items())),
        "metric_definitions": {
            "total": "All joined labeled materials, including abstentions; positives/negatives count ALL observed labels.",
            "confusion_counts": "tp, fn, tn, fp count assessed materials ONLY. Abstentions are neither false negatives nor true negatives.",
            "sensitivity": "tp / (tp + fn); assessed observed positives only.",
            "specificity": "tn / (tn + fp); assessed observed negatives only.",
            "ppv": "tp / (tp + fp); assessed predicted positives only (positive predictive value).",
            "npv": "tn / (tn + fn); assessed predicted negatives only (negative predictive value).",
            "balanced_accuracy": "(sensitivity + specificity) / 2; defined only when both class denominators exist.",
            "coverage": "assessed / total; assessed excludes every abstention reason.",
            "end_to_end_positive_detection_fraction": "tp / positives; includes ALL observed positives in the denominator, including positive abstentions.",
            "undefined": "Zero denominators produce JSON null and blank CSV fields, never zero or perfect performance.",
        },
        "summary": {
            "total": baseline["total"], "positives": baseline["positives"], "negatives": baseline["negatives"],
            "assessed": baseline["assessed"], "abstentions": len(abstentions),
            "abstentions_by_label": {"0": baseline["abstentions_negative"], "1": baseline["abstentions_positive"]},
            "score_status_counts": {status: status_counts[status] for status in _SCORE_STATUSES},
            "applicability_counts": {domain: domains[domain] for domain in ("in_domain", "unknown", "out_of_domain", "invalid")},
            "unknown_applicability_assessed": sum(row["applicability"] == "unknown" and not row["reasons"] for row in records),
            "abstention_reason_counts": dict(sorted(Counter(reason for row in abstentions for reason in row["reasons"]).items())),
            "abstention_reason_counting": "Reasons may overlap; each material is counted once in abstentions and once per applicable reason.",
        },
        "abstention_records": [
            {key: row[key] for key in ("id", "label", "score_text", "score_status", "reasons")}
            for row in abstentions
        ],
        "baseline_metrics": baseline,
        "threshold_sweep": sweep,
        "histogram": {
            "bin_edges": list(HISTOGRAM_EDGES),
            "bin_rule": "Left-inclusive, right-exclusive; the last bin includes score 1.0.",
            "y_unit": "Labeled material counts, not normalized densities; eligible scores only.",
            "groups": {str(label): _distribution(records, label) for label in (0, 1)},
        },
        "limitations": [
            "Public TDC AMES may overlap ADMET-AI training/pretraining; overlap is unknown. These are NOT independent generalization/validation results. Resplitting the same records cannot establish independence and is not performed.",
            "Observed Ames labels and supplied model/version/source claims are not independently verified. Score distributions describe these materials, not assay replicates or a representative impurity population.",
            "Scores are not established calibrated probabilities. Unknown applicability is allowed and counted, not evidence of being in domain. Predictive values depend on this dataset's class prevalence and coverage.",
            "Only saved scores are used; no inference, fitting, threshold optimization, automatic threshold application or application-default change occurs. Saved binary directions do not determine sweep classifications; saved abstentions/failures remain abstentions.",
            "This single-model exploratory analysis does not establish an ICH M7 complementary expert-rule/statistical (Q)SAR assessment with applicability and expert review, impurity classification, acceptable intake, safety or regulatory compliance. No ICH M7 numerical score cutoff is asserted, including 0.5.",
        ],
    }


def _write_histogram(report, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ValueError("PNG output requires matplotlib; use external_tools/admet-env/Scripts/python.exe") from exc
    figure, axes = plt.subplots(figsize=(10, 6))
    try:
        for label, color, description in (("0", "#2874A6", "Observed Ames negative"), ("1", "#C0392B", "Observed Ames positive")):
            group = report["histogram"]["groups"][label]
            axes.stairs(
                group["histogram_counts"], report["histogram"]["bin_edges"],
                fill=True, alpha=0.45, color=color,
                label=f"{description}: assessed {group['assessed_scores']}/{group['observed_materials']}",
            )
        axes.axvline(BASELINE_THRESHOLD, color="black", linestyle="--", label="0.5 baseline (not an optimum)")
        axes.set(xlim=(0, 1), xlabel="Saved AMES score (not an established calibrated probability)",
                 ylabel="Labeled materials per bin (counts)",
                 title="Saved scores stratified by observed Ames label\nDescriptive/exploratory only; TDC training overlap unknown")
        axes.set_ylim(bottom=0)
        axes.legend()
        figure.text(0.5, 0.02, f"Abstentions omitted from bars, retained in metrics: {report['summary']['abstentions']}. Not ICH M7 compliance.", ha="center")
        figure.tight_layout(rect=(0, 0.05, 1, 1))
        figure.savefig(path, format="png", dpi=160)
    finally:
        plt.close(figure)


def _input_receipt(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "ames_evaluation.csv")
    parser.add_argument("--predictions", type=Path, default=ROOT / "admet_predictions.csv", help="Saved ID-bearing AMES scores CSV, not the raw model matrix")
    parser.add_argument("--output", type=Path, default=ROOT / "ames_threshold_metrics.json")
    parser.add_argument("--sweep-output", type=Path, default=ROOT / "ames_threshold_sweep.csv")
    parser.add_argument("--histogram-output", type=Path, default=PROJECT / 'evaluation' / 'figures' / 'ames_score_histogram.png')
    args = parser.parse_args(argv)
    try:
        inputs = {args.data.resolve(), args.predictions.resolve()}
        outputs = [args.output.resolve(), args.sweep_output.resolve(), args.histogram_output.resolve()]
        if any(path in inputs for path in outputs) or len(set(outputs)) != len(outputs):
            raise ValueError("Output paths must differ from input paths and each other")
        report = analyze(load_joined_rows(args.data, args.predictions))
        report["inputs"] = {"data": _input_receipt(args.data), "saved_predictions": _input_receipt(args.predictions), "score_column": "score", "fresh_inference": False}
        # Fail on a missing plotting dependency before writing the metric files.
        _write_histogram(report, args.histogram_output)
        args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        with args.sweep_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["threshold_sweep"][0]))
            writer.writeheader()
            writer.writerows(report["threshold_sweep"])
    except (ValueError, OSError, csv.Error, UnicodeError) as exc:
        print(f"AMES threshold analysis error: {exc}", file=sys.stderr)
        return 2
    print(f"Descriptive/exploratory only: {report['summary']['total']} materials, {report['summary']['abstentions']} abstentions; 101 fixed thresholds; 0.5 baseline retained, no optimum selected.")
    print("Training overlap unknown; not independent validation, calibrated probabilities or ICH M7 compliance.")
    print(f"JSON: {args.output}\nCSV: {args.sweep_output}\nPNG: {args.histogram_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
