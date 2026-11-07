"""Isolated worker for the in-house Ames XGBoost model. Only scores leave here.

This runs inside `external_tools/ames-env`, whose package versions match the
environment the model was fitted in. The model's own inference code refuses to
run when the RDKit version differs, because its descriptor block would silently
change meaning, so the version check is the model's and is not bypassed here.

The structures arriving here were already canonicalized by the application. The
model normalizes again with its own rules before featurising, which may strip a
salt the application kept, so the form the model actually scored is reported
back rather than assumed to be the form that was sent.
"""

import argparse
import hashlib
import json
from numbers import Real
from pathlib import Path
import sys

MAX_DISTINCT_VALID = 256
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "ames_xgboost_balanced"


class _WorkerError(ValueError):
    pass


def _unknown_model():
    return {"tool": "in-house Ames XGBoost", "version": None, "endpoint": "ames",
            "weights_sha256": {}}


def _validate_request(payload):
    if not isinstance(payload, dict) or set(payload) != {"canonical_smiles"}:
        raise _WorkerError("invalid_request")
    smiles = payload["canonical_smiles"]
    if (not isinstance(smiles, list) or not 1 <= len(smiles) <= MAX_DISTINCT_VALID
            or any(not isinstance(value, str) or not value for value in smiles)):
        raise _WorkerError("invalid_request")
    if len(set(smiles)) != len(smiles):
        raise _WorkerError("duplicate_request")
    return smiles


def _provenance(model_dir):
    """Hash the files that decide the prediction, so a swap is visible."""
    files = ("model.joblib", "model_spec.json", "feature_manifest.json")
    weights = {}
    for name in files:
        path = model_dir / name
        if not path.is_file():
            raise _WorkerError("model_resources_missing")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        weights[name] = digest.hexdigest()
    spec = json.loads((model_dir / "model_spec.json").read_text(encoding="utf-8"))
    return {
        "tool": "in-house Ames XGBoost",
        "version": spec.get("experiment_id"),
        "endpoint": "ames",
        "weights_sha256": weights,
        # Recorded so a reader can tell which measured figures apply to the
        # weights that produced a given report.
        "validation_roc_auc": spec.get("roc_auc"),
        "training_structures": spec.get("train_n"),
        "feature_blocks": spec.get("blocks"),
    }


def _score_rows(smiles, model_dir):
    sys.path.insert(0, str(model_dir))
    try:
        import pandas as pd
        from predict_expanded import predict_frame
    except ImportError as exc:
        raise _WorkerError(f"backend_missing:{exc}") from exc
    frame = pd.DataFrame({"smiles": smiles})
    try:
        # The threshold is applied by the caller, not here: this worker reports
        # the score and lets the policy decide where the line sits.
        predicted = predict_frame(frame, model_dir, "smiles", threshold=0.5)
    except RuntimeError as exc:
        raise _WorkerError(f"backend_error:{exc}") from exc
    if len(predicted) != len(smiles):
        raise _WorkerError("unexpected_index")

    rows = []
    for position, item in enumerate(smiles):
        record = predicted.iloc[position]
        score = record["ames_score"]
        status = str(record["structure_status"])
        if status != "valid":
            rows.append({"canonical_smiles": item, "status": "failed", "score": None,
                         "model_canonical_smiles": None,
                         "error": f"model_rejected_structure:{status}"})
            continue
        if score is None or not isinstance(score, Real) or score != score:
            rows.append({"canonical_smiles": item, "status": "failed", "score": None,
                         "model_canonical_smiles": str(record["canonical_smiles"]),
                         "error": "invalid_score"})
            continue
        value = float(score)
        if not 0.0 <= value <= 1.0:
            rows.append({"canonical_smiles": item, "status": "failed", "score": None,
                         "model_canonical_smiles": str(record["canonical_smiles"]),
                         "error": "invalid_score"})
            continue
        rows.append({"canonical_smiles": item, "status": "success", "score": value,
                     "model_canonical_smiles": str(record["canonical_smiles"]),
                     "error": None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    output = Path(args.output)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        smiles = _validate_request(payload)
        model_dir = Path(args.model_dir)
        provenance = _provenance(model_dir)
        rows = _score_rows(smiles, model_dir)
    except _WorkerError as exc:
        output.write_text(json.dumps({"error": str(exc), "model": _unknown_model(),
                                      "assessments": []}, ensure_ascii=False),
                          encoding="utf-8")
        return 1
    except Exception as exc:  # noqa: BLE001 - reported, never silently negative
        output.write_text(json.dumps({"error": f"backend_error:{exc}",
                                      "model": _unknown_model(), "assessments": []},
                                     ensure_ascii=False), encoding="utf-8")
        return 1
    output.write_text(json.dumps({"model": provenance, "assessments": rows},
                                 ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
