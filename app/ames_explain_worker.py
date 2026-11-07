"""Why the in-house Ames model scored a structure the way it did.

Runs inside `external_tools/ames-env` beside the scoring worker. It asks the
booster for per-feature contributions to one prediction and reports the named
descriptors that pushed hardest each way, with the 2,048 fingerprint bits
summed into one block because a single bit has no readable meaning.

A contribution says how much a column moved this model's output, in log-odds.
It is not a mechanism: two models fitted on the same data can attribute the
same correct prediction to different columns. Scoring is unchanged; this
describes a prediction and never alters one.
"""

import argparse
import json
from pathlib import Path
import sys

MAX_DISTINCT_VALID = 64
TOP_EACH_WAY = 6
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "ames_xgboost_balanced"
FINGERPRINT_PREFIXES = ("rdkit_", "ecfp", "maccs_", "ap_", "tt_")


class _WorkerError(ValueError):
    pass


def _validate(payload):
    if not isinstance(payload, dict) or set(payload) != {"canonical_smiles"}:
        raise _WorkerError("invalid_request")
    smiles = payload["canonical_smiles"]
    if (not isinstance(smiles, list) or not 1 <= len(smiles) <= MAX_DISTINCT_VALID
            or any(not isinstance(value, str) or not value for value in smiles)):
        raise _WorkerError("invalid_request")
    return smiles


def _explain(smiles, model_dir):
    sys.path.insert(0, str(model_dir))
    try:
        import joblib
        import numpy as np
        import xgboost
        from chemistry import normalize_structure
        from features import RECIPES
        from predict import featurize
    except ImportError as exc:
        raise _WorkerError(f"backend_missing:{exc}") from exc

    spec = json.loads((model_dir / "model_spec.json").read_text(encoding="utf-8"))
    meta = json.loads((model_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    selected = json.loads((model_dir / "selected_features.json").read_text(
        encoding="utf-8"))["raw"]

    structures = [normalize_structure(item) for item in smiles]
    good = [index for index, row in enumerate(structures) if row["status"] == "valid"]
    rows = [{"canonical_smiles": item, "status": "failed", "error":
             f"model_rejected_structure:{structures[index]['status']}",
             "pushed_positive": [], "pushed_negative": []}
            for index, item in enumerate(smiles)]
    if not good:
        return rows

    recipe = spec["recipe"]
    if spec.get("blocks"):
        RECIPES["CUSTOM"] = spec["blocks"]
        recipe = "CUSTOM"
    matrix = featurize([structures[index]["canonical_smiles"] for index in good],
                       [structures[index]["parent_smiles"] for index in good],
                       recipe, meta["descriptor_names"])
    pipeline = joblib.load(model_dir / "model.joblib")
    steps = list(pipeline.named_steps.values())
    prepared = matrix
    for step in steps[:-1]:
        prepared = step.transform(prepared)
    booster = steps[-1].get_booster()
    if hasattr(prepared, "toarray"):
        prepared = prepared.toarray()
    contributions = booster.predict(xgboost.DMatrix(np.asarray(prepared)),
                                    pred_contribs=True)
    if contributions.shape[1] != len(selected) + 1:
        raise _WorkerError("unexpected_contribution_shape")

    for position, index in enumerate(good):
        values = contributions[position]
        bias = float(values[-1])
        fingerprint_total = 0.0
        named = []
        for column, name in enumerate(selected):
            amount = float(values[column])
            if name.startswith(FINGERPRINT_PREFIXES):
                fingerprint_total += amount
                continue
            if amount:
                named.append({"feature": name, "contribution": amount})
        named.sort(key=lambda row: row["contribution"], reverse=True)
        positive = [row for row in named if row["contribution"] > 0][:TOP_EACH_WAY]
        negative = [row for row in reversed(named)
                    if row["contribution"] < 0][:TOP_EACH_WAY]
        rows[index] = {
            "canonical_smiles": smiles[index], "status": "success", "error": None,
            "baseline_log_odds": bias,
            "fingerprint_block_contribution": fingerprint_total,
            "named_feature_contribution": float(sum(row["contribution"]
                                                    for row in named)),
            "pushed_positive": positive, "pushed_negative": negative,
        }
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
        smiles = _validate(payload)
        rows = _explain(smiles, Path(args.model_dir))
    except _WorkerError as exc:
        output.write_text(json.dumps({"error": str(exc), "explanations": []},
                                     ensure_ascii=False), encoding="utf-8")
        return 1
    except Exception as exc:  # noqa: BLE001 - reported, never silently empty
        output.write_text(json.dumps({"error": f"backend_error:{exc}",
                                      "explanations": []}, ensure_ascii=False),
                          encoding="utf-8")
        return 1
    output.write_text(json.dumps({"explanations": rows}, ensure_ascii=False),
                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
