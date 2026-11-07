"""Score and explain with the descriptor-only Ames model, in its own process.

Runs inside `external_tools/ames-env`. Every column here has a name, so the
score and the reason come back in one call, with no unnamed block to hide the
decision in (benchmark log 6-6).

The descriptor column list is read from the saved spec, never recomputed. The
fit dropped columns that did not vary over the training rows, so recomputing
the list for a handful of query structures would featurise differently from
how the model was fitted.

A contribution is log-odds movement in this model, not a mechanism.
"""

import argparse
import json
from pathlib import Path
import sys

MAX_DISTINCT_VALID = 256
TOP_EACH_WAY = 6
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "ames_descriptors_balanced"


class _WorkerError(ValueError):
    pass


def _validate(payload):
    if not isinstance(payload, dict) or set(payload) != {"canonical_smiles"}:
        raise _WorkerError("invalid_request")
    smiles = payload["canonical_smiles"]
    if (not isinstance(smiles, list) or not 1 <= len(smiles) <= MAX_DISTINCT_VALID
            or any(not isinstance(value, str) or not value for value in smiles)):
        raise _WorkerError("invalid_request")
    if len(set(smiles)) != len(smiles):
        raise _WorkerError("duplicate_request")
    return smiles


def _weights_digest(model_dir):
    from hashlib import sha256

    digest = {}
    for name in ("model.joblib", "model_spec.json"):
        path = model_dir / name
        if path.is_file():
            digest[name] = sha256(path.read_bytes()).hexdigest()
    return digest


def _score(smiles, model_dir):
    try:
        import joblib
        import numpy as np
        import xgboost
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Descriptors import CalcMolDescriptors
    except ImportError as exc:
        raise _WorkerError(f"backend_missing:{exc}") from exc

    RDLogger.DisableLog("rdApp.*")
    spec_path = model_dir / "model_spec.json"
    weights_path = model_dir / "model.joblib"
    if not spec_path.is_file() or not weights_path.is_file():
        raise _WorkerError("model_resources_missing")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    names = spec["descriptor_names"]

    rows, usable, matrix = [], [], []
    for item in smiles:
        molecule = Chem.MolFromSmiles(item.strip())
        if molecule is None:
            rows.append({"canonical_smiles": item, "status": "failed",
                         "score": None, "error": "model_rejected_structure:unparsed",
                         "pushed_positive": [], "pushed_negative": []})
            continue
        described = CalcMolDescriptors(molecule)
        values = [float(described.get(name) or 0.0) for name in names]
        matrix.append(values)
        usable.append(len(rows))
        rows.append({"canonical_smiles": item, "status": "pending",
                     "score": None, "error": None,
                     "pushed_positive": [], "pushed_negative": []})
    if not usable:
        return rows, spec

    prepared = np.nan_to_num(np.array(matrix, dtype=np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
    model = joblib.load(weights_path)
    scores = model.predict_proba(prepared)[:, 1]
    contributions = model.get_booster().predict(xgboost.DMatrix(prepared),
                                                pred_contribs=True)
    if contributions.shape[1] != len(names) + 1:
        raise _WorkerError("unexpected_contribution_shape")

    for position, index in enumerate(usable):
        amounts = contributions[position]
        named = [{"feature": name, "contribution": float(amounts[column])}
                 for column, name in enumerate(names) if amounts[column]]
        named.sort(key=lambda row: row["contribution"], reverse=True)
        rows[index].update({
            "status": "success",
            "score": float(scores[position]),
            "baseline_log_odds": float(amounts[-1]),
            "named_feature_contribution": float(sum(row["contribution"]
                                                    for row in named)),
            "pushed_positive": [row for row in named
                                if row["contribution"] > 0][:TOP_EACH_WAY],
            "pushed_negative": [row for row in reversed(named)
                                if row["contribution"] < 0][:TOP_EACH_WAY],
        })
    return rows, spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    output = Path(args.output)
    model_dir = Path(args.model_dir)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        smiles = _validate(payload)
        rows, spec = _score(smiles, model_dir)
    except _WorkerError as exc:
        output.write_text(json.dumps({"error": str(exc), "assessments": []},
                                     ensure_ascii=False), encoding="utf-8")
        return 1
    except Exception as exc:  # noqa: BLE001 - reported, never silently empty
        output.write_text(json.dumps({"error": f"backend_error:{exc}",
                                      "assessments": []}, ensure_ascii=False),
                          encoding="utf-8")
        return 1
    output.write_text(json.dumps({
        "model": {"tool": spec.get("tool", "in-house Ames descriptors"),
                  "version": None, "endpoint": "ames",
                  "features": spec.get("features"),
                  "columns": spec.get("columns"),
                  "weights_sha256": _weights_digest(model_dir)},
        "assessments": rows}, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
