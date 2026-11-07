"""Isolated local ADMET-AI worker. Only AMES scores leave this process."""

import argparse
import hashlib
import importlib.metadata
import json
import math
from numbers import Real
from pathlib import Path
import sys


MAX_DISTINCT_VALID = 256


class _WorkerError(ValueError):
    pass


def _unknown_model():
    return {"tool": "ADMET-AI", "version": None, "endpoint": "ames", "weights_sha256": {}}


def _failed(canonical_smiles, error):
    return {"canonical_smiles": canonical_smiles, "status": "failed", "score": None, "error": error}


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


def _model_provenance(models_dir):
    """Hash the installed resource directory used by ADMETModel's default loader."""
    models_dir = Path(models_dir)
    files = sorted(path for path in models_dir.rglob("*") if path.is_file())
    if not any(path.suffix == ".pt" for path in files):
        raise _WorkerError("model_resources_missing")
    weights = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        weights[path.relative_to(models_dir).as_posix()] = digest.hexdigest()
    try:
        version = importlib.metadata.version("admet-ai")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"tool": "ADMET-AI", "version": version, "endpoint": "ames", "weights_sha256": weights}


def _align_predictions(smiles, predictions):
    """Use exact index labels, never row positions or an average of duplicate rows."""
    columns = list(predictions.columns)
    if "AMES" not in columns:
        raise _WorkerError("missing_endpoint")
    if columns.count("AMES") != 1:
        raise _WorkerError("duplicate_endpoint")
    indexed = {value: [] for value in smiles}
    for identity, score in predictions["AMES"].items():
        if not isinstance(identity, str) or identity not in indexed:
            raise _WorkerError("unexpected_index")
        indexed[identity].append(score)
    rows = []
    for identity in smiles:
        matches = indexed[identity]
        if len(matches) != 1:
            rows.append(_failed(identity, "missing_index" if not matches else "duplicate_index"))
            continue
        score = matches[0]
        if isinstance(score, bool) or not isinstance(score, Real) or not 0 <= score <= 1:
            rows.append(_failed(identity, "invalid_score"))
            continue
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            rows.append(_failed(identity, "invalid_score"))
            continue
        rows.append({"canonical_smiles": identity, "status": "success", "score": score, "error": None})
    return rows


def run_predictions(payload):
    """Use installed local weights without downloading or changing configuration."""
    model_info = _unknown_model()
    smiles = []
    try:
        smiles = _validate_request(payload)
        from admet_ai import ADMETModel
        from admet_ai.constants import DEFAULT_MODELS_DIR

        model_info = _model_provenance(DEFAULT_MODELS_DIR)
        model = ADMETModel(include_physchem=False, drugbank_path=None, num_workers=0)
        predictions = model.predict(smiles)
        return {"model": model_info, "assessments": _align_predictions(smiles, predictions), "error": None}
    except _WorkerError as exc:
        error = str(exc)
    except ImportError:
        error = "backend_missing"
    except FileNotFoundError:
        error = "model_resources_missing"
    except Exception:
        error = "backend_error"
    print(f"ADMET worker: {error}.", file=sys.stderr)
    return {"model": model_info, "assessments": [_failed(value, error) for value in smiles], "error": error}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        payload = None
    result = run_predictions(payload)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
