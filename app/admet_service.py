"""Run the installed AMES model in its own interpreter, retaining abstentions."""

import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

if __package__:
    from .toxicology import canonicalize_smiles
else:
    from toxicology import canonicalize_smiles


ROOT = Path(__file__).resolve().parents[1]
ADMET_PYTHON = ROOT / "external_tools" / "admet-env" / "Scripts" / "python.exe"
ADMET_WORKER = Path(__file__).resolve().with_name("admet_worker.py")
MAX_DISTINCT_VALID = 256

_LIMITATIONS = (
    "Research-only AMES predictions, not a safety determination or an ICH M7 classification.",
    "Scores are not clinically calibrated probabilities. Applicability domain and "
    "model-training overlap are unknown; a negative AMES prediction does not establish safety.",
    "Only AMES is reported; other toxicities, exposure, residual concentration and purge "
    "are not assessed.",
    "Shared map-free canonicalization preserves stereochemistry, charge and disconnected "
    "components; salts, protonation states, tautomers and isotopes are not merged.",
    "Invalid inputs and backend failures remain abstentions, never negative predictions.",
)
_WORKER_ERRORS = {
    "backend_missing": "The ADMET-AI backend or one of its dependencies is not installed.",
    "backend_error": "The installed ADMET-AI backend could not complete prediction.",
    "model_resources_missing": "Installed local ADMET-AI model resources are missing.",
    "missing_endpoint": "ADMET-AI did not return the AMES endpoint.",
    "duplicate_endpoint": "ADMET-AI returned an ambiguous duplicate AMES endpoint.",
    "unexpected_index": "ADMET-AI returned an unexpected structure identity.",
    "missing_index": "ADMET-AI returned no row with the exact requested structure identity.",
    "duplicate_index": "ADMET-AI returned duplicate rows for this structure identity.",
    "invalid_score": "ADMET-AI returned a non-numeric, non-finite or out-of-range AMES score.",
    "invalid_request": "The ADMET-AI worker rejected its input request.",
    "duplicate_request": "The ADMET-AI worker received duplicate structure identities.",
}


def admet_readiness() -> dict:
    """Inspect expected Windows-local files, without testing model execution.

    Missing asset identifiers are interpreter, worker, package and weights.
    Presence does not validate dependencies, weight contents or runtime success.
    """
    package_dir = ADMET_PYTHON.parent.parent / "Lib" / "site-packages" / "admet_ai"
    required_files = (
        ("interpreter", (ADMET_PYTHON,)),
        ("worker", (ADMET_WORKER,)),
        ("package", (package_dir / "__init__.py", package_dir / "constants.py")),
    )
    missing = []
    for asset, paths in required_files:
        try:
            present = all(path.is_file() for path in paths)
        except OSError:
            present = False
        if not present:
            missing.append(asset)
    try:
        models_dir = package_dir / "resources" / "models"
        weights_present = any(
            path.suffix == ".pt" and path.is_file() for path in models_dir.rglob("*")
        )
    except OSError:
        weights_present = False
    if not weights_present:
        missing.append("weights")
    return {"ready": not missing, "missing": missing}


def _unknown_model():
    return {"tool": "ADMET-AI", "version": None, "endpoint": "ames", "weights_sha256": {}}


def _number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _assessment(value, canonical_smiles):
    return {
        "canonical_smiles": canonical_smiles,
        "input_smiles": value if isinstance(value, str) else str(value),
        "status": "invalid" if canonical_smiles is None else "failed",
        "score": None,
        "prediction": "abstain",
        "applicability": "unknown",
        "error": "A complete, non-empty, sanitizable SMILES string is required."
        if canonical_smiles is None else None,
    }


def _fail(rows, error):
    for row in rows:
        row.update(status="failed", score=None, prediction="abstain", error=error)


def _worker_error(code):
    if isinstance(code, str) and code in _WORKER_ERRORS:
        return _WORKER_ERRORS[code]
    return "The ADMET-AI worker reported an unspecified failure."


def _diagnostic(stderr):
    """Describe bounded stderr without exposing paths, inputs, tokens or tracebacks."""
    if not stderr:
        return ""
    if isinstance(stderr, bytes):
        stderr = stderr[:4096].decode("utf-8", errors="replace")
    else:
        stderr = str(stderr)[:4096]
    if any(marker in stderr for marker in ("ModuleNotFoundError", "ImportError", "backend_missing")):
        kind = "a missing backend dependency"
    elif any(marker in stderr for marker in ("FileNotFoundError", "model_resources_missing")):
        kind = "a missing local resource"
    else:
        kind = "a worker diagnostic"
    return f" Stderr contained {kind}; raw diagnostic text is withheld for privacy."


def validate_admet_model(model):
    """Validate shared model identity and weight provenance before using results."""
    if not isinstance(model, dict) or model.get("tool") != "ADMET-AI" or model.get("endpoint") != "ames":
        raise ValueError("ADMET worker returned a mismatched model or endpoint identity.")
    version = model.get("version")
    if version is not None and (not isinstance(version, str) or not version.strip()):
        raise ValueError("ADMET worker returned invalid model version provenance.")
    weights = model.get("weights_sha256")
    if not isinstance(weights, dict):
        raise ValueError("ADMET worker returned invalid model weight provenance.")
    for name, digest in weights.items():
        if (not isinstance(name, str) or not name or name.startswith(("/", "\\"))
                or ":" in name or ".." in name.replace("\\", "/").split("/")
                or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError("ADMET worker returned invalid model weight provenance.")
    return {"tool": "ADMET-AI", "version": version, "endpoint": "ames", "weights_sha256": dict(weights)}


def _apply_response(report, valid, payload):
    rows = list(valid.values())
    if not isinstance(payload, dict):
        _fail(rows, "ADMET worker output must be a JSON object.")
        return
    try:
        model = validate_admet_model(payload.get("model"))
    except ValueError as exc:
        _fail(rows, str(exc))
        return
    report["model"] = model
    if payload.get("error") is not None:
        _fail(rows, _worker_error(payload["error"]))
        return
    predictions = payload.get("assessments")
    if not isinstance(predictions, list):
        _fail(rows, "ADMET worker output contains no assessment array.")
        return
    indexed = {canonical: [] for canonical in valid}
    for prediction in predictions:
        if (not isinstance(prediction, dict) or not isinstance(prediction.get("canonical_smiles"), str)
                or prediction["canonical_smiles"] not in indexed):
            _fail(rows, _worker_error("unexpected_index"))
            return
        indexed[prediction["canonical_smiles"]].append(prediction)
    for canonical, row in valid.items():
        matches = indexed[canonical]
        if len(matches) != 1:
            _fail([row], _worker_error("missing_index" if not matches else "duplicate_index"))
            continue
        prediction = matches[0]
        if prediction.get("status") != "success" or prediction.get("error") is not None:
            _fail([row], _worker_error(prediction.get("error")))
            continue
        score = prediction.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not 0 <= score <= 1 or not math.isfinite(score)):
            _fail([row], _worker_error("invalid_score"))
            continue
        if not model["weights_sha256"]:
            _fail([row], "ADMET worker supplied no model weight provenance for its prediction.")
            continue
        row.update(status="success", score=float(score),
                   prediction="positive" if score >= report["threshold"] else "negative", error=None)


def assess_admet_materials(smiles, *, threshold=0.5, timeout=300) -> dict:
    """Predict distinct valid structures; group invalid inputs by exact emitted text.

    A string is one material; otherwise an iterable supplies the materials. Original
    values are canonicalized before rendering; non-strings remain invalid. Invalid
    rows retain their text and distinct source type names in first-seen input_types
    order. Valid canonical aliases are retained. More than 256 distinct valid
    structures raises ValueError before the backend runs. Nothing is cached.
    """
    threshold = _number(threshold, "threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one, inclusive")
    timeout = _number(timeout, "timeout")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if isinstance(smiles, (str, bytes)) or smiles is None:
        values = [smiles]
    else:
        try:
            values = iter(smiles)
        except TypeError:
            values = [smiles]
    report = {"model": _unknown_model(), "threshold": threshold,
              "assessments": [], "limitations": list(_LIMITATIONS)}
    valid = {}
    invalid = {}
    for value in values:
        canonical = canonicalize_smiles(value)
        if canonical in valid:
            aliases = valid[canonical].setdefault("input_smiles_list", [valid[canonical]["input_smiles"]])
            if value not in aliases:
                aliases.append(value)
            continue
        row = _assessment(value, canonical)
        if canonical is None:
            input_type = type(value).__name__
            previous = invalid.get(row["input_smiles"])
            if previous is not None:
                if input_type not in previous["input_types"]:
                    previous["input_types"].append(input_type)
                continue
            row["input_types"] = [input_type]
            invalid[row["input_smiles"]] = row
        report["assessments"].append(row)
        if canonical is not None:
            valid[canonical] = row
            if len(valid) > MAX_DISTINCT_VALID:
                raise ValueError(f"ADMET-AI supports at most {MAX_DISTINCT_VALID} distinct valid structures per call")
    if not valid:
        return report
    rows = list(valid.values())
    try:
        if not ADMET_PYTHON.is_file():
            _fail(rows, "The local ADMET-AI Python interpreter is not installed.")
            return report
        with tempfile.TemporaryDirectory(prefix="admet_") as directory:
            run_dir = Path(directory)
            input_path = run_dir / "input.json"
            output_path = run_dir / "output.json"
            input_path.write_text(json.dumps({"canonical_smiles": list(valid)}, ensure_ascii=False), encoding="utf-8")
            env = dict(os.environ)
            env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                       TMP=directory, TEMP=directory, TMPDIR=directory,
                       MPLCONFIGDIR=str(run_dir / "matplotlib"), TORCH_HOME=str(run_dir / "torch"),
                       XDG_CACHE_HOME=str(run_dir / "cache"))
            command = [str(ADMET_PYTHON), "-B", str(ADMET_WORKER), "--input", str(input_path), "--output", str(output_path)]
            completed = subprocess.run(command, shell=False, check=False, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=timeout, cwd=directory, env=env)
            if completed.returncode != 0:
                _fail(rows, f"ADMET worker exited unsuccessfully (code {completed.returncode})."
                      + _diagnostic(completed.stderr))
                return report
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError, RecursionError):
                _fail(rows, "ADMET worker did not produce a readable, valid JSON result.")
                return report
            _apply_response(report, valid, payload)
    except subprocess.TimeoutExpired as exc:
        _fail(rows, "ADMET worker exceeded the requested timeout." + _diagnostic(exc.stderr))
    except (OSError, ValueError, OverflowError):
        _fail(rows, "ADMET worker could not be started or its temporary files could not be accessed.")
    return report
