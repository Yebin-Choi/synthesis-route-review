"""Run the in-house Ames model in its own interpreter, retaining abstentions.

Same shape as `admet_service`: one envelope with per-material assessments, a
threshold and the provenance of the weights that produced the scores. Keeping
the shape identical is what lets the screening policy treat this model and
ADMET-AI as two entries of one kind rather than two special cases.

The model lives in `models/ames_xgboost_balanced` and runs under
`external_tools/ames-env`, whose package versions match the environment it was
fitted in. That directory holds the same recipe as `models/ames_xgboost` with
the collection's 30.2% active rate carried in the training weights, which is
what lets the decision line stay at the conventional 0.50; the unweighted fit
is kept beside it as the comparison baseline. A missing interpreter, a missing
model file or a backend failure is an abstention. Nothing here turns a failure
into a negative prediction.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile

if __package__:
    from .toxicology import canonicalize_smiles
else:
    from toxicology import canonicalize_smiles


ROOT = Path(__file__).resolve().parents[1]
AMES_PYTHON = ROOT / "external_tools" / "ames-env" / "Scripts" / "python.exe"
AMES_WORKER = Path(__file__).resolve().with_name("ames_model_worker.py")
MODEL_DIR = ROOT / "models" / "ames_xgboost_balanced"
MAX_DISTINCT_VALID = 256

_LIMITATIONS = (
    "Research-only AMES predictions, not a safety determination or an ICH M7 classification.",
    "The score is an uncalibrated positive-class score from a gradient boosted tree "
    "model. It is not a chance that the structure is mutagenic. Measured on grouped "
    "folds (benchmark log 6-8), structures shown around 0.287 were active 35.0% of "
    "the time, so the low end reads lower than it is. Treat it as a ranking.",
    "Measured on 933 independent structures it had not seen, this model reached "
    "ROC-AUC 0.902 and balanced accuracy 0.847 at the 0.5 line, finding 77.4% of "
    "the positives (benchmark log 6-20).",
    "The collection is 30.2% active and that imbalance is carried in the training "
    "weights, not in the decision line, so 0.5 is near this model's own balance "
    "point of 0.43: both 0.40 and 0.50 measure 0.847 and the balance point itself "
    "reaches only 0.857. The line is adjustable and moving it trades sensitivity "
    "against specificity.",
    "Training data and the evaluation set come from the same curated Ames collection. "
    "Performance on chemistry unlike that collection is not established.",
    "Only AMES is reported; other toxicities, exposure, residual concentration and purge "
    "are not assessed.",
    "Invalid inputs and backend failures remain abstentions, never negative predictions.",
)
_WORKER_ERRORS = {
    "invalid_request": "The Ames model worker rejected its input request.",
    "duplicate_request": "The Ames model worker received duplicate structure identities.",
    "model_resources_missing": "The in-house Ames model files are missing.",
    "unexpected_index": "The Ames model returned a different number of rows than requested.",
    "invalid_score": "The Ames model returned a non-numeric or out-of-range score.",
}


def _unknown_model() -> dict:
    return {"tool": "in-house Ames XGBoost", "version": None, "endpoint": "ames",
            "weights_sha256": {}}


def ames_model_readiness() -> dict:
    """Which expected local files are missing, without running the model.

    Presence does not prove the backend runs: the model refuses to score when
    the installed RDKit differs from the one it was fitted with, and that only
    shows up on a real call.
    """
    required = (
        ("interpreter", AMES_PYTHON),
        ("worker", AMES_WORKER),
        ("weights", MODEL_DIR / "model.joblib"),
        ("spec", MODEL_DIR / "model_spec.json"),
        ("manifest", MODEL_DIR / "feature_manifest.json"),
    )
    missing = []
    for asset, path in required:
        try:
            present = path.is_file()
        except OSError:
            present = False
        if not present:
            missing.append(asset)
    return {"ready": not missing, "missing": missing,
            "interpreter": str(AMES_PYTHON), "model_dir": str(MODEL_DIR)}


def _assessment(value, canonical):
    text = value if isinstance(value, str) else repr(value)
    if canonical is None:
        return {"input_smiles": text, "canonical_smiles": None, "status": "invalid",
                "score": None, "error": "구조를 해석하지 못했습니다."}
    return {"input_smiles": text, "canonical_smiles": canonical, "status": "pending",
            "score": None, "error": None}


def _fail(rows, message):
    for row in rows:
        row["status"] = "failed"
        row["score"] = None
        row["error"] = message


def _describe(code):
    for key, message in _WORKER_ERRORS.items():
        if code == key or code.startswith(f"{key}:"):
            return message
    if code.startswith("backend_missing"):
        return "The Ames model environment is missing a dependency."
    if code.startswith("backend_error"):
        return ("The Ames model backend could not complete prediction. "
                "This is what a package-version mismatch looks like.")
    if code.startswith("model_rejected_structure"):
        return "The model could not read this structure."
    return f"The Ames model worker failed: {code}"


def score_with_worker(smiles, *, threshold, timeout, worker, model_dir,
                      prefix, label, unknown_model, limitations,
                      missing_weights=None, finish=None):
    """Run one isolated scoring worker over the distinct valid structures.

    The two in-house models differ in four things: the worker they call, the
    weights they load, the wording of their failures and the extra fields
    each result carries. Everything else here — the argument checks, folding
    duplicate structures into aliases, the 256 bound, the temporary run
    directory and the shape of the report — was written out twice and had to
    be kept in step by hand.

    `finish(row, result)` copies whatever else a model reports onto a scored
    row. A model that reports nothing extra passes None.
    """
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError("threshold must be a number")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one, inclusive")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    if isinstance(smiles, (str, bytes)) or smiles is None:
        values = [smiles]
    else:
        try:
            values = list(smiles)
        except TypeError:
            values = [smiles]

    report = {"model": unknown_model(), "threshold": float(threshold),
              "assessments": [], "limitations": list(limitations)}
    valid = {}
    for value in values:
        canonical = canonicalize_smiles(value)
        if canonical is not None and canonical in valid:
            row = valid[canonical]
            aliases = row.setdefault("input_smiles_list", [row["input_smiles"]])
            text = value if isinstance(value, str) else repr(value)
            if text not in aliases:
                aliases.append(text)
            continue
        row = _assessment(value, canonical)
        report["assessments"].append(row)
        if canonical is not None:
            valid[canonical] = row
            if len(valid) > MAX_DISTINCT_VALID:
                raise ValueError(
                    f"이 모델은 한 번에 서로 다른 유효 구조를 "
                    f"{MAX_DISTINCT_VALID}개까지만 받습니다")
    if not valid:
        return report

    rows = list(valid.values())
    if not AMES_PYTHON.is_file():
        _fail(rows, "격리된 Ames 모델 파이썬 환경이 설치되어 있지 않습니다.")
        return report
    if not (model_dir / "model.joblib").is_file():
        _fail(rows, missing_weights or f"{label} 가중치 파일이 없습니다.")
        return report

    try:
        with tempfile.TemporaryDirectory(prefix=prefix) as directory:
            run_dir = Path(directory)
            input_path = run_dir / "input.json"
            output_path = run_dir / "output.json"
            input_path.write_text(
                json.dumps({"canonical_smiles": list(valid)}, ensure_ascii=False),
                encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            subprocess.run(
                [str(AMES_PYTHON), "-X", "utf8", "-B", str(worker),
                 "--input", str(input_path), "--output", str(output_path),
                 "--model-dir", str(model_dir)],
                capture_output=True, timeout=timeout, env=env, check=False)
            if not output_path.is_file():
                _fail(rows, f"{label} 작업자가 결과를 남기지 않았습니다.")
                return report
            payload = json.loads(output_path.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        _fail(rows, f"{label} 실행이 제한 시간을 넘겼습니다.")
        return report
    except (OSError, ValueError) as exc:
        _fail(rows, f"{label}을 실행하지 못했습니다: {exc}")
        return report

    if payload.get("error"):
        _fail(rows, _describe(str(payload["error"])))
        return report
    report["model"] = payload.get("model") or unknown_model()

    scored = {row.get("canonical_smiles"): row
              for row in payload.get("assessments", [])}
    for row in rows:
        result = scored.get(row["canonical_smiles"])
        if result is None:
            row["status"] = "failed"
            row["error"] = "모델이 이 구조에 대한 결과를 돌려주지 않았습니다."
            continue
        if result.get("status") != "success":
            row["status"] = "failed"
            row["score"] = None
            row["error"] = _describe(str(result.get("error") or "backend_error"))
            continue
        row["status"] = "success"
        row["score"] = float(result["score"])
        row["error"] = None
        if finish is not None:
            finish(row, result)
    return report


def _keep_model_form(row, result):
    """The model normalizes again with its own rules, so the form it scored is
    kept when it differs from the one the application sent."""
    model_form = result.get("model_canonical_smiles")
    if model_form and model_form != row["canonical_smiles"]:
        row["model_canonical_smiles"] = model_form


def assess_ames_model_materials(smiles, *, threshold=0.5, timeout=600) -> dict:
    """Score distinct valid structures with the in-house model.

    A string is one material; otherwise an iterable supplies the materials.
    Duplicate canonical structures are scored once and their original
    spellings kept as aliases. More than 256 distinct valid structures raises
    before the backend runs. Nothing is cached.
    """
    return score_with_worker(
        smiles, threshold=threshold, timeout=timeout,
        worker=AMES_WORKER, model_dir=MODEL_DIR, prefix="ames_model_",
        label="Ames 모델", unknown_model=_unknown_model,
        limitations=_LIMITATIONS,
        missing_weights="Ames 모델 가중치 파일이 없습니다.",
        finish=_keep_model_form)


EXPLAIN_WORKER = Path(__file__).resolve().with_name("ames_explain_worker.py")
MAX_EXPLAIN = 64


def explain_ames_model_materials(smiles, *, timeout=900) -> dict:
    """Why the model scored these structures as it did.

    A contribution says how much a column moved this model's output, in
    log-odds. It is not a mechanism and not a statement about chemistry.

    Measured 2026-11-05 and worth knowing before reading the output: on this
    model the 2,048 fingerprint bits carry almost the whole decision and the
    named descriptors carry almost none of it. The split is reported so a
    reader can see that rather than mistake a tiny descriptor contribution for
    the reason. The descriptor-only variant scores 0.8999 against this one's
    0.9018, so an explainable model is available at no measurable cost.
    """
    if isinstance(smiles, (str, bytes)) or smiles is None:
        values = [smiles]
    else:
        values = list(smiles)
    canonical = []
    for value in values:
        item = canonicalize_smiles(value)
        if item is not None and item not in canonical:
            canonical.append(item)
    report = {"explanations": [], "limitations": [
        "A contribution describes this model's output, not a mechanism.",
        "The fingerprint bits are summed into one block because a single bit "
        "has no readable meaning on its own.",
        "Measured on this model, the fingerprint block carries almost the "
        "whole decision, so the named descriptors explain little.",
    ]}
    if not canonical:
        return report
    if len(canonical) > MAX_EXPLAIN:
        canonical = canonical[:MAX_EXPLAIN]
        report["truncated_to"] = MAX_EXPLAIN
    if not AMES_PYTHON.is_file() or not EXPLAIN_WORKER.is_file():
        report["error"] = "설명을 계산할 격리 환경이 없습니다."
        return report
    try:
        with tempfile.TemporaryDirectory(prefix="ames_explain_") as directory:
            run_dir = Path(directory)
            input_path = run_dir / "input.json"
            output_path = run_dir / "output.json"
            input_path.write_text(
                json.dumps({"canonical_smiles": canonical}, ensure_ascii=False),
                encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            subprocess.run(
                [str(AMES_PYTHON), "-X", "utf8", "-B", str(EXPLAIN_WORKER),
                 "--input", str(input_path), "--output", str(output_path),
                 "--model-dir", str(MODEL_DIR)],
                capture_output=True, timeout=timeout, env=env, check=False)
            if not output_path.is_file():
                report["error"] = "설명 작업자가 결과를 남기지 않았습니다."
                return report
            payload = json.loads(output_path.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        report["error"] = "설명 계산이 제한 시간을 넘겼습니다."
        return report
    except (OSError, ValueError) as exc:
        report["error"] = f"설명을 계산하지 못했습니다: {exc}"
        return report
    if payload.get("error"):
        report["error"] = _describe(str(payload["error"]))
        return report
    report["explanations"] = payload.get("explanations", [])
    return report
