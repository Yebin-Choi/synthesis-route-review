"""The explainable in-house Ames model, in the same envelope as the others.

Two in-house models are offered, not one. The model in benchmark log 6-6
decides almost entirely on unnamed fingerprint bits and cannot say why; this
one uses 215 named 2D descriptors and scores 0.8999 against its 0.9018 on the
same grouped folds, a gap smaller than the fold spread. They agree on 89.8%
of calls, so roughly one structure in ten gets a different answer, and showing
that disagreement is the point of the comparison screen rather than a defect
to be averaged away.

Everything else follows `ames_model_service`: one envelope, per-material
assessments, an adjustable line, and the provenance of the weights. A missing
interpreter, missing weights or a backend failure is an abstention. Nothing
here turns a failure into a negative prediction.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile

if __package__:
    from .ames_model_service import AMES_PYTHON, _assessment, _describe, _fail
    from .toxicology import canonicalize_smiles
else:
    from ames_model_service import AMES_PYTHON, _assessment, _describe, _fail
    from toxicology import canonicalize_smiles


ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).resolve().with_name("ames_descriptor_worker.py")
MODEL_DIR = ROOT / "models" / "ames_descriptors_balanced"
MAX_DISTINCT_VALID = 256

_LIMITATIONS = (
    "Research-only AMES predictions, not a safety determination or an ICH M7 "
    "classification.",
    "The score is an uncalibrated positive-class score, not a chance that the "
    "structure is mutagenic. Read it as a ranking. Measured on grouped folds "
    "(benchmark log 6-13): structures shown an average of 0.288 were active "
    "0.330 of the time, and the Brier score is 0.111.",
    "This model too carries the collection's 30.2% active rate in its training "
    "weights rather than in the decision line, so it runs at the conventional "
    "0.50 (benchmark log 6-20). Measured on 1,201 independent structures it "
    "reaches balanced accuracy 0.789 there, against 0.797 at its own balance "
    "point of 0.43.",
    "Measured on the same independent structures, this model reached ROC-AUC "
    "0.833 against the fingerprint model's 0.902 (benchmark log 6-20). It "
    "answers every structure, where the fingerprint model abstains on 22%.",
    "This model and the fingerprint model agree on 89.8% of calls. A "
    "disagreement is not resolved anywhere in this software.",
    "A contribution describes this model's output in log-odds. It is not a "
    "mechanism and not a statement about chemistry.",
    "Only AMES is reported; other toxicities, exposure, residual concentration "
    "and purge are not assessed.",
    "Invalid inputs and backend failures remain abstentions, never negative "
    "predictions.",
)


def _unknown_model() -> dict:
    return {"tool": "in-house Ames descriptors", "version": None,
            "endpoint": "ames", "weights_sha256": {}}


def descriptor_model_readiness() -> dict:
    """Which expected local files are missing, without running the model."""
    required = (("interpreter", AMES_PYTHON), ("worker", WORKER),
                ("weights", MODEL_DIR / "model.joblib"),
                ("spec", MODEL_DIR / "model_spec.json"))
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


def assess_ames_descriptor_materials(smiles, *, threshold=0.5, timeout=600) -> dict:
    """Score distinct valid structures, with the named reasons for each score.

    A string is one material; otherwise an iterable supplies the materials.
    Duplicate canonical structures are scored once and their original spellings
    kept as aliases. More than 256 distinct valid structures raises before the
    backend runs. Nothing is cached.
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

    report = {"model": _unknown_model(), "threshold": float(threshold),
              "assessments": [], "limitations": list(_LIMITATIONS)}
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
    if not (MODEL_DIR / "model.joblib").is_file():
        _fail(rows, "기술자 모델 가중치 파일이 없습니다. "
                    "ames/scripts/train_descriptor_model.py를 먼저 실행하세요.")
        return report

    try:
        with tempfile.TemporaryDirectory(prefix="ames_desc_") as directory:
            run_dir = Path(directory)
            input_path = run_dir / "input.json"
            output_path = run_dir / "output.json"
            input_path.write_text(
                json.dumps({"canonical_smiles": list(valid)}, ensure_ascii=False),
                encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            subprocess.run(
                [str(AMES_PYTHON), "-X", "utf8", "-B", str(WORKER),
                 "--input", str(input_path), "--output", str(output_path),
                 "--model-dir", str(MODEL_DIR)],
                capture_output=True, timeout=timeout, env=env, check=False)
            if not output_path.is_file():
                _fail(rows, "기술자 모델 작업자가 결과를 남기지 않았습니다.")
                return report
            payload = json.loads(output_path.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        _fail(rows, "기술자 모델 실행이 제한 시간을 넘겼습니다.")
        return report
    except (OSError, ValueError) as exc:
        _fail(rows, f"기술자 모델을 실행하지 못했습니다: {exc}")
        return report

    if payload.get("error"):
        _fail(rows, _describe(str(payload["error"])))
        return report
    report["model"] = payload.get("model") or _unknown_model()

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
        # The named reasons travel with the score: this model has no unnamed
        # block to put the decision in, so the two belong together.
        row["pushed_positive"] = result.get("pushed_positive", [])
        row["pushed_negative"] = result.get("pushed_negative", [])
        row["baseline_log_odds"] = result.get("baseline_log_odds")
    return report
