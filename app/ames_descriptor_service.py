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

from pathlib import Path

if __package__:
    from .ames_model_service import AMES_PYTHON, score_with_worker
else:
    from ames_model_service import AMES_PYTHON, score_with_worker


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


def _keep_named_reasons(row, result):
    """The named reasons travel with the score: this model has no unnamed
    block to put the decision in, so the two belong together."""
    row["pushed_positive"] = result.get("pushed_positive", [])
    row["pushed_negative"] = result.get("pushed_negative", [])
    row["baseline_log_odds"] = result.get("baseline_log_odds")


def assess_ames_descriptor_materials(smiles, *, threshold=0.5, timeout=600) -> dict:
    """Score distinct valid structures, with the named reasons for each score.

    A string is one material; otherwise an iterable supplies the materials.
    Duplicate canonical structures are scored once and their original
    spellings kept as aliases. More than 256 distinct valid structures raises
    before the backend runs. Nothing is cached.
    """
    return score_with_worker(
        smiles, threshold=threshold, timeout=timeout,
        worker=WORKER, model_dir=MODEL_DIR, prefix="ames_desc_",
        label="기술자 모델", unknown_model=_unknown_model,
        limitations=_LIMITATIONS,
        missing_weights="기술자 모델 가중치 파일이 없습니다. "
                        "ames/scripts/train_descriptor_model.py로 먼저 만드세요.",
        finish=_keep_named_reasons)
