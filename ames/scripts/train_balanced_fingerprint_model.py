"""Refit the shipped Ames model with the class imbalance in the weights.

Why. The collection is 30.2% positive and the shipped fit has weighting off
(`"balanced": false` in its spec), so its scores sit low and 0.5 lands far
from the balance point: measured on grouped folds it finds 65.3% of the
positives there. Published Ames benchmarks are 54 to 55% positive, which is
why 0.5 works for them and why their papers never have to move a line.

Carrying the imbalance in `scale_pos_weight` instead moves the balance point
from 0.25 to 0.35 and lifts sensitivity at 0.5 from 0.653 to 0.726, with
ROC-AUC unchanged at 0.899 against 0.902 (benchmark log 6-19). The decision
line then stays at the 0.5 everyone uses. AmesFormer's own code does the same
thing: a fixed 0.5 with class-weighted training loss.

This is the same model, not a different one. The recipe, the feature
pipeline, the hyperparameters and the seed are read out of the shipped
artefact and reused; the single change is the weight. It is fitted on the
train and validation rows, which is what the shipped model was fitted and
selected on, so both can be scored on one evaluation file. The locked test
split is not touched.

Run it with the interpreter the model's own environment uses, because the
feature block is tied to that RDKit version:

    external_tools/ames-env/Scripts/python.exe \\
      ames/scripts/train_balanced_fingerprint_model.py
"""

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "models" / "ames_xgboost"
TARGET = ROOT / "models" / "ames_xgboost_balanced"
DATA = ROOT / "data" / "ames"
# The package's own modules, which carry the normalisation and the feature
# block. They are imported from the model directory, as the worker does.
sys.path.insert(0, str(SOURCE))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rdkit  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402

from chemistry import normalize_structure  # noqa: E402
from features import RECIPES  # noqa: E402
from model_lib import dense  # noqa: E402
from predict import featurize  # noqa: E402

# Files the package needs beside the weights for the worker to import it.
PACKAGE_FILES = ("chemistry.py", "features.py", "feature_pipeline.py",
                 "model_lib.py", "predict.py", "predict_expanded.py",
                 "feature_manifest.json", "requirements.txt")


def rows(split):
    """Canonical structures and labels of one split, as the file holds them."""
    frame = pd.read_csv(DATA / f"{split}_model_ready_70.csv", encoding="utf-8-sig")
    frame = frame[frame["label"].isin([0, 1])]
    return frame["canonical_smiles"].astype(str).tolist(), frame["label"].to_numpy()


def main():
    spec = json.loads((SOURCE / "model_spec.json").read_text(encoding="utf-8"))
    manifest = json.loads((SOURCE / "feature_manifest.json").read_text(encoding="utf-8"))
    if rdkit.__version__ != manifest["rdkit_version"]:
        raise SystemExit(
            f"RDKit {rdkit.__version__} is not the {manifest['rdkit_version']} "
            "the feature block was defined against. Use the model's own "
            "environment; a silently different descriptor block is worse than "
            "a refusal.")

    smiles, labels = [], []
    for split in ("train", "validation"):
        part_smiles, part_labels = rows(split)
        smiles += part_smiles
        labels.append(part_labels)
        print(f"{split:11s} {len(part_smiles):>6,}개")
    labels = np.concatenate(labels)
    positive = float(labels.mean())
    print(f"합계 {len(smiles):,}개 · 양성 비율 {positive:.4f}")

    structures = [normalize_structure(value) for value in smiles]
    usable = [index for index, item in enumerate(structures)
              if item["status"] == "valid"]
    dropped = len(structures) - len(usable)
    if dropped:
        print(f"구조를 읽지 못해 제외: {dropped}개")
    labels = labels[usable]

    if spec.get("blocks"):
        RECIPES["CUSTOM"] = spec["blocks"]
    matrix = featurize([structures[i]["canonical_smiles"] for i in usable],
                       [structures[i]["parent_smiles"] for i in usable],
                       spec["recipe"], manifest["descriptor_names"])
    print(f"특징 {matrix.shape[1]:,}열 · 블록 {spec['blocks']}")

    # The shipped pipeline, cloned so nothing of the fitted state carries
    # over, with the weight as the only changed setting.
    shipped = joblib.load(SOURCE / "model.joblib")
    processor = clone(shipped.named_steps["features"])
    estimator = clone(shipped.named_steps["model"])
    ratio = float((labels == 0).sum() / max((labels == 1).sum(), 1))
    estimator.set_params(scale_pos_weight=ratio)
    print(f"scale_pos_weight = {ratio:.4f}")

    model = Pipeline([("features", processor),
                      ("dense", FunctionTransformer(dense, accept_sparse=True)),
                      ("model", estimator)])
    model.fit(matrix, labels)

    TARGET.mkdir(parents=True, exist_ok=True)
    for name in PACKAGE_FILES:
        shutil.copy2(SOURCE / name, TARGET / name)
    joblib.dump(model, TARGET / "model.joblib")

    carried = dict(spec)
    carried.update({
        "balanced": True,
        "scale_pos_weight": ratio,
        "params": {**spec.get("params", {}), "balanced": True},
        "fitted_on": {"splits": ["train", "validation"],
                      "structures": int(len(usable)),
                      "active_fraction": float(labels.mean())},
        "refit_of": spec.get("experiment_id"),
        "refit_reason": "The collection is 30.2% positive and the shipped fit "
                        "left that to the decision line. Carried in the "
                        "training weights instead so the line stays at 0.5. "
                        "Recipe, pipeline, hyperparameters and seed unchanged "
                        "(benchmark log 6-19).",
        "refit_rdkit": rdkit.__version__,
    })
    # Figures that described the shipped fit and not this one.
    for stale in ("roc_auc", "average_precision", "balanced_accuracy", "accuracy",
                  "sensitivity", "specificity", "precision", "f1", "mcc", "brier",
                  "log_loss", "cv_auc", "cv_auc_sd", "fold_auc", "train_auc",
                  "tn", "fp", "fn", "tp", "elapsed_s", "started_at",
                  "finished_at", "model_path", "result_file", "prediction_file"):
        carried.pop(stale, None)
    (TARGET / "model_spec.json").write_text(
        json.dumps(carried, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {TARGET}")


if __name__ == "__main__":
    main()
