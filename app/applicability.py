"""Place a material in a measured accuracy band instead of trusting one score.

The model does not hold one accuracy. Measured on grouped folds (benchmark
log 6-1) it runs from ROC-AUC 0.823 on structures unlike anything it saw to
0.965 where it has close relatives. A score without that band is a number
without its error bar.

The band is Tanimoto to the nearest training structure, mapped onto the
measured bands. It describes the model, not the material: a structure in the
weakest band is not more dangerous, only harder for the model to judge.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "ames" / "results" / "training_fingerprints.npy"

# Lower edge, upper edge, measured ROC-AUC for that band, and a plain label.
# From benchmark log 6-1, grouped folds over 19,963 structures.
BANDS = (
    (0.0, 0.30, 0.823, "매우 낮음"),
    (0.30, 0.40, 0.859, "낮음"),
    (0.40, 0.50, 0.852, "보통"),
    (0.50, 0.70, 0.910, "높음"),
    (0.70, 1.01, 0.965, "매우 높음"),
)
UNKNOWN = {"similarity": None, "band": None, "measured_auc": None,
           "label": "확인 불가", "reason": "학습 구조 목록을 읽지 못했습니다."}


@lru_cache(maxsize=1)
def _training_bits():
    """The packed training fingerprints, loaded once per process."""
    if not CACHE.is_file():
        return None
    packed = np.load(CACHE)
    return np.unpackbits(packed, axis=1).astype(np.uint8)


@lru_cache(maxsize=1)
def _popcounts():
    bits = _training_bits()
    return None if bits is None else bits.sum(axis=1)


def available() -> bool:
    return _training_bits() is not None


def _fingerprint(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    molecule = Chem.MolFromSmiles((smiles or "").strip())
    if molecule is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bits = np.zeros(2048, dtype=np.uint8)
    for bit in generator.GetFingerprint(molecule).GetOnBits():
        bits[bit] = 1
    return bits


def nearest_similarity(smiles):
    """Highest Tanimoto to any structure the model was exposed to."""
    bits = _training_bits()
    if bits is None:
        return None
    query = _fingerprint(smiles)
    if query is None:
        return None
    # Tanimoto over a bit matrix: the intersection is one matrix-vector
    # product, and the union follows from the two popcounts.
    intersection = bits @ query
    union = _popcounts() + int(query.sum()) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(union > 0, intersection / union, 0.0)
    return float(similarity.max())


def band_for(smiles):
    """Which measured accuracy band this material falls in."""
    similarity = nearest_similarity(smiles)
    if similarity is None:
        return dict(UNKNOWN)
    for low, high, auc, label in BANDS:
        if low <= similarity < high:
            return {"similarity": similarity, "band": f"{low:.2f}~{high:.2f}",
                    "measured_auc": auc, "label": label, "reason": None}
    last = BANDS[-1]
    return {"similarity": similarity, "band": f"{last[0]:.2f}~{last[1]:.2f}",
            "measured_auc": last[2], "label": last[3], "reason": None}


def summarise(materials):
    """Band counts over a list of materials, for a one-line screen summary."""
    counts = {label: 0 for _, _, _, label in BANDS}
    counts["확인 불가"] = 0
    weakest = []
    for material in materials:
        placed = band_for(material)
        counts[placed["label"]] = counts.get(placed["label"], 0) + 1
        if placed["measured_auc"] is not None and placed["measured_auc"] < 0.86:
            weakest.append(material)
    return {"counts": counts, "outside_the_strong_bands": weakest,
            "materials": len(materials)}
