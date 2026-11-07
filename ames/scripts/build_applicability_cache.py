"""Pack the model's training fingerprints so the screen can place a material.

The model holds its score only near the chemistry it was fitted on: measured
in benchmark log 6-1, ROC-AUC runs from 0.823 below Tanimoto 0.3 to 0.965
above 0.7. A score shown without saying which of those bands the material
falls in is a number without its error bar, and the regulatory guidance on
out-of-domain results (PMC7500704) asks for exactly that distinction.

Computing the fingerprints of twenty thousand training structures on every
screen refresh is not acceptable, so they are packed once into a bit array
the application memory-maps and reuses.
"""
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ames_common import DATA  # noqa: E402

OUT_BITS = ROOT.parent / 'results' / 'training_fingerprints.npy'
OUT_META = ROOT.parent / 'results' / 'training_fingerprints.json'
SIZE = 2048
RADIUS = 2


def main():
    import csv

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=SIZE)
    rows, seen = [], set()
    with DATA.open(encoding='utf-8-sig', newline='') as handle:
        for record in csv.DictReader(handle):
            # Every split, including test: the point is which chemistry the
            # model has been exposed to, not which rows it was fitted on.
            item = (record.get('canonical_smiles') or '').strip()
            if not item or item in seen:
                continue
            molecule = Chem.MolFromSmiles(item)
            if molecule is None:
                continue
            seen.add(item)
            bits = np.zeros(SIZE, dtype=np.uint8)
            for bit in generator.GetFingerprint(molecule).GetOnBits():
                bits[bit] = 1
            rows.append(np.packbits(bits))
    packed = np.vstack(rows)
    np.save(OUT_BITS, packed)
    OUT_META.write_text(json.dumps({
        'structures': len(rows), 'fingerprint': 'morgan', 'radius': RADIUS,
        'bits': SIZE, 'source': str(DATA.name),
        'note': 'Every split of the shared collection, deduplicated by '
                'canonical structure. Used only to place a material in a '
                'measured accuracy band, never for prediction.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'구조 {len(rows)}개 · {OUT_BITS} ({packed.nbytes / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
