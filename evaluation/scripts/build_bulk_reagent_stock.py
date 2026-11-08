"""Turn the readable bulk-reagent list into a stock file AiZynthFinder reads.

The list is `test/models/bulk_reagents.txt`, name and SMILES per line, so the
set of things this project calls "available" can be read and argued with. The
stock AiZynthFinder loads is an HDF5 of InChIKeys, which cannot be. This
script is the one step between them.

    C:/Users/hahab/miniconda3/envs/aizynth-env/python.exe \
        evaluation/scripts/build_bulk_reagent_stock.py

Writes test/models/bulk_reagents_stock.hdf5 and reports which entries ZINC
already had, so the list does not quietly grow to cover things that were
never missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "test" / "models"
SOURCE = MODELS / "bulk_reagents.txt"
TARGET = MODELS / "bulk_reagents_stock.hdf5"
ZINC = MODELS / "zinc_stock.hdf5"


def entries():
    for number, line in enumerate(SOURCE.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "\t" not in text:
            raise SystemExit(f"{SOURCE.name}:{number} 이름과 SMILES를 탭으로 "
                             f"나눠야 합니다: {text!r}")
        name, smiles = (part.strip() for part in text.split("\t", 1))
        yield number, name, smiles


def main():
    rows = []
    for number, name, smiles in entries():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise SystemExit(f"{SOURCE.name}:{number} 구조를 읽지 못했습니다: "
                             f"{name} {smiles!r}")
        rows.append({"name": name, "smiles": smiles,
                     "inchi_key": inchi.MolToInchiKey(molecule)})
    if not rows:
        raise SystemExit(f"{SOURCE.name}에 항목이 없습니다.")

    keys = [row["inchi_key"] for row in rows]
    if len(set(keys)) != len(keys):
        seen, repeated = set(), set()
        for key in keys:
            (repeated if key in seen else seen).add(key)
        names = [row["name"] for row in rows if row["inchi_key"] in repeated]
        raise SystemExit(f"같은 구조가 두 번 있습니다: {', '.join(sorted(set(names)))}")

    already = set()
    if ZINC.is_file():
        existing = set(pd.read_hdf(ZINC, "table")["inchi_key"])
        already = {row["name"] for row in rows if row["inchi_key"] in existing}

    frame = pd.DataFrame({"inchi_key": keys})
    frame.to_hdf(TARGET, key="table", mode="w")

    print(f"{len(rows)}종을 {TARGET.relative_to(ROOT)}에 넣었습니다.")
    if already:
        print("ZINC에 이미 있던 것 (빼도 됩니다): " + ", ".join(sorted(already)))
    else:
        print("전부 ZINC에 없던 것입니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
