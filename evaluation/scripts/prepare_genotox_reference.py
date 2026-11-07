"""Build the reference sets for the two endpoints Ames does not cover.

Both sources are CC-BY 4.0 and are read as data, never executed. Structures
are canonicalised with RDKit and deduplicated; a structure whose duplicates
disagree on the label is dropped rather than resolved, because picking one
would invent an answer the source does not give.

Carcinogenicity carries a training-overlap flag. ADMET-AI's carcinogenicity
model is fitted on TDC Carcinogens_Lagunin, so any structure of that set must
be excluded before the model is compared with tools that never saw it. This
is the same rule the Ames work follows.
"""

import argparse
import csv
import io
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canonical(smiles):
    molecule = Chem.MolFromSmiles((smiles or "").strip())
    return None if molecule is None else Chem.MolToSmiles(molecule)


def collapse(pairs):
    """One row per structure. A structure labelled both ways is dropped."""
    seen = {}
    for key, label in pairs:
        seen.setdefault(key, set()).add(label)
    kept = {key: next(iter(labels)) for key, labels in seen.items() if len(labels) == 1}
    return kept, sorted(key for key, labels in seen.items() if len(labels) > 1)


def read(path, delimiter=","):
    with io.open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write(path, rows, columns):
    with io.open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carcinogenicity", type=Path, required=True)
    parser.add_argument("--micronucleus", type=Path, required=True)
    parser.add_argument("--tdc-carcinogens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    trained = set()
    for row in read(args.tdc_carcinogens, delimiter="\t"):
        key = canonical((row["Drug"] or "").strip().strip('"'))
        if key:
            trained.add(key)
    print(f"TDC Carcinogens_Lagunin structures: {len(trained)}")

    pairs, unreadable = [], 0
    for row in read(args.carcinogenicity):
        key = canonical(row["SMILES"])
        label = (row["Carcinogenicity"] or "").strip().lower()
        if key is None or label not in ("yes", "no"):
            unreadable += 1
            continue
        pairs.append((key, "1" if label == "yes" else "0"))
    kept, conflicted = collapse(pairs)
    rows = [{"id": f"carc-{index:04d}", "smiles": key, "label": label,
             "training_overlap": "yes" if key in trained else "no"}
            for index, (key, label) in enumerate(sorted(kept.items()), start=1)]
    write(args.out / "carcinogenicity_reference.csv", rows,
          ["id", "smiles", "label", "training_overlap"])
    unseen = [row for row in rows if row["training_overlap"] == "no"]
    print(f"carcinogenicity: {len(rows)} structures "
          f"(unreadable {unreadable}, conflicting {len(conflicted)}), "
          f"labels {Counter(row['label'] for row in rows)}")
    print(f"  not in the ADMET-AI training set: {len(unseen)}, "
          f"labels {Counter(row['label'] for row in unseen)}")

    pairs, unreadable = [], 0
    for row in read(args.micronucleus, delimiter="\t"):
        key = canonical(row["SMILES"])
        label = (row["Experimental value"] or "").strip().lower()
        if key is None or label not in ("active", "inactive"):
            unreadable += 1
            continue
        pairs.append((key, "1" if label == "active" else "0"))
    kept, conflicted = collapse(pairs)
    rows = [{"id": f"mnvit-{index:04d}", "smiles": key, "label": label}
            for index, (key, label) in enumerate(sorted(kept.items()), start=1)]
    write(args.out / "micronucleus_reference.csv", rows, ["id", "smiles", "label"])
    print(f"micronucleus: {len(rows)} structures "
          f"(unreadable {unreadable}, conflicting {len(conflicted)}), "
          f"labels {Counter(row['label'] for row in rows)}")


if __name__ == "__main__":
    main()
