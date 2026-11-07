"""Measure one ADMET-AI endpoint against a reference set.

Runs in ADMET-AI's own interpreter. The model returns a probability per
endpoint; the call is taken at the conventional 0.50, the same line the Ames
screens use, and the raw score is kept so an AUC can be computed over the
rows the model answered.

A structure the model could not score is an abstention, never a negative.
"""

import argparse
import csv
import io
from pathlib import Path


def auc(pairs):
    """ROC-AUC by rank, ties given the average rank."""
    pairs = sorted(pairs, key=lambda pair: pair[0])
    ranks, index = {}, 0
    while index < len(pairs):
        last = index
        while last + 1 < len(pairs) and pairs[last + 1][0] == pairs[index][0]:
            last += 1
        rank = (index + last) / 2 + 1
        for position in range(index, last + 1):
            ranks[position] = rank
        index = last + 1
    positives = [position for position, pair in enumerate(pairs) if pair[1] == 1]
    n_pos, n_neg = len(positives), len(pairs) - len(positives)
    if not n_pos or not n_neg:
        return None
    total = sum(ranks[position] for position in positives)
    return (total - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--column", required=True, help="ADMET-AI endpoint column")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unseen-only", action="store_true")
    args = parser.parse_args()

    with args.reference.open(encoding="utf-8", newline="") as handle:
        reference = list(csv.DictReader(handle))
    if args.unseen_only:
        reference = [row for row in reference if row.get("training_overlap") != "yes"]
    print(f"{args.column}: {len(reference)} structures")

    from admet_ai import ADMETModel

    model = ADMETModel(include_physchem=False, drugbank_path=None, num_workers=0)
    predictions = model.predict([row["smiles"] for row in reference])
    if args.column not in predictions.columns:
        raise SystemExit(f"ADMET-AI did not return {args.column!r}; "
                         f"available: {sorted(predictions.columns)[:40]}")
    scores = list(predictions[args.column])

    rows, scored = [], []
    for row, score in zip(reference, scores):
        value = None
        try:
            value = float(score)
            if value != value or not 0.0 <= value <= 1.0:
                value = None
        except (TypeError, ValueError):
            value = None
        rows.append({"id": row["id"], "smiles": row["smiles"], "label": row["label"],
                     "score": "" if value is None else f"{value:.6f}",
                     "call": "" if value is None else int(value >= 0.5)})
        if value is not None:
            scored.append((value, int(row["label"])))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "smiles", "label", "score", "call"])
        writer.writeheader()
        writer.writerows(rows)

    tp = sum(1 for value, truth in scored if value >= 0.5 and truth == 1)
    fn = sum(1 for value, truth in scored if value < 0.5 and truth == 1)
    tn = sum(1 for value, truth in scored if value < 0.5 and truth == 0)
    fp = sum(1 for value, truth in scored if value >= 0.5 and truth == 0)
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    print(f"  answered {len(scored)}/{len(rows)} ({len(scored)/len(rows):.3f})")
    print(f"  sensitivity  {sensitivity:.3f}" if sensitivity is not None else "  sensitivity  n/a")
    print(f"  specificity  {specificity:.3f}" if specificity is not None else "  specificity  n/a")
    if sensitivity is not None and specificity is not None:
        print(f"  balanced     {(sensitivity + specificity) / 2:.3f}")
    area = auc(scored)
    print(f"  AUC          {area:.3f}" if area is not None else "  AUC          n/a")
    print(f"  tp {tp}  fp {fp}  tn {tn}  fn {fn}")


if __name__ == "__main__":
    main()
