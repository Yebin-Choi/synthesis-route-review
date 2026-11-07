"""Measure a Toxtree rulebase against a reference set, keeping abstentions.

One module per run. The call is read from the module's own verdict column,
never from the alert counter, because an alert count of zero is not the same
as the tree reaching a negative conclusion.

A record the tree could not evaluate is an abstention, not a negative. That
is the same rule the Ames screens follow, and it is why `answered` is
reported beside every figure here.
"""

import argparse
import csv
import io
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOXTREE_HOME = ROOT / "external_tools" / "toxtree-home"
TOXTREE_JAR = "Toxtree-3.1.0.1851.jar"

# Per module: the verdict column, the text that means "negative", and the
# column that marks a record the tree failed on.
MODULES = {
    "mutant.BB_CarcMutRules": {
        "negative_columns": ("Negative for genotoxic carcinogenicity",
                             "Negative for nongenotoxic carcinogenicity"),
        "error_column": "Error when applying the decision tree",
        "label": "Toxtree Benigni/Bossa (carcinogenicity)",
    },
    "mic.MICRules": {
        "verdict_column": "Structure Alerts for the in vivo micronucleus assay in rodents",
        "negative_text": "No alerts",
        "error_column": None,
        "label": "Toxtree in vivo micronucleus alerts",
    },
}


def run_toxtree(module, structures, timeout):
    with tempfile.TemporaryDirectory(prefix="toxtree_eval_") as directory:
        work = Path(directory)
        source, result = work / "input.csv", work / "output.csv"
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["evaluation_id", "SMILES"])
            writer.writeheader()
            for identity, smiles in structures:
                writer.writerow({"evaluation_id": identity, "SMILES": smiles})
        completed = subprocess.run(
            ["java", "-Xmx3072m", "-Djava.awt.headless=true", "-Dfile.encoding=UTF-8",
             "-jar", str(TOXTREE_HOME / TOXTREE_JAR), "-n",
             "-i", str(source), "-o", str(result), "-m", module],
            cwd=TOXTREE_HOME, capture_output=True, text=True, errors="replace",
            timeout=timeout, check=False)
        if not result.is_file():
            raise SystemExit(f"Toxtree produced no output:\n{completed.stderr[-2000:]}")
        with result.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


def call_of(row, spec):
    """1 positive, 0 negative, None not evaluated."""
    error_column = spec.get("error_column")
    if error_column and (row.get(error_column) or "").strip().upper() == "YES":
        return None
    if "negative_columns" in spec:
        values = [(row.get(name) or "").strip().upper() for name in spec["negative_columns"]]
        if any(value not in ("YES", "NO") for value in values):
            return None
        return 0 if all(value == "YES" for value in values) else 1
    verdict = (row.get(spec["verdict_column"]) or "").strip()
    if not verdict:
        return None
    return 0 if spec["negative_text"].casefold() in verdict.casefold() else 1


def metrics(pairs):
    tp = sum(1 for call, truth in pairs if call == 1 and truth == 1)
    fn = sum(1 for call, truth in pairs if call == 0 and truth == 1)
    tn = sum(1 for call, truth in pairs if call == 0 and truth == 0)
    fp = sum(1 for call, truth in pairs if call == 1 and truth == 0)
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = None
    if sensitivity is not None and specificity is not None:
        balanced = (sensitivity + specificity) / 2
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "sensitivity": sensitivity,
            "specificity": specificity, "balanced": balanced}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--module", choices=sorted(MODULES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--unseen-only", action="store_true",
                        help="drop rows flagged training_overlap=yes")
    args = parser.parse_args()
    spec = MODULES[args.module]

    with args.reference.open(encoding="utf-8", newline="") as handle:
        reference = list(csv.DictReader(handle))
    if args.unseen_only:
        reference = [row for row in reference if row.get("training_overlap") != "yes"]
    truth = {row["id"]: int(row["label"]) for row in reference}
    structures = [(row["id"], row["smiles"]) for row in reference]
    print(f"{spec['label']}: {len(structures)} structures")

    results = run_toxtree(args.module, structures, args.timeout)
    calls = {row.get("evaluation_id"): call_of(row, spec) for row in results}

    rows = [{"id": identity, "smiles": smiles, "label": truth[identity],
             "call": "" if calls.get(identity) is None else calls[identity]}
            for identity, smiles in structures]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "smiles", "label", "call"])
        writer.writeheader()
        writer.writerows(rows)

    answered = [(calls[identity], truth[identity]) for identity, _ in structures
                if calls.get(identity) is not None]
    figures = metrics(answered)
    share = len(answered) / len(structures) if structures else 0
    print(f"  answered {len(answered)}/{len(structures)} ({share:.3f})")
    for name in ("sensitivity", "specificity", "balanced"):
        value = figures[name]
        print(f"  {name:12s} {'n/a' if value is None else f'{value:.3f}'}")
    print(f"  tp {figures['tp']}  fp {figures['fp']}  tn {figures['tn']}  fn {figures['fn']}")


if __name__ == "__main__":
    main()
