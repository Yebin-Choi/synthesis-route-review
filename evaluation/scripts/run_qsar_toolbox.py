"""Run the OECD QSAR Toolbox Ames alert profilers over a labelled structure set.

Calls the Toolbox's own WebAPI on this machine. Both profilers are run in one
pass because they share the structure search, which is the slow half of the
call.

An abstention is never scored as a negative: structures the Toolbox declined
are written as `abstain` and left out of sensitivity and specificity, exactly
as the other tools' runners do.

    python evaluation/scripts/run_qsar_toolbox.py \
        --input evaluation/results/ames_independent_unseen.csv \
        --out-prefix evaluation/results/qsar_toolbox_independent
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "app"))

import qsar_toolbox_service as toolbox  # noqa: E402
from toxicology import canonicalize_smiles  # noqa: E402

FIELDS = ["id", "smiles", "tool", "model", "version", "endpoint", "prediction",
          "applicability", "source", "reason", "alerts"]

PROFILERS = {
    "qsar_toolbox_oasis": toolbox.OASIS_AMES_PROFILER,
    "qsar_toolbox_iss": toolbox.ISS_AMES_PROFILER,
}


def load_inputs(path, limit):
    with io.open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "smiles"}.issubset(reader.fieldnames or []):
            raise ValueError("Input CSV requires id and smiles columns")
        rows, seen = [], set()
        for row in reader:
            if not row["id"] or not row["smiles"] or row["id"] in seen:
                raise ValueError("Input IDs must be unique and nonempty")
            seen.add(row["id"])
            rows.append(row)
            if limit is not None and len(rows) == limit:
                break
    if not rows:
        raise ValueError("Input CSV has no molecules")
    return rows


def score(records):
    """Sensitivity, specificity and balanced accuracy over answered rows only."""
    tp = tn = fp = fn = 0
    answered = 0
    for record in records:
        label, prediction = record["label"], record["prediction"]
        if prediction == "abstain" or label not in ("0", "1"):
            continue
        answered += 1
        if label == "1":
            if prediction == "1":
                tp += 1
            else:
                fn += 1
        else:
            if prediction == "1":
                fp += 1
            else:
                tn += 1
    positives, negatives = tp + fn, tn + fp
    sensitivity = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    balanced = (None if sensitivity is None or specificity is None
                else (sensitivity + specificity) / 2)
    return {"n_total": len(records), "n_answered": answered,
            "answered_rate": answered / len(records) if records else None,
            "true_positive": tp, "true_negative": tn,
            "false_positive": fp, "false_negative": fn,
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced": balanced}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    rows = load_inputs(Path(args.input), args.limit)
    versions = toolbox.toolbox_versions()
    if not versions.get("toolbox"):
        raise SystemExit("QSAR Toolbox WebAPI에 연결하지 못했습니다. "
                         "LMC.Toolbox.WebAPI.exe가 실행 중인지 확인하세요.")

    guids = {tool_id: toolbox.profiler_guid(caption)
             for tool_id, caption in PROFILERS.items()}
    print(f"Toolbox {versions['toolbox']} / WebAPI v{versions['webapi']} "
          f"at {toolbox.base_url()}", flush=True)
    for tool_id, guid in guids.items():
        print(f"  {tool_id}: {PROFILERS[tool_id]}  {guid}", flush=True)

    collected = {tool_id: [] for tool_id in PROFILERS}
    started = time.time()
    for index, row in enumerate(rows, 1):
        canonical = canonicalize_smiles(row["smiles"])
        if canonical is None:
            outcomes = {guid: {"status": "invalid", "verdict": None, "alerts": [],
                               "reason": "unreadable_structure"}
                        for guid in guids.values()}
        else:
            outcomes = toolbox.profile_multi(canonical, list(guids.values()),
                                             timeout=args.timeout)
        for tool_id, guid in guids.items():
            outcome = outcomes[guid]
            if outcome["status"] == "success":
                prediction = "1" if outcome["verdict"] == "positive" else "0"
                applicability = "in_domain"
            else:
                prediction = "abstain"
                applicability = "declined"
            collected[tool_id].append({
                "id": row["id"], "smiles": row["smiles"],
                "tool": "OECD QSAR Toolbox",
                "model": PROFILERS[tool_id],
                "version": versions["toolbox"],
                "endpoint": "ames",
                "prediction": prediction,
                "applicability": applicability,
                "source": toolbox.SOURCE,
                "reason": outcome.get("reason") or "",
                "alerts": "; ".join(outcome.get("alerts") or []),
                "label": row.get("label", ""),
            })
        if index % 50 == 0 or index == len(rows):
            rate = index / max(time.time() - started, 1e-9)
            print(f"  {index}/{len(rows)}  {rate:.1f}/s", flush=True)

    summary = {"input": str(args.input), "n": len(rows),
               "toolbox_version": versions["toolbox"],
               "webapi_version": versions["webapi"],
               "elapsed_seconds": round(time.time() - started, 1),
               "profilers": {}}
    for tool_id, records in collected.items():
        out_csv = Path(f"{args.out_prefix}_{tool_id}.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with io.open(out_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        metrics = score(records)
        reasons = {}
        for record in records:
            if record["prediction"] == "abstain":
                reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
        summary["profilers"][tool_id] = {
            "profiler": PROFILERS[tool_id], "guid": guids[tool_id],
            "predictions_csv": str(out_csv), "metrics": metrics,
            "abstain_reasons": reasons}
        print(f"\n{tool_id}  {PROFILERS[tool_id]}")
        print(f"  답한 비율 {metrics['answered_rate']:.3f} "
              f"({metrics['n_answered']}/{metrics['n_total']})")
        if metrics["balanced"] is not None:
            print(f"  민감도 {metrics['sensitivity']:.3f} · "
                  f"특이도 {metrics['specificity']:.3f} · "
                  f"균형 정확도 {metrics['balanced']:.3f}")
        if reasons:
            print(f"  기권 사유 {reasons}")

    manifest = Path(f"{args.out_prefix}_manifest.json")
    with io.open(manifest, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"\n기록: {manifest}")


if __name__ == "__main__":
    main()
