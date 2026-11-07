"""Can the fingerprint model answer the mixtures it currently refuses?

The model answers 933 of the 1,201 independent structures. Of the 268 it
refuses, 166 are refused for holding more than one organic fragment and 96
for holding no carbon at all. The refusal is a policy, not a limit: the
feature block would compute on a mixture, the recipe just was not defined
for one, so the model declines rather than guess which fragment the label
belongs to.

For a bacterial mutagenicity call there is a defensible way round it. A
mixture is mutagenic if any component is, so each organic fragment can be
scored on its own and the highest score taken. This asks whether that is
true in the data rather than only in the argument: it scores the fragments,
takes the highest, and compares the calls with the measured labels on those
rows and on the whole file.

Nothing is fitted here and the model file is not touched. Run it with the
model's own environment, which carries the RDKit the feature block needs:

    external_tools/ames-env/Scripts/python.exe -I \\
      evaluation/scripts/evaluate_component_fallback.py
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "ames_xgboost_balanced"
RESULTS = ROOT / "evaluation" / "results"
sys.path.insert(0, str(MODEL))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402

from chemistry import normalize_structure  # noqa: E402
# The entry point that registers the shipped recipe's own feature blocks,
# which is the one the worker uses. `predict.predict` does not, so it raises
# on this model.
from predict_expanded import predict_frame  # noqa: E402

THRESHOLD = 0.50


def organic_fragments(smiles):
    """The carbon-bearing fragments of a structure, largest first."""
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return []
    pieces = []
    for fragment in Chem.GetMolFrags(mol, asMols=True):
        if any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms()):
            pieces.append((fragment.GetNumHeavyAtoms(),
                           Chem.MolToSmiles(fragment, canonical=True,
                                            isomericSmiles=True)))
    pieces.sort(reverse=True)
    return [smiles for _, smiles in pieces]


def balanced(frame, call_column="call"):
    decided = frame[frame[call_column].isin([0, 1])]
    positives = decided[decided["label"] == 1]
    negatives = decided[decided["label"] == 0]
    if positives.empty or negatives.empty:
        return None, None, None
    sensitivity = float((positives[call_column] == 1).mean())
    specificity = float((negatives[call_column] == 0).mean())
    return sensitivity, specificity, (sensitivity + specificity) / 2


def auc(labels, scores):
    mask = ~np.isnan(scores)
    y, s = np.asarray(labels)[mask], np.asarray(scores)[mask]
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_scores = s[order]
    index = 0
    while index < len(s):
        stop = index
        while stop + 1 < len(s) and sorted_scores[stop + 1] == sorted_scores[index]:
            stop += 1
        ranks[order[index:stop + 1]] = (index + stop) / 2.0 + 1.0
        index = stop + 1
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0)
                 / (positives * negatives))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", default=str(RESULTS / "ames_independent_unseen.csv"))
    parser.add_argument("--out", default=str(RESULTS / "component_fallback.json"))
    args = parser.parse_args()

    truth = pd.read_csv(args.truth, encoding="utf-8-sig")
    truth = truth[truth["label"].isin([0, 1])].reset_index(drop=True)
    statuses = [normalize_structure(value)["status"]
                for value in truth["smiles"].astype(str)]
    truth["status"] = statuses
    print(truth["status"].value_counts().to_dict())

    # What the model answers today.
    shipped = predict_frame(truth, MODEL, "smiles", threshold=THRESHOLD)
    truth["shipped_score"] = shipped["ames_score"].to_numpy(dtype=float)
    truth["shipped_call"] = [
        int(value) if pd.notna(value) else -1
        for value in shipped["ames_prediction"]]

    mixtures = truth[truth["status"] == "multiple_organic_components"]
    print(f"혼합물 {len(mixtures)}종 · 양성 {int((mixtures['label'] == 1).sum())}")

    # One row per fragment, scored in one call so the feature block runs once.
    rows = []
    for position, record in mixtures.iterrows():
        for order, fragment in enumerate(organic_fragments(record["smiles"])):
            rows.append({"row": position, "order": order, "smiles": fragment})
    pieces = pd.DataFrame(rows)
    print(f"조각 {len(pieces)}개를 따로 점수 낸다")
    scored = predict_frame(pieces, MODEL, "smiles", threshold=THRESHOLD)
    pieces["score"] = scored["ames_score"].to_numpy(dtype=float)
    pieces["status"] = scored["structure_status"].to_numpy()
    unreadable = int(pieces["score"].isna().sum())
    print(f"조각 중 읽히지 않은 것 {unreadable}개")

    # Three ways of turning several fragment scores into one. Taking the
    # highest follows the argument above; the other two are here because the
    # argument could be wrong, and a mixture's label may well have been
    # measured on whichever component dominates it by mass.
    ways = {
        "최댓값": pieces.groupby("row")["score"].max(),
        "평균": pieces.groupby("row")["score"].mean(),
        "가장 큰 조각": pieces[pieces["order"] == 0].set_index("row")["score"],
    }
    report_ways = {}
    for way, series in ways.items():
        answered = series.dropna()
        frame = truth.loc[answered.index].copy()
        frame["score"] = answered.to_numpy()
        frame["call"] = (frame["score"] >= THRESHOLD).astype(int)
        sensitivity, specificity, value = balanced(frame)
        # Where the line would have to sit for these rows to balance, which
        # says whether the ranking is sound and only the cutoff is wrong.
        best_cut, best_value = None, -1.0
        for cutoff in np.unique(np.round(frame["score"].to_numpy(), 3)):
            trial = frame.assign(
                call=(frame["score"] >= cutoff).astype(int))
            _, _, trial_value = balanced(trial)
            if trial_value is not None and trial_value > best_value:
                best_cut, best_value = float(cutoff), trial_value
        report_ways[way] = {
            "rows": int(len(frame)),
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": value,
            "auc": auc(frame["label"], frame["score"].to_numpy()),
            "own_balance_point": {"cutoff": best_cut,
                                  "balanced_accuracy": best_value}}
        entry = report_ways[way]
        print(f"  {way:8s} {entry['rows']}행 민감도 {sensitivity:.3f} "
              f"특이도 {specificity:.3f} 균형 {value:.3f} "
              f"AUC {entry['auc']:.3f} · 자기 균형점 "
              f"{best_cut:.2f}에서 {best_value:.3f}")

    highest = ways["최댓값"]
    answered = highest.dropna()
    recovered = truth.loc[answered.index].copy()
    recovered["score"] = answered.to_numpy()
    recovered["call"] = (recovered["score"] >= THRESHOLD).astype(int)

    report = {"truth_file": Path(args.truth).name,
              "rows": int(len(truth)),
              "threshold": THRESHOLD,
              "status_counts": truth["status"].value_counts().to_dict(),
              "fragments": {"rows_tried": int(len(mixtures)),
                            "fragments_scored": int(len(pieces)),
                            "fragments_unreadable": unreadable,
                            "rows_recovered": int(len(recovered))}}

    report["ways_of_combining_fragments"] = report_ways
    sensitivity, specificity, value = balanced(recovered)
    report["recovered_rows"] = {
        "rows": int(len(recovered)),
        "positives": int((recovered["label"] == 1).sum()),
        "negatives": int((recovered["label"] == 0).sum()),
        "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": value,
        "auc": auc(recovered["label"], recovered["score"].to_numpy())}
    print(f"되찾은 {len(recovered)}행에서 민감도 {sensitivity:.3f} "
          f"특이도 {specificity:.3f} 균형 {value:.3f} "
          f"AUC {report['recovered_rows']['auc']:.3f}")

    # The whole file, before and after.
    for name, frame in (("shipped", truth.assign(
                             call=truth["shipped_call"],
                             score=truth["shipped_score"])),
                        ("with_fallback", None)):
        if frame is None:
            merged = truth.copy()
            merged["score"] = merged["shipped_score"]
            merged["call"] = merged["shipped_call"]
            merged.loc[recovered.index, "score"] = recovered["score"]
            merged.loc[recovered.index, "call"] = recovered["call"]
            frame = merged
        sensitivity, specificity, value = balanced(frame)
        decided = int(frame["call"].isin([0, 1]).sum())
        report[name] = {
            "decided": decided, "coverage": decided / len(truth),
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": value,
            "auc": auc(frame["label"], frame["score"].to_numpy(dtype=float)),
            "caught_of_all_positives": float(
                ((frame["label"] == 1) & (frame["call"] == 1)).sum()
                / (frame["label"] == 1).sum())}
        entry = report[name]
        print(f"{name:14s} 답한 비율 {entry['coverage']:.3f} 민감도 "
              f"{entry['sensitivity']:.3f} 특이도 {entry['specificity']:.3f} "
              f"균형 {entry['balanced_accuracy']:.3f} AUC {entry['auc']:.3f} "
              f"· 전체 양성 중 잡은 비율 "
              f"{entry['caught_of_all_positives']:.3f}")

    report["notes"] = [
        "조각별 점수의 최댓값을 쓴다. 혼합물은 성분 하나가 변이원성이면 "
        "변이원성이라는 전제이고, 이 전제가 자료에서 맞는지를 재는 것이 "
        "이 파일이다.",
        "모델 파일은 고치지 않았다. 못 읽겠다고 답하던 행에 같은 모델을 "
        "조각별로 돌린 것뿐이다.",
        "탄소가 없는 구조는 그대로 답하지 않는다. 지문이 만들어지지 않는다.",
        "음성 예측은 선별 결과이며 안전성의 근거가 아니다.",
    ]

    destination = Path(args.out)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"기록: {destination}")


if __name__ == "__main__":
    main()
