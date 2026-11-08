"""Figures for the two in-house Ames models' development write-up.

Everything here is drawn from files already in the repository. Nothing is
re-fitted and no model is run: the prediction CSVs were written by earlier
measurement runs, and the sweep tables are transcribed from the benchmark log
sections they cite. A figure that cannot be drawn from a stored file is not
drawn.

    external_tools/ames-env/Scripts/python.exe -I -X utf8 \
        evaluation/scripts/build_model_development_figures.py

Output: evaluation/figures/model_development/
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evaluation" / "results"
OUT = ROOT / "evaluation" / "figures" / "model_development"

# Korean labels need a Korean face. Malgun Gothic ships with Windows; without
# it every Hangul glyph would come out as a box.
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 160
matplotlib.rcParams["savefig.bbox"] = "tight"

INK = "#1f2933"
GRID = "#d8dee4"
MODEL1 = "#2563eb"
MODEL2 = "#d97706"
OTHER = "#7b8794"
ACCENT = "#059669"

# The panel's own labels are too long to sit side by side under a bar group.
SHORT_NAMES = {
    "rules": "전문가 규칙\n(ICH M7)", "qsar_toolbox": "QSAR\nToolbox 4.9",
    "epa_test": "EPA TEST\n5.1.2", "toxtree": "Toxtree\n3.1.0",
    "admet": "ADMET-AI\n2.0.1", "ames_desc": "자체 모델 2\n기술자만",
    "ames_xgb": "자체 모델 1\n지문+기술자",
}

TRUTH = RESULTS / "ames_independent_unseen.csv"
PANEL = RESULTS / "tool_panel_wide.json"

SCORED = {
    "ames_xgb": ("자체 모델 1: 지문+기술자",
                 RESULTS / "ames_balanced_independent_predictions.csv", MODEL1),
    "ames_desc": ("자체 모델 2: 기술자만",
                  RESULTS / "ames_descriptor_balanced_independent_predictions.csv", MODEL2),
    "admet": ("ADMET-AI 2.0.1",
              RESULTS / "admet_independent_predictions.csv", OTHER),
}


def load_truth():
    with TRUTH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["id"]: int(row["label"]) for row in rows}


def load_scores(path):
    """One structure's score, or nothing when the tool did not answer.

    An abstention is left out rather than written as 0. A missing score read
    as a low score would turn "could not judge" into "judged negative".
    """
    scores = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("score") or "").strip()
            if row.get("prediction") not in {"0", "1"} or not raw:
                continue
            try:
                scores[row["id"]] = float(raw)
            except ValueError:
                continue
    return scores


def roc_points(labels, scores):
    """ROC by rank, ties kept together, so a flat score region cannot be
    walked through one row at a time to flatter the curve."""
    order = np.argsort(-scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    positives, negatives = labels.sum(), (1 - labels).sum()
    tpr, fpr = [0.0], [0.0]
    index = 0
    seen_pos = seen_neg = 0
    while index < len(scores):
        stop = index
        while stop < len(scores) and scores[stop] == scores[index]:
            stop += 1
        seen_pos += labels[index:stop].sum()
        seen_neg += (1 - labels[index:stop]).sum()
        tpr.append(seen_pos / positives)
        fpr.append(seen_neg / negatives)
        index = stop
    return np.array(fpr), np.array(tpr)


def auc_of(labels, scores):
    fpr, tpr = roc_points(labels, scores)
    return float(np.trapezoid(tpr, fpr))


def save(figure, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def style(axis):
    axis.set_axisbelow(True)
    axis.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK, labelsize=9)


def figure_roc(truth):
    """ROC on the rows all three models answered.

    Model 1 abstains on 268 of the 1,201, so curves drawn on each model's own
    answered rows would be three different questions. The shared rows make it
    one.
    """
    loaded = {key: load_scores(path) for key, (_, path, _) in SCORED.items()}
    shared = sorted(set(truth) & set.intersection(*(set(v) for v in loaded.values())))
    labels = np.array([truth[i] for i in shared], dtype=float)

    figure, axis = plt.subplots(figsize=(5.2, 5.0))
    style(axis)
    for key, (label, _, colour) in SCORED.items():
        scores = np.array([loaded[key][i] for i in shared], dtype=float)
        fpr, tpr = roc_points(labels, scores)
        axis.plot(fpr, tpr, color=colour, linewidth=2.0,
                  label=f"{label} (AUC {auc_of(labels, scores):.3f})")
    axis.plot([0, 1], [0, 1], color=GRID, linewidth=1.0, linestyle="--")
    axis.set_xlabel("1 - 특이도 (음성을 양성이라 한 비율)", color=INK, fontsize=10)
    axis.set_ylabel("민감도 (양성을 양성이라 한 비율)", color=INK, fontsize=10)
    axis.set_title(f"세 모델이 모두 답한 {len(shared)}종에서의 ROC",
                   color=INK, fontsize=11.5, pad=10)
    axis.legend(loc="lower right", fontsize=8.5, frameon=False)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    return save(figure, "fig1_roc_shared_rows.png"), len(shared)


def figure_balanced_interval(panel):
    """Every tool's balanced accuracy with the 95% interval.

    Printed as a ranked list the rows read as an order. The intervals say how
    much of that order the measurement supports.
    """
    order = ["rules", "qsar_toolbox", "epa_test", "toxtree", "admet",
             "ames_desc", "ames_xgb"]
    rows = [panel["single"][key] for key in order]
    labels = [row["label"] for row in rows]
    values = [row["balanced_accuracy"] for row in rows]
    low = [row["balanced_interval"]["low"] for row in rows]
    high = [row["balanced_interval"]["high"] for row in rows]
    colours = [MODEL1 if key == "ames_xgb" else MODEL2 if key == "ames_desc"
               else OTHER for key in order]

    figure, axis = plt.subplots(figsize=(6.6, 4.0))
    style(axis)
    y = np.arange(len(rows))
    axis.hlines(y, low, high, color=colours, linewidth=2.2, alpha=0.55)
    axis.scatter(values, y, color=colours, s=46, zorder=3)
    for index, value in enumerate(values):
        axis.annotate(f"{value:.3f}", (value, index), textcoords="offset points",
                      xytext=(0, 9), ha="center", fontsize=8.5, color=INK)
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=9)
    axis.set_xlabel("균형 정확도 (민감도와 특이도의 평균), 95% 구간",
                    color=INK, fontsize=10)
    axis.set_title(f"미학습 구조 {panel['rows']:,}종에서 잰 일곱 도구",
                   color=INK, fontsize=11.5, pad=10)
    axis.set_xlim(0.60, 0.92)
    return save(figure, "fig2_balanced_accuracy_intervals.png")


def figure_sensitivity_specificity(panel):
    """Sensitivity and specificity apart, because the average hides which one
    a tool bought its score with."""
    order = ["rules", "qsar_toolbox", "epa_test", "toxtree", "admet",
             "ames_desc", "ames_xgb"]
    rows = [panel["single"][key] for key in order]
    labels = [SHORT_NAMES[key] for key in order]
    sensitivity = [row["sensitivity"] for row in rows]
    specificity = [row["specificity"] for row in rows]
    coverage = [row["coverage"] for row in rows]

    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    style(axis)
    x = np.arange(len(rows))
    width = 0.27
    axis.bar(x - width, sensitivity, width, color=MODEL1, label="민감도")
    axis.bar(x, specificity, width, color=MODEL2, label="특이도")
    axis.bar(x + width, coverage, width, color=OTHER, label="답한 비율")
    for index in range(len(rows)):
        for offset, value in ((-width, sensitivity[index]), (0, specificity[index]),
                              (width, coverage[index])):
            axis.annotate(f"{value:.2f}", (index + offset, value),
                          textcoords="offset points", xytext=(0, 3),
                          ha="center", fontsize=7.2, color=INK)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=8.5)
    axis.set_ylabel("비율", color=INK, fontsize=10)
    axis.set_ylim(0, 1.12)
    axis.set_title("민감도와 특이도를 나눠 보면 도구마다 사는 방식이 다르다",
                   color=INK, fontsize=11.5, pad=10)
    axis.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper left")
    return save(figure, "fig3_sensitivity_specificity.png")


def figure_two_models(panel, overlap):
    """What each in-house model covers, and where they disagree."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.6))

    style(left)
    both = overlap["both_answered"]
    only_two = overlap["only_model_2"]
    total = both + only_two
    left.barh([1], [both], color=MODEL1, height=0.46)
    left.barh([1], [only_two], left=[both], color="#eceff3", height=0.46)
    left.barh([0], [total], color=MODEL2, height=0.46)
    left.set_yticks([1, 0])
    left.set_yticklabels(["자체 모델 1", "자체 모델 2"], fontsize=9.5)
    left.set_xlim(0, total * 1.02)
    left.set_xlabel(f"답한 구조 수 (전체 {total:,}종)", color=INK, fontsize=9.5)
    left.annotate(f"{both:,}", (both / 2, 1), ha="center", va="center",
                  fontsize=9.5, color="white")
    left.annotate(f"기권 {only_two:,}", (both + only_two / 2, 1), ha="center",
                  va="center", fontsize=8.5, color=INK)
    left.annotate(f"{total:,}", (total / 2, 0), ha="center", va="center",
                  fontsize=9.5, color="white")
    left.set_title("답하는 범위", color=INK, fontsize=10.5, pad=8)

    style(right)
    agree = overlap["agree"]
    split_1 = overlap["split_model_1_right"]
    split_2 = overlap["split_model_2_right"]
    right.bar([0], [agree], color=ACCENT, width=0.5)
    right.bar([1], [split_1], color=MODEL1, width=0.5)
    right.bar([2], [split_2], color=MODEL2, width=0.5)
    for index, value in enumerate((agree, split_1, split_2)):
        right.annotate(f"{value:,}", (index, value), textcoords="offset points",
                       xytext=(0, 3), ha="center", fontsize=9, color=INK)
    right.set_xticks([0, 1, 2])
    right.set_xticklabels(["같은 답", "갈렸고\n모델 1이 맞음", "갈렸고\n모델 2가 맞음"],
                          fontsize=8.5)
    right.set_ylabel("구조 수", color=INK, fontsize=9.5)
    right.set_title(f"둘 다 답한 {both:,}종에서의 일치 "
                    f"({agree / both:.1%})", color=INK, fontsize=10.5, pad=8)

    figure.suptitle("두 모델은 좋은 쪽과 나쁜 쪽이 아니라 범위가 다른 두 도구",
                    color=INK, fontsize=11.5, y=1.02)
    return save(figure, "fig4_model1_vs_model2.png")


# Transcribed from the benchmark log. The sweeps ran on the scaffold-grouped
# cross-validation of the 19,963 training structures, not on the independent
# set, so these numbers never share an axis with the figures above.
FEATURE_SWEEP = [
    ("기술자 10개", 10, 0.8117, 0.0155),
    ("Morgan 2048", 2048, 0.8676, 0.0139),
    ("RDKit 지문 2048", 2048, 0.8752, 0.0128),
    ("MACCS 167", 167, 0.8855, 0.0111),
    ("Morgan 2048 + 기술자", 2263, 0.8978, 0.0086),
    ("기술자 215개 (모델 2)", 215, 0.8999, 0.0066),
    ("RDKit 지문 + 기술자 (모델 1)", 2263, 0.9018, 0.0075),
]
DESCRIPTOR_CURVE = [(5, 0.7433, 0.0233), (10, 0.7960, 0.0232), (20, 0.8521, 0.0138),
                    (40, 0.8861, 0.0079), (80, 0.8958, 0.0067), (160, 0.8983, 0.0070),
                    (215, 0.8998, 0.0070)]


def figure_feature_choice():
    """Why model 2 drops the fingerprint, and how few descriptors it needs."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.4, 4.0),
                                         gridspec_kw={"width_ratios": [1.25, 1]})

    style(left)
    labels = [row[0] for row in FEATURE_SWEEP]
    values = [row[2] for row in FEATURE_SWEEP]
    errors = [row[3] for row in FEATURE_SWEEP]
    colours = [MODEL1 if "모델 1" in name else MODEL2 if "모델 2" in name else OTHER
               for name in labels]
    y = np.arange(len(labels))
    left.barh(y, values, color=colours, height=0.6,
              xerr=errors, error_kw={"ecolor": INK, "elinewidth": 0.9, "capsize": 2.5})
    left.set_yticks(y)
    left.set_yticklabels(labels, fontsize=8.5)
    left.set_xlim(0.75, 0.93)
    left.set_xlabel("AUC, 골격 묶음 교차검증 (±는 겹 간 편차)", color=INK, fontsize=9.5)
    left.set_title("지문을 빼도 성적이 내려가지 않는다", color=INK, fontsize=10.5, pad=8)

    style(right)
    counts = [row[0] for row in DESCRIPTOR_CURVE]
    aucs = [row[1] for row in DESCRIPTOR_CURVE]
    spread = [row[2] for row in DESCRIPTOR_CURVE]
    right.errorbar(counts, aucs, yerr=spread, color=MODEL2, linewidth=1.8,
                   marker="o", markersize=4.5, capsize=2.5, elinewidth=0.9)
    right.axvline(80, color=GRID, linestyle="--", linewidth=1.0)
    right.annotate("80개에서 꺾인다", (80, 0.8958), xytext=(-74, -28),
                   textcoords="offset points", fontsize=8.5, color=INK)
    right.set_xscale("log")
    right.set_xticks(counts)
    right.set_xticklabels([str(count) for count in counts], fontsize=8.5)
    right.set_xlabel("쓴 기술자 수", color=INK, fontsize=9.5)
    right.set_ylabel("AUC", color=INK, fontsize=9.5)
    right.set_title("기술자를 줄여도 80개까지는 버틴다", color=INK, fontsize=10.5, pad=8)

    return save(figure, "fig5_feature_choice.png")


# Benchmark log 6-20: the same 933 independent structures, the shipped
# fingerprint model with and without class weights, swept across cutoffs.
THRESHOLD_SWEEP = {
    "가중치 켬 (현재 쓰는 학습본)": [
        (0.20, 0.881, 0.720), (0.25, 0.867, 0.770), (0.30, 0.853, 0.819),
        (0.40, 0.819, 0.876), (0.50, 0.774, 0.921), (0.60, 0.734, 0.950),
        (0.70, 0.686, 0.978)],
}
UNWEIGHTED_AT_050 = (0.698, 0.950)


def figure_threshold():
    """Why the cutoff stayed at 0.50 once the imbalance moved into training."""
    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    style(axis)
    rows = THRESHOLD_SWEEP["가중치 켬 (현재 쓰는 학습본)"]
    cutoffs = [row[0] for row in rows]
    balanced = [(row[1] + row[2]) / 2 for row in rows]
    axis.plot(cutoffs, balanced, color=MODEL1, linewidth=2.0, marker="o",
              markersize=4.5, label="가중치 켬 (현재)")
    axis.scatter([0.50], [(UNWEIGHTED_AT_050[0] + UNWEIGHTED_AT_050[1]) / 2],
                 color=OTHER, s=52, zorder=3, marker="s",
                 label="가중치 없음, 같은 0.50")
    axis.axvline(0.50, color=GRID, linestyle="--", linewidth=1.0)
    axis.annotate("화면이 쓰는 선", (0.50, 0.866), xytext=(-64, 0),
                  textcoords="offset points", fontsize=8.5, color=INK)
    axis.set_xlabel("판정 기준선", color=INK, fontsize=10)
    axis.set_ylabel("균형 정확도", color=INK, fontsize=10)
    axis.set_ylim(0.79, 0.87)
    axis.set_title("학습에서 불균형을 처리하니 0.50이 꼭대기에 들어왔다",
                   color=INK, fontsize=11, pad=10)
    axis.legend(fontsize=8.5, frameon=False, loc="lower left")
    return save(figure, "fig6_threshold.png")


def overlap_counts(truth):
    """Model 1's coverage gap and the two models' agreement, recounted here."""
    one = load_scores(SCORED["ames_xgb"][1])
    two = load_scores(SCORED["ames_desc"][1])
    call = {}
    for key, scores in (("one", one), ("two", two)):
        call[key] = {i: int(value >= 0.5) for i, value in scores.items()}
    shared = [i for i in truth if i in call["one"] and i in call["two"]]
    only_two = [i for i in truth if i not in call["one"] and i in call["two"]]
    agree = sum(call["one"][i] == call["two"][i] for i in shared)
    split = [i for i in shared if call["one"][i] != call["two"][i]]
    return {
        "both_answered": len(shared),
        "only_model_2": len(only_two),
        "agree": agree,
        "split_model_1_right": sum(call["one"][i] == truth[i] for i in split),
        "split_model_2_right": sum(call["two"][i] == truth[i] for i in split),
        "only_model_2_right": sum(call["two"][i] == truth[i] for i in only_two),
    }


def main():
    truth = load_truth()
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    overlap = overlap_counts(truth)

    _, shared_rows = figure_roc(truth)
    figure_balanced_interval(panel)
    figure_sensitivity_specificity(panel)
    figure_two_models(panel, overlap)
    figure_feature_choice()
    figure_threshold()

    summary = {"truth_rows": len(truth), "roc_shared_rows": shared_rows,
               **overlap}
    (OUT / "figure_counts.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
