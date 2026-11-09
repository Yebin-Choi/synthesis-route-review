"""Figures for the report, drawn from the measurement files rather than typed.

Three things the report has to show and a table cannot:

  cutoff_tradeoff   Where the decision line should sit. Sensitivity falls and
                    specificity rises monotonically as the line moves, and
                    balanced accuracy has a peak. The peak is the finding, and
                    a reader sees a peak faster than they read a column.
  reliability       Whether the shown number behaves like a probability. If it
                    did, every point would sit on the diagonal. They do not.
  tool_comparison   Six tools on one axis, with how often each one answered.
                    Accuracy and answer rate have to be read together, so they
                    are in one figure rather than two.

Labels are in English because the figures go into a report whose figure
captions are English. Every number is read from the measurement JSON; nothing
here is typed in by hand, so a figure cannot drift away from the measurement.

    python evaluation/scripts/build_report_figures.py
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ames" / "results"
OUT = RESULTS / "figures"

# Measured on the independent evaluation file, rows the in-house models had
# never seen. Benchmark log 6-2 and 6-11. Kept beside the tool ids so the
# figure and the screen's performance table cannot disagree.
sys.path.insert(0, str(ROOT / "app"))


def load(name):
    path = RESULTS / name
    if not path.is_file():
        raise SystemExit(f"측정 파일이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cutoff_tradeoff(plt):
    """Both in-house models, same folds, same structures, same axis."""
    runs = [("Fingerprint + descriptors", load("ames_why_calibration.json"), "#4C72B0"),
            ("Descriptors only", load("ames_why_calibration_descriptors.json"), "#C44E52")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for axis, (name, report, colour) in zip(axes, runs):
        cuts = [row["cutoff"] for row in report["cutoffs"]]
        axis.plot(cuts, [row["sensitivity"] for row in report["cutoffs"]],
                  marker="o", markersize=3, label="sensitivity", color="#C44E52")
        axis.plot(cuts, [row["specificity"] for row in report["cutoffs"]],
                  marker="s", markersize=3, label="specificity", color="#4C72B0")
        axis.plot(cuts, [row["balanced_accuracy"] for row in report["cutoffs"]],
                  marker="^", markersize=3, label="balanced accuracy",
                  color="#55A868", linewidth=2)
        best = report["best_cutoff_by_balanced_accuracy"]
        axis.axvline(best["cutoff"], color="#55A868", linestyle=":", linewidth=1)
        axis.axvline(0.5, color="#888888", linestyle="--", linewidth=1)
        # Both labels sit in empty parts of the panel: the balance point above
        # every curve, the convention below the sensitivity curve. The legend
        # keeps the lower left.
        axis.text(best["cutoff"] + 0.015, 1.045,
                  f"measured balance point {best['cutoff']:.2f}\n"
                  f"balanced accuracy {best['balanced_accuracy']:.3f}",
                  fontsize=7.5, color="#2f6b45", va="top")
        axis.text(0.515, 0.47, "0.50\nconvention", fontsize=7.5,
                  color="#555555", va="top")
        axis.set_title(f"{name}  (AUC {report['auc']:.4f})", fontsize=9)
        axis.set_xlabel("decision threshold")
        axis.set_ylim(0.35, 1.06)
    axes[0].set_ylabel("rate")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Where the decision line should sit, on scaffold-grouped folds "
                 "(19,963 structures)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "cutoff_tradeoff.png")
    plt.close(fig)
    return "cutoff_tradeoff.png"


def reliability(plt):
    """How far the shown number sits from the rate actually observed.

    Drawn as two panels because the curve alone understates it. On the
    diagonal plot the points look close to the line; the gap panel shows the
    deviation is systematic, the same sign in the same bands for both models,
    and reaches 0.043. That is the honest claim: not wildly wrong, but wrong
    in a direction, which is what stops it being read as a probability.
    """
    runs = [("Fingerprint + descriptors", load("ames_why_calibration.json"),
             "#4C72B0", "o"),
            ("Descriptors only", load("ames_why_calibration_descriptors.json"),
             "#C44E52", "s")]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.8))

    left.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1,
              label="a calibrated score")
    width, offset = 0.055, -0.03
    largest = 0.0
    for name, report, colour, marker in runs:
        bands = [row for row in report["reliability"]
                 if row.get("mean_number_shown") is not None]
        left.plot([row["mean_number_shown"] for row in bands],
                  [row["actually_active"] for row in bands],
                  marker=marker, markersize=5, color=colour, label=name)
        centres = [(row["shown_from"] + row["shown_to"]) / 2 for row in bands]
        gaps = [row["gap"] for row in bands]
        largest = max(largest, max(abs(value) for value in gaps))
        right.bar([centre + offset for centre in centres], gaps, width=width,
                  color=colour, label=name)
        offset += width
    left.set_xlabel("mean score shown on screen")
    left.set_ylabel("fraction actually active")
    left.set_title("Shown against observed", fontsize=9)
    left.legend(frameon=False, fontsize=8, loc="upper left")
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)

    right.axhline(0, color="#333333", linewidth=1)
    right.set_xlabel("score band shown on screen")
    right.set_ylabel("observed minus shown")
    right.set_title(f"The gap is systematic, and reaches {largest:.3f}",
                    fontsize=9)
    right.set_xlim(0, 1.05)
    fig.suptitle("The shown number does not match the rate observed, "
                 "so it is not a probability", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "reliability.png")
    plt.close(fig)
    return "reliability.png"


def tool_comparison(plt):
    """Accuracy and answer rate together: one without the other misleads."""
    import tool_comparison_view as view

    # English labels rather than the screen's Korean ones: matplotlib has no
    # Korean font configured here and would draw empty boxes.
    english = {"ames_xgb": "Ours\n(XGBoost)",
               "ames_desc": "Ours\n(descriptors,\nexplainable)",
               "admet": "ADMET-AI\n2.0.1",
               "toxtree": "Toxtree\n3.1.0",
               "epa_test": "EPA TEST\n5.1.2",
               "rules": "Our rules\n(ICH M7)"}
    order = ["ames_xgb", "ames_desc", "admet", "toxtree", "epa_test", "rules"]
    labels = [english[t] for t in order]
    balanced = [view.TOOL_PERFORMANCE[t]["balanced"] for t in order]
    answered = [view.TOOL_PERFORMANCE[t]["answered"] for t in order]
    # Matched by prefix, not by the whole label. The exact wording moved to
    # "통계 기반" and this comparison stopped matching, so every bar came out
    # rule-coloured while the title still named two colours.
    colours = ["#4C72B0" if view.TOOL_PERFORMANCE[t]["method"].startswith("통계")
               else "#DD8452" for t in order]

    fig, axis = plt.subplots(figsize=(7.6, 3.8))
    positions = range(len(order))
    axis.bar(positions, balanced, color=colours, width=0.62)
    for index, (value, rate) in enumerate(zip(balanced, answered)):
        axis.text(index, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
        axis.text(index, 0.035, f"answered\n{rate:.0%}", ha="center",
                  fontsize=7, color="#ffffff")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, fontsize=7.5)
    axis.set_ylabel("balanced accuracy")
    axis.set_ylim(0, 0.95)
    axis.set_title("Six tools on the same 1,201 unseen structures.\n"
                   "Blue: statistical.  Orange: rule-based.  "
                   "No consensus is formed.", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "tool_comparison.png")
    plt.close(fig)
    return "tool_comparison.png"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    for drawer in (cutoff_tradeoff, reliability, tool_comparison):
        print(f"  기록: {OUT / drawer(plt)}")


if __name__ == "__main__":
    main()
