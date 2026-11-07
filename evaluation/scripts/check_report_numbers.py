"""Do the numbers in the explanation document still match what was measured?

A document that drifts away from the measurement is worse than no document,
because it is read as if it were current. Every figure in it either comes
from a measurement file in `ames/results/` or from the performance table the
screen itself uses, so both can be compared directly rather than by eye.

    python evaluation/scripts/check_report_numbers.py

It prints each mismatch and exits non-zero if there is one. It is a check,
not a fixer: a mismatch means either the document or the measurement moved,
and which one is wrong is a judgement, not something to patch automatically.
"""

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))

import engine  # noqa: E402
import tool_comparison_view as view  # noqa: E402

DOCUMENT = ROOT / "evaluation" / "notes" / "소프트웨어_설명_2026-11-06.md"
CALIBRATIONS = (("지문+기술자 모델", "ames_why_calibration.json"),
                ("기술자 모델", "ames_why_calibration_descriptors.json"))


def tool_rows(document, lines):
    """Each tool's measured figures, against the row that reports them."""
    problems = []
    for info in view.TOOL_PERFORMANCE.values():
        # The document writes a label in its own words, so the row is found
        # by the name stem plus the figure that identifies it.
        stem = re.split(r"[ (]", info["label"])[0]
        row = next((line for line in lines
                    if line.startswith("|") and stem in line
                    and f"{info['balanced']:.3f}" in line), None)
        if row is None:
            problems.append(f"{info['label']}: 성능표 행을 찾지 못했습니다")
            continue
        for field in ("sensitivity", "specificity", "balanced", "answered"):
            if f"{info[field]:.3f}" not in row:
                problems.append(
                    f"{info['label']} {field}={info[field]:.3f} 가 문서와 다릅니다")
    return problems


def measured_figures(document):
    """The cross-validation figures the document quotes."""
    problems = []
    for name, filename in CALIBRATIONS:
        path = ROOT / "ames" / "results" / filename
        if not path.is_file():
            problems.append(f"{name}: 측정 파일이 없습니다 ({filename})")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        best = report["best_cutoff_by_balanced_accuracy"]
        for label, value in (("AUC", f"{report['auc']:.4f}"),
                             ("균형 정확도", f"{best['balanced_accuracy']:.3f}"),
                             ("기준선", f"{best['cutoff']:.2f}")):
            if value not in document:
                problems.append(f"{name} {label} {value} 가 문서에 없습니다")
    if str(engine.MAX_JSON_DEPTH) not in document:
        problems.append(f"중첩 한도 {engine.MAX_JSON_DEPTH} 가 문서에 없습니다")
    return problems


def main():
    if not DOCUMENT.is_file():
        raise SystemExit(f"문서가 없습니다: {DOCUMENT}")
    document = DOCUMENT.read_text(encoding="utf-8")
    lines = document.splitlines()
    problems = tool_rows(document, lines) + measured_figures(document)
    if problems:
        print(f"불일치 {len(problems)}건")
        for item in problems:
            print("  -", item)
        raise SystemExit(1)
    print(f"문서의 수치가 측정과 모두 일치합니다: {DOCUMENT.name}")


if __name__ == "__main__":
    main()
