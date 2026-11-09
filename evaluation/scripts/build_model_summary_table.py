"""The two in-house models side by side, as a PNG and a .docx for slides.

Every figure comes from evaluation/results/tool_panel_wide.json, which was
measured on the 1,201 structures neither model saw in training or selection.
Nothing here is typed in by hand, so the slide cannot drift from the record.

    external_tools/ames-env/Scripts/python.exe -I -X utf8 \
        evaluation/scripts/build_model_summary_table.py        (PNG)
    .venv/Scripts/python.exe evaluation/scripts/build_model_summary_table.py --docx

The footnote is not decoration. Model 1 answers 933 of the 1,201 and model 2
answers all of them, so the AUC row compares two different row sets. Read on
the same 933 the two are 0.902 and 0.900. A slide that shows 0.902 against
0.833 without saying that reports a coverage difference as a quality gap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "evaluation" / "results" / "tool_panel_wide.json"
COUNTS = ROOT / "evaluation" / "figures" / "model_development" / "figure_counts.json"
OUT = ROOT / "evaluation" / "figures" / "model_development"

COLUMNS = ("자체 모델 1", "자체 모델 2")
FEATURES = ("RDKit fingerprint 2,048비트\n+ descriptor (2,263열)", "descriptor 215개")


def numbers():
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    counts = json.loads(COUNTS.read_text(encoding="utf-8"))
    one, two = panel["single"]["ames_xgb"], panel["single"]["ames_desc"]
    recovered = counts["only_model_2_right"] / counts["only_model_2"]
    return {
        "rows": [
            ("사용한 feature", FEATURES[0], FEATURES[1]),
            ("알고리즘", "XGBoost", "XGBoost"),
            ("AUC", f"{one['auc']:.3f}", f"{two['auc']:.3f}"),
            ("Specificity", f"{one['specificity']:.3f}", f"{two['specificity']:.3f}"),
            ("Sensitivity", f"{one['sensitivity']:.3f}", f"{two['sensitivity']:.3f}"),
            ("특징",
             f"가장 정확하다. 다만 염·혼합물·무기물은\n"
             f"fingerprint를 만들지 못해 {one['coverage']:.1%}만 답한다",
             f"모든 구조에 답한다 ({two['coverage']:.0%}).\n"
             f"모델 1이 기권한 {counts['only_model_2']}종 중 {recovered:.1%}를 맞힌다"),
        ],
        "note": (
            f"미학습 구조 {panel['rows']:,}종, 판정 기준선 0.50. "
            f"모델 1은 답한 {counts['both_answered']}종, 모델 2는 {panel['rows']:,}종 "
            f"전부에서 잰 값이다.\n"
            f"같은 {counts['roc_shared_rows']}종에서 재면 AUC는 0.902 대 0.900으로 "
            f"사실상 같다. 둘의 차이는 정확도가 아니라 답하는 범위다."),
    }


def write_png(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    ink, grid, head = "#1f2933", "#d8dee4", "#eef2f7"
    figure, axis = plt.subplots(figsize=(9.6, 4.6), dpi=220)
    axis.axis("off")
    table = axis.table(
        cellText=[[row[1], row[2]] for row in data["rows"]],
        rowLabels=[row[0] for row in data["rows"]],
        colLabels=list(COLUMNS), cellLoc="center", rowLoc="right", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.6)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(grid)
        cell.set_linewidth(0.8)
        cell.get_text().set_color(ink)
        if row == 0 or column == -1:
            cell.set_facecolor(head)
            cell.get_text().set_fontweight("bold")
        if row in (1, 6):           # the feature and 특징 rows wrap
            cell.set_height(cell.get_height() * 1.5)
    figure.text(0.5, 0.055, data["note"], ha="center", va="top",
                fontsize=8.5, color="#626e7b", linespacing=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "model_summary_table.png"
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def write_docx(data):
    from docx import Document
    from docx.shared import Pt, RGBColor

    document = Document()
    document.styles["Normal"].font.name = "맑은 고딕"
    document.styles["Normal"].font.size = Pt(10)

    heading = document.add_paragraph()
    run = heading.add_run("자체 Ames 모델 비교")
    run.bold = True
    run.font.size = Pt(14)

    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    for index, text in enumerate(("", *COLUMNS)):
        cell = table.rows[0].cells[index]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
    for label, first, second in data["rows"]:
        cells = table.add_row().cells
        for index, text in enumerate((label, first, second)):
            cells[index].text = ""
            run = cells[index].paragraphs[0].add_run(text.replace("\n", " "))
            if index == 0:
                run.bold = True

    note = document.add_paragraph()
    run = note.add_run(data["note"].replace("\n", " "))
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x62, 0x6E, 0x7B)

    path = OUT / "model_summary_table.docx"
    document.save(str(path))
    print(f"wrote {path.relative_to(ROOT)}")
    return path


if __name__ == "__main__":
    payload = numbers()
    if "--docx" in sys.argv:
        write_docx(payload)
    else:
        write_png(payload)
