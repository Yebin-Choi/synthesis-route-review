"""Turn one of this project's Markdown reports into a .docx.

Covers only what these reports use: ATX headings, pipe tables, images,
ordered and unordered lists, horizontal rules, fenced code, and inline
`code` and **bold**. Anything else is written as plain text rather than
silently dropped.

    .venv/Scripts/python.exe evaluation/scripts/markdown_to_docx.py \
        <input.md> <output.docx>

Image paths resolve against the Markdown file's own folder.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

BODY_FONT = "맑은 고딕"
CODE_FONT = "Consolas"
INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x62, 0x6E, 0x7B)

IMAGE = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
ORDERED = re.compile(r"^(\d+)\.\s+(.*)$")
BULLET = re.compile(r"^[-*]\s+(.*)$")
# **bold** and `code`, kept as alternatives so one pass splits on either.
INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_runs(paragraph, text):
    """Write text into a paragraph, honouring **bold** and `code`."""
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            run = paragraph.add_run(piece[2:-2])
            run.bold = True
        elif piece.startswith("`") and piece.endswith("`"):
            run = paragraph.add_run(piece[1:-1])
            run.font.name = CODE_FONT
            run.font.size = Pt(9.5)
        else:
            run = paragraph.add_run(piece)
        run.font.color.rgb = INK


def split_row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_divider(line):
    cells = split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells)


def add_table(document, header, body):
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    for index, text in enumerate(header):
        cell = table.rows[0].cells[index]
        cell.text = ""
        paragraph = cell.paragraphs[0]
        add_runs(paragraph, text)
        for run in paragraph.runs:
            run.bold = True
    for row in body:
        cells = table.add_row().cells
        for index in range(len(header)):
            cells[index].text = ""
            add_runs(cells[index].paragraphs[0],
                     row[index] if index < len(row) else "")
    document.add_paragraph()
    return table


def convert(source: Path, target: Path):
    lines = source.read_text(encoding="utf-8").splitlines()
    document = Document()

    normal = document.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("---") and set(stripped) == {"-"}:
            document.add_paragraph()
            index += 1
            continue

        match = IMAGE.match(stripped)
        if match:
            path = (source.parent / match.group(1)).resolve()
            if path.is_file():
                document.add_picture(str(path), width=Inches(6.2))
                document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                document.add_paragraph(f"[그림 없음: {match.group(1)}]")
            index += 1
            continue

        match = HEADING.match(stripped)
        if match:
            level = len(match.group(1))
            paragraph = document.add_heading("", level=min(level, 4))
            add_runs(paragraph, match.group(2))
            for run in paragraph.runs:
                run.font.name = BODY_FONT
                run.font.color.rgb = INK
            index += 1
            continue

        if stripped.startswith("```"):
            index += 1
            block = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            paragraph = document.add_paragraph()
            run = paragraph.add_run("\n".join(block))
            run.font.name = CODE_FONT
            run.font.size = Pt(9)
            run.font.color.rgb = MUTED
            continue

        if stripped.startswith("|") and index + 1 < len(lines) \
                and is_divider(lines[index + 1]):
            header = split_row(stripped)
            index += 2
            body = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                body.append(split_row(lines[index].strip()))
                index += 1
            add_table(document, header, body)
            continue

        match = ORDERED.match(stripped)
        if match:
            paragraph = document.add_paragraph(style="List Number")
            add_runs(paragraph, match.group(2))
            index += 1
            continue

        match = BULLET.match(stripped)
        if match:
            paragraph = document.add_paragraph(style="List Bullet")
            add_runs(paragraph, match.group(1))
            index += 1
            continue

        if stripped.startswith(">"):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.3)
            add_runs(paragraph, stripped.lstrip("> ").strip())
            for run in paragraph.runs:
                run.font.color.rgb = MUTED
            index += 1
            continue

        # A paragraph runs until a blank line; Markdown's soft wraps are joined.
        block = [stripped]
        index += 1
        while index < len(lines):
            nxt = lines[index].strip()
            if not nxt or nxt.startswith(("#", "|", "-", "*", ">", "!", "```")) \
                    or ORDERED.match(nxt):
                break
            block.append(nxt)
            index += 1
        add_runs(document.add_paragraph(), " ".join(block))

    target.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(target))
    print(f"wrote {target}")


if __name__ == "__main__":
    convert(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
