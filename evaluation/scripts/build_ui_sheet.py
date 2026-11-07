"""Render every verdict chip and both table shapes on one page, in both themes.

The route screens cannot be opened without a finished route search, so this
is how the shared presentation vocabulary gets looked at: the same CSS the
app ships, with one example of every state it can show.

It is also the reference to put in the report when explaining why a verdict
looks the way it does. Regenerate it after changing `app/ui_kit.py` or the
theme, otherwise the sheet stops describing the software:

    python evaluation/scripts/build_ui_sheet.py

Every structure and score below is sample material for the drawing, never an
AMES measurement and never a toxicity determination.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))

import presentation  # noqa: E402
import ui_kit  # noqa: E402

OUTPUT = ROOT / "evaluation" / "notes" / "화면_표현_규칙.html"


def board():
    chips = "  ".join(
        ui_kit.verdict_chip(name) for name in
        ("positive", "negative", "no_alert", "indeterminate",
         "out_of_domain", "not_run"))
    detailed = (
        ui_kit.verdict_chip("positive", aside="0.81") + "  "
        + ui_kit.verdict_chip("indeterminate", note="적용 범위 밖입니다") + "  "
        + ui_kit.chip("전문가 검토 필요", ui_kit.REVIEW, hint="설명이 여기 붙습니다"))
    facts = ui_kit.facts([
        ("판정 대상", "Ames 변이원성"),
        ("도구", "ADMET-AI 2.0.1 (이 컴퓨터에서 실행)"),
        ("판정 기준선", "0.50 · 이 도구의 측정된 균형점"),
        ("적용 범위 평가", "이 검사에서는 하지 않음"),
    ])
    table = ui_kit.table(
        ["구조", "우리 모델", "ADMET-AI", "자체 규칙", "모델 적용 범위"],
        [{"cells": ["C1CO1",
                    ui_kit.verdict_chip("positive", aside="0.93"),
                    ui_kit.verdict_chip("positive", aside="0.71"),
                    ui_kit.chip("위험 구조", ui_kit.CONCERN, note="에폭사이드 고리"),
                    ui_kit.chip("매우 높음", ui_kit.CLEAR, aside="0.84",
                                note="이 구간 측정 AUC 0.965")]},
         {"cells": ["CCO",
                    ui_kit.verdict_chip("negative", aside="0.04"),
                    ui_kit.verdict_chip("negative", aside="0.11"),
                    ui_kit.verdict_chip("no_alert"),
                    ui_kit.chip("높음", ui_kit.CLEAR, aside="0.62",
                                note="이 구간 측정 AUC 0.910")]},
         {"cells": ["N=Nc1ccccc1",
                    ui_kit.verdict_chip("positive", aside="0.58"),
                    ui_kit.verdict_chip("negative", aside="0.22"),
                    ui_kit.chip("위험 구조", ui_kit.CONCERN, note="아조기"),
                    ui_kit.chip("낮음", ui_kit.CONCERN, aside="0.34",
                                note="이 구간 측정 AUC 0.859")],
          "mark": True},
         {"cells": ["not-a-structure",
                    ui_kit.verdict_chip("indeterminate", note="구조를 읽지 못했습니다"),
                    ui_kit.verdict_chip("indeterminate", note="구조를 읽지 못했습니다"),
                    ui_kit.verdict_chip("indeterminate", note="구조를 읽지 못했습니다"),
                    ui_kit.chip("확인 불가", ui_kit.ABSENT)]}])
    return f"""
    <h2>판정 어휘</h2>
    <p>{chips}</p>
    <p class="why">색만으로 뜻을 전하지 않습니다. 칩마다 낱말이 들어 있어
    흑백으로 인쇄해도, 색을 구분하지 못하는 사람이 읽어도 뜻이 남습니다
    (WCAG 2.1 성공기준 1.4.1).</p>
    <p class="why">"경보 없음"은 음성이 아닙니다. 구조 경보가 걸리지 않았다는
    뜻일 뿐입니다. "판정 보류"와 "적용 범위 밖"도 음성이 아닙니다.</p>
    <h2>점수·이유·설명이 붙은 칩</h2>
    <p>{detailed}</p>
    <h2>출처 기록</h2>
    {facts}
    <h2>도구 비교표 (갈린 행은 바탕색으로 표시)</h2>
    {table}
    """


def main():
    sections = []
    for name, light in (("어두운 화면", False), ("밝은 화면", True)):
        sections.append(
            f'<section class="theme">{presentation.theme_css(light)}'
            f'{ui_kit.KIT_CSS}<h1>{name}</h1>{board()}</section>')
    OUTPUT.write_text(f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>화면 표현 규칙</title>
<style>
 body {{ margin:0; font-family:"Malgun Gothic",-apple-system,sans-serif; }}
 section.theme {{ padding:24px 32px; }}
 section.theme:first-of-type {{ background:#0b1422; color:#f1f5f9;
   --review-panel:#162337; --review-muted:#cbd5e1; --review-accent:#7dd3fc; }}
 section.theme:last-of-type {{ background:#ffffff; color:#172033;
   --review-panel:#f1f5f9; --review-muted:#475569; --review-accent:#175b91; }}
 h1 {{ font-size:20px; }}
 h2 {{ font-size:15px; margin:22px 0 6px; font-weight:600; opacity:0.75; }}
 p.why {{ font-size:13px; opacity:0.72; max-width:48rem; margin:4px 0; }}
</style></head><body>{''.join(sections)}</body></html>
""", encoding="utf-8")
    print(f"기록: {OUTPUT}")


if __name__ == "__main__":
    main()
