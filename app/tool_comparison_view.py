"""Show what every genotoxicity tool said about the same materials.

The software does not decide which tool is right. It runs the ones the user
picked over the materials of the route under review and lays the answers side
by side, with each tool's measured performance beside it so the reader has
something to judge with. Where a tool did not answer, the reason it gave is
shown instead of a blank, because "no alert" and "could not tell" are
different statements and only one of them is reassuring.

One column is not a tool: the consensus is the mean of the three statistical
models' scores. It is here because it was measured to be better than any one
of them (balanced accuracy 0.864 against 0.847, ROC-AUC 0.916 against 0.902,
benchmark log 6-56), and because averaging is not a judgement about which
tool is right. The three it averages stay on screen beside it.

The decision of which routes stay selectable is made elsewhere, by the single
tool named in the screening policy. This screen is for comparison.

Layout follows the convention used by published prediction servers: a metric
strip, then a verdict matrix where the call is carried by a coloured chip
rather than by a sentence, then the evidence behind the tools. Explanations
live in a collapsed section so the screen is read rather than waded through.
"""

import csv
import io

import altair as alt
import pandas as pd
import streamlit as st

if __package__:
    from . import applicability, screening_judges, ui_kit
    from .presentation import GLOSSARY, reason_text, show_structure
else:
    import applicability
    import screening_judges
    import ui_kit
    from presentation import GLOSSARY, reason_text, show_structure


# Measured, not estimated, and all five on ONE set: the 1,201 structures of
# the independent evaluation file that the in-house model had never seen.
#
# The full file holds 3,091 structures, but 1,890 of them are in the shared
# collection the in-house model was fitted, selected and reported on. Scoring
# it there would be scoring it partly on its own training data, and the
# difference is not small: it reads 0.863 on the full file and 0.824 here.
# Recorded in evaluation/tool_benchmark_log.md section 6-2.
# "auc" is ROC-AUC on the same 1,201 unseen structures, over the rows each
# tool answered, computed from the stored scores in evaluation/results (log
# 6-49). A tool that answers with an alert rather than a score has none, so
# it is None for the rule screens, exactly like "line".
MEASURED_ON = "외부 검증셋 1,201종"

# Every score-returning tool now runs at 0.50, the line published work uses.
#
# It used to be otherwise. The collection the in-house models are fitted on is
# 30.2% active, and leaving that imbalance to the decision line pushed their
# balance point down to 0.25, so the screen ran them there and had to explain
# why. The models are now fitted with the imbalance carried in the training
# weights instead (benchmark log 6-20), which moves the balance point to 0.43
# and flattens the top of the curve: 0.40 and 0.50 both measure 0.847, and the
# balance point itself only reaches 0.857. There is almost nothing left to win
# by moving the line, so it stays where everyone else's is.
#
# A rule tool returns a verdict, so a line does not apply, and EPA TEST
# publishes its own call.
TOOL_PERFORMANCE = {
    "rules": {"label": "전문가 규칙 (ICH M7)", "method": "전문가 규칙 기반", "line": None,
              "sensitivity": 0.552, "specificity": 0.769, "balanced": 0.661,
              "balanced_interval": (0.633, 0.689),
              "answered": 1.000, "measured_on": MEASURED_ON},
    "toxtree": {"label": "Toxtree 3.1.0", "method": "전문가 규칙 기반", "line": None,
                "sensitivity": 0.691, "specificity": 0.790, "balanced": 0.741,
                "balanced_interval": (0.711, 0.773),
                "answered": 0.777, "measured_on": MEASURED_ON},
    "admet": {"label": "ADMET-AI 2.0.1", "method": "통계 기반", "line": 0.50,
              "sensitivity": 0.678, "specificity": 0.820, "balanced": 0.749,
              "balanced_interval": (0.723, 0.776),
              "auc": 0.818, "auc_interval": (0.792, 0.844),
              "answered": 1.000, "measured_on": MEASURED_ON},
    # EPA TEST publishes its own call beside its consensus value, and that
    # call is what was measured. We draw no line through its number, so
    # `line` is None and the table says whose line it is instead.
    "epa_test": {"label": "EPA TEST 5.1.2", "method": "통계 기반", "line": None,
                 "own_line": True,
                 "sensitivity": 0.629, "specificity": 0.798, "balanced": 0.714,
                 "balanced_interval": (0.680, 0.746),
                 "auc": 0.770, "auc_interval": (0.735, 0.805),
                 "answered": 0.611, "measured_on": MEASURED_ON},
    "ames_xgb": {"label": "자체 모델 1: fingerprint", "method": "통계 기반", "line": 0.50,
                 "sensitivity": 0.774, "specificity": 0.921, "balanced": 0.847,
                 "balanced_interval": (0.821, 0.871),
                 "auc": 0.902, "auc_interval": (0.879, 0.924),
                 "answered": 0.777, "measured_on": MEASURED_ON},
    "ames_desc": {"label": "자체 모델 2: descriptor", "method": "통계 기반",
                  "line": 0.50,
                  "sensitivity": 0.648, "specificity": 0.930, "balanced": 0.789,
                  "balanced_interval": (0.766, 0.812),
                  "auc": 0.833, "auc_interval": (0.808, 0.859),
                  "answered": 1.000, "measured_on": MEASURED_ON},
    # Not a tool of its own: the average of the three scores above, judged at
    # the same line they are. Every figure here was measured on the same
    # 1,201 structures as the rest of the table (benchmark log 6-56). The
    # averaging is the whole method, so there is no fitted weight that could
    # have been tuned on the structures these figures come from.
    "consensus": {"label": "합의 점수: 세 모델 평균", "short_label": "합의 점수",
                  "method": "통계 기반", "line": 0.50,
                  "sensitivity": 0.791, "specificity": 0.936,
                  "balanced": 0.864, "balanced_interval": (0.841, 0.887),
                  "auc": 0.916, "auc_interval": (0.894, 0.936),
                  "answered": 0.777, "measured_on": MEASURED_ON,
                  "derived_from": ("ames_xgb", "ames_desc", "admet")},
    # An alert set, so there is no line to draw and `line` is None. The
    # OASIS alert set is the one registered: the Toolbox's other Ames
    # profiler (ISS) agreed with Toxtree on 99.2% of the structures both
    # answered, so it would have added a second copy of a tool already here
    # (benchmark log 6-34).
    "qsar_toolbox": {"label": "QSAR Toolbox 4.9", "method": "전문가 규칙 기반",
                     "line": None,
                     "sensitivity": 0.423, "specificity": 0.979, "balanced": 0.701,
                     "balanced_interval": (0.676, 0.725),
                     "answered": 0.940, "measured_on": MEASURED_ON},
}


def independent_tool_ids():
    """The judges that are tools in their own right, not derived from others.

    The consensus column is the mean of three of them, so counting it as an
    eighth tool would overstate how many pieces of software are behind the
    screen.
    """
    return tuple(tool_id for tool_id in screening_judges.implemented_judge_ids()
                 if not TOOL_PERFORMANCE.get(tool_id, {}).get("derived_from"))


def measured_line(tool_id):
    """The measured decision line for a tool, or None if a line does not apply."""
    return TOOL_PERFORMANCE.get(tool_id, {}).get("line")
# The two that shell out to a separate program. Every other tool runs in
# this process or in a worker that is already warm.
EXTERNAL_TOOLS = ("toxtree", "epa_test", "qsar_toolbox")
# Both in-house models open by default: they disagree about one structure
# in seven (benchmark log 6-11), and showing that is what the screen is for.
#
# The consensus opens with them and costs nothing extra: it is the average
# of exactly these three scores, so selecting it runs nothing the screen was
# not already running.
DEFAULT_TOOLS = ("rules", "consensus", "ames_xgb", "ames_desc", "admet")

# Moving the line still trades sensitivity for specificity, and these are the
# measured rows so a reader can see the size of the trade rather than take it
# on trust. Measured on the 933 independent structures the model had not seen
# (benchmark log 6-20), not on cross-validation folds.
MEASURED_CUTOFF = 0.50
CUTOFF_TRADE_OFF = (
    (0.20, 0.881, 0.720, 0.801),
    (0.25, 0.867, 0.770, 0.819),
    (0.30, 0.853, 0.819, 0.836),
    (0.40, 0.819, 0.876, 0.847),
    (0.50, 0.774, 0.921, 0.847),
    (0.60, 0.734, 0.950, 0.842),
    (0.70, 0.686, 0.978, 0.832),
)


CALL_LABEL = {"1": "양성", "0": "음성", "abstain": "예측 결과 없음"}
CALL_CLASS = {"1": ui_kit.CONCERN, "0": ui_kit.CLEAR, "abstain": ui_kit.UNKNOWN}

# A rule tool that raises nothing has not predicted a negative, so it does not
# get the word. The ISS Ames profiler derived from Toxtree's mutagenicity
# module states this directly, and our own measurement agrees: the rule
# screen's sensitivity is 0.552, so nearly half the real positives raise no
# alert. https://qsardb.food.dtu.dk/download/profilers/Ames_ISS.pdf
#
# Only the word changes. The call underneath stays a negative, because the
# sensitivity and specificity in the performance table were counted that way,
# and relabelling the call would make those figures describe something else.
RULE_NEGATIVE_LABEL = "경고구조 없음"


def call_label(call, tool_id=None):
    if (call == "0" and tool_id is not None
            and TOOL_PERFORMANCE.get(tool_id, {}).get("method") == "전문가 규칙 기반"):
        return RULE_NEGATIVE_LABEL
    return CALL_LABEL.get(call, call)


def route_materials(report):
    """Distinct materials of the report, in the order they first appear."""
    seen, materials = set(), []
    for route in report.get("routes", []):
        for step in route.get("steps", []):
            for material in step.get("materials", []):
                key = material.get("canonical_smiles") or material.get("input_smiles")
                if not key or key in seen:
                    continue
                seen.add(key)
                materials.append(key)
    return materials


def materials_of_route(report, route_id):
    """Distinct materials of one route, in the order they first appear."""
    seen, materials = set(), []
    for route in report.get("routes", []):
        if route.get("id") != route_id:
            continue
        for step in route.get("steps", []):
            for material in step.get("materials", []):
                key = material.get("canonical_smiles") or material.get("input_smiles")
                if not key or key in seen:
                    continue
                seen.add(key)
                materials.append(key)
    return materials


def run_tools(materials, tool_ids, *, threshold=None):
    """One call per tool, failures kept as a tool-level error rather than raised.

    A tool that cannot run must not stop the others from being shown: the
    whole point of the screen is the comparison that survives.

    With `threshold` left out, every tool uses its own measured line. Passing
    one applies that single line to all of them, which is what the override on
    screen does; it is not the default because the measured lines differ.
    """
    # The consensus is an average of three of the other judges, so its
    # members are run whether or not the reader asked to see them, and their
    # columns are dropped again at the end if they were not asked for.
    requested = tuple(tool_ids)
    derived = tuple(tool_id for tool_id in requested
                    if screening_judges.get_judge(tool_id).get("derived_from"))
    needed = list(requested)
    for tool_id in derived:
        for member in screening_judges.get_judge(tool_id)["derived_from"]:
            if member not in needed:
                needed.append(member)

    results = {}
    for tool_id in needed:
        if tool_id in derived:
            continue
        entry = screening_judges.get_judge(tool_id)
        try:
            line = measured_line(tool_id) if threshold is None else threshold
            if entry["uses_threshold"] and line is not None:
                results[tool_id] = screening_judges.run_judge(
                    tool_id, materials, threshold=line)
            else:
                # Either the tool takes no line at all, or we have no measured
                # line of our own to apply and it keeps its own. Falling back
                # to our model's 0.25 here is how EPA TEST briefly ran at 0.25
                # while its performance row came from its own verdicts, which
                # reversed the tool on about half the structures.
                results[tool_id] = screening_judges.run_judge(tool_id, materials)
        except Exception as exc:  # noqa: BLE001 - surfaced per tool, never hidden
            results[tool_id] = {"judge": tool_id, "calls": [], "evidence": {},
                                "tool_error": f"{type(exc).__name__}: {exc}"}

    for tool_id in derived:
        try:
            line = measured_line(tool_id) if threshold is None else threshold
            results[tool_id] = screening_judges.consensus_from_results(
                materials, results, threshold=line)
        except Exception as exc:  # noqa: BLE001 - surfaced per tool, never hidden
            results[tool_id] = {"judge": tool_id, "calls": [], "evidence": {},
                                "tool_error": f"{type(exc).__name__}: {exc}"}

    # Back in the order asked for, so the columns do not reshuffle because
    # one of them had to wait for its members.
    return {tool_id: results[tool_id] for tool_id in requested}


def build_comparison(materials, results):
    """One row per material, one column per tool, plus why a tool abstained."""
    indexed = {}
    for tool_id, result in results.items():
        indexed[tool_id] = {call["input_smiles"]: call
                            for call in result.get("calls", [])}
    rows = []
    for material in materials:
        row = {"물질": material}
        calls = {}
        for tool_id in results:
            call = indexed[tool_id].get(material)
            if call is None:
                row[TOOL_PERFORMANCE[tool_id]["label"]] = "실행 안 됨"
                continue
            calls[tool_id] = call
            text = call_label(call["call"], tool_id)
            if call["call"] == "abstain" and call.get("error"):
                text = f"예측 결과 없음: {reason_text(call['error'])}"
            elif call.get("score") is not None:
                text = f"{text} ({call['score']:.2f})"
            row[TOOL_PERFORMANCE[tool_id]["label"]] = text
        # A derived column is not a second opinion. The consensus is the mean
        # of three columns already here, so counting it as a tool that
        # disagreed would report one disagreement as two.
        decided = {call["call"] for tool_id, call in calls.items()
                   if call["call"] != "abstain"
                   and not TOOL_PERFORMANCE.get(tool_id, {}).get("derived_from")}
        row["_calls"] = calls
        row["_disagree"] = len(decided) > 1
        row["_answered"] = len(decided)
        rows.append(row)
    return rows


def interval_text(bounds):
    """A measured 95% range, or a dash when none was measured.

    Every figure in the table came off 1,201 structures, so each one could
    have fallen somewhere else had the 1,201 been different structures. The
    range says how far. Two tools whose ranges overlap are not ranked by the
    table, whatever order the rows happen to be in.
    """
    if not bounds:
        return "측정 안 함"
    low, high = bounds
    return f"{low:.3f} - {high:.3f}"


def performance_rows(tool_ids):
    rows = []
    for tool_id in tool_ids:
        info = TOOL_PERFORMANCE[tool_id]
        rows.append({
            "도구": info["label"], "방법론": info["method"],
            "판정 기준선": (f"{info['line']:.2f}" if info["line"] is not None
                        else ("도구 자체 판정" if info.get("own_line")
                              else "없음")),
            "민감도": info["sensitivity"], "특이도": info["specificity"],
            "균형 정확도": info["balanced"],
            "균형 정확도 95% 구간": interval_text(info.get("balanced_interval")),
            # A rule screen answers with an alert, not a score, so there is no
            # curve to take an area under. Saying so beats a blank cell.
            "AUC": (f"{info['auc']:.3f}" if info.get("auc") is not None
                    else "점수를 내지 않음"),
            "AUC 95% 구간": (interval_text(info.get("auc_interval"))
                          if info.get("auc") is not None
                          else "점수를 내지 않음"),
            "판정 비율": info["answered"],
            "측정 자료": info["measured_on"],
        })
    return rows


# Neither red nor green: those two carry the verdict in every chip on this
# screen, and a tool wearing one of them would be read as a call. Mid-tone
# hues, because the screen has a light mode and a dark one and the same dot
# has to be visible on both.
TOOL_COLOURS = ("#3b82f6", "#8b5cf6", "#d97706", "#64748b")


def short_label(smiles, *, width=16):
    """A y-axis label. A full SMILES pushes the plot area to nothing."""
    return smiles if len(smiles) <= width else smiles[:width - 1] + "…"


def score_points(rows, tool_ids):
    """One point per score, for the tools judged at our own line.

    EPA TEST and the rule tools are left out. A rule tool returns no number,
    and EPA TEST publishes its own call beside its value, so drawing its
    number against our line would invite a comparison nobody measured.
    """
    scored = [tool_id for tool_id in tool_ids
              if TOOL_PERFORMANCE.get(tool_id, {}).get("line") is not None]
    points = []
    for row in rows:
        for tool_id in scored:
            call = row["_calls"].get(tool_id)
            if call is None or call.get("score") is None:
                continue
            points.append({"물질": short_label(row["물질"]),
                           "도구": TOOL_PERFORMANCE[tool_id]["label"],
                           "점수": round(float(call["score"]), 3),
                           "판정": CALL_LABEL.get(call["call"], "예측 결과 없음"),
                           "전체 구조": row["물질"]})
    return points


def comparison_csv(rows, tool_ids):
    """The matrix as a file, so a reader can take it into a report."""
    columns = ["물질"] + [TOOL_PERFORMANCE[tool_id]["label"] for tool_id in tool_ids]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns + ["엇갈림"])
    writer.writeheader()
    for row in rows:
        record = {column: row.get(column, "") for column in columns}
        record["엇갈림"] = "예" if row["_disagree"] else "아니오"
        writer.writerow(record)
    return buffer.getvalue()


def _chip(call, tool_id=None):
    """One verdict as a coloured chip, with its score or its reason beneath."""
    if call is None:
        return ui_kit.chip("실행 안 됨", ui_kit.ABSENT)
    label = call_label(call["call"], tool_id)
    return ui_kit.chip(
        label, CALL_CLASS.get(call["call"], ui_kit.UNKNOWN),
        # The glossary line for a word a reader cannot guess, on hover rather
        # than as a sentence under every cell of the column.
        hint=GLOSSARY.get(label),
        aside=None if call.get("score") is None else f"{call['score']:.2f}",
        note=reason_text(call.get("error")) if call["call"] == "abstain"
        else None)


def _band_cell(placed):
    """How much standing the model has to judge this material.

    This is a statement about the model, not about the material: a structure
    in a weak band is not more dangerous, it is one the model has seen less
    like before.
    """
    if placed is None or placed.get("measured_auc") is None:
        return ui_kit.chip("확인 불가", ui_kit.ABSENT)
    strength = ui_kit.CLEAR if placed["measured_auc"] >= 0.90 else (
        ui_kit.UNKNOWN if placed["measured_auc"] >= 0.86 else ui_kit.CONCERN)
    return ui_kit.chip(placed["label"], strength,
                       hint=GLOSSARY.get("모델 적용 범위"),
                       aside=f"{placed['similarity']:.2f}",
                       note=f"이 구간 측정 AUC {placed['measured_auc']:.3f}")


# One line per tool, shown on hover over its column so the header can stay
# short. A reader should not have to know what "전문가 규칙" is to read the row.
TOOL_HINTS = {
    "rules": "이 소프트웨어가 직접 넣은 규칙입니다. ICH M7(R2)이 이름을 올린 "
             "물질 목록과, 변이원에서 자주 보이는 부분 구조를 대조합니다.",
    "toxtree": "독성 경고구조를 판정하는 공개 프로그램입니다. 따로 설치해서 "
               "이 컴퓨터에서 실행합니다.",
    "epa_test": "미국 환경보호청이 배포하는 독성 예측 프로그램입니다. 구조가 "
                "자기 적용 범위를 벗어나면 스스로 판정을 내지 않습니다.",
    "admet": "공개된 약물성·독성 예측 모델입니다. 이 컴퓨터에서 실행하며 "
             "구조가 밖으로 나가지 않습니다.",
    "ames_xgb": "직접 학습시킨 모델입니다. 지문 비트와 기술자를 함께 쓰며, "
                "왜 그렇게 판정했는지는 설명하지 못합니다. 염·혼합물·무기물은 "
                "지문을 만들지 못해 답하지 않습니다.",
    "ames_desc": "직접 학습시킨 두 번째 모델입니다. 이름 있는 기술자만 써서 "
                 "어떤 성질이 판정을 밀었는지 말할 수 있습니다. 모델 1이 읽지 "
                 "못하는 구조도 읽습니다.",
    "consensus": "도구가 아니라 위 세 모델의 점수를 평균한 값입니다. 셋이 "
                 "모두 답한 물질에서만 나오고, 셋 중 하나라도 답하지 않으면 "
                 "평균을 내지 않습니다.",
    "qsar_toolbox": "OECD가 배포하는 규제용 프로그램입니다. 이 컴퓨터에서 "
                    "실행하며 구조가 밖으로 나가지 않습니다. 고분자처럼 "
                    "경고구조를 적용할 수 없는 물질은 스스로 판정을 내지 않습니다.",
}


def _matrix_html(rows, tool_ids, bands=None):
    columns = ["물질"] + [TOOL_PERFORMANCE[tool_id]["label"] for tool_id in tool_ids]
    hints = {TOOL_PERFORMANCE[tool_id]["label"]: TOOL_HINTS.get(tool_id)
             for tool_id in tool_ids}
    hints["모델 적용 범위"] = ("자체 모델이 이 물질과 비슷한 구조를 얼마나 "
                         "봤는지입니다. 물질이 위험한지에 대한 말이 아닙니다.")
    if bands:
        columns.append("모델 적용 범위")
    built = []
    for row in rows:
        cells = [ui_kit.text(row["물질"])]
        cells += [_chip(row["_calls"].get(tool_id), tool_id) for tool_id in tool_ids]
        if bands:
            cells.append(_band_cell(bands.get(row["물질"])))
        built.append({"cells": cells, "mark": row["_disagree"]})
    return ui_kit.table(columns, built, hints=hints)


def explanation_lines(entry):
    """The explanation as text, with the block split stated first.

    The split matters more than the list. On this model the fingerprint bits
    carry almost the whole decision, so a reader shown only a descriptor list
    would take a contribution of 0.006 for the reason behind the call.
    """
    if entry is None or entry.get("status") != "success":
        return ["설명을 계산하지 못했습니다."]
    fingerprint = entry["fingerprint_block_contribution"]
    named = entry["named_feature_contribution"]
    share = abs(fingerprint) / max(abs(fingerprint) + abs(named), 1e-9)
    lines = [f"구조 지문 전체가 {fingerprint:+.2f}, 이름 있는 기술자 전체가 "
             f"{named:+.2f}만큼 밀었습니다 (log-odds).",
             f"이 판정의 {share:.0%}는 지문에서 나왔습니다."]
    if share > 0.8:
        lines.append("지문이 거의 전부를 결정했으므로 아래 기술자 목록은 "
                     "판정 이유라기보다 곁가지입니다.")
    for direction, key in (("양성 쪽", "pushed_positive"),
                           ("음성 쪽", "pushed_negative")):
        items = entry.get(key) or []
        if items:
            lines.append(f"{direction}으로 민 기술자: " + ", ".join(
                f"{row['feature']} {row['contribution']:+.3f}" for row in items[:5]))
    return lines


def _tool_picker(available):
    """The tool multiselect and the run button, side by side."""
    controls, action = st.columns([4, 1])
    with controls:
        chosen = st.multiselect(
            "도구", available,
            default=[tool for tool in DEFAULT_TOOLS if tool in available],
            format_func=lambda tool_id: TOOL_PERFORMANCE[tool_id]["label"],
            key="comparison_tools", label_visibility="collapsed",
            placeholder="비교할 도구를 고르세요")
    with action:
        run = st.button("실행", key="comparison_run", type="primary",
                        width="stretch")
    return chosen, run


def _summary_metrics(rows, disagreed, silent, bands):
    """The four-number strip over the matrix.

    Short labels with the explanation on hover. A label long enough to say
    what it means wraps in a narrow column and pushes its number onto a
    second line, and then the strip stops lining up.
    """
    weak = sum(1 for placed in bands.values()
               if placed.get("measured_auc") is not None
               and placed["measured_auc"] < 0.86)
    first, second, third, fourth = st.columns(4)
    first.metric("물질", len(rows), help="이 경로에서 검토한 비목표 물질 수입니다.")
    second.metric("엇갈림", len(disagreed),
                  help="도구마다 답이 다른 물질 수입니다. 어느 쪽이 맞는지는 "
                       "소프트웨어가 정하지 않습니다.")
    third.metric("무응답", len(silent),
                 help="어느 도구도 답하지 않은 물질 수입니다. 음성이 아니라 "
                      "판정이 없는 것입니다.")
    fourth.metric("근거 부족", weak if bands else "없음",
                  help="자체 모델이 비슷한 구조를 거의 보지 못한 물질 수입니다. "
                       "위험하다는 뜻이 아니라 모델이 판단할 근거가 적다는 뜻입니다.")


def _model_explanations(disagreed, chosen):
    """Descriptor contributions for the shown structures, if asked for."""
    if "ames_xgb" not in chosen:
        return {}
    if not st.toggle("자체 모델이 왜 그렇게 판정했는지 보기", key="comparison_explain"):
        return {}
    from ames_model_service import explain_ames_model_materials

    with st.spinner("설명을 계산하고 있습니다"):
        payload = explain_ames_model_materials([row["물질"] for row in disagreed])
    if payload.get("error"):
        st.warning(payload["error"])
    return {entry["canonical_smiles"]: entry
            for entry in payload.get("explanations", [])}


def _show_disagreements(disagreed, chosen, bands, *, limit=6):
    """The structures the tools split on, drawn one per row."""
    st.subheader(f"엇갈린 물질 {len(disagreed)}개")
    shown = disagreed[:limit]
    explained = _model_explanations(shown, chosen)
    for row in shown:
        left, right = st.columns([1, 3])
        with left:
            show_structure(row["물질"])
        with right:
            st.markdown(_matrix_html([row], chosen, bands), unsafe_allow_html=True)
            if explained:
                for line in explanation_lines(explained.get(row["물질"])):
                    st.caption(line)
    if len(disagreed) > limit:
        st.caption(f"위 표에 전부 있습니다. 여기에는 앞 {limit}개만 펼쳤습니다.")


def _show_scores(rows, chosen, threshold):
    """Every score on one 0-1 axis with the line drawn through it.

    The matrix already prints each number under its chip, but a number does
    not say how close it came. 0.48 and 0.52 take opposite chips and are the
    same reading; a point just off the line shows that and a chip cannot.
    """
    points = score_points(rows, chosen)
    if not points:
        return
    line = MEASURED_CUTOFF if threshold is None else threshold
    st.subheader("점수와 기준선")
    frame = pd.DataFrame(points)
    # Red and green are the verdict colours everywhere else on this screen. A
    # tool drawn in red would read as a positive call whatever its score says,
    # so the tools take hues that mean nothing here.
    tools = list(dict.fromkeys(point["도구"] for point in points))
    rule = alt.Chart(pd.DataFrame({"기준선": [line]})).mark_rule(
        strokeDash=[5, 4], color="#64748b", strokeWidth=1.5).encode(
        x=alt.X("기준선:Q", title=None))
    dots = alt.Chart(frame).mark_circle(size=130, opacity=0.85).encode(
        x=alt.X("점수:Q", title="점수", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".1f", tickCount=6)),
        y=alt.Y("물질:N", title=None, sort=None,
                axis=alt.Axis(labelLimit=180)),
        color=alt.Color("도구:N", title="도구",
                        scale=alt.Scale(
                            domain=tools,
                            range=[TOOL_COLOURS[index % len(TOOL_COLOURS)]
                                   for index in range(len(tools))]),
                        legend=alt.Legend(orient="bottom", columns=1)),
        tooltip=["전체 구조", "도구", "점수", "판정"],
    ).properties(height=alt.Step(26))
    st.altair_chart(rule + dots, width="stretch")
    # Name only the tools the reader actually picked. The line used to say
    # the rule tools and EPA TEST were left out whether or not either was in
    # the run, which sent a reader looking for a column that was never there.
    left_out = [TOOL_PERFORMANCE[tool_id]["label"] for tool_id in chosen
                if TOOL_PERFORMANCE.get(tool_id, {}).get("line") is None]
    missing = (" " + ", ".join(left_out) + "는 우리 기준선으로 자를 점수를 "
               "내지 않아 이 그림에 없습니다." if left_out else "")
    st.caption(f"가로축은 점수. 기준선 {line:.2f}보다 오른쪽이 양성입니다. 선에 붙은 점은 "
               "어느 쪽으로도 갈 수 있는 값입니다. 점수는 확률이 아니라 순서를 "
               "매기는 값입니다." + missing)


def _show_performance(chosen, *, with_note=False):
    st.subheader("도구별 성능")
    measured = [{"도구": TOOL_PERFORMANCE[tool_id]["label"],
                 "균형 정확도": TOOL_PERFORMANCE[tool_id]["balanced"]}
                for tool_id in chosen if tool_id in TOOL_PERFORMANCE]
    if measured:
        # The axis runs the whole 0-1, not from 0.5. Cutting the left off
        # stretches a 0.03 gap into half the plot, and the gaps here are
        # small enough that it would have flattered somebody.
        coin = alt.Chart(pd.DataFrame({"동전 던지기": [0.5]})).mark_rule(
            strokeDash=[5, 4], color="#64748b", strokeWidth=1.5).encode(
            x=alt.X("동전 던지기:Q", title=None))
        bars = alt.Chart(pd.DataFrame(measured)).mark_bar(
            color=TOOL_COLOURS[0], opacity=0.9).encode(
            x=alt.X("균형 정확도:Q", title="균형 정확도",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".1f", tickCount=6)),
            y=alt.Y("도구:N", title=None, sort="-x",
                    axis=alt.Axis(labelLimit=220)),
            tooltip=["도구", "균형 정확도"],
        ).properties(height=alt.Step(30))
        st.altair_chart(bars + coin, width="stretch")
        st.caption("가로축은 균형 정확도. 점선 0.5는 동전 던지기와 같습니다.")
        if {"ames_xgb", "ames_desc"} <= set(chosen):
            # The two in-house models are not a better and a worse version of
            # one thing. The answered column already carries the difference
            # (77.7% against 100%); this says what it is.
            st.caption("모델 1이 답하지 않은 구조는 대부분 염·혼합물·무기물입니다. "
                       "모델 2가 그 자리를 메웁니다.")
    # The graph answers which tool did better; the table answers by how much
    # and how often it answered at all. Printing both on the surface was the
    # same numbers twice, so the detail sits one click away.
    with st.expander("숫자로 보기"):
        st.dataframe(performance_rows(chosen), hide_index=True, width="stretch")
        if with_note:
            st.caption("모든 도구를 같은 자료에서 쟀습니다. 판정 비율이 낮으면 "
                       "정확도가 높아도 그만큼 자주 답하지 않는다는 뜻입니다.")


def show_route_tool_table(workspace):
    """The tool-by-tool table for the route the reader is already looking at.

    The risk screen gives each material one chip, and that chip is the rule
    screen alone: its sensitivity is 0.552, so half the real positives carry
    no mark there. A reader who wanted the model calls had to leave for the
    comparison tab and find the same material again. The picker and the table
    are here too now, scoped to this route.
    """
    report = (workspace or {}).get("report")
    route_id = (workspace or {}).get("route_id")
    if not report or not route_id:
        return
    materials = materials_of_route(report, route_id)
    if not materials:
        return

    st.subheader("도구별 판정")
    st.caption("위 카드의 칩은 규칙 판정 하나입니다. 여기서 도구를 골라 같은 "
               "물질에 대한 다른 판정을 함께 볼 수 있습니다.")
    available = list(screening_judges.implemented_judge_ids())
    controls, action = st.columns([4, 1])
    with controls:
        chosen = st.multiselect(
            "도구", available,
            default=[tool_id for tool_id in DEFAULT_TOOLS if tool_id in available],
            format_func=lambda tool_id: TOOL_PERFORMANCE[tool_id]["label"],
            key="route_tools", label_visibility="collapsed",
            placeholder="실행할 도구를 고르세요")
    with action:
        run = st.button("실행", key="route_tools_run", type="primary",
                        width="stretch")
    if not chosen:
        st.caption("도구를 하나 이상 고르세요.")
        return

    state_key = "_route_tool_result"
    if run:
        with st.spinner(f"도구 {len(chosen)}개를 물질 {len(materials)}개에 "
                        "실행하고 있습니다"):
            st.session_state[state_key] = {
                "results": run_tools(materials, chosen), "materials": materials,
                "tools": chosen, "route_id": route_id}
    stored = st.session_state.get(state_key)
    # A stored run belongs to one route and one tool set. Switching either
    # must show the button again rather than the previous route's answers.
    if (not stored or stored["tools"] != chosen or stored["route_id"] != route_id
            or stored["materials"] != materials):
        slow = [TOOL_PERFORMANCE[tool_id]["label"] for tool_id in chosen
                if tool_id in EXTERNAL_TOOLS]
        note = (" " + ", ".join(slow) + "는 외부 프로그램을 쓰므로 시간이 걸립니다."
                if slow else "")
        st.caption(f"물질 {len(materials)}개, 도구 {len(chosen)}개." + note)
        return

    for tool_id, result in stored["results"].items():
        if result.get("tool_error"):
            st.error(f"{TOOL_PERFORMANCE[tool_id]['label']}를 실행하지 못했습니다: "
                     f"{result['tool_error']}")
    rows = build_comparison(stored["materials"], stored["results"])
    st.markdown(_matrix_html(rows, chosen), unsafe_allow_html=True)
    disagreed = [row for row in rows if row["_disagree"]]
    if disagreed:
        st.caption(f"도구마다 답이 다른 물질 {len(disagreed)}개는 표에서 "
                   "표시했습니다. 어느 쪽이 맞는지는 정하지 않습니다.")
    st.download_button("이 표 내려받기 (CSV)", comparison_csv(rows, chosen),
                       file_name="route_tool_calls.csv", mime="text/csv",
                       key="route_tools_csv")


def show_tool_comparison(workspace):
    """The comparison screen. Runs nothing until the user asks for it."""
    report = (workspace or {}).get("report")
    if not report:
        st.info("경로를 먼저 불러오세요.")
        return
    materials = route_materials(report)
    if not materials:
        st.info("비교할 물질이 없습니다.")
        return

    ui_kit.use()
    chosen, run = _tool_picker(list(screening_judges.implemented_judge_ids()))
    threshold = None
    if not chosen:
        st.warning("도구를 하나 이상 고르세요.")
        return

    state_key = "_comparison_result"
    if run:
        with st.spinner(f"도구 {len(chosen)}개를 물질 {len(materials)}개에 "
                        "실행하고 있습니다"):
            st.session_state[state_key] = {
                "results": run_tools(materials, chosen, threshold=threshold),
                "materials": materials, "tools": chosen, "threshold": threshold}
    stored = st.session_state.get(state_key)
    if not stored or stored["tools"] != chosen:
        # Only name the slow ones when one of them is actually in the run.
        # The line used to warn about Toxtree and EPA TEST whether or not they
        # had been picked, which made a fast run look like a slow one.
        slow = [TOOL_PERFORMANCE[tool_id]["label"] for tool_id in chosen
                if tool_id in EXTERNAL_TOOLS]
        # "띄우므로" would be wrong for the Toolbox: it does not launch it but
        # calls one that has to be running already.
        note = (" " + ", ".join(slow) + "는 외부 프로그램을 쓰므로 시간이 걸립니다."
                if slow else "")
        st.caption(f"물질 {len(materials)}개, 도구 {len(chosen)}개." + note)
        _show_performance(chosen)
        return

    for tool_id, result in stored["results"].items():
        if result.get("tool_error"):
            st.error(f"{TOOL_PERFORMANCE[tool_id]['label']}를 실행하지 못했습니다: "
                     f"{result['tool_error']}")

    rows = build_comparison(stored["materials"], stored["results"])
    disagreed = [row for row in rows if row["_disagree"]]
    silent = [row for row in rows if row["_answered"] == 0]
    bands = ({material: applicability.band_for(material)
              for material in stored["materials"]}
             if applicability.available() else {})

    _summary_metrics(rows, disagreed, silent, bands)
    st.markdown(_matrix_html(rows, chosen, bands), unsafe_allow_html=True)
    _show_scores(rows, chosen, stored["threshold"])
    st.download_button("비교표 내려받기 (CSV)", comparison_csv(rows, chosen),
                       file_name="tool_comparison.csv", mime="text/csv")
    if disagreed:
        _show_disagreements(disagreed, chosen, bands)

    _show_performance(chosen, with_note=True)
