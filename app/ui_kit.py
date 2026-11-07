"""One presentation vocabulary shared by every screen.

Rules every screen follows:

- A verdict is a chip carrying a word, never a bare status code or a sentence.
- An abstention is its own kind, distinct from a negative, and carries its
  reason. "No alert found" and "could not tell" are different statements.
- Colour never carries the verdict alone, so the screen reads in grayscale.
- A measured number is shown with what it was measured on, or not shown.
- Explanations and the raw record live in collapsed blocks.

`use()` emits the CSS per screen rather than once in the entry script, so a
screen renders correctly when mounted alone, which is how the tests drive it.
"""

from html import escape

import streamlit as st

# Verdict kinds. The names say what the reader sees, not what the backend
# called it, so a screen cannot leak an internal code through the style name.
CONCERN, CLEAR, REVIEW, UNKNOWN, ABSENT = "pos", "neg", "warn", "abs", "none"

# The call vocabulary of ICH M7 (Q)SAR reporting, which separates the two
# non-answers a screen most often collapses into "negative". The paper on
# handling out-of-domain and indeterminate results under ICH M7 states that
# neither is a negative, and that an overall negative conclusion generally
# needs expert review when one is present.
# https://pmc.ncbi.nlm.nih.gov/articles/PMC7500704/
#
# `no_alert` is its own word rather than "음성" because a rule screen finding
# nothing is not a negative prediction. The ISS Ames profiler derived from
# Toxtree's mutagenicity module says so outright: "the 'No structural alert'
# flag is not equivalent to a negative prediction."
# https://qsardb.food.dtu.dk/download/profilers/Ames_ISS.pdf
# Our own measurement agrees: the rule screen's sensitivity is 0.552, so it
# raises nothing on nearly half of the real positives. The underlying call
# stays a negative, because that is how the performance figures were counted;
# only the word on screen changes.
VERDICT = {
    "positive": ("양성", CONCERN),
    "negative": ("음성", CLEAR),
    "no_alert": ("경고구조 없음", CLEAR),
    "indeterminate": ("상충된 결과", UNKNOWN),
    "out_of_domain": ("적용 범위 밖", UNKNOWN),
    "not_run": ("실행 안 됨", ABSENT),
}


def verdict_chip(name, **kwargs):
    """A chip for one of the ICH M7 call categories, by name."""
    label, kind = VERDICT[name]
    return chip(label, kind, **kwargs)

KIT_CSS = """<style>
.uk-chip { display:inline-block; border-radius:999px; padding:0.08rem 0.5rem;
  font-size:0.78rem; font-weight:600; white-space:nowrap; }
.uk-pos { background:#fde2e2; color:#8c1c1c; }
.uk-neg { background:#ddf2e4; color:#14532d; }
.uk-warn { background:#fdf0d5; color:#7c4a03; }
.uk-abs { background:#e8eaee; color:#475569; }
.uk-none { background:transparent; color:var(--review-muted,#64748b); }
.uk-aside { color:var(--review-muted,#64748b); font-size:0.74rem;
  margin-left:0.25rem; }
.uk-hint { display:inline-flex; align-items:center; justify-content:center;
  width:15px; height:15px; margin-left:4px; border:1px solid var(--review-muted,#94a3b8);
  border-radius:50%; color:var(--review-muted,#64748b); font-size:10px;
  line-height:1; font-weight:600; vertical-align:middle; cursor:help;
  user-select:none; }
.uk-note { display:block; color:var(--review-muted,#64748b);
  font-size:0.72rem; margin-top:0.15rem; max-width:19rem; }
.uk-facts { display:grid; grid-template-columns:auto 1fr; gap:0.18rem 0.9rem;
  margin:0.2rem 0 0.5rem; font-size:0.84rem; }
.uk-facts dt { color:var(--review-muted,#64748b); white-space:nowrap; }
.uk-facts dd { margin:0; overflow-wrap:anywhere; }
.uk-wrap { overflow-x:auto; }
.uk-table { border-collapse:collapse; width:100%; font-size:0.86rem; }
.uk-table th, .uk-table td {
  border-bottom:1px solid var(--review-panel,#e2e8f0);
  padding:0.42rem 0.55rem; text-align:left; vertical-align:top; }
.uk-table th { font-weight:600; color:var(--review-muted,#475569);
  white-space:nowrap; border-bottom-width:2px; }
.uk-table td.uk-mono { font-family:ui-monospace,Consolas,monospace;
  font-size:0.78rem; word-break:break-all; max-width:22rem; }
.uk-table tr.uk-mark td { background:color-mix(in srgb,
  var(--review-accent,#175b91) 7%, transparent); }
</style>"""


def use():
    """Emit the shared styles. Safe and expected to be called once per screen."""
    st.markdown(KIT_CSS, unsafe_allow_html=True)


def chip(label, kind=ABSENT, *, aside=None, note=None, hint=None):
    """One verdict as a chip: a word, optionally a number, optionally a reason.

    `aside` is for a figure that qualifies the word, such as the score behind
    it. `note` is for the reason an abstention gave, which belongs with the
    abstention rather than in a separate column where it would be read as a
    comment on a verdict that was never made. `hint` is the glossary line for
    a term the reader cannot guess from the word, shown on hover so it does
    not take a line of the screen.
    """
    piece = f'<span class="uk-chip uk-{kind}">{escape(str(label))}</span>'
    if hint:
        piece += f'<span class="uk-hint" title="{escape(str(hint))}">?</span>'
    if aside is not None:
        piece += f'<span class="uk-aside">{escape(str(aside))}</span>'
    if note:
        piece += f'<span class="uk-note">{escape(str(note))}</span>'
    return piece


def text(value):
    """A plain value made safe to place in a cell or a fact.

    Cells and fact values are markup by design, so a chip can sit in one.
    Anything that came from a structure string, a file or a backend message
    goes through here first.
    """
    return escape(str(value))


def facts(pairs):
    """A compact label/value grid, in place of a run of captions.

    Four captions in a column read as four unrelated remarks. The same four
    as a grid read as one record, which is what they are.

    Values are markup, so a chip can be one. Pass anything else through
    `text()`; labels are escaped here.
    """
    items = [(label, value) for label, value in pairs if value not in (None, "")]
    if not items:
        return ""
    body = "".join(f"<dt>{escape(str(label))}</dt><dd>{value}</dd>"
                   for label, value in items)
    return f'<dl class="uk-facts">{body}</dl>'


def show_facts(pairs):
    markup = facts(pairs)
    if markup:
        st.markdown(markup, unsafe_allow_html=True)


def table(columns, rows, *, mono_first=True, hints=None):
    """A table whose cells are already markup, so verdicts stay chips.

    `rows` holds dicts with `cells` and an optional `mark` flag for a row the
    reader should notice, such as one where the tools disagreed. Cells are
    markup; pass plain values through `text()` first. `hints` maps a column
    name to its glossary line, shown on hover so the header can stay short.
    """
    hints = hints or {}
    head = ""
    for column in columns:
        label = escape(str(column))
        note = hints.get(column)
        if note:
            label += f'<span class="uk-hint" title="{escape(str(note))}">?</span>'
        head += f"<th>{label}</th>"
    body = []
    for row in rows:
        cells = []
        for index, cell in enumerate(row["cells"]):
            kind = ' class="uk-mono"' if mono_first and index == 0 else ""
            cells.append(f"<td{kind}>{cell}</td>")
        mark = " class='uk-mark'" if row.get("mark") else ""
        body.append(f"<tr{mark}>{''.join(cells)}</tr>")
    return (f'<div class="uk-wrap"><table class="uk-table">'
            f'<thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def show_table(columns, rows, **kwargs):
    st.markdown(table(columns, rows, **kwargs), unsafe_allow_html=True)


def metric_row(items):
    """A metric strip. Each item is (label, value) or (label, value, help)."""
    if not items:
        return
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        label, value = item[0], item[1]
        column.metric(label, value, help=item[2] if len(item) > 2 else None)


def raw_record(payload, *, title="원본 기록"):
    """Keep the exact record reachable without printing it onto the page."""
    with st.expander(title):
        st.json(payload, expanded=False)


__all__ = [
    "ABSENT", "CLEAR", "CONCERN", "KIT_CSS", "REVIEW", "UNKNOWN",
    "VERDICT", "chip", "facts", "metric_row", "raw_record",
    "show_facts", "show_table", "table", "text", "use", "verdict_chip",
]
