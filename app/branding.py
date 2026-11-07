"""The product's own header, status strip and footer.

The screen used to open with a bare `st.title`, which is what a script looks
like rather than what a tool looks like. The published prediction servers all
open the same way instead: a mark, the product's name, what it does in one
line, and the figures that say how far to trust it, before any control.

That last part is the one worth keeping honest. The strip under the name
carries each in-house model's measured accuracy and the decision line it
runs at, so the first thing on the screen is what the software can do rather
than a claim about it.

The logo is read from `app/assets/` when it is there and drawn as a mark
when it is not, so the header is never broken by a missing file and the real
asset can be dropped in without touching this code.
"""

from pathlib import Path

import streamlit as st

if __package__:
    from . import ui_kit
else:
    import ui_kit

NAME = "ReRx"
SUBTITLE = "합성 경로 유전독성 검토"
TAGLINE = "RETROSYNTHESIS · AMES GENOTOXICITY WORKBENCH"
VERSION = "v0.3"
ASSETS = Path(__file__).resolve().parent / "assets"
# Any of these is used as the logo, in order. The first is the brand asset;
# the SVG beside it is the fallback this module ships.
LOGO_NAMES = ("rerx_logo.png", "rerx_logo.svg", "logo.png", "logo.svg")

# The mark drawn when no asset file is present: the R, the orbit around it
# and the cross, in the brand's two blues. Replaced by the real file the
# moment one is dropped into app/assets/.
FALLBACK_MARK = """
<svg viewBox="0 0 150 104" width="44" height="31" role="img"
     aria-label="ReRx 로고" xmlns="http://www.w3.org/2000/svg">
  <!-- The orbit: one ellipse, tilted, drawn with a gap so it reads as a
       swoosh passing behind the letter rather than a closed ring. -->
  <ellipse cx="62" cy="52" rx="56" ry="31" fill="none" stroke="#2F7FD4"
           stroke-width="11" stroke-linecap="round"
           transform="rotate(-18 62 52)"
           stroke-dasharray="150 110" stroke-dashoffset="34"/>
  <!-- The cross, inside the orbit and to the left of the letter. -->
  <path d="M26 44h10V34h11v10h10v11H47v10H36V55H26z" fill="#2F7FD4"/>
  <!-- The letter, over the orbit. -->
  <path d="M58 22h33c15 0 24 8 24 19 0 9-6 15-17 18l21 25h-20L80 60h-6v24H58z
           M74 36v12h16c6 0 9-2 9-6s-3-6-9-6z" fill="#1B3A7A"/>
</svg>
"""

HEADER_CSS = """<style>
.rx-head { display:flex; align-items:center; gap:14px; margin:0 0 2px; }
.rx-mark { display:flex; align-items:center; flex:0 0 auto; }
.rx-mark img { height:30px; width:auto; display:block; }
.rx-mark.rx-plate { background:#fff; border-radius:7px; padding:4px 7px; }
.rx-mark svg { display:block; }
.rx-mark p { margin:0; display:contents; }
.rx-names { display:flex; flex-direction:column; gap:1px; min-width:0; }
.rx-line { display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; }
.rx-name { font-size:26px; font-weight:700; letter-spacing:-0.01em;
  color:var(--review-text); line-height:1.1; }
.rx-sub { font-size:18px; font-weight:600; color:var(--review-text); }
.rx-version { font-size:11px; font-weight:600; padding:1px 7px;
  border-radius:999px; background:var(--review-panel);
  color:var(--review-muted); border:1px solid var(--review-line); }
.rx-tag { font-size:10.5px; letter-spacing:0.14em; text-transform:uppercase;
  color:var(--review-muted); }
.rx-strip { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 22px; }
.rx-stat { display:flex; align-items:baseline; gap:6px; font-size:12px;
  padding:5px 11px; border-radius:8px; background:var(--review-panel);
  border:1px solid var(--review-line); color:var(--review-muted);
  white-space:nowrap; }
.rx-stat b { color:var(--review-text); font-weight:600; }
.rx-stat .rx-key { color:var(--review-accent); font-weight:600; }
.rx-foot { display:flex; justify-content:space-between; flex-wrap:wrap;
  gap:8px; margin-top:6px; font-size:11.5px; color:var(--review-muted); }
.rx-foot .rx-dot { color:#2f9e63; }
@media(max-width:700px) {
  .rx-name { font-size:21px; } .rx-sub { font-size:15px; }
  .rx-strip { gap:6px; } .rx-stat { font-size:11px; padding:4px 9px; }
}
</style>"""


def logo_path():
    """The brand asset on disk, or None when none has been dropped in."""
    for name in LOGO_NAMES:
        candidate = ASSETS / name
        if candidate.is_file():
            return candidate
    return None


def _one_line(markup):
    """Collapse the markup to one line.

    A newline before a tag makes Streamlit's markdown open a paragraph around
    it, which takes the mark out of the header's flex row and drops it onto a
    line of its own.
    """
    return " ".join(part.strip() for part in markup.splitlines() if part.strip())


def logo_markup():
    """The logo as markup: the asset when present, the drawn mark otherwise."""
    path = logo_path()
    if path is None:
        return _one_line(FALLBACK_MARK)
    if path.suffix.lower() == ".svg":
        return _one_line(path.read_text(encoding="utf-8"))
    import base64

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    kind = "png" if path.suffix.lower() == ".png" else "jpeg"
    return (f'<img src="data:image/{kind};base64,{encoded}" '
            f'alt="{ui_kit.text(NAME)}">')


def stat_chips(entries):
    """The strip under the name: one chip per figure worth opening with.

    Each entry is (key, label, value). The key is what the figure is about,
    so a reader can tell at a glance which model a number belongs to.
    """
    chips = []
    for key, label, value in entries:
        middle = f"{ui_kit.text(label)} " if label else ""
        chips.append(
            f'<span class="rx-stat"><span class="rx-key">{ui_kit.text(key)}</span>'
            f'{middle}<b>{ui_kit.text(value)}</b></span>')
    return f'<div class="rx-strip">{"".join(chips)}</div>' if chips else ""


def header_markup(stats=(), plate=True):
    """The header. The brand asset is a lockup that already carries the name.

    So when it is present the name is not also typed out beside it; the image
    is the name, and the word stays in its alt text. Without the asset the
    drawn mark is only a mark, so the name is set in type next to it.

    `plate` puts the lockup on a white rounded field. The navy in it is close
    enough to the dark background that it would otherwise half disappear,
    and on the light background the field cannot be seen anyway.
    """
    has_asset = logo_path() is not None
    classes = "rx-mark rx-plate" if has_asset and plate else "rx-mark"
    name = "" if has_asset else f'<span class="rx-name">{ui_kit.text(NAME)}</span>'
    return (HEADER_CSS
            + '<div class="rx-head">'
            + f'<span class="{classes}">{logo_markup()}</span>'
            + '<span class="rx-names">'
            + f'<span class="rx-line">{name}'
            + f'<span class="rx-sub">{ui_kit.text(SUBTITLE)}</span>'
            + f'<span class="rx-version">{ui_kit.text(VERSION)}</span></span>'
            + f'<span class="rx-tag">{ui_kit.text(TAGLINE)}</span>'
            + '</span></div>'
            + stat_chips(stats))


def show_header(stats=()):
    """Draw the header. The theme toggle sits beside it, not inside it."""
    heading, theme = st.columns([5, 1], vertical_alignment="center")
    with heading:
        st.markdown(header_markup(stats), unsafe_allow_html=True)
    with theme:
        st.toggle("Light mode", key="light_mode")


def footer_markup(notes=()):
    left = ", ".join(
        [f'<span class="rx-dot">●</span> {ui_kit.text(NAME)} {ui_kit.text(VERSION)}']
        + [ui_kit.text(note) for note in notes if note])
    return ('<div class="rx-foot"><span>' + left + "</span>"
            "<span>연구용 선별 결과이며 안전성 판정이 아닙니다</span></div>")


def show_footer(notes=()):
    st.markdown(footer_markup(notes), unsafe_allow_html=True)


__all__ = ["NAME", "SUBTITLE", "TAGLINE", "VERSION", "footer_markup",
           "header_markup", "logo_markup", "logo_path", "show_footer",
           "show_header", "stat_chips"]
