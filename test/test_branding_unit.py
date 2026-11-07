"""The header, the strip under it and the footer.

The screen opens with these, so two things have to hold. The figures in the
strip have to be the measured ones rather than a second copy that can drift,
and a missing logo file has to leave a header rather than a hole.
"""

from pathlib import Path
import re
import sys
import unittest
import unittest.mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import branding
import tool_comparison_view as view


class LogoTests(unittest.TestCase):
    def test_a_missing_asset_still_draws_a_mark(self):
        """The header is never broken by a file nobody has dropped in yet."""
        with unittest.mock.patch.object(branding, "logo_path", return_value=None):
            markup = branding.logo_markup()
        self.assertIn("<svg", markup)
        self.assertIn("ReRx", markup)

    def test_the_name_is_typed_only_when_the_lockup_is_absent(self):
        """The brand asset carries the word, so it is not set twice."""
        with unittest.mock.patch.object(branding, "logo_path", return_value=None):
            drawn = branding.header_markup()
        self.assertIn('class="rx-name"', drawn)
        if branding.logo_path() is not None:
            self.assertNotIn('class="rx-name"', branding.header_markup())
            self.assertIn(f'alt="{branding.NAME}"', branding.header_markup())

    def test_the_mark_is_one_line_so_markdown_leaves_it_inline(self):
        """A newline before a tag opens a paragraph, which drops the mark out
        of the header's row and onto a line of its own."""
        self.assertNotIn("\n", branding.logo_markup())

    def test_an_asset_would_be_preferred_over_the_drawn_mark(self):
        self.assertEqual(branding.LOGO_NAMES[0], "rerx_logo.png")
        self.assertTrue(str(branding.ASSETS).endswith("assets"))


class HeaderTests(unittest.TestCase):
    def stats(self):
        return [(label, "균형 정확도",
                 f"{view.TOOL_PERFORMANCE[tool]['balanced']:.3f} · "
                 f"기준선 {view.TOOL_PERFORMANCE[tool]['line']:.2f}")
                for label, tool in (("자체 모델", "ames_xgb"),
                                    ("기술자 모델", "ames_desc"))]

    def test_the_name_and_what_it_does_are_both_there(self):
        markup = branding.header_markup()
        for piece in (branding.NAME, branding.SUBTITLE, branding.VERSION,
                      branding.TAGLINE):
            with self.subTest(piece):
                self.assertIn(piece, markup)

    def test_the_strip_carries_the_measured_figures_not_a_second_copy(self):
        markup = branding.header_markup(self.stats())
        for tool in ("ames_xgb", "ames_desc"):
            with self.subTest(tool):
                info = view.TOOL_PERFORMANCE[tool]
                self.assertIn(f"{info['balanced']:.3f}", markup)
                self.assertIn(f"기준선 {info['line']:.2f}", markup)

    def test_a_chip_without_a_label_reads_as_one_phrase(self):
        markup = branding.stat_chips([("연동 도구", "", "6개")])
        text = re.sub(r"<[^>]+>", " ", markup)
        self.assertEqual(" ".join(text.split()), "연동 도구 6개")

    def test_a_value_from_outside_cannot_inject_markup(self):
        markup = branding.stat_chips([("x", "y", "<script>x</script>")])
        self.assertNotIn("<script>", markup)


class FooterTests(unittest.TestCase):
    def test_it_names_the_product_version_and_refuses_to_claim_safety(self):
        markup = branding.footer_markup(["목표 물질 25개"])
        self.assertIn(branding.VERSION, markup)
        self.assertIn("목표 물질 25개", markup)
        self.assertIn("안전성 판정이 아닙니다", markup)


if __name__ == "__main__":
    unittest.main()
