"""The shared presentation vocabulary.

What these tests are for: the kit exists so one verdict is said one way on
every screen. Two properties have to hold or the kit is worse than the
duplication it replaced.

First, a non-answer must never read as a negative. ICH M7 (Q)SAR reporting
separates out-of-domain and indeterminate from negative and says neither is
one (PMC7500704), and the ISS Ames profiler says a rule screen's "no
structural alert" is not a negative prediction either.

Second, the verdict must survive losing colour. WCAG 2.1 SC 1.4.1 asks that
colour not be the only means of conveying information, so every chip has to
carry its word.
"""

from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import ui_kit


def strip_tags(markup):
    return re.sub(r"<[^>]+>", " ", markup)


class VerdictVocabularyTests(unittest.TestCase):
    def test_every_verdict_carries_its_word_not_only_a_colour(self):
        for name in ui_kit.VERDICT:
            with self.subTest(name):
                label, _ = ui_kit.VERDICT[name]
                self.assertTrue(label.strip())
                self.assertIn(label, strip_tags(ui_kit.verdict_chip(name)))

    def test_a_non_answer_never_carries_the_word_for_negative(self):
        negative, _ = ui_kit.VERDICT["negative"]
        for name in ("indeterminate", "out_of_domain", "not_run"):
            with self.subTest(name):
                self.assertNotIn(negative, strip_tags(ui_kit.verdict_chip(name)))

    def test_a_rule_screen_finding_nothing_is_not_called_a_negative(self):
        no_alert, _ = ui_kit.VERDICT["no_alert"]
        negative, _ = ui_kit.VERDICT["negative"]
        self.assertNotEqual(no_alert, negative)
        self.assertNotIn(negative, no_alert)

    def test_the_two_non_answers_are_told_apart(self):
        self.assertNotEqual(ui_kit.VERDICT["indeterminate"][0],
                            ui_kit.VERDICT["out_of_domain"][0])


class ChipTests(unittest.TestCase):
    def test_a_reason_travels_with_the_chip_that_has_no_verdict(self):
        markup = ui_kit.chip("예측 결과 없음", ui_kit.UNKNOWN, note="적용 범위 밖")
        self.assertIn("적용 범위 밖", strip_tags(markup))

    def test_a_score_is_an_aside_and_never_replaces_the_word(self):
        markup = ui_kit.chip("양성", ui_kit.CONCERN, aside="0.81")
        text = strip_tags(markup)
        self.assertIn("양성", text)
        self.assertIn("0.81", text)

    def test_a_label_from_outside_cannot_inject_markup(self):
        markup = ui_kit.chip("<script>x</script>")
        self.assertNotIn("<script>", markup)


class FactsAndTableTests(unittest.TestCase):
    def test_an_empty_value_is_dropped_rather_than_shown_as_blank(self):
        markup = ui_kit.facts([("도구", "ADMET-AI"), ("버전", None), ("출처", "")])
        self.assertIn("ADMET-AI", markup)
        self.assertNotIn("버전", markup)
        self.assertNotIn("출처", markup)

    def test_no_facts_at_all_renders_nothing(self):
        self.assertEqual(ui_kit.facts([("버전", None)]), "")

    def test_a_marked_row_is_distinguishable_in_the_markup(self):
        markup = ui_kit.table(
            ["물질", "판정"],
            [{"cells": ["CCO", ui_kit.verdict_chip("negative")]},
             {"cells": ["C1CO1", ui_kit.verdict_chip("positive")], "mark": True}])
        self.assertEqual(markup.count("uk-mark"), 1)

    def test_a_structure_string_is_escaped_through_the_text_helper(self):
        self.assertEqual(ui_kit.text("a<b&c"), "a&lt;b&amp;c")


if __name__ == "__main__":
    unittest.main()
