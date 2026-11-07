"""Presentation must preserve chemical uncertainty and evidenced connectivity."""
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from rdkit.Chem import Draw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from engine import DISCLAIMER
import presentation
import ui_kit
from presentation import (
    GLOSSARY, STATUS, STATUS_KINDS, material_reason, route_dot, structure_png,
    term, theme_css,
)
from toxicology import assess_material


class PresentationTests(unittest.TestCase):
    def test_structure_image_and_rejected_input(self):
        self.assertTrue(structure_png("CCO").startswith(b"\x89PNG"))
        for value in ("", "일반문장", "C1CC", "C" * 4097, "C" * 257, None):
            with self.subTest(value=str(value)[:12]):
                self.assertIsNone(structure_png(value))

    def test_drawing_oserror_is_a_missing_image_not_an_uncaught_failure(self):
        structure_png.clear()
        self.addCleanup(structure_png.clear)
        with patch.object(Draw, "MolToImage", side_effect=OSError("drawing unavailable")) as draw:
            self.assertIsNone(structure_png("CCO"))
            draw.assert_called_once()

    def test_png_save_oserror_is_a_missing_image_not_an_uncaught_failure(self):
        structure_png.clear()
        self.addCleanup(structure_png.clear)
        with patch.object(Image.Image, "save", side_effect=OSError("PNG encoding unavailable")) as save:
            self.assertIsNone(structure_png("CCO"))
            save.assert_called_once()

    def test_a_finding_is_named_and_a_non_finding_adds_no_second_sentence(self):
        """The chip carries the verdict, so the line beside it must add to it.

        It used to repeat the chip: "해당 없음" with "검출된 신호 없음" under
        it, which is the same statement twice and reads as reassurance.
        """
        alert = material_reason(assess_material("CN(C)N=O"))
        self.assertIn("N-니트로소기", alert)
        self.assertNotIn("안전", alert)
        self.assertNotIn("독성 확정", alert)
        for smiles in ("CCO", "C1CC"):
            with self.subTest(smiles):
                self.assertEqual(material_reason(assess_material(smiles)), "")

    def test_every_status_label_carries_a_hover_explanation(self):
        for smiles in ("CCO", "CN(C)N=O", "ClCC1CO1", "C1CC"):
            with self.subTest(smiles=smiles):
                label = STATUS[assess_material(smiles)["status"]]
                marked = term(label)
                self.assertIn('class="review-term"', marked)
                self.assertIn(GLOSSARY[label].replace('"', "&quot;"), marked)

    def test_safety_limitation_wording_stays_in_the_data_layer(self):
        self.assertIn("안전성 입증이 아니며", DISCLAIMER)
        negative = assess_material("CCO")
        self.assertIn("does not mean safe or class 5",
                      " ".join(negative["limitations"]).lower())
        self.assertIn("not evidence of safety or class 5",
                      negative["intake_basis"].lower())

    def test_graph_connects_exact_structures_not_step_order(self):
        graph = route_dot({"steps": [
            {"id": "first", "reactants": ["CCO"], "products": ["CC=O"]},
            {"id": "disconnected", "reactants": ["N"], "products": ["NN"]},
            {"id": "continuation", "reactants": ["CC=O"], "products": ["CC(=O)O"]},
        ]})
        self.assertIn("reaction_0 -> material_1;", graph)
        self.assertIn("material_1 -> reaction_2;", graph)
        self.assertNotIn("material_1 -> reaction_1;", graph)
        self.assertEqual(graph.count("->"), 6)

    def test_graph_does_not_join_invalid_structures(self):
        graph = route_dot({"steps": [
            {"id": "a", "reactants": ["N"], "products": ["invalid"]},
            {"id": "b", "reactants": ["invalid"], "products": ["NN"]},
        ]})
        self.assertIn("reaction_0 -> material_1;", graph)
        self.assertIn("material_2 -> reaction_1;", graph)

    def test_theme_text_and_accent_contrast(self):
        def luminance(color):
            values = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in values]
            return sum(a * b for a, b in zip(linear, (.2126, .7152, .0722)))

        for light in (False, True):
            tokens = dict(re.findall(r"--review-(\w+):(#\w+)", theme_css(light)))
            for token in ("text", "muted", "accent"):
                for background in ("bg", "panel"):
                    levels = sorted((luminance(tokens[token]), luminance(tokens[background])))
                    with self.subTest(light=light, token=token, background=background):
                        self.assertGreaterEqual((levels[1] + .05) / (levels[0] + .05), 4.5)

    def test_summary_states_and_plain_text_bind_both_active_palette_colors(self):
        summaries = (
            '[data-testid="stExpander"] summary',
            '[data-testid="stExpander"] summary:hover',
            '[data-testid="stExpander"] summary:focus',
            '[data-testid="stExpander"] summary:focus-visible',
            '[data-testid="stExpander"] summary:focus-within',
            '[data-testid="stExpander"] details[open] > summary',
        )
        for light in (False, True):
            css = theme_css(light)
            tokens = dict(re.findall(r"--review-(\w+):(#\w+)", css))
            rules = re.findall(r"([^{}]+)\{([^{}]*)\}", css)
            cases = [(selector, "panel") for selector in summaries]
            cases.extend((
                ('[data-testid="stText"]', "bg"),
                ('[data-testid="stText"] pre', "bg"),
                ('[data-testid="stText"] span', "bg"),
            ))
            for selector, background in cases:
                with self.subTest(light=light, selector=selector):
                    self.assertTrue(any(
                        selector in [part.strip() for part in selectors.split(",")]
                        and f"background:{tokens[background]}!important" in body
                        and f"color:{tokens['text']}!important" in body
                        for selectors, body in rules
                    ))
            self.assertIn("font-size:17px!important;line-height:1.5!important", css)
            self.assertIn("min-height:48px!important", css)
            self.assertIn(f"outline:3px solid {tokens['accent']}!important", css)


class VerdictColourTests(unittest.TestCase):
    """One word, one colour. The card and the table are the same screen.

    "경고구조 없음" used to be grey on a card and green in the excluded-route
    and byproduct tables, which is where a reader skims fastest. The rule
    screen's sensitivity is 0.552: it raises nothing on nearly half of the
    real positives, so green there says something the measurement does not.
    """

    def test_the_tables_use_the_same_colours_as_the_cards(self):
        import review_views
        self.assertEqual(review_views.MATERIAL_KINDS, STATUS_KINDS)

    def test_no_alert_is_never_the_colour_a_negative_gets(self):
        self.assertNotEqual(STATUS_KINDS["no_alert_detected"], ui_kit.CLEAR)

    def test_the_model_verdicts_keep_their_own_colours(self):
        """ui_kit.VERDICT is counted as a negative in the performance table,
        so it stays green on purpose and this change must not touch it."""
        self.assertEqual(ui_kit.VERDICT["no_alert"][1], ui_kit.CLEAR)


if __name__ == "__main__":
    unittest.main()


class ReasonTextTests(unittest.TestCase):
    """A tool's reason for not answering has to be readable where it is read.

    The services write those reasons in English because the same strings go
    into the saved record and the provenance, where the exact wording is
    checked. The screen is Korean, so the translation happens at the screen
    and the record is left alone.
    """

    def test_a_known_backend_reason_is_shown_in_the_screen_language(self):
        self.assertEqual(
            presentation.reason_text("The model could not read this structure."),
            "모델이 이 구조를 읽지 못했습니다.")

    def test_an_unknown_reason_is_shown_as_it_came_rather_than_dropped(self):
        self.assertEqual(presentation.reason_text("something new happened"),
                         "something new happened")

    def test_no_reason_stays_no_reason(self):
        for value in (None, ""):
            with self.subTest(value):
                self.assertEqual(presentation.reason_text(value), value)

    def test_the_generation_errors_a_reader_can_act_on_are_translated(self):
        """A stuck reader needs the message in the language of the screen.

        These reached the screen in English. The worst was the boundary one,
        which is an intermittent failure a retry usually clears, and which
        said nothing about retrying.
        """
        import generation_service

        for code in ("path_escape", "startup_failed", "package_missing",
                     "interpreter_missing", "not_configured", "config_invalid",
                     "config_changed", "assets_missing", "no_routes",
                     "timeout", "backend_failed", "missing_key",
                     "target_invalid"):
            with self.subTest(code):
                english = generation_service._ERRORS[code]
                korean = presentation.reason_text(english)
                self.assertNotEqual(korean, english, "번역되지 않았습니다")
                self.assertTrue(any("가" <= ch <= "힣" for ch in korean))

    def test_the_retry_hint_is_on_the_error_that_a_retry_clears(self):
        import generation_service

        korean = presentation.reason_text(generation_service._ERRORS["path_escape"])
        self.assertIn("다시 확인", korean)

    def test_every_mapped_reason_is_korean_and_not_an_english_passthrough(self):
        for marker, korean in presentation.REASON_TEXT:
            with self.subTest(marker):
                self.assertTrue(any("가" <= ch <= "힣" for ch in korean))
                self.assertNotIn(marker.casefold(), korean.casefold())
