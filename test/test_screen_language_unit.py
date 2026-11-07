"""Nothing internal reaches the screen, in any language but the screen's.

Leaks of this kind were found one at a time by opening the app and reading
it: raw reason codes, `status=success`, `applicability=unknown`,
`secondary_amine_nitrite:` at the front of a sentence, English backend
messages and English model limitations. Each was fixed where it was found,
which is no guarantee about the next one.

So the rule is checked instead of inspected. The text the screens draw is
swept for the shapes internal strings take, and the raw-record blocks, which
are English on purpose and reachable by design, are excluded.
"""

from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "test"))

import admet_service
import ames_descriptor_service
import ames_model_service
import presentation

# Identifiers that are the reader's words, not the code's.
ALLOWED_SNAKE = {"model_spec", "run_np_screen", "preview_with_example"}
# key=value shapes that are chemistry or a URL, not an internal field.
ALLOWED_PAIR = re.compile(r"^(?:[A-Za-z]{1,3}=[A-Za-z0-9()\[\]@+-]+|page=\d+)$")

SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
# No trailing \b: a Korean particle right after the value is a word character
# too, so "status=success는" would slip past one.
PAIR = re.compile(r"\b[A-Za-z_]{3,}=[A-Za-z0-9_.]+")
KOREAN = re.compile(r"[가-힣]")


def internal_tokens(text):
    """Internal-looking fragments in one piece of screen text."""
    found = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith(".uk-") or line.startswith("[data-testid"):
            continue
        found += [token for token in SNAKE.findall(line)
                  if token not in ALLOWED_SNAKE]
        found += [token for token in PAIR.findall(line)
                  if not ALLOWED_PAIR.match(token)]
    return found


class SweepHelperTests(unittest.TestCase):
    """The sweep has to catch what it is for, or it proves nothing."""

    def test_it_catches_the_shapes_that_leaked_before(self):
        for leak in ("사유: prohibited_material", "부산물 status=success는",
                     "applicability=unknown입니다",
                     "secondary_amine_nitrite: 같은 단계에"):
            with self.subTest(leak):
                self.assertTrue(internal_tokens(leak), f"놓침: {leak}")

    def test_it_does_not_object_to_ordinary_screen_text(self):
        for fine in ("경고구조 없음", "판정 기준선 0.25", "CC(=O)[O-].[Na+]",
                     "ROC-AUC 0.9015", "ICH M7 목록 물질",
                     "https://example.invalid/doc?page=18"):
            with self.subTest(fine):
                self.assertEqual(internal_tokens(fine), [])


class ModelLimitationTests(unittest.TestCase):
    """Every limitation a model ships has to be readable on the screen.

    They are written in English because the same strings travel in the saved
    envelope, where the wording is checked. The screen says them in Korean.
    """

    SERVICES = (admet_service, ames_model_service, ames_descriptor_service)

    def english_limitations(self):
        return [item for service in self.SERVICES
                for item in service._LIMITATIONS if not KOREAN.search(item)]

    def test_the_scan_finds_limitations_to_check(self):
        self.assertGreaterEqual(len(self.english_limitations()), 10)

    def test_every_one_of_them_has_a_korean_rendering(self):
        missing = [item for item in self.english_limitations()
                   if presentation.limitation_text(item) == item]
        self.assertEqual(missing, [], f"번역 없는 한계 문구: {missing}")

    def test_each_rendering_is_korean_and_keeps_its_numbers(self):
        for item in self.english_limitations():
            with self.subTest(item[:40]):
                korean = presentation.limitation_text(item)
                self.assertTrue(KOREAN.search(korean))
                # Compared as numbers, so 0.50 and 0.5 count as the same
                # figure while a dropped or altered one still fails.
                shown = {float(value) for value in re.findall(r"\d+\.\d+", korean)}
                for number in re.findall(r"\d+\.\d+", item):
                    self.assertIn(float(number), shown,
                                  f"{number}이(가) 번역에서 빠졌습니다")

    def test_an_unmapped_limitation_is_shown_rather_than_dropped(self):
        self.assertEqual(presentation.limitation_text("brand new wording"),
                         "brand new wording")
        for empty in (None, ""):
            with self.subTest(empty):
                self.assertEqual(presentation.limitation_text(empty), empty)


if __name__ == "__main__":
    unittest.main()
