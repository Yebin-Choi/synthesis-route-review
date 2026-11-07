"""Every reason code the engine can emit has to have a Korean label.

The screen prints a reason's label. When a code had none it printed as
itself, so a route dropped for a prohibited material reported
"prohibited_material" to the reader, and a conditional warning reported
"conditional_warning". Both are findings a reviewer has to act on, and
neither is readable as a code.

This test reads the literals out of the modules that build the reasons, so a
new code added later fails here until it is classified rather than appearing
raw on screen.
"""

from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import review_views

# Where reasons are built, and the names in those files that are not codes.
SOURCES = ("route_selection.py", "ames_route_review.py")
NOT_CODES = {
    "code", "codes", "message", "messages", "reason", "reasons", "status",
    "input_smiles", "canonical_smiles", "step_id", "smiles", "routes",
    "steps", "materials", "policy", "counts", "limitations", "excluded_smiles",
    "rank_penalty", "assessments", "threshold", "model", "schema_version",
    "byproducts", "candidates", "conditions", "description", "source",
    "name", "role", "alerts", "origin", "source_url", "id", "target",
}


def emitted_codes():
    """Code literals passed to a reason builder in the two source modules."""
    found = set()
    for name in SOURCES:
        text = (ROOT / "app" / name).read_text(encoding="utf-8")
        for call in re.findall(r'reason\(\s*"([a-z_]{4,})"', text):
            found.add(call)
        for literal in re.findall(r'"code":\s*"([a-z_]{4,})"', text):
            found.add(literal)
    return found - NOT_CODES


class ReasonCodeCoverageTests(unittest.TestCase):
    def setUp(self):
        self.labelled = (set(review_views.DECIDING_CODES)
                         | set(review_views.ROUTINE_CODES)
                         | set(review_views.QUIET_CODES))

    def test_the_source_scan_actually_finds_codes(self):
        """A scan that silently found nothing would pass every other test."""
        found = emitted_codes()
        self.assertGreaterEqual(len(found), 8)
        self.assertIn("prohibited_material", found)
        self.assertIn("conditional_warning", found)

    def test_every_emitted_code_has_a_label(self):
        missing = sorted(emitted_codes() - self.labelled)
        self.assertEqual(missing, [], f"라벨 없는 사유 코드: {missing}")

    def test_the_status_codes_the_report_reuses_are_labelled_too(self):
        # route_selection turns a material status into a reason code directly.
        for status in ("known_concern", "structural_alert"):
            with self.subTest(status):
                self.assertIn(status, self.labelled)

    def test_a_code_belongs_to_exactly_one_group(self):
        groups = (set(review_views.DECIDING_CODES),
                  set(review_views.ROUTINE_CODES),
                  set(review_views.QUIET_CODES))
        for first in range(len(groups)):
            for second in range(first + 1, len(groups)):
                with self.subTest(first=first, second=second):
                    self.assertEqual(groups[first] & groups[second], set())

    def test_an_unlabelled_code_never_reaches_the_screen_as_itself(self):
        self.assertTrue(review_views.UNKNOWN_CODE.strip())
        self.assertNotIn("_", review_views.UNKNOWN_CODE)

    def test_a_deciding_label_is_never_an_internal_looking_string(self):
        for code, label in {**review_views.DECIDING_CODES,
                            **review_views.ROUTINE_CODES}.items():
            with self.subTest(code):
                self.assertNotIn("_", label)
                self.assertTrue(any("가" <= ch <= "힣" for ch in label))


if __name__ == "__main__":
    unittest.main()
