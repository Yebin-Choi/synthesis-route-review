"""The genotoxicity screen inside SynthEx's Critic. No LLM, no network.

Every structure here is software test material, never an Ames measurement.

What these tests are for: the Editor only reworks a step when the Critic
returns a blocking finding, so a flagged precursor must produce one and must
also clear `is_satisfactory`; and a structure that could not be read must not
come back looking clean.
"""

import asyncio
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import toxicity_critic  # noqa: E402

# Software test material.
EPOXIDE = "ClCC1CO1"        # epichlorohydrin, carries a structural alert
PLAIN = "CCO"               # ethanol, nothing to raise
UNREADABLE = "not-a-molecule"


def _rendered(*steps):
    return {"route": [{"id": step_id, "smiles": smiles} for step_id, smiles in steps]}


@dataclass
class _StepFinding:
    step_id: object
    severity: str
    feedback: str


class _Critique:
    """Stands in for the pydantic model, with the two fields that are read."""

    def __init__(self, is_satisfactory=True, problematic_steps=()):
        self.is_satisfactory = is_satisfactory
        self.problematic_steps = list(problematic_steps)

    def model_copy(self, *, update):
        copy = _Critique(self.is_satisfactory, self.problematic_steps)
        for key, value in update.items():
            setattr(copy, key, value)
        return copy


class _BaseCritic:
    prompt_version = "critic-v2-atom-accounting"

    def __init__(self, critique=None):
        self.critique = critique or _Critique()

    async def run(self, request, context):
        return self.critique


class _Route:
    def __init__(self, rendered):
        self._rendered = rendered

    def to_op_plan(self):
        return self._rendered


class _Request:
    def __init__(self, rendered):
        self.route = _Route(rendered)


def _run(critique, rendered):
    critic_class = toxicity_critic.build(_BaseCritic, _StepFinding, lambda plan: plan)
    critic = critic_class(critique)
    return asyncio.run(critic.run(_Request(rendered), context=None))


class PrecursorReadingTests(unittest.TestCase):
    def test_precursors_are_taken_from_the_right_side_of_the_arrow(self):
        pairs = toxicity_critic.precursors_of(
            _rendered(("S1", f"{PLAIN}>>{EPOXIDE}.{PLAIN}")))
        self.assertEqual(pairs, [("S1", EPOXIDE), ("S1", PLAIN)])

    def test_a_step_without_an_arrow_contributes_nothing(self):
        self.assertEqual(toxicity_critic.precursors_of(_rendered(("S1", PLAIN))), [])

    def test_the_product_is_never_screened_as_a_precursor(self):
        """The target is what the route makes, not something charged into it."""
        pairs = toxicity_critic.precursors_of(_rendered(("S1", f"{EPOXIDE}>>{PLAIN}")))
        self.assertEqual(pairs, [("S1", PLAIN)])


class ScreeningTests(unittest.TestCase):
    def test_an_alerting_precursor_blocks(self):
        blocking, unreadable = toxicity_critic.screen_precursors(
            _rendered(("S1", f"{PLAIN}>>{EPOXIDE}")))
        self.assertEqual(len(blocking), 1)
        self.assertEqual(blocking[0][0], "S1")
        self.assertIn(EPOXIDE, blocking[0][1])
        self.assertEqual(unreadable, [])

    def test_a_clean_precursor_produces_nothing(self):
        self.assertEqual(
            toxicity_critic.screen_precursors(_rendered(("S1", f"{EPOXIDE}>>{PLAIN}"))),
            ([], []))

    def test_an_unreadable_precursor_is_reported_not_passed(self):
        blocking, unreadable = toxicity_critic.screen_precursors(
            _rendered(("S1", f"{PLAIN}>>{UNREADABLE}")))
        self.assertEqual(blocking, [])
        self.assertEqual(len(unreadable), 1)
        self.assertIn("not screened", unreadable[0][1])

    def test_the_same_precursor_twice_in_a_step_is_reported_once(self):
        blocking, _ = toxicity_critic.screen_precursors(
            _rendered(("S1", f"{PLAIN}>>{EPOXIDE}.{EPOXIDE}")))
        self.assertEqual(len(blocking), 1)

    def test_the_same_precursor_in_two_steps_is_reported_for_each(self):
        blocking, _ = toxicity_critic.screen_precursors(
            _rendered(("S1", f"{PLAIN}>>{EPOXIDE}"), ("S2", f"{PLAIN}>>{EPOXIDE}")))
        self.assertEqual([step for step, _ in blocking], ["S1", "S2"])


class CritiqueTests(unittest.TestCase):
    def test_a_clean_route_is_returned_untouched(self):
        critique = _Critique(is_satisfactory=True)
        result = _run(critique, _rendered(("S1", f"{EPOXIDE}>>{PLAIN}")))
        self.assertIs(result, critique)

    def test_a_flagged_precursor_forces_an_edit(self):
        """A blocking finding alone is not enough: is_resolved reads both."""
        result = _run(_Critique(is_satisfactory=True),
                      _rendered(("S1", f"{PLAIN}>>{EPOXIDE}")))
        self.assertFalse(result.is_satisfactory)
        self.assertEqual([f.severity for f in result.problematic_steps], ["blocking"])

    def test_the_llm_findings_are_kept_alongside_ours(self):
        existing = _StepFinding(step_id="S1", severity="blocking", feedback="화학 문제")
        result = _run(_Critique(is_satisfactory=False, problematic_steps=[existing]),
                      _rendered(("S1", f"{PLAIN}>>{EPOXIDE}")))
        self.assertIn(existing, result.problematic_steps)
        self.assertEqual(len(result.problematic_steps), 2)

    def test_an_unreadable_precursor_does_not_force_an_edit(self):
        """We cannot call a structure we failed to parse dangerous."""
        result = _run(_Critique(is_satisfactory=True),
                      _rendered(("S1", f"{PLAIN}>>{UNREADABLE}")))
        self.assertTrue(result.is_satisfactory)
        self.assertEqual([f.severity for f in result.problematic_steps], ["suggestion"])

    def test_our_screen_never_clears_a_route_the_llm_objected_to(self):
        result = _run(_Critique(is_satisfactory=False),
                      _rendered(("S1", f"{PLAIN}>>{EPOXIDE}")))
        self.assertFalse(result.is_satisfactory)

    def test_the_prompt_version_records_that_the_screen_ran(self):
        critic_class = toxicity_critic.build(_BaseCritic, _StepFinding, lambda plan: plan)
        self.assertIn("ich-m7-rule-screen", critic_class.prompt_version)
        self.assertTrue(critic_class.prompt_version.startswith(_BaseCritic.prompt_version))


if __name__ == "__main__":
    unittest.main()
