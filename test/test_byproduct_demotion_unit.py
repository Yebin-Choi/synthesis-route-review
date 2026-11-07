"""A byproduct candidate demotes a route. It never removes one.

Every structure here is a software fixture, never an AMES measurement and
never a toxicity determination.

Why this rule exists, measured 2026-11-04 on a live propranolol search: one
step's ASKCOS prediction returned 11 byproduct candidates, 7 of them carrying
a structural alert, and counting them as exclusions took the surviving routes
from 1 of 10 to 0 of 10. Applying every step would empty the list faster. A
reviewer left with no routes has not been given a safety finding, only an
empty page.

The asymmetry is in what the two kinds of material are. A reagent or an
intermediate is charged into the reaction. A byproduct candidate is a
prediction, and neither how much of it forms nor whether work-up removes it
is known here. So the concern is kept, reported and ranked against, and the
route stays on the list for a person to judge.

The one thing that still removes a route is the user naming a structure
themselves. That is an instruction, not a prediction.
"""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import engine
from ames_route_review import review_route_eligibility
from route_selection import select_routes

TARGET = "CCOC(C)=O"
ALERT = "C1CO1"  # an epoxide: the rule screen raises a structural alert
CLEAN = "CCO"


def payload(reactant, *byproducts, reagents=()):
    return {"schema_version": "1.0", "routes": [{
        "id": "route-1", "target": TARGET, "source": "Software-only synthetic fixture",
        "steps": [{
            "id": "step-1", "reactants": [reactant], "products": [TARGET],
            "reagents": list(reagents), "solvents": [],
            "conditions": {"status": "observed", "source": "Software-only fixture",
                           "description": "Synthetic fixture; not a laboratory observation."},
            "byproducts": {"status": "success", "candidates": list(byproducts),
                           "source": "Synthetic completed-attempt label; not a real prediction."},
        }],
    }]}


def report(reactant, *byproducts, reagents=()):
    return engine.analyze_routes(payload(reactant, *byproducts, reagents=reagents))


def codes(route):
    return {item["code"] for item in route["reasons"]}


class RuleScreenTests(unittest.TestCase):
    def test_a_byproduct_alert_demotes_rather_than_excludes(self):
        route = select_routes(report(CLEAN, ALERT))["routes"][0]
        self.assertEqual(route["status"], "needs_review")
        self.assertIn("byproduct_concern", codes(route))

    def test_strict_alerts_still_does_not_remove_it(self):
        """The conservative setting makes alerts exclude. Not this one.

        A setting that empties the list is not conservative, it is unusable.
        """
        route = select_routes(report(CLEAN, ALERT), strict_alerts=True)["routes"][0]
        self.assertEqual(route["status"], "needs_review")
        self.assertIn("byproduct_concern", codes(route))

    def test_the_same_structure_as_a_reagent_is_excluded_under_strict_alerts(self):
        """The asymmetry is the whole point, so it is asserted directly."""
        route = select_routes(report(CLEAN, reagents=[ALERT]),
                              strict_alerts=True)["routes"][0]
        self.assertEqual(route["status"], "excluded")
        self.assertNotIn("byproduct_concern", codes(route))

    def test_the_finding_itself_is_still_on_record(self):
        """Demoting is not the same as not noticing."""
        route = select_routes(report(CLEAN, ALERT))["routes"][0]
        self.assertIn("structural_alert", codes(route))
        alert = next(item for item in route["reasons"]
                     if item["code"] == "structural_alert")
        self.assertIn("부산물", alert["message"])

    def test_a_structure_the_user_named_still_removes_the_route(self):
        """An explicit exclusion is an instruction, not a prediction."""
        route = select_routes(report(CLEAN, ALERT),
                              excluded_smiles=[ALERT])["routes"][0]
        self.assertEqual(route["status"], "excluded")
        self.assertIn("prohibited_material", codes(route))

    def test_many_alerting_byproducts_do_not_empty_the_list(self):
        """The 2026-11-04 case: seven alerting candidates on one step."""
        alerts = ["C1CO1", "ClCC1CO1", "BrCCBr", "ClCCCl", "ICC", "BrCC", "ClCC"]
        result = select_routes(report(CLEAN, *alerts))
        route = result["routes"][0]
        # Every one of them really did raise an alert, or this would pass
        # for the wrong reason.
        self.assertEqual(
            sum(1 for item in route["reasons"]
                if item["code"] == "byproduct_concern"), len(alerts))
        self.assertEqual(result["counts"]["excluded"], 0)
        self.assertEqual(route["status"], "needs_review")


class ModelEvidenceTests(unittest.TestCase):
    """The model's side of the same rule, so the two cannot disagree."""

    @staticmethod
    def evidence(*rows):
        return {"model": {"tool": "ADMET-AI", "version": "software-fixture",
                          "endpoint": "ames", "weights_sha256": {"fixture.pt": "a" * 64}},
                "threshold": 0.5,
                "assessments": [
                    {"input_smiles": smiles, "canonical_smiles": engine.canonicalize_smiles(smiles),
                     "status": "success", "score": score,
                     "prediction": "positive" if score >= 0.5 else "negative",
                     "applicability": "unknown", "error": None}
                    for smiles, score in rows],
                "limitations": ["Software fixture; not an AMES measurement."]}

    def test_a_model_positive_byproduct_demotes_rather_than_excludes(self):
        result = review_route_eligibility(
            report(CLEAN, ALERT), self.evidence((CLEAN, 0.1), (ALERT, 0.9)))
        route = result["routes"][0]
        self.assertEqual(route["status"], "needs_review")
        self.assertEqual(result["counts"]["excluded"], 0)
        self.assertIn("byproduct_concern", codes(route))

    def test_the_model_finding_is_kept_beside_the_demotion_note(self):
        route = review_route_eligibility(
            report(CLEAN, ALERT),
            self.evidence((CLEAN, 0.1), (ALERT, 0.9)))["routes"][0]
        self.assertIn("admet_positive", codes(route))

    def test_a_byproduct_demotes_no_harder_than_the_same_concern_elsewhere(self):
        """Demoting is not punishing.

        The demotion note annotates the finding beside it; it does not charge
        again. Charging twice would push a byproduct concern below an
        identical concern on a reagent, which is the opposite of the point.
        """
        rows = [(CLEAN, 0.1), (ALERT, 0.9)]
        as_byproduct = review_route_eligibility(
            report(CLEAN, ALERT), self.evidence(*rows))["routes"][0]
        as_reagent = review_route_eligibility(
            report(CLEAN, reagents=[ALERT]), self.evidence(*rows))["routes"][0]
        self.assertEqual(as_byproduct["rank_penalty"], as_reagent["rank_penalty"])
        self.assertGreater(as_byproduct["rank_penalty"], 0)

    def test_an_all_exclude_policy_still_cannot_remove_it(self):
        """The strictest shipped configuration does not get to empty the list."""
        every_kind = {"severity": {
            "known_concern": "exclude", "structural_alert": "exclude",
            "conditional_warning": "exclude", "model_positive": "exclude"}}
        result = review_route_eligibility(
            report(CLEAN, ALERT), self.evidence((CLEAN, 0.1), (ALERT, 0.9)),
            policy=every_kind)
        self.assertEqual(result["routes"][0]["status"], "needs_review")
        self.assertEqual(result["counts"]["excluded"], 0)

    def test_the_concern_costs_the_route_rank(self):
        """Demotion has to move the route, or it is only a label."""
        clean = review_route_eligibility(
            report(CLEAN), self.evidence((CLEAN, 0.1)))["routes"][0]
        concerning = review_route_eligibility(
            report(CLEAN, ALERT),
            self.evidence((CLEAN, 0.1), (ALERT, 0.9)))["routes"][0]
        self.assertGreater(concerning["rank_penalty"], clean["rank_penalty"])


if __name__ == "__main__":
    unittest.main()
