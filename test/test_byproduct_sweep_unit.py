"""The whole-route ASKCOS sweep: what it sends, what it skips, what it keeps.

Every structure here is a software fixture, never an AMES measurement and
never a toxicity determination. No network call is made: the predictor is
supplied by the test, and a test that let a real one through would be
sending structures to a public server.

The sweep replaces one button press per reaction with one tick in the
generation settings. The things that must not change when consent moves are
what gets sent, that a failure stays a failure rather than becoming "no
byproducts", and that nothing is sent for a route the policy already dropped.
"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import engine
import review_views
import workspace as context

TARGET = "CCOC(C)=O"
ALERT = "C1CO1"


def payload(*steps):
    """One route whose steps are (reactants, product) pairs."""
    return {"schema_version": "1.0", "routes": [{
        "id": "route-1", "target": TARGET,
        "source": "소프트웨어 시험용 · 실제 합성 근거 아님",
        "steps": [{
            "id": str(index), "reactants": list(reactants), "products": list(products),
            "reagents": [], "solvents": [],
            "conditions": {"status": "unknown"},
            "byproducts": {"status": "not_run", "candidates": []},
        } for index, (reactants, products) in enumerate(steps, 1)],
    }]}


def loaded(payload_dict):
    state = context.load_routes(context.new_workspace(TARGET), payload_dict)
    return state


class PlanTests(unittest.TestCase):
    def test_every_step_of_a_kept_route_is_planned_in_order(self):
        state = loaded(payload((["CCO", "CC(=O)O"], [TARGET]),
                               (["CCC"], ["CCO"])))
        plan = review_views.byproduct_plan(state)
        self.assertEqual([(entry["step_id"], entry["product"]) for entry in plan],
                         [("1", TARGET), ("2", "CCO")])

    def test_a_step_with_two_products_is_planned_without_one_so_it_can_be_skipped(self):
        state = loaded(payload((["CCO"], [TARGET, "CCC"])))
        self.assertIsNone(review_views.byproduct_plan(state)[0]["product"])

    def test_an_excluded_route_is_not_planned(self):
        state = loaded(payload((["CCO"], [TARGET])))
        state = deepcopy(state)
        for route in state["selection"]["routes"]:
            route["status"] = "excluded"
        self.assertEqual(review_views.byproduct_plan(state), [])

    def test_a_workspace_without_routes_plans_nothing(self):
        self.assertEqual(review_views.byproduct_plan({}), [])


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.state = loaded(payload((["CCO", "CC(=O)O"], [TARGET]),
                                    (["CCC"], ["CCO"])))
        progress = mock.patch.object(review_views.st, "progress")
        self.progress = progress.start()
        self.addCleanup(progress.stop)
        forbidden = mock.patch.object(
            engine, "predict_byproducts",
            side_effect=AssertionError("the sweep must use the injected predictor"))
        self.forbidden = forbidden.start()
        self.addCleanup(forbidden.stop)

    def _ok(self, *candidates):
        def predict(reactants, product, timeout=None):
            self.sent.append((reactants, product))
            return {"status": "success", "source": engine.ASKCOS_API, "error": None,
                    "candidates": [{"smiles": smiles, "mode": "test"}
                                   for smiles in candidates],
                    "limitations": ["소프트웨어 시험용"]}
        self.sent = []
        return predict

    def test_each_reaction_is_sent_once_with_its_own_reactants_and_product(self):
        predict = self._ok()
        state, skipped = review_views.sweep_byproducts(self.state, predict=predict)
        self.assertEqual(self.sent, [("CCO.CC(=O)O", TARGET), ("CCC", "CCO")])
        self.assertEqual(skipped, [])
        self.assertEqual(len(state["byproduct_attempts"]), 2)
        self.forbidden.assert_not_called()

    def test_a_returned_candidate_reaches_the_report(self):
        state, _ = review_views.sweep_byproducts(self.state, predict=self._ok(ALERT))
        found = [material["input_smiles"]
                 for route in state["report"]["routes"]
                 for step in route["steps"] for material in step["materials"]
                 if material["role"] == "byproducts"]
        self.assertIn(ALERT, found)

    def test_a_failed_reaction_is_recorded_and_the_others_still_run(self):
        calls = []

        def predict(reactants, product, timeout=None):
            calls.append(product)
            if product == TARGET:
                raise TimeoutError("공개 서버 응답 없음")
            return {"status": "success", "source": engine.ASKCOS_API, "error": None,
                    "candidates": [], "limitations": []}

        state, skipped = review_views.sweep_byproducts(self.state, predict=predict)
        self.assertEqual(calls, [TARGET, "CCO"])
        self.assertEqual(skipped, [])
        attempts = state["byproduct_attempts"]
        self.assertEqual(len(attempts), 2)
        # The failure stays a failure. Nothing here may read as "no byproducts".
        self.assertFalse(attempts[0]["applied"])
        self.assertIn("공개 서버 응답 없음", str(attempts[0]["result"]))

    def test_a_two_product_step_is_skipped_with_a_reason_and_nothing_is_sent(self):
        state = loaded(payload((["CCO"], [TARGET, "CCC"])))
        predict = self._ok()
        _, skipped = review_views.sweep_byproducts(state, predict=predict)
        self.assertEqual(self.sent, [])
        self.assertEqual(len(skipped), 1)
        self.assertIn("주생성물이 하나가 아니", skipped[0]["reason"])

    def test_the_selection_comes_back_to_where_it_started(self):
        before = (self.state["route_id"], self.state["step_id"])
        state, _ = review_views.sweep_byproducts(self.state, predict=self._ok())
        self.assertEqual((state["route_id"], state["step_id"]), before)

    def test_nothing_is_sent_when_there_is_nothing_to_ask_about(self):
        predict = self._ok()
        state, skipped = review_views.sweep_byproducts(
            {"selection": {"routes": []}, "active_input": {"routes": []}},
            predict=predict)
        self.assertEqual(self.sent, [])
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
