"""Synthetic policy fixtures for the replaceable screening policy, not evidence.

Every route and call here is a software fixture. Nothing in this file is an
AMES measurement, a toxicity determination or a regulatory judgement.
"""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "scripts"))
import engine
import screening_judges
import screening_policy
from ames_route_review import review_ames_routes, review_route_eligibility, review_routes_multi
from compare_tools import combine as reference_combine
from screening_policy import (
    COMBINATION_RULES, DEFAULT_POLICY, SEVERITY_KINDS, combine, is_ranking_active,
    position_weight, validate_policy,
)
from toxicology import canonicalize_smiles


TARGET = "CCOC(C)=O"
# The severity policy shipped before the default softened: every kind of
# evidence deleted a route. Kept as an explicit configuration so that old
# behaviour stays reachable and under test.
ALL_EXCLUDE = {"severity": {kind: "exclude" for kind in SEVERITY_KINDS}}


def _payload(*steps):
    """One route whose steps each carry one supplied non-target material."""
    return {"schema_version": "1.0", "routes": [{
        "id": "route-1", "target": TARGET, "source": "Software-only synthetic fixture",
        "steps": [
            {"id": f"step-{index}", "reactants": [material], "products": [TARGET],
             "reagents": [], "solvents": [],
             "conditions": {"status": "observed", "source": "Software-only fixture",
                            "description": "Synthetic branch fixture; not a laboratory observation."},
             "byproducts": {"status": "success", "candidates": [],
                            "source": "Synthetic completed-attempt label; not a real prediction."}}
            for index, material in enumerate(steps, 1)
        ],
    }]}


def _report(*steps):
    return engine.analyze_routes(_payload(*steps))


def _byproduct_report(material, *byproducts):
    """One route whose single step also carries predicted byproducts."""
    payload = _payload(material)
    payload["routes"][0]["steps"][0]["byproducts"] = {
        "status": "success", "candidates": list(byproducts),
        "source": "Synthetic completed-attempt label; not a real prediction.",
    }
    return engine.analyze_routes(payload)


def _row(smiles, score=0.2, *, status=None, threshold=0.5):
    canonical = canonicalize_smiles(smiles)
    if status is None:
        status = "success" if canonical is not None else "invalid"
    return {
        "canonical_smiles": canonical, "input_smiles": smiles, "status": status,
        "score": score if status == "success" else None,
        "prediction": ("positive" if score >= threshold else "negative") if status == "success" else "abstain",
        "applicability": "unknown", "error": None if status == "success" else "Synthetic abstention",
    }


def _admet(*rows, threshold=0.5, digest="a"):
    return {
        "model": {"tool": "ADMET-AI", "version": "synthetic-fixture", "endpoint": "ames",
                  "weights_sha256": {"fixture.pt": digest * 64}},
        "threshold": threshold, "assessments": list(rows),
        "limitations": ["Supplied synthetic model limitation; no empirical validation."],
    }


def _codes(route):
    return {reason["code"] for reason in route["reasons"]}


class CombinationRuleTests(unittest.TestCase):
    """The three combination rules must mean exactly what the experiment means."""

    CASES = (
        ("1",), ("0",), ("abstain",),
        ("1", "1"), ("1", "0"), ("0", "0"), ("1", "abstain"), ("0", "abstain"),
        ("abstain", "abstain"),
        ("1", "1", "0"), ("1", "0", "0"), ("1", "1", "1"), ("0", "0", "0"),
        ("1", "0", "abstain"), ("1", "1", "abstain"), ("0", "0", "abstain"),
        ("1", "abstain", "abstain"), ("abstain", "abstain", "abstain"),
    )

    def test_every_rule_matches_the_experiment_combine_on_every_call_pattern(self):
        for rule in COMBINATION_RULES:
            for calls in self.CASES:
                with self.subTest(rule=rule, calls=calls):
                    self.assertEqual(combine(calls, rule), reference_combine(list(calls), rule))

    def test_any_positive_never_calls_negative_while_a_judge_abstains(self):
        self.assertEqual(combine(["1", "abstain"], "any_positive"), "1")
        self.assertEqual(combine(["0", "abstain"], "any_positive"), "abstain")
        self.assertEqual(combine(["0", "0"], "any_positive"), "0")

    def test_unanimous_requires_every_judge_to_decide_and_to_agree(self):
        self.assertEqual(combine(["1", "1"], "unanimous"), "1")
        self.assertEqual(combine(["0", "0"], "unanimous"), "0")
        self.assertEqual(combine(["1", "0"], "unanimous"), "abstain")
        self.assertEqual(combine(["1", "abstain"], "unanimous"), "abstain")

    def test_majority_abstains_on_a_tie_and_on_any_abstention(self):
        self.assertEqual(combine(["1", "1", "0"], "majority"), "1")
        self.assertEqual(combine(["1", "0", "0"], "majority"), "0")
        self.assertEqual(combine(["1", "0"], "majority"), "abstain")
        self.assertEqual(combine(["1", "1", "abstain"], "majority"), "abstain")

    def test_unknown_rules_and_unknown_calls_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "조합 규칙"):
            combine(["1"], "best_of")
        with self.assertRaisesRegex(ValueError, "판정 결과"):
            combine(["positive"], "any_positive")


class PolicyValidationTests(unittest.TestCase):
    def test_default_policy_contents_are_the_shipped_behaviour(self):
        self.assertEqual(DEFAULT_POLICY, {
            "tools": ["rules", "admet"],
            "combination": "any_positive",
            "thresholds": {"admet": 0.5},
            "severity": {"known_concern": "exclude", "structural_alert": "demote",
                         "conditional_warning": "demote", "model_positive": "demote"},
            "position_weighting": False,
        })
        self.assertEqual(validate_policy(None), DEFAULT_POLICY)
        # Only a named ICH M7 material deletes a route; the three graded kinds
        # are demoted instead, so the default ranks and carries rank_penalty.
        self.assertTrue(is_ranking_active(validate_policy(None)))
        self.assertFalse(is_ranking_active(validate_policy(ALL_EXCLUDE)))

    def test_validation_returns_a_copy_and_never_mutates_the_default(self):
        before = copy.deepcopy(DEFAULT_POLICY)
        validated = validate_policy({"combination": "majority"})
        validated["tools"].append("rules")
        validated["severity"]["known_concern"] = "demote"
        self.assertEqual(DEFAULT_POLICY, before)
        self.assertEqual(validate_policy(None)["severity"]["known_concern"], "exclude")

    def test_unknown_keys_ids_rules_and_thresholds_are_rejected_not_defaulted(self):
        for policy, pattern in (
            ({"tool": ["rules"]}, "정책 키"),
            ({"combination": "any"}, "조합 규칙"),
            ({"combination": None}, "조합 규칙"),
            ({"tools": ["rules", "nope"]}, "판정기 id"),
            ({"tools": []}, "tools"),
            ({"tools": ["rules", "rules"]}, "중복"),
            ({"thresholds": {"unknown_tool": 0.5}}, "판정기 id"),
            ({"thresholds": {"admet": 1.5}}, "임계값"),
            ({"thresholds": {"admet": True}}, "임계값"),
            ({"severity": {"unknown_kind": "exclude"}}, "근거 종류"),
            ({"severity": {"known_concern": "ignore"}}, "처리"),
            ({"position_weighting": "yes"}, "bool"),
        ):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, pattern):
                validate_policy(policy)

    def test_unintegrated_judges_have_a_slot_but_cannot_be_enabled_or_run(self):
        registered = {"rules", "admet", "ames_xgb", "ames_desc", "toxtree",
                      "epa_test", "qsar_toolbox"}
        self.assertEqual(set(screening_judges.judge_ids()), registered)
        # Every registered slot now has execution code behind it. The slot
        # mechanism stays because a tool can be added before it is wired, and
        # the guard below is what keeps an unwired slot from inventing a call.
        self.assertEqual(set(screening_judges.implemented_judge_ids()), registered)
        for judge_id in sorted(registered):
            with self.subTest(judge=judge_id):
                self.assertTrue(screening_judges.get_judge(judge_id)["implemented"])
                # Named on its own: the validator rejects a repeated tool, so
                # pairing every id with "rules" would fail on "rules" itself.
                self.assertEqual(validate_policy({"tools": [judge_id]})["tools"],
                                 [judge_id])
        with self.assertRaisesRegex(ValueError, "판정기 id"):
            screening_judges.get_judge("toxtree-v3")

    def test_an_unwired_slot_refuses_to_run_rather_than_inventing_a_call(self):
        original = dict(screening_judges.JUDGES)
        screening_judges.JUDGES["pretend"] = {
            "tool": "Pretend", "kind": "model", "implemented": False,
            "uses_threshold": False,
            "run": screening_judges._unintegrated("pretend", "Pretend")}
        try:
            self.assertNotIn("pretend", screening_judges.implemented_judge_ids())
            with self.assertRaisesRegex(ValueError, "연동되지 않았습니다"):
                validate_policy({"tools": ["rules", "pretend"]})
            with self.assertRaisesRegex(NotImplementedError, "연동되지 않았습니다"):
                screening_judges.run_judge("pretend", ["CCO"])
        finally:
            screening_judges.JUDGES.clear()
            screening_judges.JUDGES.update(original)


class CombinationInRouteReviewTests(unittest.TestCase):
    def test_default_policy_matches_an_explicitly_named_any_positive_union(self):
        for material in ("CCO", "C1CO1", "CC=O"):
            for score in (0.2, 0.8):
                with self.subTest(material=material, score=score):
                    report = _report(material)
                    evidence = _admet(_row(material, score))
                    shipped = review_route_eligibility(report, evidence)
                    explicit = review_route_eligibility(
                        report, evidence, policy={"combination": "any_positive"})
                    self.assertEqual(shipped["routes"][0], explicit["routes"][0])
                    # The default demotes, so every route carries a ranking score.
                    self.assertIn("rank_penalty", shipped["routes"][0])

    def test_unanimous_and_majority_change_the_verdict_on_disagreement(self):
        # rule alert "1" plus model "1" plus model "0": any_positive and
        # majority both call the material positive, unanimous cannot decide.
        # What a positive then does to the route is the severity's business:
        # a structural alert demotes, a named ICH M7 material excludes.
        expected = {
            "C1CO1": {"any_positive": "needs_review", "unanimous": "insufficient_data",
                      "majority": "needs_review"},
            "CC=O": {"any_positive": "excluded", "unanimous": "insufficient_data",
                     "majority": "excluded"},
        }
        for material, by_rule in expected.items():
            report = _report(material)
            both = [_admet(_row(material, 0.8)), _admet(_row(material, 0.2), digest="b")]
            verdicts = {}
            for rule in COMBINATION_RULES:
                with self.subTest(material=material, rule=rule):
                    result = review_routes_multi(report, both, excluded_smiles=(), strict_alerts=True,
                                                 union=True, policy={"combination": rule})
                    verdicts[rule] = result["routes"][0]["status"]
                    self.assertEqual(result["policy"]["screening_policy"]["combination"], rule)
                    self.assertEqual(len(result["policy"]["admet_models"]), 2)
            self.assertEqual(verdicts, by_rule)

    def test_unanimous_abstains_instead_of_excluding_on_a_lone_model_positive(self):
        report = _report("CCO")
        evidence = [_admet(_row("CCO", 0.8)), _admet(_row("CCO", 0.2), digest="b")]
        result = review_routes_multi(report, evidence, excluded_smiles=(), strict_alerts=True,
                                     union=True, policy={"combination": "unanimous"})
        route = result["routes"][0]
        self.assertEqual(route["status"], "insufficient_data")
        self.assertTrue({"admet_positive", "admet_negative"} <= _codes(route))
        self.assertNotIn("current_model_positive_exclusion", _codes(route))

    def test_a_single_envelope_is_identical_through_the_wrapper_and_the_list(self):
        report = _report("C1CO1")
        evidence = _admet(_row("C1CO1", 0.8))
        self.assertEqual(
            review_route_eligibility(report, evidence),
            review_routes_multi(report, [evidence], excluded_smiles=(),
                                strict_alerts=True, union=True),
        )
        self.assertEqual(
            review_ames_routes(report, evidence),
            review_routes_multi(report, [evidence], excluded_smiles=(),
                                strict_alerts=False, union=False),
        )

    def test_an_abstaining_model_list_never_becomes_a_negative(self):
        report = _report("CCO")
        evidence = [_admet(_row("CCO", status="failed")), _admet(digest="b")]
        for rule in COMBINATION_RULES:
            with self.subTest(rule=rule):
                route = review_routes_multi(report, evidence, excluded_smiles=(),
                                            strict_alerts=True, union=True,
                                            policy={"combination": rule})["routes"][0]
                self.assertEqual(route["status"], "insufficient_data")
                self.assertNotIn("admet_negative", _codes(route))
                self.assertNotIn("candidate_within_scope", _codes(route))


class SeverityTests(unittest.TestCase):
    def test_default_severity_excludes_a_named_ich_m7_material_in_every_case(self):
        for evidence in (_admet(_row("CC=O", 0.8)), _admet(_row("CC=O", 0.2)), _admet(), None):
            with self.subTest(evidence=evidence):
                route = review_route_eligibility(_report("CC=O"), evidence)["routes"][0]
                self.assertEqual(route["status"], "excluded")
                self.assertIn("known_concern", _codes(route))

    def test_default_severity_retains_a_route_whose_only_signal_is_an_alert(self):
        result = review_route_eligibility(_report("C1CO1"), _admet(_row("C1CO1", 0.2)))
        route = result["routes"][0]
        self.assertEqual(route["status"], "needs_review")
        self.assertIn("structural_alert", _codes(route))
        self.assertEqual(route["rank_penalty"], 1.0)
        self.assertEqual(result["counts"],
                         {"excluded": 0, "needs_review": 1, "insufficient_data": 0, "candidate": 0})

    def test_default_severity_retains_a_route_whose_only_signal_is_the_model(self):
        result = review_route_eligibility(_report("CCO"), _admet(_row("CCO", 0.8)))
        route = result["routes"][0]
        self.assertEqual(route["status"], "needs_review")
        self.assertIn("admet_positive", _codes(route))
        self.assertNotIn("current_model_positive_exclusion", _codes(route))
        self.assertEqual(route["rank_penalty"], 1.0)
        self.assertEqual(result["counts"],
                         {"excluded": 0, "needs_review": 1, "insufficient_data": 0, "candidate": 0})

    def test_default_severity_ranks_a_retained_route_higher_with_fewer_signals(self):
        cases = (("CCO", 0.2, 0.0), ("CCO", 0.8, 1.0), ("C1CO1", 0.2, 1.0), ("C1CO1", 0.8, 2.0))
        penalties = []
        for material, score, expected in cases:
            with self.subTest(material=material, score=score):
                route = review_route_eligibility(_report(material),
                                                 _admet(_row(material, score)))["routes"][0]
                self.assertEqual(route["rank_penalty"], expected)
                penalties.append(route["rank_penalty"])
        self.assertEqual(penalties, sorted(penalties))
        self.assertLess(penalties[0], penalties[-1])

    def test_a_byproduct_alert_or_model_positive_does_not_exclude_the_route(self):
        report = _byproduct_report("CCO", "C1CO1")
        result = review_route_eligibility(report, _admet(_row("CCO", 0.2), _row("C1CO1", 0.8)))
        route = result["routes"][0]
        self.assertEqual(route["status"], "needs_review")
        self.assertTrue({"structural_alert", "admet_positive"} <= _codes(route))
        self.assertEqual(result["counts"]["excluded"], 0)
        self.assertEqual([entry["id"] for entry in result["routes"]],
                         [entry["id"] for entry in report["routes"]])

    def test_the_old_all_exclude_behaviour_is_reproducible_by_configuration(self):
        for material in ("CC=O", "C1CO1", "CCO"):
            with self.subTest(material=material):
                report = _report(material)
                evidence = _admet(_row(material, 0.8))
                route = review_route_eligibility(report, evidence, policy=ALL_EXCLUDE)["routes"][0]
                self.assertEqual(route["status"], "excluded")
                self.assertEqual(set(route), {"id", "status", "reasons"})
                self.assertIn("current_model_positive_exclusion", _codes(route))
                self.assertEqual(review_route_eligibility(report, evidence)["routes"][0]["status"],
                                 "excluded" if material == "CC=O" else "needs_review")

    def test_the_old_all_exclude_behaviour_also_covers_a_conditional_warning(self):
        payload = _payload("CNC")
        payload["routes"][0]["steps"][0]["reagents"] = ["[Na+].[O-]N=O"]
        report = engine.analyze_routes(payload)
        strict = review_route_eligibility(report, policy=ALL_EXCLUDE)["routes"][0]
        self.assertEqual(strict["status"], "excluded")
        self.assertIn("conditional_warning", _codes(strict))
        default = review_route_eligibility(report)["routes"][0]
        self.assertEqual(default["status"], "needs_review")
        self.assertIn("conditional_warning", _codes(default))

    def test_demoted_structural_alert_keeps_the_route_and_its_reason(self):
        report = _report("C1CO1")
        policy = {"severity": {"structural_alert": "demote"}}
        route = review_route_eligibility(report, _admet(_row("C1CO1", 0.2)),
                                         policy=policy)["routes"][0]
        self.assertEqual(route["status"], "needs_review")
        self.assertIn("structural_alert", _codes(route))
        self.assertEqual(route["rank_penalty"], 1.0)
        excluded = review_route_eligibility(report, _admet(_row("C1CO1", 0.2)),
                                            policy=ALL_EXCLUDE)["routes"][0]
        self.assertEqual(excluded["status"], "excluded")

    def test_demoted_model_positive_keeps_the_route_without_claiming_exclusion(self):
        report = _report("CCO")
        evidence = _admet(_row("CCO", 0.8))
        policy = {"severity": {"model_positive": "demote"}}
        route = review_route_eligibility(report, evidence, policy=policy)["routes"][0]
        self.assertEqual(route["status"], "needs_review")
        self.assertIn("admet_positive", _codes(route))
        self.assertNotIn("current_model_positive_exclusion", _codes(route))
        self.assertEqual(route["rank_penalty"], 1.0)

    def test_known_concern_still_excludes_while_a_weaker_kind_is_demoted(self):
        policy = {"severity": {"structural_alert": "demote", "model_positive": "demote"}}
        route = review_route_eligibility(_report("CC=O"), _admet(_row("CC=O", 0.8)),
                                         policy=policy)["routes"][0]
        self.assertEqual(route["status"], "excluded")
        self.assertIn("known_concern", _codes(route))

    def test_demotion_preserves_excluded_routes_and_all_of_their_reasons(self):
        report = _report("CC=O", "C1CO1")
        policy = {"severity": {"structural_alert": "demote"}}
        result = review_route_eligibility(report, _admet(_row("CC=O", 0.8), _row("C1CO1", 0.2)),
                                          policy=policy)
        route = result["routes"][0]
        self.assertEqual(route["status"], "excluded")
        self.assertTrue({"known_concern", "structural_alert"} <= _codes(route))
        self.assertEqual(result["counts"]["excluded"], 1)

    def test_demoted_same_material_and_comparison_stops_excluding(self):
        report = _report("C1CO1")
        evidence = _admet(_row("C1CO1", 0.8))
        self.assertEqual(review_ames_routes(report, evidence,
                                            policy=ALL_EXCLUDE)["routes"][0]["status"], "excluded")
        route = review_ames_routes(report, evidence,
                                   policy={"severity": {"model_positive": "demote"}})["routes"][0]
        self.assertEqual(route["status"], "needs_review")
        self.assertNotIn("rule_admet_joint_positive", _codes(route))
        self.assertIn("structural_alert", _codes(route))


class PositionWeightingTests(unittest.TestCase):
    def test_weighting_is_off_by_default_and_never_discounts_an_early_step(self):
        policy = validate_policy(None)
        self.assertFalse(policy["position_weighting"])
        # The alert sits in the first of two steps, so position weighting would
        # halve it. The default must leave every piece of evidence at 1.0.
        report = _report("C1CO1", "CCO")
        evidence = _admet(_row("C1CO1", 0.2), _row("CCO", 0.2))
        self.assertEqual(review_route_eligibility(report, evidence)["routes"][0]["rank_penalty"], 1.0)
        weighted = review_route_eligibility(report, evidence, policy={"position_weighting": True})
        self.assertEqual(weighted["routes"][0]["rank_penalty"], 0.5)

    def test_weight_formula_is_the_linear_step_position_ramp(self):
        self.assertEqual(position_weight(0, 1), 1.0)
        self.assertEqual(position_weight(2, 3), 1.0)
        self.assertEqual(position_weight(0, 2), 0.5)
        self.assertEqual(position_weight(None, 3), 1.0)
        with self.assertRaisesRegex(ValueError, "단계 색인"):
            position_weight(3, 3)

    def test_the_same_alert_weighs_more_in_a_later_step_when_weighting_is_on(self):
        early = _report("C1CO1", "CCO")
        late = _report("CCO", "C1CO1")
        evidence = _admet(_row("C1CO1", 0.2), _row("CCO", 0.2))
        policy = {"position_weighting": True, "severity": {"structural_alert": "demote"}}
        penalties = [review_route_eligibility(report, evidence, policy=policy)["routes"][0]["rank_penalty"]
                     for report in (early, late)]
        self.assertEqual(penalties, [0.5, 1.0])
        self.assertLess(penalties[0], penalties[1])

    def test_weighting_alone_ranks_without_changing_any_route_status(self):
        report = _report("CCO", "C1CO1")
        evidence = _admet(_row("CCO", 0.2), _row("C1CO1", 0.2))
        shipped = review_route_eligibility(report, evidence)
        weighted = review_route_eligibility(report, evidence, policy={"position_weighting": True})
        self.assertEqual([route["status"] for route in weighted["routes"]],
                         [route["status"] for route in shipped["routes"]])
        self.assertEqual(weighted["counts"], shipped["counts"])
        self.assertEqual(weighted["routes"][0]["rank_penalty"], 1.0)

    def test_model_positives_are_weighted_by_the_step_they_appear_in(self):
        report = _report("CCO", "CCC")
        evidence = _admet(_row("CCO", 0.8), _row("CCC", 0.8))
        policy = {"position_weighting": True, "severity": {"model_positive": "demote"}}
        route = review_route_eligibility(report, evidence, policy=policy)["routes"][0]
        self.assertEqual(route["rank_penalty"], 1.5)
        self.assertEqual(route["status"], "needs_review")


class JudgeRegistryTests(unittest.TestCase):
    def test_the_rule_judge_returns_calls_in_the_shared_shape(self):
        result = screening_judges.run_judge("rules", ["CC=O", "C1CO1", "CCO", "not-a-molecule"])
        self.assertEqual(result["judge"], "rules")
        self.assertEqual([row["call"] for row in result["calls"]], ["1", "1", "0", "abstain"])
        self.assertEqual([row["kind"] for row in result["calls"]],
                         ["known_concern", "structural_alert", None, None])
        self.assertEqual(result["evidence"]["tool"], "local-rule-screen")
        self.assertTrue(result["evidence"]["source"])
        for row in result["calls"]:
            self.assertEqual(set(row), {"input_smiles", "canonical_smiles", "call", "score",
                                        "kind", "error"})
            self.assertIsNone(row["score"])

    def test_invalid_material_abstains_and_is_never_reported_negative(self):
        row = screening_judges.run_judge("rules", ["not-a-molecule"])["calls"][0]
        self.assertEqual(row["call"], "abstain")
        self.assertIsNone(row["canonical_smiles"])
        self.assertNotEqual(row["call"], "0")

    def test_the_model_judge_abstains_without_running_the_backend_on_invalid_input(self):
        result = screening_judges.run_judge("admet", ["not-a-molecule"], threshold=0.5)
        self.assertEqual(result["judge"], "admet")
        self.assertEqual([row["call"] for row in result["calls"]], ["abstain"])
        self.assertEqual(result["evidence"]["tool"], "ADMET-AI")
        self.assertEqual(result["evidence"]["threshold"], 0.5)
        self.assertIn("model_result", result)

    def test_the_threshold_a_policy_sets_reaches_the_model_judge(self):
        self.assertEqual(screening_policy.model_threshold(validate_policy(None)), 0.5)
        self.assertEqual(
            screening_policy.model_threshold(validate_policy({"thresholds": {"admet": 0.3}})), 0.3)


if __name__ == "__main__":
    unittest.main()
