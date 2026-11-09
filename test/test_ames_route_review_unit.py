"""Synthetic policy/validation fixtures, not empirical AMES or safety evidence."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import ames_route_review
import engine
from ames_route_review import review_ames_routes, review_route_eligibility
from route_selection import select_routes
from toxicology import canonicalize_smiles


TARGET = "CCOC(C)=O"
# The severity policy shipped before the default softened: every kind of
# evidence deleted a route. Kept as an explicit configuration so that old
# behaviour, and the reasons it emits, stay reachable and under test.
ALL_EXCLUDE = {"severity": {"known_concern": "exclude", "structural_alert": "exclude",
                            "conditional_warning": "exclude", "model_positive": "exclude"}}


def _payload(*materials):
    return {"schema_version": "1.0", "routes": [
        {"id": f"route-{index}", "target": TARGET, "source": "Software-only synthetic fixture",
         "steps": [{"id": "step-1", "reactants": [material], "products": [TARGET],
                    "reagents": [], "solvents": [],
                    "conditions": {"status": "observed", "source": "Software-only fixture",
                                   "description": "Synthetic branch fixture; not a laboratory observation."},
                    "byproducts": {"status": "success", "candidates": [],
                                   "source": "Synthetic completed-attempt label; not a real prediction."}}]}
        for index, material in enumerate(materials, 1)
    ]}


def _report(*materials):
    return engine.analyze_routes(_payload(*materials))


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


def _admet(*rows, threshold=0.5):
    return {
        "model": {"tool": "ADMET-AI", "version": "synthetic-fixture", "endpoint": "ames",
                  "weights_sha256": {"fixture.pt": "a" * 64}},
        "threshold": threshold, "assessments": list(rows),
        "limitations": ["Supplied synthetic model limitation; no empirical validation."],
    }


def _codes(route):
    return {reason["code"] for reason in route["reasons"]}


class AmesRoutePolicyTests(unittest.TestCase):
    def test_repeated_invalid_service_input_preserves_all_route_occurrences(self):
        from admet_service import assess_admet_materials
        report = _report("CC(", "CC(", "bad-smiles")
        with patch("admet_service.subprocess.run") as worker:
            admet = assess_admet_materials(["CC(", "CC(", "bad-smiles"])
        worker.assert_not_called()
        self.assertEqual(len(admet["assessments"]), 2)
        result = review_ames_routes(report, admet)
        self.assertEqual(result["counts"]["insufficient_data"], 3)
        self.assertEqual([route["id"] for route in result["routes"]],
                         [route["id"] for route in report["routes"]])
        for route in result["routes"]:
            self.assertIn("admet_unassessed", _codes(route))

    def test_mixed_invalid_service_sources_supply_unique_rows_for_all_route_occurrences(self):
        from admet_service import assess_admet_materials
        materials = ("None", "None", "12", "CC(")
        report = _report(*materials)
        with patch("admet_service.subprocess.run") as worker:
            admet = assess_admet_materials([None, None, "None", 12, "12", "CC("])
        worker.assert_not_called()
        self.assertEqual([(row["input_smiles"], row["input_types"]) for row in admet["assessments"]],
                         [("None", ["NoneType", "str"]), ("12", ["int", "str"]), ("CC(", ["str"])])
        result = review_ames_routes(report, admet)
        self.assertEqual(result["counts"],
                         {"excluded": 0, "needs_review": 0, "insufficient_data": 4, "candidate": 0})
        self.assertEqual([route["id"] for route in result["routes"]],
                         [route["id"] for route in report["routes"]])
        for route, material in zip(result["routes"], materials):
            unassessed = [reason for reason in route["reasons"] if reason["code"] == "admet_unassessed"]
            self.assertEqual(len(unassessed), 1)
            self.assertEqual(unassessed[0]["smiles"], material)
            self.assertIn("invalid", unassessed[0]["message"])
            self.assertNotIn("admet_negative", _codes(route))

    def test_success_requires_weight_provenance_but_failure_remains_abstention(self):
        report = _report("CCO")
        result = _admet(_row("CCO"))
        result["model"]["weights_sha256"] = {}
        with self.assertRaisesRegex(ValueError, "weight provenance"):
            review_ames_routes(report, result)
        result["assessments"] = [_row("CCO", status="failed")]
        self.assertEqual(review_ames_routes(report, result)["routes"][0]["status"], "insufficient_data")

    def test_rule_and_model_truth_table_including_known_and_custom_alerts(self):
        # The same-material AND comparison excludes only when both the rule
        # kind and model_positive are configured to delete a route. Under the
        # default, where model_positive only demotes, the joint finding is
        # reported for expert review instead of removing the route.
        for material, rule_code in (("CCO", None), ("C1CO1", "structural_alert"), ("CC=O", "known_concern")):
            for score in (0.2, 0.8):
                for policy in (None, ALL_EXCLUDE):
                    with self.subTest(material=material, score=score, policy=policy):
                        route = review_ames_routes(_report(material), _admet(_row(material, score)),
                                                   policy=policy)["routes"][0]
                        joint = bool(rule_code) and score >= 0.5
                        expected = ("excluded" if joint and policy is ALL_EXCLUDE
                                    else "needs_review" if rule_code or score >= 0.5 else "candidate")
                        self.assertEqual(route["status"], expected)
                        self.assertIn("admet_positive" if score >= 0.5 else "admet_negative", _codes(route))
                        self.assertEqual("rule_admet_joint_positive" in _codes(route), expected == "excluded")
                        self.assertEqual("candidate_within_scope" in _codes(route), expected == "candidate")
                        if rule_code:
                            baseline = select_routes(_report(material))["routes"][0]["reasons"]
                            self.assertTrue(all(reason in route["reasons"] for reason in baseline))

    def test_all_statuses_exact_contract_and_original_ids(self):
        report = _report("CCO", "CCC", "not-a-molecule", "C1CO1")
        report["routes"][1]["id"] = "  supplied-id  "
        admet = _admet(_row("CCO"), _row("CCC", 0.8), _row("not-a-molecule"), _row("C1CO1", 0.8))
        result = review_ames_routes(report, admet)
        self.assertEqual(set(result), {"routes", "counts", "policy", "limitations"})
        self.assertEqual([route["id"] for route in result["routes"]], [route["id"] for route in report["routes"]])
        self.assertEqual([route["status"] for route in result["routes"]],
                         ["candidate", "needs_review", "insufficient_data", "needs_review"])
        self.assertEqual(result["counts"], {"excluded": 0, "needs_review": 2, "insufficient_data": 1, "candidate": 1})
        for route in result["routes"]:
            # The default demotes, so a ranking score joins the route contract.
            self.assertEqual(set(route), {"id", "status", "reasons", "rank_penalty"})
            for reason in route["reasons"]:
                self.assertEqual(set(reason), {"code", "message", "step_id", "smiles"})
                self.assertTrue(reason["message"])
        strict = review_ames_routes(report, admet, policy=ALL_EXCLUDE)
        self.assertEqual([route["status"] for route in strict["routes"]],
                         ["candidate", "needs_review", "insufficient_data", "excluded"])
        self.assertEqual(strict["counts"],
                         {"excluded": 1, "needs_review": 1, "insufficient_data": 1, "candidate": 1})
        for route in strict["routes"]:
            self.assertEqual(set(route), {"id", "status", "reasons"})
        self.assertEqual(result["policy"]["rule_admet_exclusion"], "same_material_AND")
        self.assertFalse(result["policy"]["strict_alerts"])
        self.assertTrue(result["policy"]["research_only"])
        self.assertEqual(result["policy"]["admet_threshold"], 0.5)
        self.assertEqual(result["policy"]["admet_model"], admet["model"])

    def test_explicit_ban_overrides_both_negative_and_missing_model(self):
        report = _report("CCO")
        for admet in (_admet(_row("CCO")), _admet()):
            with self.subTest(admet=admet):
                result = review_ames_routes(report, admet, excluded_smiles=["OCC", "CCO"])
                route = result["routes"][0]
                self.assertEqual(route["status"], "excluded")
                self.assertIn("prohibited_material", _codes(route))
                self.assertNotIn("candidate_within_scope", _codes(route))
                self.assertEqual(result["policy"]["excluded_smiles"], ["CCO"])
                baseline = select_routes(report, excluded_smiles=["CCO"])["routes"][0]
                self.assertTrue(all(reason in route["reasons"] for reason in baseline["reasons"]))

    def test_prohibited_component_retains_baseline_matching_policy(self):
        smiles = "CCO.[Na+]"
        route = review_ames_routes(_report(smiles), _admet(_row(smiles)), excluded_smiles=["CCO"])["routes"][0]
        self.assertEqual(route["status"], "excluded")
        self.assertIn("prohibited_component", _codes(route))

    def test_rule_and_model_positives_must_be_on_the_same_material(self):
        payload = _payload("C1CO1")
        payload["routes"][0]["steps"][0]["reagents"] = ["CCO"]
        report = engine.analyze_routes(payload)
        result = review_ames_routes(report, _admet(_row("C1CO1", 0.2), _row("CCO", 0.8)))
        route = result["routes"][0]
        self.assertEqual(route["status"], "needs_review")
        self.assertTrue({"structural_alert", "admet_positive", "admet_negative"} <= _codes(route))
        self.assertNotIn("rule_admet_joint_positive", _codes(route))

    def test_conditional_warning_does_not_invent_a_structure_or_joint_positive(self):
        for score in (0.2, 0.8):
            with self.subTest(score=score):
                payload = _payload("CNC")
                nitrite = "[Na+].[O-]N=O"
                payload["routes"][0]["steps"][0]["reagents"] = [nitrite]
                route = review_ames_routes(engine.analyze_routes(payload),
                                           _admet(_row("CNC", score), _row(nitrite)))["routes"][0]
                self.assertEqual(route["status"], "needs_review")
                self.assertIn("conditional_warning", _codes(route))
                self.assertNotIn("rule_admet_joint_positive", _codes(route))
                warning = next(reason for reason in route["reasons"] if reason["code"] == "conditional_warning")
                self.assertIsNone(warning["smiles"])
                self.assertEqual({reason["smiles"] for reason in route["reasons"] if reason["code"].startswith("admet_")},
                                 {canonicalize_smiles("CNC"), canonicalize_smiles(nitrite)})

    def test_missing_failed_and_invalid_model_evidence_never_becomes_negative(self):
        cases = (("CCO", _admet()), ("CCO", _admet(_row("CCO", status="failed"))),
                 ("not-a-molecule", _admet(_row("not-a-molecule"))), ("not-a-molecule", _admet()))
        for material, admet in cases:
            with self.subTest(material=material, admet=admet):
                route = review_ames_routes(_report(material), admet)["routes"][0]
                self.assertEqual(route["status"], "insufficient_data")
                self.assertIn("admet_unassessed", _codes(route))
                self.assertNotIn("admet_negative", _codes(route))
                self.assertNotIn("candidate_within_scope", _codes(route))

    def test_all_abstain_and_unknown_baseline_data_has_no_candidates(self):
        payload = _payload("CCO", "C1CO1")
        for route in payload["routes"]:
            route["steps"][0]["conditions"] = {"status": "unknown"}
            route["steps"][0]["byproducts"] = {"status": "not_run", "candidates": []}
        admet = _admet(_row("CCO", status="failed"), _row("C1CO1", status="failed"))
        admet["model"].update(version=None, weights_sha256={})
        result = review_ames_routes(engine.analyze_routes(payload), admet)
        self.assertEqual(result["counts"]["candidate"], 0)
        self.assertEqual([route["status"] for route in result["routes"]], ["insufficient_data", "needs_review"])
        for route in result["routes"]:
            self.assertTrue({"admet_unassessed", "conditions_not_observed",
                             "byproducts_incomplete"} <= _codes(route))

    def test_report_wide_gaps_are_recorded_without_deciding_the_route(self):
        # Unobserved conditions and unrun byproducts are true of every
        # generated route, so a status built on them is the same on every
        # route and separates none of them. They stay in the reasons.
        for field, value, expected_code in (
                ("conditions", {"status": "unknown"}, "conditions_not_observed"),
                ("byproducts", {"status": "failed", "candidates": []}, "byproducts_incomplete"),
                ("missing_information", ["Synthetic missing condition detail"], "missing_information")):
            with self.subTest(field=field):
                report = _report("CCO")
                report["routes"][0]["steps"][0][field] = value
                route = review_ames_routes(report, _admet(_row("CCO")))["routes"][0]
                self.assertEqual(route["status"], "candidate")
                self.assertIn(expected_code, _codes(route))

    def test_a_structure_that_cannot_be_read_still_holds_the_route_back(self):
        report = _report("not-a-molecule")
        route = review_ames_routes(report, _admet())["routes"][0]
        self.assertEqual(route["status"], "insufficient_data")
        self.assertIn("invalid_material", _codes(route))

    def test_joint_positive_preserves_all_missing_invalid_warning_and_failed_reasons(self):
        payload = _payload("C1CO1")
        step = payload["routes"][0]["steps"][0]
        step["reagents"] = ["not-a-molecule", "CNC", "[Na+].[O-]N=O"]
        step["conditions"] = {"status": "unknown"}
        step["byproducts"] = {"status": "failed", "candidates": []}
        report = engine.analyze_routes(payload)
        baseline = select_routes(report)["routes"][0]
        # Under the default the joint finding demotes; configured to exclude it
        # deletes the route. Either way every baseline reason is preserved.
        for policy, status in ((None, "needs_review"), (ALL_EXCLUDE, "excluded")):
            with self.subTest(policy=policy):
                route = review_ames_routes(report, _admet(_row("C1CO1", 0.8)), policy=policy)["routes"][0]
                self.assertEqual(route["status"], status)
                self.assertTrue(all(reason in route["reasons"] for reason in baseline["reasons"]))
                self.assertTrue({"admet_positive", "admet_unassessed", "invalid_material",
                                 "missing_information", "conditional_warning",
                                 "byproducts_incomplete"} <= _codes(route))
                self.assertEqual(sum(reason["code"] == "admet_unassessed" for reason in route["reasons"]), 3)
                self.assertEqual("rule_admet_joint_positive" in _codes(route), status == "excluded")
        route = review_ames_routes(report, _admet(_row("C1CO1", 0.8)), policy=ALL_EXCLUDE)["routes"][0]
        joint = next(reason for reason in route["reasons"] if reason["code"] == "rule_admet_joint_positive")
        self.assertIn("공동 연구 정책", joint["message"])
        self.assertIn("입증된 독성", joint["message"])

    def test_complete_coverage_is_required_in_every_step(self):
        payload = _payload("CCO")
        step = copy.deepcopy(payload["routes"][0]["steps"][0])
        step.update(id="second-step", reactants=["CCC"])
        payload["routes"][0]["steps"].append(step)
        route = review_ames_routes(engine.analyze_routes(payload), _admet(_row("CCO")))["routes"][0]
        self.assertEqual(route["status"], "insufficient_data")
        unassessed = next(reason for reason in route["reasons"] if reason["code"] == "admet_unassessed")
        self.assertEqual((unassessed["step_id"], unassessed["smiles"]), ("second-step", "CCC"))

    def test_empty_non_target_material_scope_never_becomes_candidate(self):
        result = review_ames_routes(_report(TARGET), _admet())
        self.assertEqual(result["routes"][0]["status"], "insufficient_data")
        self.assertIn("no_materials", _codes(result["routes"][0]))
        self.assertEqual(result["counts"]["candidate"], 0)

    def test_all_excluded_routes_are_retained_without_replacement_or_fallback(self):
        report = _report("C1CO1", "CC=O")
        admet = _admet(_row("C1CO1", 0.8), _row("CC=O", 0.8))
        # Nothing is replaced or back-filled whether the signals delete the
        # routes or only demote them: the same ids come back in the same order.
        for policy, statuses in ((ALL_EXCLUDE, ["excluded", "excluded"]),
                                 (None, ["needs_review", "needs_review"])):
            with self.subTest(policy=policy):
                result = review_ames_routes(report, admet, policy=policy)
                self.assertEqual([route["id"] for route in result["routes"]],
                                 [route["id"] for route in report["routes"]])
                self.assertEqual([route["status"] for route in result["routes"]], statuses)
                self.assertEqual(result["counts"][statuses[0]], 2)
                self.assertEqual(sum(result["counts"].values()), 2)

    def test_canonical_aliases_maps_and_repeated_materials_share_one_assessment(self):
        mapped = "[CH3:1][CH2:2][OH:3]"
        report = _report("OCC", mapped)
        report["routes"][0]["steps"][0]["materials"].append(
            copy.deepcopy(report["routes"][0]["steps"][0]["materials"][0]))
        row = _row("CCO")
        row["input_smiles_list"] = ["CCO", "OCC", mapped]
        result = review_ames_routes(report, _admet(row))
        self.assertEqual(result["counts"]["candidate"], 2)
        self.assertEqual(sum(reason["code"] == "admet_negative" for reason in result["routes"][0]["reasons"]), 2)

    def test_invalid_rows_match_exact_input_names_not_other_invalid_structures(self):
        report = _report("not-a-molecule", "CC(")
        result = review_ames_routes(report, _admet(_row("not-a-molecule"), _row("CC(")))
        self.assertEqual(result["counts"]["insufficient_data"], 2)
        for route, material in zip(result["routes"], ("not-a-molecule", "CC(")):
            self.assertTrue(all(reason["smiles"] == material for reason in route["reasons"]
                                if reason["code"] == "admet_unassessed"))

    def test_threshold_endpoints_and_ties_use_greater_than_or_equal(self):
        for threshold, score, expected in ((0, 0, "needs_review"), (1, 1, "needs_review"),
                                           (1, 0.999, "candidate"), (0.5, 0.5, "needs_review")):
            with self.subTest(threshold=threshold, score=score):
                result = review_ames_routes(_report("CCO"),
                                            _admet(_row("CCO", score, threshold=threshold), threshold=threshold))
                self.assertEqual(result["routes"][0]["status"], expected)

    def test_immutability_includes_nested_model_provenance_and_returned_evidence(self):
        report = _report("CCO", "C1CO1", "not-a-molecule")
        admet = _admet(_row("CCO"), _row("C1CO1", 0.8), _row("not-a-molecule"))
        exclusions = ["OCC"]
        before = copy.deepcopy((report, admet, exclusions))
        result = review_ames_routes(report, admet, excluded_smiles=exclusions)
        self.assertEqual((report, admet, exclusions), before)
        result["routes"][0]["reasons"][0]["message"] = "Changed output only"
        result["policy"]["admet_model"]["weights_sha256"]["fixture.pt"] = "b" * 64
        result["policy"]["excluded_smiles"].append("O")
        result["limitations"].append("Changed output only")
        self.assertEqual((report, admet, exclusions), before)

    def test_limitations_preserve_unknown_applicability_and_no_regulatory_or_probability_claim(self):
        result = review_ames_routes(_report("CCO"), _admet(_row("CCO")))
        wording = " ".join(result["limitations"])
        # Said in words rather than as the field name the backend uses.
        self.assertIn("적용 범위 안에 있는지 알려주지 않습니다", wording)
        self.assertNotIn("applicability=", wording)
        self.assertIn("보정된 독성 확률이 아닙니다", wording)
        self.assertIn("Class 5 또는 규제 적합성 판정이 아닙니다", wording)
        self.assertIn("Supplied synthetic model limitation", wording)


class RouteEligibilityPolicyTests(unittest.TestCase):
    def test_default_union_differs_from_separately_named_and_truth_table(self):
        # Default presentation: only a named ICH M7 material deletes a route.
        # A structural alert or a model positive demotes it to expert review.
        for material, known in (("CCO", False), ("CC=O", True), ("C1CO1", False)):
            rule_positive = material in ("CC=O", "C1CO1")
            for score in (0.2, 0.8):
                with self.subTest(material=material, score=score):
                    report = _report(material)
                    evidence = _admet(_row(material, score))
                    presented = review_route_eligibility(report, evidence)
                    comparison = review_ames_routes(report, evidence)
                    self.assertEqual(presented["routes"][0]["status"],
                                     "excluded" if known else
                                     "needs_review" if rule_positive or score >= 0.5 else "candidate")
                    self.assertEqual(comparison["routes"][0]["status"],
                                     "needs_review" if rule_positive or score >= 0.5 else "candidate")
                    self.assertEqual(presented["policy"]["rule_admet_exclusion"], "rule_OR_current_model")
                    self.assertEqual(comparison["policy"]["rule_admet_exclusion"], "same_material_AND")
                    self.assertTrue(presented["policy"]["strict_alerts"])
                    codes = _codes(presented["routes"][0])
                    self.assertNotIn("current_model_positive_exclusion", codes)
                    self.assertNotIn("rule_admet_joint_positive", codes)
                    strict = review_route_eligibility(report, evidence, policy=ALL_EXCLUDE)
                    self.assertEqual(strict["routes"][0]["status"],
                                     "excluded" if rule_positive or score >= 0.5 else "candidate")
                    self.assertEqual("current_model_positive_exclusion" in _codes(strict["routes"][0]),
                                     score >= 0.5)

    def test_relaxing_alerts_keeps_excluding_a_listed_material(self):
        # The alert checkbox relaxes how strictly structural alerts are read.
        # Being on the ICH M7 list is a prohibition, not an alert, so turning
        # the checkbox off must not turn a listed material into a demotion.
        for material, relaxed in (("CC=O", "excluded"), ("C1CO1", "needs_review")):
            for evidence in (None, _admet(_row(material, 0.2))):
                with self.subTest(material=material, model=evidence is not None):
                    report = _report(material)
                    strict = review_route_eligibility(report, evidence, strict_alerts=True)
                    loose = review_route_eligibility(report, evidence, strict_alerts=False)
                    self.assertEqual(strict["routes"][0]["status"],
                                     "excluded" if material == "CC=O" else "needs_review")
                    self.assertEqual(loose["routes"][0]["status"], relaxed)
                    expected = "known_concern" if material == "CC=O" else "structural_alert"
                    self.assertIn(expected, _codes(loose["routes"][0]))
        # The comparison view runs without the rule judge on purpose and is
        # not touched by the checkbox.
        comparison = review_ames_routes(_report("CC=O"), _admet(_row("CC=O", 0.2)))
        self.assertEqual(comparison["routes"][0]["status"], "needs_review")

    def test_conditional_warning_is_demoted_without_model_or_matching_rule_material(self):
        payload = _payload("CNC")
        nitrite = "[Na+].[O-]N=O"
        payload["routes"][0]["steps"][0]["reagents"] = [nitrite]
        report = engine.analyze_routes(payload)
        for evidence in (None, _admet(_row("CNC"), _row(nitrite))):
            for policy, status in ((None, "needs_review"), (ALL_EXCLUDE, "excluded")):
                with self.subTest(model=evidence is not None, policy=policy):
                    result = review_route_eligibility(report, evidence, policy=policy)
                    self.assertEqual(result["routes"][0]["status"], status)
                    self.assertIn("conditional_warning", _codes(result["routes"][0]))
                    warning = next(reason for reason in result["routes"][0]["reasons"]
                                   if reason["code"] == "conditional_warning")
                    self.assertIsNone(warning["smiles"])

    def test_missing_failed_stale_and_partial_evidence_never_create_a_negative(self):
        report = _report("CCO", "CCC", "CC=O")
        for status in ("missing", "failed", "stale", "invalid"):
            with self.subTest(status=status):
                result = review_route_eligibility(report, model_status=status, model_error="Synthetic unavailable")
                self.assertEqual([route["status"] for route in result["routes"]],
                                 ["insufficient_data", "insufficient_data", "excluded"])
                self.assertTrue(all("admet_negative" not in _codes(route) for route in result["routes"]))
                self.assertTrue(all("admet_unassessed" in _codes(route) for route in result["routes"]))
        partial = review_route_eligibility(report, _admet(_row("CCO", 0.8), _row("CCC", status="failed")))
        # A model positive alone demotes; the named ICH M7 material still goes.
        self.assertEqual([route["status"] for route in partial["routes"]],
                         ["needs_review", "insufficient_data", "excluded"])
        self.assertIn("Synthetic abstention", str(partial["routes"][1]["reasons"]))

    def test_union_reuses_strict_baseline_and_retains_every_reason_and_original(self):
        payload = _payload("CCO", "CC=O")
        payload["routes"][0]["steps"][0]["reagents"] = ["not-a-molecule"]
        payload["routes"][0]["steps"][0]["conditions"] = {"status": "unknown"}
        report = engine.analyze_routes(payload)
        original = copy.deepcopy(report)
        evidence = _admet(_row("CCO", 0.8))
        evidence_copy = copy.deepcopy(evidence)
        baseline = select_routes(report, strict_alerts=True)
        with patch.object(ames_route_review, "select_routes", wraps=select_routes) as select:
            result = review_route_eligibility(report, evidence)
        select.assert_called_once_with(report, excluded_smiles=(), strict_alerts=True)
        self.assertEqual([route["id"] for route in result["routes"]], [route["id"] for route in report["routes"]])
        self.assertEqual([route["status"] for route in result["routes"]], ["needs_review", "excluded"])
        self.assertEqual(result["counts"]["excluded"], 1)
        for route, rules in zip(result["routes"], baseline["routes"]):
            self.assertTrue(all(reason in route["reasons"] for reason in rules["reasons"]
                                if reason["code"] != "candidate_within_scope"))
        self.assertTrue({"invalid_material", "conditions_not_observed", "admet_unassessed",
                         "admet_positive"} <= _codes(result["routes"][0]))
        self.assertIn("known_concern", _codes(result["routes"][1]))
        self.assertEqual(report, original)
        self.assertEqual(evidence, evidence_copy)

    def test_union_still_rejects_invalid_membership_provenance_and_threshold(self):
        for mutate in (
            lambda result: result["assessments"].append(_row("CCC", 0.8)),
            lambda result: result["model"].update(weights_sha256={}),
            lambda result: result.update(threshold=float("nan")),
            lambda result: result["assessments"][0].update(prediction="negative"),
        ):
            evidence = _admet(_row("CCO", 0.8))
            mutate(evidence)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                review_route_eligibility(_report("CCO"), evidence)


class AmesRouteValidationTests(unittest.TestCase):
    def test_report_validation_and_non_strict_baseline_are_reused(self):
        report = _report("CCO")
        with patch.object(ames_route_review, "select_routes", wraps=select_routes) as baseline:
            review_ames_routes(report, _admet(_row("CCO")), excluded_smiles=["CCC"])
        baseline.assert_called_once_with(report, excluded_smiles=["CCC"], strict_alerts=False)
        for malformed in (None, {}, {"schema_version": "1.0", "routes": []}):
            with self.subTest(report=malformed), self.assertRaises(ValueError):
                review_ames_routes(malformed, _admet())

    def test_prohibited_target_and_target_component_keep_baseline_error(self):
        with self.assertRaisesRegex(ValueError, "목표물 target"):
            review_ames_routes(_report("CCO"), _admet(_row("CCO")), excluded_smiles=[TARGET])
        payload = _payload("CCC")
        payload["routes"][0]["target"] = "CCO.[Na+]"
        payload["routes"][0]["steps"][0]["products"] = ["CCO.[Na+]"]
        with self.assertRaisesRegex(ValueError, "목표물 target"):
            review_ames_routes(engine.analyze_routes(payload), _admet(_row("CCC")), excluded_smiles=["CCO"])

    def test_malformed_root_and_required_fields_are_rejected(self):
        for result in (None, [], {}, {"model": None}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                review_ames_routes(_report("CCO"), result)
        for field in ("model", "threshold", "assessments", "limitations"):
            result = _admet(_row("CCO"))
            del result[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                review_ames_routes(_report("CCO"), result)
        for field, values in (("assessments", (None, {}, "CCO", [None])),
                              ("limitations", (None, {}, "unknown", [1], [""]))):
            for value in values:
                result = _admet(_row("CCO"))
                result[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    review_ames_routes(_report("CCO"), result)

    def test_invalid_model_provenance_and_wrong_endpoint_are_rejected(self):
        changes = (("tool", "Other"), ("endpoint", "toxicity"), ("version", 2), ("version", ""),
                   ("weights_sha256", []), ("weights_sha256", {"fixture.pt": "not-a-sha256"}))
        for field, value in changes:
            result = _admet(_row("CCO"))
            result["model"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                review_ames_routes(_report("CCO"), result)
        for field in ("tool", "version", "endpoint", "weights_sha256"):
            result = _admet(_row("CCO"))
            del result["model"][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                review_ames_routes(_report("CCO"), result)

    def test_threshold_and_success_score_require_finite_numeric_unit_interval(self):
        values = (None, True, False, "0.5", "invalid", float("nan"), float("inf"), -0.01, 1.01, [], {}, 10 ** 1000)
        for value in values:
            for field in ("threshold", "score"):
                result = _admet(_row("CCO"))
                if field == "threshold":
                    result[field] = value
                else:
                    result["assessments"][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    review_ames_routes(_report("CCO"), result)

    def test_prediction_threshold_mismatch_is_an_error_not_silently_repaired(self):
        for score, threshold, prediction in ((0.8, 0.5, "negative"), (0.2, 0.5, "positive"),
                                             (0.5, 0.5, "negative"), (0.8, 0.9, "positive"),
                                             (0.2, 0.5, "abstain")):
            result = _admet(_row("CCO", score, threshold=threshold), threshold=threshold)
            result["assessments"][0]["prediction"] = prediction
            with self.subTest(score=score, threshold=threshold, prediction=prediction):
                with self.assertRaisesRegex(ValueError, "prediction.*score.*threshold"):
                    review_ames_routes(_report("CCO"), result)

    def test_all_duplicate_identities_are_rejected_including_alias_conflicts(self):
        for second in (_row("CCO"), _row("OCC", 0.8), _row("CCO", status="failed")):
            with self.subTest(second=second), self.assertRaisesRegex(ValueError, "Duplicate"):
                review_ames_routes(_report("CCO"), _admet(_row("CCO"), second))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            review_ames_routes(_report("not-a-molecule"), _admet(_row("not-a-molecule"), _row("not-a-molecule")))

    def test_stale_extra_valid_invalid_and_target_rows_are_rejected(self):
        for stale in (_row("CCC"), _row(TARGET), _row("another-invalid-input")):
            with self.subTest(stale=stale), self.assertRaisesRegex(ValueError, "stale evidence"):
                review_ames_routes(_report("CCO"), _admet(_row("CCO"), stale))
        with self.assertRaisesRegex(ValueError, "stale evidence"):
            review_ames_routes(_report("not-a-molecule"), _admet(_row("CC(")))

    def test_salts_charge_stereo_isotopes_and_tautomers_are_not_merged_for_model_matching(self):
        pairs = (("CCO.[Na+]", "CCO"), ("C[NH2+]C", "CNC"),
                 ("C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"),
                 ("[13CH3]CO", "CCO"), ("C=CO", "CC=O"))
        for material, different in pairs:
            with self.subTest(material=material, different=different):
                with self.assertRaisesRegex(ValueError, "stale evidence"):
                    review_ames_routes(_report(material), _admet(_row(different)))

    def test_malformed_and_inconsistent_assessment_fields_are_rejected(self):
        changes = (("input_smiles", None), ("input_smiles", ""), ("input_smiles", "not-a-molecule"),
                   ("canonical_smiles", None), ("canonical_smiles", "OCC"), ("canonical_smiles", "CCC"),
                   ("canonical_smiles", []), ("status", "unknown"), ("status", "invalid"), ("status", []),
                   ("applicability", "validated"), ("applicability", None), ("error", "Conflicting success error"),
                   ("error", []), ("input_smiles_list", []), ("input_smiles_list", "CCO"),
                   ("input_smiles_list", ["CCC"]), ("input_smiles_list", [None]))
        for field, value in changes:
            result = _admet(_row("CCO"))
            result["assessments"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                review_ames_routes(_report("CCO"), result)
        for field in ("canonical_smiles", "input_smiles", "status", "score", "prediction", "applicability", "error"):
            result = _admet(_row("CCO"))
            del result["assessments"][0][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                review_ames_routes(_report("CCO"), result)

    def test_failed_and_invalid_assessments_cannot_carry_negative_or_numeric_evidence(self):
        for material, status in (("CCO", "failed"), ("not-a-molecule", "invalid")):
            for field, value in (("score", 0.1), ("prediction", "negative"), ("prediction", "positive")):
                result = _admet(_row(material, status=status))
                result["assessments"][0][field] = value
                with self.subTest(material=material, field=field, value=value), self.assertRaises(ValueError):
                    review_ames_routes(_report(material), result)
        for status in ("success", "failed"):
            result = _admet(_row("not-a-molecule"))
            result["assessments"][0]["status"] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                review_ames_routes(_report("not-a-molecule"), result)

    def test_source_type_provenance_does_not_override_structure_or_status_checks(self):
        for canonical in (None, "CCO"):
            row = _row("CCO", status="invalid")
            row.update(canonical_smiles=canonical, input_types=["SmilesLike"])
            with self.subTest(canonical=canonical), self.assertRaises(ValueError):
                review_ames_routes(_report("CCO"), _admet(row))


if __name__ == "__main__":
    unittest.main()
