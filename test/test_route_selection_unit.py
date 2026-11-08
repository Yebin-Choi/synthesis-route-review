"""Software-only route-selection fixtures; no reaction or safety observations."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import engine
from route_selection import select_routes
from toxicology import canonicalize_smiles


TARGET = "CCOC(C)=O"


def _payload(*materials):
    return {"schema_version": "1.0", "routes": [
        {"id": f"route-{index}", "target": TARGET, "source": "Software-only synthetic fixture",
         "steps": [{"id": "step-1", "reactants": [material], "products": [TARGET],
                    "reagents": [], "solvents": [],
                    "conditions": {"status": "observed", "source": "Software-only fixture",
                                   "description": "Synthetic observed label for branch testing; not an experiment."},
                    "byproducts": {"status": "success", "candidates": [],
                                   "source": "Synthetic completed-attempt label; no real prediction."}}]}
        for index, material in enumerate(materials, 1)
    ]}


def _report(*materials):
    return engine.analyze_routes(_payload(*materials))


def _codes(route):
    return {reason["code"] for reason in route["reasons"]}


class RouteSelectionTests(unittest.TestCase):
    def test_all_four_statuses_and_exact_output_contract(self):
        report = _report("CCO", "C1CO1", "not-a-molecule", "CCC")
        result = select_routes(report, excluded_smiles=["CCC"])
        self.assertEqual(set(result), {"routes", "counts", "policy", "limitations"})
        self.assertEqual([route["status"] for route in result["routes"]],
                         ["candidate", "needs_review", "insufficient_data", "excluded"])
        self.assertEqual(result["counts"], {"excluded": 1, "needs_review": 1,
                                            "insufficient_data": 1, "candidate": 1})
        self.assertEqual(result["policy"], {"excluded_smiles": ["CCC"], "strict_alerts": False})
        self.assertEqual([route["id"] for route in result["routes"]],
                         [route["id"] for route in report["routes"]])
        for route in result["routes"]:
            self.assertEqual(set(route), {"id", "status", "reasons"})
            self.assertTrue(route["reasons"])
            for reason in route["reasons"]:
                self.assertEqual(set(reason), {"code", "message", "step_id", "smiles"})
                self.assertIsInstance(reason["message"], str)
        candidate = result["routes"][0]["reasons"][0]
        self.assertEqual(candidate, {"code": "candidate_within_scope",
                                    "message": "입력된 평가 범위에서 선별 기준을 통과한 후보; 안전 보장 아님",
                                    "step_id": None, "smiles": None})

    def test_known_and_structural_alerts_review_by_default_exclude_in_strict_mode(self):
        report = _report("CC=O", "C1CO1")
        for strict, expected in ((False, "needs_review"), (True, "excluded")):
            with self.subTest(strict=strict):
                routes = select_routes(report, strict_alerts=strict)["routes"]
                self.assertEqual([route["status"] for route in routes], [expected, expected])
                self.assertIn("known_concern", _codes(routes[0]))
                self.assertIn("structural_alert", _codes(routes[1]))
                if strict:
                    # The message names the setting as the screen labels it,
                    # and still refuses to claim the route is toxic.
                    for route in routes:
                        message = route["reasons"][0]["message"]
                        self.assertIn("규칙에 걸린 경로도 제외", message)
                        self.assertIn("증명이 아닙니다", message)
                        self.assertNotIn("strict_alerts", message)

    def test_explicit_policy_normalizes_atom_maps_and_deduplicates(self):
        mapped = "[CH3:1][CH2:2][OH:3]"
        result = select_routes(_report("OCC"), excluded_smiles=[mapped, "CCO", " OCC "])
        self.assertEqual(result["policy"]["excluded_smiles"], ["CCO"])
        route = result["routes"][0]
        self.assertEqual(route["status"], "excluded")
        self.assertEqual(_codes(route), {"prohibited_material"})
        self.assertEqual(route["reasons"][0]["smiles"], "CCO")
        route = select_routes(_report(mapped), excluded_smiles=["CCO"])["routes"][0]
        self.assertEqual(route["status"], "excluded")
        self.assertEqual(_codes(route), {"prohibited_material"})

    def test_disconnected_material_component_match_is_explicitly_named(self):
        result = select_routes(_report("[Na+].CCO"), excluded_smiles=["OCC"])
        route = result["routes"][0]
        self.assertEqual(route["status"], "excluded")
        reason = route["reasons"][0]
        self.assertEqual(reason["code"], "prohibited_component")
        self.assertEqual(reason["step_id"], "step-1")
        self.assertEqual(reason["smiles"], canonicalize_smiles("[Na+].CCO"))
        self.assertIn("성분 CCO", reason["message"])
        self.assertIn("전체 물질과 성분의 동일성을 가정하지 않습니다", reason["message"])

    def test_multicomponent_policy_matches_full_material_not_a_subset(self):
        report = _report("CCO.[Na+]", "CCO.[Na+].O", "CCO", "CCC")
        result = select_routes(report, excluded_smiles=["[Na+].OCC", "CC"])
        self.assertEqual([route["status"] for route in result["routes"]],
                         ["excluded", "candidate", "candidate", "candidate"])
        self.assertEqual(_codes(result["routes"][0]), {"prohibited_material"})

    def test_charge_stereo_isotope_and_tautomer_are_not_merged(self):
        pairs = (("C[NH2+]C", "CNC"),
                 ("CC(=O)[O-].[Na+]", "CC(=O)O"),
                 ("C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"),
                 ("C[C@H](O)C(=O)O", "CC(O)C(=O)O"),
                 ("[13CH3]CO", "CCO"), ("C=CO", "CC=O"))
        for material, prohibited in pairs:
            with self.subTest(material=material, prohibited=prohibited):
                report = _report(material)
                self.assertEqual(select_routes(report, excluded_smiles=[prohibited])["routes"][0]["status"],
                                 "candidate")
                self.assertEqual(select_routes(report, excluded_smiles=[material])["routes"][0]["status"],
                                 "excluded")

    def test_disconnected_component_matching_also_preserves_stereo(self):
        left = "C[C@H](O)C(=O)O"
        right = "C[C@@H](O)C(=O)O"
        report = _report(f"{left}.[Na+]")
        self.assertEqual(select_routes(report, excluded_smiles=[right])["routes"][0]["status"], "candidate")
        route = select_routes(report, excluded_smiles=[left])["routes"][0]
        self.assertEqual(route["status"], "excluded")
        self.assertEqual(_codes(route), {"prohibited_component"})
        self.assertIn(canonicalize_smiles(left), route["reasons"][0]["message"])

    def test_intermediates_and_supplied_byproduct_candidates_are_selectable(self):
        payload = _payload("CCO")
        first = payload["routes"][0]["steps"][0]
        second = copy.deepcopy(first)
        first["products"] = ["CCC"]
        first["byproducts"]["candidates"] = ["CO"]
        second.update(id="step-2", reactants=["CCC"])
        payload["routes"][0]["steps"].append(second)
        report = engine.analyze_routes(payload)
        route = select_routes(report, excluded_smiles=["CCC", "CO"])["routes"][0]
        self.assertEqual(route["status"], "excluded")
        self.assertEqual({(reason["step_id"], reason["smiles"]) for reason in route["reasons"]},
                         {("step-1", "CCC"), ("step-1", "CO"), ("step-2", "CCC")})
        self.assertEqual(_codes(route), {"prohibited_material"})

    def test_empty_material_coverage_is_not_a_candidate(self):
        report = _report(TARGET)
        self.assertEqual(report["routes"][0]["steps"][0]["materials"], [])
        route = select_routes(report)["routes"][0]
        self.assertEqual(route["status"], "insufficient_data")
        self.assertIn("no_materials", _codes(route))

    def test_a_step_without_materials_is_incomplete_even_with_other_coverage(self):
        payload = _payload("CCO")
        step = copy.deepcopy(payload["routes"][0]["steps"][0])
        step.update(id="empty-step", reactants=[TARGET])
        payload["routes"][0]["steps"].append(step)
        route = select_routes(engine.analyze_routes(payload))["routes"][0]
        self.assertEqual(route["status"], "insufficient_data")
        self.assertEqual([reason["step_id"] for reason in route["reasons"]
                          if reason["code"] == "no_materials"], ["empty-step"])

    def test_unresolved_conditions_are_recorded_without_deciding_the_route(self):
        # Every generated route has unobserved conditions, so this cannot
        # separate one route from another. Measured on ten aspirin routes:
        # 10 steps of 10. It stays in the reasons and leaves the status to
        # findings that differ between routes.
        for state in ("unknown", "proposed"):
            with self.subTest(state=state):
                report = _report("CCO")
                step = report["routes"][0]["steps"][0]
                step["conditions"]["status"] = state
                step["missing_information"] = []
                for strict in (False, True):
                    route = select_routes(report, strict_alerts=strict)["routes"][0]
                    self.assertEqual(route["status"], "candidate")
                    self.assertEqual(_codes(route),
                                     {"conditions_not_observed", "candidate_within_scope"})
                    self.assertIn("추천·제안 조건은 관측 조건이 아닙니다", route["reasons"][0]["message"])
                self.assertEqual(step["conditions"]["status"], state)

    def test_unrun_byproducts_are_recorded_without_deciding_the_route(self):
        for state in ("failed", "not_run"):
            with self.subTest(state=state):
                report = _report("CCO")
                step = report["routes"][0]["steps"][0]
                step["byproducts"]["status"] = state
                step["missing_information"] = []
                route = select_routes(report)["routes"][0]
                self.assertEqual(route["status"], "candidate")
                self.assertEqual(_codes(route),
                                 {"byproducts_incomplete", "candidate_within_scope"})

    def test_missing_information_is_kept_even_though_it_does_not_decide(self):
        report = _report("CCO")
        report["routes"][0]["steps"][0]["missing_information"] = ["중간체 근거 미확인"]
        route = select_routes(report)["routes"][0]
        self.assertEqual(route["status"], "candidate")
        self.assertIn({"code": "missing_information", "message": "중간체 근거 미확인",
                       "step_id": "step-1", "smiles": None}, route["reasons"])

    def test_a_structure_that_cannot_be_read_still_holds_the_route_back(self):
        # This one is about the route in hand, not about the whole report,
        # so it keeps its say.
        report = _report("not-a-molecule")
        route = select_routes(report)["routes"][0]
        self.assertEqual(route["status"], "insufficient_data")
        self.assertIn("invalid_material", _codes(route))

    def test_conditional_warnings_review_or_conservatively_exclude(self):
        payload = _payload("CNC")
        payload["routes"][0]["steps"][0]["reagents"] = ["[Na+].[O-]N=O"]
        report = engine.analyze_routes(payload)
        self.assertEqual(report["routes"][0]["conditional_warning_count"], 1)
        for strict, expected in ((False, "needs_review"), (True, "excluded")):
            with self.subTest(strict=strict):
                route = select_routes(report, strict_alerts=strict)["routes"][0]
                self.assertEqual(route["status"], expected)
                self.assertEqual(_codes(route), {"conditional_warning"})
                # The warning's own code stays in the report entry, not at
                # the front of the sentence the reader is shown.
                message = route["reasons"][0]["message"]
                self.assertNotIn("secondary_amine_nitrite", message)
                self.assertEqual(
                    report["routes"][0]["steps"][0]["conditional_warnings"][0]["code"],
                    "secondary_amine_nitrite")
                self.assertIn("생성 확정이나 생성량 예측은 아닙니다", message)
                self.assertIsNone(route["reasons"][0]["smiles"])

    def test_precedence_never_erases_missing_data_or_other_reasons(self):
        payload = _payload("CC=O")
        step = payload["routes"][0]["steps"][0]
        step["conditions"]["status"] = "proposed"
        step["byproducts"]["status"] = "failed"
        step["reagents"] = ["not-a-molecule", "CNC", "[Na+].[O-]N=O"]
        report = engine.analyze_routes(payload)
        missing = report["routes"][0]["steps"][0]["missing_information"]
        for policy, expected in (({}, "needs_review"), ({"strict_alerts": True}, "excluded"),
                                 ({"excluded_smiles": ["CC=O"]}, "excluded")):
            with self.subTest(policy=policy):
                route = select_routes(report, **policy)["routes"][0]
                self.assertEqual(route["status"], expected)
                self.assertTrue({"known_concern", "conditional_warning", "invalid_material", "missing_information",
                                 "conditions_not_observed", "byproducts_incomplete"}.issubset(_codes(route)))
                self.assertEqual([reason["message"] for reason in route["reasons"]
                                  if reason["code"] == "missing_information"], missing)
                if "excluded_smiles" in policy:
                    self.assertIn("prohibited_material", _codes(route))

    def test_all_rejected_retains_same_route_set_without_fallback(self):
        report = _report("CCO", "CCC")
        result = select_routes(report, excluded_smiles=["OCC", "CCC"])
        self.assertEqual([route["id"] for route in result["routes"]], ["route-1", "route-2"])
        self.assertEqual([route["status"] for route in result["routes"]], ["excluded", "excluded"])
        self.assertEqual(result["counts"], {"excluded": 2, "needs_review": 0,
                                            "insufficient_data": 0, "candidate": 0})

    def test_report_and_policy_are_not_mutated_for_any_selection_mode(self):
        report = _report("CCO", "CC=O", "C1CO1")
        before = copy.deepcopy(report)
        policy = ["[CH3:1][CH2:2][OH:3]", "OCC"]
        policy_before = list(policy)
        for options in ({}, {"strict_alerts": True}, {"excluded_smiles": policy}):
            result = select_routes(report, **options)
            self.assertEqual(report, before)
            self.assertEqual(policy, policy_before)
            self.assertEqual([route["id"] for route in result["routes"]],
                             [route["id"] for route in report["routes"]])
            result["routes"][0]["reasons"][0]["message"] = "modified return value"
            result["policy"]["excluded_smiles"].append("O")
            self.assertEqual(report, before)
            self.assertEqual(policy, policy_before)

    def test_limitations_keep_success_and_candidates_evidence_scoped(self):
        result = select_routes(_report("CCO"))
        wording = " ".join(result["limitations"])
        # The wording names what happened rather than the field it is stored
        # in. "status=success" was on screen until 2026-11-06.
        self.assertIn("평가를 끝까지 돌렸다는 뜻일 뿐", wording)
        self.assertIn("부산물을 다 찾았다는 뜻이 아닙니다", wording)
        self.assertNotIn("status=", wording)
        self.assertIn("생성 여부·생성량·잔류량·노출량·정제 효과를 예측하지 않으며", wording)
        self.assertIn("후보 판정은 안전 보장이 아닙니다", wording)


class RouteSelectionValidationTests(unittest.TestCase):
    def test_explicit_prohibited_target_is_rejected_not_falsely_avoided(self):
        report = _report("CCO")
        mapped_target = "[CH3:1][CH2:2][O:3][C:4]([CH3:5])=[O:6]"
        for target in (TARGET, mapped_target):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "목표물 target"):
                select_routes(report, excluded_smiles=[target])

    def test_prohibited_component_of_disconnected_target_is_also_rejected(self):
        payload = _payload("CCC")
        payload["routes"][0]["target"] = "CCO.[Na+]"
        payload["routes"][0]["steps"][0]["products"] = ["CCO.[Na+]"]
        report = engine.analyze_routes(payload)
        with self.assertRaisesRegex(ValueError, "성분 CCO.*목표물 target"):
            select_routes(report, excluded_smiles=["CCO"])

    def test_policy_requires_real_bool(self):
        for strict in (None, "true", "False", 0, 1, [], {}):
            with self.subTest(strict=strict), self.assertRaises(ValueError):
                select_routes(_report("CCO"), strict_alerts=strict)

    def test_empty_collections_are_valid_but_empty_or_invalid_entries_are_not(self):
        report = _report("CCO")
        for policy in ([], (), set(), frozenset()):
            with self.subTest(policy=policy):
                self.assertEqual(select_routes(report, excluded_smiles=policy)["routes"][0]["status"], "candidate")
        for entry in ("", " ", ".", "CC(", "CCO trailing-name", "CCO\x00junk", None, 1, []):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                select_routes(report, excluded_smiles=[entry])
        for policy in (None, "CCO", "", b"CCO", 123, {}, {"CCO": True}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                select_routes(report, excluded_smiles=policy)

    def test_unordered_policy_collections_have_deterministic_normalized_receipt(self):
        result = select_routes(_report("CCC"), excluded_smiles={"OCC", "CCO", "O"})
        self.assertEqual(result["policy"]["excluded_smiles"], ["CCO", "O"])

    def test_malformed_report_roots_are_rejected(self):
        for report in (None, [], {}, {"schema_version": "2", "routes": []},
                       {"schema_version": "1.0", "routes": []},
                       {"schema_version": "1.0", "routes": None},
                       {"schema_version": "1.0", "routes": {}},
                       {"schema_version": "1.0", "routes": [None]}):
            with self.subTest(report=report), self.assertRaises(ValueError):
                select_routes(report)

    def test_malformed_route_fields_and_duplicate_ids_are_rejected(self):
        for field, value in (("id", ""), ("id", []), ("target", "invalid"), ("target", None),
                             ("steps", []), ("steps", None), ("steps", "step-1"), ("steps", [None])):
            with self.subTest(field=field, value=value):
                report = _report("CCO")
                report["routes"][0][field] = value
                with self.assertRaises(ValueError):
                    select_routes(report)
        report = _report("CCO", "CCC")
        report["routes"][1]["id"] = " route-1 "
        with self.assertRaises(ValueError):
            select_routes(report)
        report = _report("CCO")
        report["routes"][0]["steps"].append(copy.deepcopy(report["routes"][0]["steps"][0]))
        with self.assertRaises(ValueError):
            select_routes(report)

    def test_malformed_step_fields_are_rejected(self):
        changes = (("id", ""), ("id", None), ("materials", None), ("materials", "CCO"),
                   ("materials", [None]), ("conditional_warnings", None),
                   ("conditional_warnings", [None]), ("conditional_warnings", [{}]),
                   ("conditional_warnings", [{"code": "warning", "message": ""}]),
                   ("missing_information", None), ("missing_information", "unknown"),
                   ("missing_information", [""]), ("missing_information", [1]),
                   ("conditions", None), ("conditions", {}),
                   ("conditions", {"status": ["observed"]}),
                   ("conditions", {"status": "suggested"}),
                   ("conditions", {"status": "observed"}),
                   ("conditions", {"status": "proposed", "source": "fixture"}),
                   ("byproducts", None), ("byproducts", {}),
                   ("byproducts", {"status": ["success"], "candidates": []}),
                   ("byproducts", {"status": "success", "candidates": []}),
                   ("byproducts", {"status": "not_run", "candidates": "CCO"}),
                   ("byproducts", {"status": "failed", "candidates": ["CCO"]}),
                   ("byproducts", {"status": "not_run", "candidates": [None]}))
        for field, value in changes:
            with self.subTest(field=field, value=value):
                report = _report("CCO")
                report["routes"][0]["steps"][0][field] = value
                with self.assertRaises(ValueError):
                    select_routes(report)
        for field in ("id", "materials", "conditional_warnings", "missing_information", "conditions", "byproducts"):
            with self.subTest(missing_field=field):
                report = _report("CCO")
                del report["routes"][0]["steps"][0][field]
                with self.assertRaises(ValueError):
                    select_routes(report)

    def test_unknown_or_inconsistent_material_states_never_fail_open(self):
        changes = (("status", "safe"), ("status", None), ("status", ["known_concern"]),
                   ("status", "invalid"), ("input_smiles", None), ("input_smiles", ""),
                   ("input_smiles", "not-a-molecule"), ("canonical_smiles", None),
                   ("canonical_smiles", "CCC"), ("canonical_smiles", ["CCO"]),
                   ("alerts", None), ("alerts", [""]), ("alerts", ["Unexpected alert"]),
                   ("special_concern", "false"), ("special_concern", True), ("name", "Unexpected name"),
                   ("status", "structural_alert"), ("status", "known_concern"))
        for field, value in changes:
            with self.subTest(field=field, value=value):
                report = _report("CCO")
                report["routes"][0]["steps"][0]["materials"][0][field] = value
                with self.assertRaises(ValueError):
                    select_routes(report)
        for field in ("status", "input_smiles", "canonical_smiles", "alerts", "special_concern", "name"):
            with self.subTest(missing_field=field):
                report = _report("not-a-molecule")
                del report["routes"][0]["steps"][0]["materials"][0][field]
                with self.assertRaises(ValueError):
                    select_routes(report)
        report = _report("not-a-molecule")
        report["routes"][0]["steps"][0]["materials"][0]["status"] = "no_alert_detected"
        with self.assertRaises(ValueError):
            select_routes(report)
        report = _report("CC=O")
        report["routes"][0]["steps"][0]["materials"][0]["status"] = "no_alert_detected"
        with self.assertRaises(ValueError):
            select_routes(report)


if __name__ == "__main__":
    unittest.main()
