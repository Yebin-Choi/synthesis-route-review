"""Observable offline behavior: python -m unittest discover -s test -p '*_unit.py'."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))
import env_support
import engine


class RouteScreeningTests(unittest.TestCase):
    def setUp(self):
        self.payload = engine.load_example()

    def test_known_conditional_and_missing_cases_are_distinct(self):
        result = engine.analyze_routes(self.payload)
        known, conditional, missing = result["routes"]
        self.assertEqual(known["known_concern_count"], 1)
        self.assertEqual(conditional["conditional_warning_count"], 1)
        self.assertEqual(missing["known_concern_count"], 0)
        self.assertEqual(missing["missing_step_count"], 1)
        self.assertIn("미평가", missing["review_status"])
        self.assertNotIn("safe", result)
        self.assertEqual([r["id"] for r in result["routes"]], [r["id"] for r in self.payload["routes"]])

    def test_missing_conditions_and_byproducts_count_separately(self):
        route = self.payload["routes"][0]
        route["steps"] += [copy.deepcopy(route["steps"][0]) for _ in range(2)]
        for index, step in enumerate(route["steps"]):
            step["id"] = str(index)
        route["steps"][1]["byproducts"]["status"] = "failed"
        result = engine.analyze_routes(self.payload)["routes"][0]
        self.assertEqual(result["missing_step_count"], 3)
        self.assertEqual(result["missing_information_count"], 6)
        self.assertEqual(result["byproduct_coverage"], 0)

    def test_proposed_conditions_not_observed(self):
        result = engine.analyze_routes(self.payload)["routes"][1]
        self.assertEqual(result["condition_coverage"], 0)
        step = self.payload["routes"][1]["steps"][0]
        step["conditions"]["status"] = "observed"
        step["byproducts"] = {"status": "success", "candidates": [], "source": "test fixture"}
        result = engine.analyze_routes(self.payload)["routes"][1]
        self.assertEqual(result["condition_coverage"], 1)
        self.assertEqual(result["byproduct_coverage"], 1)
        self.assertIn("정확도", engine.analyze_routes(self.payload)["comparison_note"])

    def test_invalid_material_is_visible(self):
        self.payload["routes"][0]["steps"][0]["reagents"].append("not-a-molecule")
        result = engine.analyze_routes(self.payload)["routes"][0]
        self.assertEqual(result["invalid_material_count"], 1)
        self.assertEqual(result["evaluated_material_count"], result["material_count"] - 1)
        self.assertTrue(any("해석 실패" in s for s in result["steps"][0]["missing_information"]))

    def test_oversize_target_rejected_before_parser_on_direct_import_and_generation(self):
        target = "C" * 4097
        payload = copy.deepcopy(self.payload)
        payload["routes"][0]["target"] = target
        rendered = {"route": [{"smiles": "CCO>>C"}]}
        for operation in (
            lambda: engine.validate_routes(payload),
            lambda: engine.analyze_routes(payload),
            lambda: engine.generate_routes(target),
            lambda: engine.from_synthex_rendered(rendered, "oversize", target),
            lambda: engine.import_synthex({"format_version": "routejson-1.0", "target": target}),
        ):
            with self.subTest(operation=operation):
                with patch.object(engine, "canonicalize_smiles") as parser:
                    with self.assertRaisesRegex(ValueError, "4096"):
                        operation()
                    parser.assert_not_called()

    def test_target_length_includes_raw_whitespace_and_keeps_4096_boundary(self):
        boundary = "C" + " " * 4095
        self.assertEqual(engine.validate_target_smiles(boundary), "C")
        with patch.object(engine, "canonicalize_smiles") as parser:
            with self.assertRaisesRegex(ValueError, "4096"):
                engine.validate_target_smiles(boundary + " ")
            parser.assert_not_called()

    def test_map_variants_deduplicated_and_api_excluded(self):
        step = self.payload["routes"][0]["steps"][0]
        step["reactants"].append("[CH3:1][CH2:2][OH:3]")
        result = engine.analyze_routes(self.payload)["routes"][0]
        self.assertEqual(result["material_count"], 3)
        self.assertFalse(any(m["canonical_smiles"] == "CCOC(C)=O" for m in result["steps"][0]["materials"]))

    def test_intermediates_retained(self):
        route = self.payload["routes"][0]
        route["steps"].append({"id": "2", "reactants": ["CC=O"], "products": ["CCO"],
                               "reagents": [], "solvents": [], "conditions": {"status": "unknown"},
                               "byproducts": {"status": "not_run", "candidates": []}})
        result = engine.analyze_routes(self.payload)["routes"][0]
        self.assertTrue(any(m["canonical_smiles"] == "CCO" and m["role"] == "products"
                            for m in result["steps"][1]["materials"]))

    def test_unrelated_step_species_not_combined(self):
        route = self.payload["routes"][1]
        route["steps"][0]["reagents"] = ["CNC"]
        other = copy.deepcopy(route["steps"][0])
        other["id"] = "2"
        other["reagents"] = ["[Na+].[O-]N=O"]
        route["steps"].append(other)
        result = engine.analyze_routes(self.payload)
        self.assertEqual(result["routes"][1]["conditional_warning_count"], 0)
        self.assertIn("단계 간 잔류", result["disclaimer"])

    def test_strict_schema(self):
        for mutate in (
            lambda p: p.update(schema_version="2"),
            lambda p: p["routes"].append(copy.deepcopy(p["routes"][0])),
            lambda p: p["routes"][0].update(target="not-smiles"),
            lambda p: p["routes"][0].update(target="CCCC"),
            lambda p: p["routes"][0]["steps"][0].update(reactants="CCO"),
            lambda p: p["routes"][0]["steps"][0]["byproducts"].update(candidates=["CC=O"]),
            lambda p: p["routes"][0]["steps"][0].update(conditions={"status": "observed"}),
        ):
            with self.subTest(mutate=mutate):
                payload = copy.deepcopy(self.payload)
                mutate(payload)
                with self.assertRaises(ValueError):
                    engine.analyze_routes(payload)

    def test_retained_metadata_rejects_nonfinite_values_before_analysis(self):
        for field, key in (("conditions", "description"), ("byproducts", "error")):
            for value in (float("nan"), float("inf"), float("-inf")):
                for nested in (False, True):
                    with self.subTest(field=field, value=value, nested=nested):
                        payload = copy.deepcopy(self.payload)
                        payload["routes"][0]["steps"][0][field][key] = (
                            {"measurements": [1.0, {"value": value}]} if nested else value)
                        with patch.object(engine, "assess_material") as assess:
                            with self.assertRaisesRegex(ValueError, field + ".*JSON"):
                                engine.analyze_routes(payload)
                            assess.assert_not_called()

    def test_retained_metadata_rejects_unsupported_and_cyclic_values(self):
        cyclic_list = []
        cyclic_list.append(cyclic_list)
        cyclic_dict = {}
        cyclic_dict["self"] = cyclic_dict
        deeply_nested = []
        for _ in range(2000):
            deeply_nested = [deeply_nested]
        for field, key in (("conditions", "description"), ("byproducts", "error")):
            for value in ({"not-json"}, object(), cyclic_list, cyclic_dict, deeply_nested):
                with self.subTest(field=field, value_type=type(value).__name__):
                    payload = copy.deepcopy(self.payload)
                    payload["routes"][0]["steps"][0][field][key] = {"nested": [value]}
                    with patch.object(engine, "assess_material") as assess:
                        with self.assertRaisesRegex(ValueError, field + ".*JSON"):
                            engine.analyze_routes(payload)
                        assess.assert_not_called()

    def test_retained_metadata_rejects_mixed_keys_before_report_hashing(self):
        for field, key in (("conditions", "description"), ("byproducts", "error")):
            with self.subTest(field=field):
                payload = copy.deepcopy(self.payload)
                payload["routes"][0]["steps"][0][field][key] = {1: "a", "detail": "b"}
                with patch.object(engine, "assess_material") as assess:
                    with self.assertRaisesRegex(ValueError, field + ".*JSON"):
                        engine.analyze_routes(payload)
                    assess.assert_not_called()

    def test_retained_metadata_reports_are_strict_json_without_mutation(self):
        step = self.payload["routes"][0]["steps"][0]
        step["conditions"]["description"] = {
            "measurements": [23.5, -4, 0, None, True], "note": "  raw condition evidence  "}
        step["byproducts"]["error"] = {
            "attempts": [1, {"duration": 0.25, "successful": False}], "detail": None}
        original = copy.deepcopy(self.payload)
        report = engine.analyze_routes(self.payload)
        reported_step = report["routes"][0]["steps"][0]
        for field in ("conditions", "byproducts"):
            self.assertEqual(reported_step[field], original["routes"][0]["steps"][0][field])
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False, sort_keys=True)), report)
        self.assertEqual(self.payload, original)

    def test_unused_metadata_does_not_require_json_serialization(self):
        self.payload["unused"] = {"not-json"}
        self.payload["routes"][0]["unused"] = float("nan")
        step = self.payload["routes"][0]["steps"][0]
        step["material_provenance"] = {
            "reagents": {"status": "observed", "source": "test fixture", "unused": object()}}
        report = engine.analyze_routes(self.payload)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_metadata_status_validation_still_rejects_invalid_values(self):
        for field in ("conditions", "byproducts"):
            for value in ("unsupported", None, 1, False, [], {}):
                with self.subTest(field=field, value=value):
                    payload = copy.deepcopy(self.payload)
                    payload["routes"][0]["steps"][0][field]["status"] = value
                    with self.assertRaisesRegex(ValueError, field + r"\.status"):
                        engine.analyze_routes(payload)

    def test_array_validation_preserves_raw_materials_and_provenance(self):
        route = self.payload["routes"][0]
        route["source"] = "  raw route source \n"
        step = route["steps"][0]
        step["reactants"][0] = "  CCO \t"
        step["products"][0] = "  CCOC(C)=O \t"
        step["material_provenance"] = {}
        for role in (*engine.ROLES[2:], *engine.ADDITIONAL_ROLES):
            step[role] = ["  O \t"]
            step["material_provenance"][role] = {
                "status": "observed", "source": f"  raw {role} evidence \n"}
        step["byproducts"] = {
            "status": "success", "candidates": ["  CC=O \t"], "source": "  raw byproduct evidence \n"}
        original = copy.deepcopy(self.payload)
        report = engine.analyze_routes(self.payload)
        reported_route = report["routes"][0]
        reported_step = reported_route["steps"][0]
        self.assertEqual(reported_route["source"], route["source"])
        self.assertEqual(reported_step["reaction_structures"], {
            "reactants": step["reactants"], "products": step["products"]})
        for role in ("reactants", *engine.ROLES[2:], *engine.ADDITIONAL_ROLES, "byproducts"):
            expected = step["byproducts"]["candidates"] if role == "byproducts" else step[role]
            self.assertEqual([material["input_smiles"] for material in reported_step["materials"]
                              if material["role"] == role], expected)
        for material in reported_step["materials"]:
            role = material["role"]
            self.assertEqual(material["route_source"], route["source"])
            if role == "byproducts":
                self.assertEqual(material["origin"], step["byproducts"]["source"])
            elif role in step["material_provenance"]:
                self.assertEqual(material["origin"], step["material_provenance"][role]["source"])
        self.assertEqual(self.payload, original)

    def test_byproduct_provenance_is_explicit_and_missing_evidence_remains_unknown(self):
        step = self.payload["routes"][0]["steps"][0]
        step["byproducts"] = {
            "status": "success", "candidates": ["CC=O", "not-a-molecule"],
            "source": "Synthetic evaluation attempt",
        }
        report = engine.analyze_routes(self.payload)
        candidates = [material for material in report["routes"][0]["steps"][0]["materials"]
                      if material["role"] == "byproducts"]
        self.assertTrue(all(material["evidence_status"] == "unknown" for material in candidates))
        step["material_provenance"] = {
            "byproducts": {"status": "proposed", "source": "  Synthetic prediction evidence \n"}}
        original = copy.deepcopy(self.payload)
        report = engine.analyze_routes(self.payload)
        candidates = [material for material in report["routes"][0]["steps"][0]["materials"]
                      if material["role"] == "byproducts"]
        self.assertEqual([material["input_smiles"] for material in candidates], ["CC=O", "not-a-molecule"])
        self.assertTrue(all(material["evidence_status"] == "proposed" for material in candidates))
        self.assertTrue(all(material["origin"] == step["material_provenance"]["byproducts"]["source"]
                            for material in candidates))
        self.assertEqual(candidates[1]["status"], "invalid")
        self.assertEqual(report["routes"][0]["invalid_material_count"], 1)
        self.assertEqual(self.payload, original)

    def test_byproduct_provenance_rejects_unsupported_roles_statuses_and_missing_source(self):
        for provenance in (
            {"unsupported": {"status": "proposed", "source": "fixture"}},
            {"byproducts": {"status": "success", "source": "fixture"}},
            {"byproducts": {"status": [], "source": "fixture"}},
            {"byproducts": {"status": "proposed"}},
        ):
            with self.subTest(provenance=provenance):
                payload = copy.deepcopy(self.payload)
                payload["routes"][0]["steps"][0]["material_provenance"] = provenance
                with self.assertRaises(ValueError):
                    engine.analyze_routes(payload)

    def test_all_material_arrays_reject_blank_and_nonstring_items(self):
        for role in (*engine.ROLES, *engine.ADDITIONAL_ROLES, "byproducts"):
            for value in ("", " \t\n", None, 1, {}, []):
                with self.subTest(role=role, value=value):
                    payload = copy.deepcopy(self.payload)
                    step = payload["routes"][0]["steps"][0]
                    if role == "byproducts":
                        step[role] = {"status": "success", "candidates": [value], "source": "test fixture"}
                    else:
                        step[role] = [value]
                    with self.assertRaises(ValueError):
                        engine.analyze_routes(payload)

    def test_synthex_direction_and_missing_conditions(self):
        payload = engine.from_synthex_rendered({"route": [{"id": "s1", "smiles": "CCOC(C)=O>>CCO.CC(=O)O"}]}, "sx", "CCOC(C)=O")
        self.assertEqual(payload["routes"][0]["steps"][0]["reactants"], ["CCO", "CC(=O)O"])
        self.assertEqual(engine.analyze_routes(payload)["routes"][0]["condition_coverage"], 0)
        with self.assertRaises(ValueError):
            engine.from_synthex_rendered({"route": [{"smiles": "bad"}]}, "sx", "CCO")

    def test_synthex_external_summary_preserves_product_not_precursor_groups(self):
        target = "C[NH3+].[Cl-]"
        # The right side could denote a salt plus water or three fragments;
        # an external summary carries no association between those fragments.
        payload = engine.from_synthex_rendered(
            {"route": [{"smiles": target + ">>C[NH3+].[Cl-].O"}]}, "salt", target)
        step = payload["routes"][0]["steps"][0]
        self.assertEqual(step["products"], [target])
        self.assertEqual(step["reactants"], ["C[NH3+]", "[Cl-]", "O"])
        # Canonicalization preserves an explicitly supplied salt as one input;
        # it does not supply missing associations between reactant entries.
        self.assertEqual(engine.canonicalize_smiles("[Cl-].C[NH3+]"), target)
        report = engine.analyze_routes(payload)
        self.assertEqual(report["routes"][0]["target"], engine.canonicalize_smiles(target))
        materials = report["routes"][0]["steps"][0]["materials"]
        self.assertFalse(any(m["canonical_smiles"] == engine.canonicalize_smiles(target) for m in materials))

    @env_support.requires("pydantic")
    def test_synthex_routejson_renderer_imports_connected_charged_precursors(self):
        # Idempotent hydrogen setting isolates the real renderer's fragment
        # contract. This is a synthetic software control, not a synthesis.
        fixture = {
            "format_version": "routejson-1.0",
            "target": "[CH3:1][NH3+:2].[Cl-:3]",
            "steps": [{"id": "S1", "operations": [
                {"op": "set_explicit_h", "map_idx": 2, "n": 3}]}],
        }
        payload = engine.import_synthex(fixture)
        from synthex.routejson import Route, render_plan

        rendered = render_plan(Route.model_validate(fixture).to_op_plan())
        product, precursors = rendered["route"][0]["smiles"].split(">>")
        self.assertEqual(product, "C[NH3+].[Cl-]")
        self.assertCountEqual(precursors.split("."), ["C[NH3+]", "[Cl-]"])
        frontier = rendered["steps"][0]["rendered"]["frontier"]
        self.assertCountEqual([engine.canonicalize_smiles(m["smi"]) for m in frontier],
                              ["C[NH3+]", "[Cl-]"])
        self.assertTrue(all("." not in m["smi"] for m in frontier))
        route = payload["routes"][0]
        step = route["steps"][0]
        self.assertEqual(engine.canonicalize_smiles(route["target"]), product)
        self.assertEqual(step["products"], [product])
        self.assertEqual(step["reactants"], precursors.split("."))
        self.assertNotIn(product, step["reactants"])

    def test_unspecified_atoms_are_invalid_not_evaluated_materials_or_targets(self):
        for smiles in ("*", "C*", "[*:1]C"):
            with self.subTest(smiles=smiles):
                payload = engine.from_synthex_rendered(
                    {"route": [{"smiles": "CCO>>" + smiles}]}, "unknown", "CCO")
                route = engine.analyze_routes(payload)["routes"][0]
                self.assertEqual(route["invalid_material_count"], 1)
                self.assertEqual(route["evaluated_material_count"], 0)
                with self.assertRaises(ValueError):
                    engine.from_synthex_rendered(
                        {"route": [{"smiles": smiles + ">>CCO"}]}, "unknown-target", smiles)

    def test_aizynth_tree_import(self):
        steps = []
        engine._tree_steps({"type": "mol", "smiles": "CCOC(C)=O", "children": [
            {"type": "reaction", "children": [{"type": "mol", "smiles": "CCO"},
                                               {"type": "mol", "smiles": "CC(=O)O"}]}]}, steps)
        self.assertEqual(steps[0]["products"], ["CCOC(C)=O"])
        self.assertEqual(steps[0]["conditions"]["status"], "unknown")


class ExternalResponseTests(unittest.TestCase):
    @patch.object(engine.requests, "post")
    def test_network_failure_is_not_no_alert(self, post):
        post.side_effect = requests.Timeout("test timeout")
        result = engine.predict_byproducts("CCO.CC(=O)O", "CCOC(C)=O")
        self.assertEqual(result["status"], "failed")
        self.assertIn("Timeout", result["error"])
        self.assertIsNone(result["raw_response"])
        self.assertEqual(result["source"], engine.ASKCOS_API)

    @patch.object(engine.requests, "post")
    def test_main_product_and_duplicates_excluded(self, post):
        post.return_value = Mock(json=lambda: {"result": {"predict_expand": [
            {"prd_smiles": "CCOC(C)=O"}, {"prd_smiles": "CC=O"}, {"prd_smiles": "O=CC"}]}})
        result = engine.predict_byproducts("CCO.CC(=O)O", "CCOC(C)=O")
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["assessment"]["status"], "known_concern")

    @patch.object(engine.requests, "post")
    def test_malformed_response_failed(self, post):
        for payload in ({}, {"result": {"predict_expand": [None]}}, None):
            post.return_value = Mock(json=lambda: payload)
            result = engine.predict_byproducts("CCO", "CC=O")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["raw_response"], payload)

    @patch.object(engine.requests, "post")
    def test_response_keeps_raw_method_and_original_score_fields(self, post):
        row = {"prd_smiles": "CC=O", "modes_name": "Synthetic method", "score": 0.42,
               "provider_score": {"value": 0.6, "basis": "Synthetic control"}}
        payload = {"result": {"predict_expand": [row]}, "provider_version": "fixture"}
        post.return_value = Mock(json=lambda: payload)
        result = engine.predict_byproducts("CCO", "CCOC(C)=O")
        self.assertEqual(result["raw_response"], payload)
        self.assertEqual(result["source"], engine.ASKCOS_API)
        self.assertEqual(result["candidates"][0]["mode"], row["modes_name"])
        self.assertEqual(result["candidates"][0]["score"], row["score"])
        self.assertEqual(result["candidates"][0]["provider_score"], row["provider_score"])
        post.return_value = Mock(json=lambda: {"result": {"predict_expand": [{"prd_smiles": "CC=O"}]}})
        no_score = engine.predict_byproducts("CCO", "CCOC(C)=O")
        self.assertNotIn("score", no_score["candidates"][0])

    @patch.object(engine.requests, "post")
    def test_invalid_candidate_smiles_remains_visible_as_unevaluated(self, post):
        post.return_value = Mock(json=lambda: {"result": {"predict_expand": [{"prd_smiles": "not-a-molecule"}]}})
        result = engine.predict_byproducts("CCO", "CCOC(C)=O")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["candidates"][0]["smiles"], "not-a-molecule")
        self.assertEqual(result["candidates"][0]["assessment"]["status"], "invalid")

    @patch.object(engine.requests, "post")
    def test_invalid_input_never_sent(self, post):
        with self.assertRaises(ValueError):
            engine.predict_byproducts("bad", "CC=O")
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
