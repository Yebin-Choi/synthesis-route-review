"""Offline shared-context contracts; synthetic routes are not chemistry evidence."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import engine
import toxicology
import workspace


TARGET = "CCOC(C)=O"


def _step(step_id, target=TARGET):
    return {
        "id": step_id,
        "reactants": ["CCO", "CC(=O)O"],
        "products": [target],
        "reagents": [],
        "solvents": [],
        "conditions": {
            "status": "observed", "source": "Synthetic unit-test control only",
            "description": "Not an experimental observation or synthesis claim.",
        },
        "byproducts": {"status": "not_run", "candidates": []},
    }


def _payload(target=TARGET):
    return {
        "schema_version": "1.0",
        "routes": [
            {"id": route_id, "target": target, "source": "Synthetic unit-test route",
             "steps": [_step("s1", target), _step("s2", target)]}
            for route_id in ("r1", "r2")
        ],
    }


def _loaded():
    return workspace.load_routes(workspace.new_workspace(TARGET), _payload())


def _success(*smiles):
    return {
        "status": "success", "error": None,
        "candidates": [{"smiles": value, "mode": "synthetic control", "score": 0.25}
                       for value in smiles],
        "source": engine.ASKCOS_API,
        "raw_response": {"result": {"predict_expand": [{"prd_smiles": value} for value in smiles]}},
    }


def _model_row(smiles, score=0.2, status="success"):
    canonical = engine.canonicalize_smiles(smiles)
    return {
        "input_smiles": smiles, "canonical_smiles": canonical, "status": status,
        "score": score if status == "success" else None,
        "prediction": ("positive" if score >= 0.5 else "negative") if status == "success" else "abstain",
        "applicability": "unknown", "error": None if status == "success" else "Synthetic failed row",
    }


def _model_evidence(state, *rows):
    model = {"tool": "ADMET-AI", "version": "synthetic-workspace-control", "endpoint": "ames",
             "weights_sha256": {"fixture.pt": "a" * 64}}
    return {
        "schema_version": 1,
        "report_sha256": state["report_sha256"],
        "report_generation": state["report_generation"],
        "model_result": {"model": model, "threshold": 0.5, "assessments": list(rows),
                         "limitations": ["Synthetic software control, not scientific validation."]},
        "metadata": {
            "evaluation_id": str(uuid4()), "completed_at_utc": "2026-11-04T00:00:00+00:00",
            "report_sha256": state["report_sha256"], "report_generation": state["report_generation"],
            "source": "local_ADMET-AI_model_inference", "model": copy.deepcopy(model), "threshold": 0.5,
        },
        "error": None,
    }


def _filter_payload(*materials):
    template = _payload()["routes"][0]
    routes = []
    for index, material in enumerate(materials, 1):
        route = copy.deepcopy(template)
        route["id"] = f"r{index}"
        for step in route["steps"]:
            step["reactants"] = [material]
        routes.append(route)
    return {"schema_version": "1.0", "routes": routes}


class RawSmilesBoundaryTests(unittest.TestCase):
    def test_direct_material_rejects_raw_4097_before_trimming_or_parsing(self):
        for raw in ("C" * 4097, " " * 4094 + "CCO"):
            with self.subTest(raw_prefix=raw[:5]):
                with patch.object(toxicology.Chem, "MolFromSmiles") as parser:
                    self.assertIsNone(toxicology.canonicalize_smiles(raw))
                    result = toxicology.assess_material(raw)
                    parser.assert_not_called()
                self.assertEqual(result["input_smiles"], raw)
                self.assertEqual(result["status"], "invalid")
                self.assertIsNone(result["canonical_smiles"])
                self.assertIsNone(result["name"])
                self.assertIsNone(result["intake_ug_day"])
                self.assertTrue(any("4096" in item and "not parsed or assessed" in item
                                    for item in result["limitations"]))
                self.assertTrue(any("not truncated" in item for item in result["limitations"]))

    def test_raw_4096_boundary_and_valid_recovery_use_the_existing_parser(self):
        raw = " " * 4093 + "CCO"
        parser_function = toxicology.Chem.MolFromSmiles
        with patch.object(toxicology.Chem, "MolFromSmiles", wraps=parser_function) as parser:
            self.assertIsNone(toxicology.canonicalize_smiles(" " + raw))
            parser.assert_not_called()
            self.assertEqual(toxicology.canonicalize_smiles(raw), "CCO")
            result = toxicology.assess_material(raw)
        self.assertEqual(parser.call_count, 2)
        self.assertTrue(all(call.args[0] == "CCO" for call in parser.call_args_list))
        self.assertEqual(result["input_smiles"], raw)
        self.assertEqual(result["canonical_smiles"], "CCO")
        self.assertNotEqual(result["status"], "invalid")

    def test_every_imported_material_role_retains_overlength_as_unassessed_and_recovers(self):
        parser_function = toxicology.Chem.MolFromSmiles

        def bounded_parser(smiles, *args, **kwargs):
            self.assertLessEqual(len(smiles), 4096, "Overlength SMILES reached RDKit")
            return parser_function(smiles, *args, **kwargs)

        for raw in ("C" * 4097, " " * 4094 + "CCO"):
            for role in (*engine.ROLES, *engine.ADDITIONAL_ROLES, "byproducts"):
                with self.subTest(role=role, raw_prefix=raw[:5]):
                    payload = _payload()
                    step = payload["routes"][0]["steps"][0]
                    if role == "byproducts":
                        step[role] = {"status": "success", "candidates": [raw],
                                      "source": "Synthetic overlength candidate"}
                        values = step[role]["candidates"]
                    else:
                        values = step.setdefault(role, [])
                        values.append(raw)
                    original = copy.deepcopy(payload)
                    initial = workspace.new_workspace(TARGET)
                    with patch.object(toxicology.Chem, "MolFromSmiles", side_effect=bounded_parser) as parser:
                        loaded = workspace.load_routes(initial, payload)
                    self.assertTrue(parser.called)
                    reported_step = loaded["report"]["routes"][0]["steps"][0]
                    materials = [item for item in reported_step["materials"]
                                 if item["role"] == role and item["input_smiles"] == raw]
                    self.assertEqual(len(materials), 1)
                    self.assertEqual(materials[0]["status"], "invalid")
                    self.assertIsNone(materials[0]["canonical_smiles"])
                    self.assertIsNone(materials[0]["intake_ug_day"])
                    self.assertTrue(any("4096" in item for item in materials[0]["limitations"]))
                    self.assertEqual(loaded["report"]["routes"][0]["invalid_material_count"], 1)
                    self.assertTrue(any(raw in item for item in reported_step["missing_information"]))
                    self.assertTrue(any(reason["code"] == "invalid_material" and reason["smiles"] == raw
                                        for reason in loaded["selection"]["routes"][0]["reasons"]))
                    self.assertEqual(payload, original)
                    self.assertEqual(loaded["original_input"], original)
                    self.assertEqual(loaded["active_input"], original)
                    self.assertIsNone(initial["report"])
                    previous = copy.deepcopy(loaded)
                    values[-1] = "CO"
                    with patch.object(toxicology.Chem, "MolFromSmiles", side_effect=bounded_parser):
                        recovered = workspace.load_routes(loaded, payload)
                    self.assertEqual(recovered["report"]["routes"][0]["invalid_material_count"], 0)
                    self.assertTrue(any(item["role"] == role and item["canonical_smiles"] == "CO"
                                        for item in recovered["report"]["routes"][0]["steps"][0]["materials"]))
                    self.assertEqual(loaded, previous)
                    self.assertEqual(loaded["original_input"], original)

    def test_byproduct_application_cannot_bypass_raw_limit_and_preserves_original(self):
        parser_function = toxicology.Chem.MolFromSmiles

        def bounded_parser(smiles, *args, **kwargs):
            self.assertLessEqual(len(smiles), 4096, "Overlength SMILES reached RDKit")
            return parser_function(smiles, *args, **kwargs)

        for raw in ("C" * 4097, " " * 4094 + "CCO"):
            with self.subTest(raw_prefix=raw[:5]):
                state = _loaded()
                original = copy.deepcopy(state)
                request = workspace.make_byproduct_request(state, TARGET)
                with patch.object(toxicology.Chem, "MolFromSmiles", side_effect=bounded_parser):
                    updated = workspace.apply_byproduct_result(state, request, _success(raw))
                self.assertTrue(updated["byproduct_attempts"][-1]["applied"])
                self.assertEqual(workspace.selected_step(updated)["byproducts"]["candidates"], [raw])
                candidate = next(item for item in updated["report"]["routes"][0]["steps"][0]["materials"]
                                 if item["role"] == "byproducts")
                self.assertEqual(candidate["input_smiles"], raw)
                self.assertEqual(candidate["status"], "invalid")
                self.assertEqual(candidate["evidence_status"], "proposed")
                self.assertIsNone(candidate["canonical_smiles"])
                self.assertEqual(updated["report"]["routes"][0]["invalid_material_count"], 1)
                self.assertEqual(updated["original_input"], original["original_input"])
                self.assertEqual(state, original)
                request = workspace.make_byproduct_request(updated, TARGET)
                recovered = workspace.apply_byproduct_result(updated, request, _success("CO"))
                self.assertTrue(recovered["byproduct_attempts"][-1]["applied"])
                self.assertEqual(recovered["report"]["routes"][0]["invalid_material_count"], 0)
                self.assertEqual(recovered["original_input"], original["original_input"])
                self.assertEqual(workspace.selected_step(updated)["byproducts"]["candidates"], [raw])

    def test_overlength_reactants_or_products_are_rejected_before_external_request(self):
        raw = " " * 4094 + "CCO"
        with patch.object(engine.requests, "post") as post:
            for reactants, product in ((raw, TARGET), ("CCO", raw)):
                with self.subTest(reactants=reactants[:5], product=product[:5]):
                    with self.assertRaises(ValueError):
                        engine.predict_byproducts(reactants, product)
            post.assert_not_called()
        payload = _payload()
        payload["routes"][0]["steps"][0]["reactants"].append(raw)
        state = workspace.load_routes(workspace.new_workspace(TARGET), payload)
        with self.assertRaises(ValueError):
            workspace.make_byproduct_request(state, TARGET)


class TargetAndHashTests(unittest.TestCase):
    def test_new_context_retains_exact_raw_text_without_fake_evidence(self):
        raw = "  [CH3:1][CH2:2][OH:3] \t"
        state = workspace.new_workspace(raw)
        self.assertEqual(state["raw_target"], raw)
        self.assertEqual(state["target"], "CCO")
        self.assertEqual(set(state), {
            "workspace_id", "context_revision", "raw_target", "target",
            "original_input", "active_input", "report", "report_sha256", "report_generation",
            "route_id", "step_id", "policy", "selection", "byproduct_attempts",
            "model_evidence", "model_evidence_state", "model_evidence_history",
        })
        for key in ("original_input", "active_input", "report", "report_sha256", "route_id",
                    "step_id", "selection", "model_evidence", "report_generation"):
            self.assertIsNone(state[key], key)
        self.assertEqual(state["byproduct_attempts"], [])
        self.assertEqual(state["model_evidence_history"], [])
        self.assertEqual(workspace.model_evidence_status(state)["status"], "missing")
        self.assertTrue(state["policy"]["strict_alerts"])
        self.assertIsNone(workspace.selected_step(state))
        self.assertNotIn("consent", state)
        other = workspace.new_workspace("CCO")
        self.assertEqual(str(UUID(state["workspace_id"])), state["workspace_id"])
        self.assertEqual(state["context_revision"], 0)
        self.assertNotEqual(state["workspace_id"], other["workspace_id"])
        state["policy"]["excluded_smiles"].append("O")
        self.assertEqual(other["policy"]["excluded_smiles"], [])

    def test_invalid_targets_raise_korean_errors(self):
        for value in (None, 1, {}, [], "", " \t", "일반문장", "C1CC", "C(C", "C*", "CCO extra"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "[가-힣]"):
                    workspace.validate_target(value)

    def test_raw_length_is_guarded_before_the_parser_without_truncation(self):
        boundary = "C" + " " * 4095
        self.assertEqual(workspace.validate_target(boundary), "C")
        self.assertEqual(workspace.new_workspace(boundary)["raw_target"], boundary)
        for value in ("C" * 4097, " " + "C" * 4096):
            with self.subTest(length=len(value)):
                with patch.object(engine, "canonicalize_smiles") as parser:
                    with self.assertRaisesRegex(ValueError, "4096"):
                        workspace.validate_target(value)
                    parser.assert_not_called()
        with patch.object(engine, "canonicalize_smiles", return_value="C") as parser:
            value = "C" * 4096
            workspace.validate_target(value)
            parser.assert_called_once_with(value)

    def test_identity_does_not_merge_salts_charge_isotopes_stereo_or_tautomers(self):
        for left, right in (
            ("CC(=O)[O-].[Na+]", "CC(=O)O"),
            ("[NH4+]", "N"),
            ("[13CH3]CO", "CCO"),
            ("F[C@H](Cl)Br", "F[C@@H](Cl)Br"),
            ("CC=O", "C=CO"),
        ):
            with self.subTest(left=left, right=right):
                self.assertNotEqual(workspace.validate_target(left), workspace.validate_target(right))
        self.assertEqual(workspace.validate_target("[CH3:1][CH2:2][OH:3]"), "CCO")
        self.assertEqual(workspace.validate_target("[Na+].CC(=O)[O-]"),
                         workspace.validate_target("CC(=O)[O-].[Na+]"))

    def test_report_hash_is_strict_deterministic_json_and_value_sensitive(self):
        report = {"z": [1, None, True, "한글"], "a": {"b": 2.5}}
        original = copy.deepcopy(report)
        expected = hashlib.sha256(json.dumps(
            report, ensure_ascii=False, sort_keys=True, allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        self.assertEqual(workspace.report_hash(report), expected)
        self.assertEqual(workspace.report_hash({"a": {"b": 2.5}, "z": report["z"]}), expected)
        self.assertEqual(report, original)
        changed = copy.deepcopy(report)
        changed["z"].reverse()
        self.assertNotEqual(workspace.report_hash(changed), expected)
        cyclic = {}
        cyclic["self"] = cyclic
        for invalid in (None, [], {1: "non-string key"}, {"tuple": (1, 2)}, cyclic,
                        {"set": {1}}, {"x": float("nan")}, {"x": float("inf")},
                        {"x": float("-inf")}, {"x": object()}):
            with self.subTest(kind=type(invalid).__name__):
                with self.assertRaises(ValueError):
                    workspace.report_hash(invalid)


class InputAndSelectionTests(unittest.TestCase):
    def test_load_analyses_whole_input_and_keeps_independent_source_snapshots(self):
        state = workspace.new_workspace("  " + TARGET + "  ")
        payload = _payload()
        before = copy.deepcopy(payload)
        with patch.object(engine, "analyze_routes", wraps=engine.analyze_routes) as analyze:
            with patch.object(workspace, "review_route_eligibility", wraps=workspace.review_route_eligibility) as select:
                loaded = workspace.load_routes(state, payload)
        analyze.assert_called_once()
        self.assertEqual(analyze.call_args.args[0], before)
        select.assert_called_once_with(loaded["report"], None, strict_alerts=True, excluded_smiles=[],
                                       model_status="missing", model_error=None)
        self.assertEqual(loaded["original_input"], before)
        self.assertEqual(loaded["active_input"], before)
        self.assertEqual(payload, before)
        self.assertIsNone(state["report"])
        self.assertEqual((loaded["route_id"], loaded["step_id"]), ("r1", "s1"))
        self.assertEqual(loaded["report_sha256"], workspace.report_hash(loaded["report"]))
        payload["routes"][0]["steps"][0]["reactants"].append("O")
        loaded["active_input"]["routes"][0]["steps"][0]["conditions"]["description"] = "changed active input"
        self.assertEqual(loaded["original_input"], before)
        self.assertEqual(loaded["report"]["routes"][0]["steps"][0]["conditions"],
                         before["routes"][0]["steps"][0]["conditions"])

    def test_target_matching_uses_exact_canonical_identity(self):
        payload = _payload("CCO")
        payload["routes"][0]["target"] = "[CH3:1][CH2:2][OH:3]"
        state = workspace.load_routes(workspace.new_workspace("CCO"), payload)
        self.assertEqual(state["target"], "CCO")
        self.assertEqual(state["original_input"]["routes"][0]["target"], payload["routes"][0]["target"])
        for target, wrong in (("CCO", TARGET), ("CC(=O)O", "CC(=O)[O-].[Na+]"),
                              ("[NH4+]", "N"), ("[13CH3]CO", "CCO"),
                              ("F[C@H](Cl)Br", "F[C@@H](Cl)Br")):
            with self.subTest(target=target, imported=wrong):
                original = workspace.new_workspace(target)
                with patch.object(engine, "analyze_routes") as analyze:
                    with self.assertRaisesRegex(ValueError, "목표물"):
                        workspace.load_routes(original, _payload(wrong))
                    analyze.assert_not_called()
                self.assertIsNone(original["report"])

    def test_mixed_or_unrelated_payload_does_not_switch_target_or_mutate_success(self):
        state = _loaded()
        original = copy.deepcopy(state)
        mixed = _payload()
        mixed["routes"][1]["target"] = "CCO"
        mixed["routes"][1]["steps"][0]["products"] = ["CCO"]
        for payload in (mixed, _payload("CCO")):
            with self.subTest(targets=[route["target"] for route in payload["routes"]]):
                with self.assertRaisesRegex(ValueError, "목표물"):
                    workspace.load_routes(state, payload)
                self.assertEqual(state, original)
        switched = workspace.new_workspace("CCO")
        self.assertIsNone(switched["report"])
        self.assertEqual(workspace.load_routes(switched, _payload("CCO"))["target"], "CCO")

    def test_reload_resets_derived_evidence_but_preserves_policy(self):
        state = workspace.set_policy(_loaded(), True, ["O"])
        state = workspace.set_model_evidence(state, _model_evidence(state, _model_row("CCO")))
        state.update(byproduct_attempts=[{"applied": False, "error": "old attempt"}])
        reloaded = workspace.load_routes(state, _payload())
        self.assertEqual(reloaded["policy"], state["policy"])
        for key in ("model_evidence",):
            self.assertIsNone(reloaded[key])
        self.assertEqual(reloaded["byproduct_attempts"], state["byproduct_attempts"])
        self.assertTrue(reloaded["model_evidence_history"])
        before = copy.deepcopy(state)
        with patch.object(engine, "analyze_routes", side_effect=RuntimeError("analysis control")):
            with self.assertRaisesRegex(RuntimeError, "analysis control"):
                workspace.load_routes(state, _payload())
        self.assertEqual(state, before)

    def test_selected_source_step_keeps_target_and_cannot_mutate_workspace(self):
        state = _loaded()
        step = workspace.selected_step(state)
        self.assertEqual(step, state["active_input"]["routes"][0]["steps"][0])
        self.assertIn(TARGET, step["products"])
        self.assertFalse(any(material["canonical_smiles"] == TARGET
                             for material in state["report"]["routes"][0]["steps"][0]["materials"]))
        step["reactants"].append("O")
        self.assertNotIn("O", workspace.selected_step(state)["reactants"])

    def test_step_and_route_switch_invalidate_only_stage_bound_values(self):
        state = _loaded()
        state = workspace.set_model_evidence(state, _model_evidence(state, _model_row("CCO")))
        state.update(byproduct_attempts=[{"applied": False}])
        before = copy.deepcopy(state)
        for route_id, step_id in (("r1", "s2"), ("r2", "s1")):
            with self.subTest(route=route_id, step=step_id):
                switched = workspace.select_step(state, route_id, step_id)
                for key in ("report", "report_sha256", "active_input", "original_input", "selection",
                            "model_evidence", "byproduct_attempts"):
                    self.assertEqual(switched[key], state[key], key)
                self.assertEqual(switched["workspace_id"], state["workspace_id"])
                self.assertEqual(switched["context_revision"], state["context_revision"] + 1)
                self.assertNotEqual(workspace.context_key(switched), workspace.context_key(state))
        self.assertEqual(workspace.select_step(state, "r1", "s1"), state)
        for route_id, step_id in (("missing", "s1"), ("r1", "missing"), (None, "s1")):
            with self.assertRaises(ValueError):
                workspace.select_step(state, route_id, step_id)
        with self.assertRaises(ValueError):
            workspace.select_step(workspace.new_workspace(TARGET), "r1", "s1")
        self.assertEqual(state, before)

    def test_stage_return_and_target_return_never_restore_an_old_context_key(self):
        initial = _loaded()
        other_step = workspace.select_step(initial, "r1", "s2")
        returned = workspace.select_step(other_step, "r1", "s1")
        self.assertEqual(returned["context_revision"], initial["context_revision"] + 2)
        self.assertEqual(returned["report_sha256"], initial["report_sha256"])
        self.assertNotEqual(workspace.context_key(returned), workspace.context_key(initial))
        recreated = _loaded()
        self.assertEqual(recreated["context_revision"], initial["context_revision"])
        self.assertEqual(recreated["report_sha256"], initial["report_sha256"])
        self.assertNotEqual(recreated["workspace_id"], initial["workspace_id"])
        self.assertNotEqual(workspace.context_key(recreated), workspace.context_key(initial))

    def test_each_accepted_load_creates_a_new_bound_report_generation(self):
        initial = workspace.new_workspace(TARGET)
        loaded = workspace.load_routes(initial, _payload())
        self.assertEqual(loaded["context_revision"], initial["context_revision"] + 1)
        reloaded = workspace.load_routes(loaded, _payload())
        self.assertEqual(reloaded["context_revision"], loaded["context_revision"] + 1)
        self.assertNotEqual(reloaded["report_generation"], loaded["report_generation"])
        self.assertEqual(reloaded["report_sha256"], loaded["report_sha256"])
        self.assertNotEqual(workspace.context_key(reloaded), workspace.context_key(loaded))
        changed_input = _payload()
        changed_input["routes"][0]["source"] = "A different synthetic source"
        changed = workspace.load_routes(loaded, changed_input)
        self.assertEqual(changed["context_revision"], loaded["context_revision"] + 1)
        selected = workspace.select_step(loaded, "r2", "s2")
        reset = workspace.load_routes(selected, _payload())
        self.assertEqual(reset["context_revision"], selected["context_revision"] + 1)

    def test_policy_reuses_frozen_evidence_without_analysis_model_or_network(self):
        payload = _payload()
        payload["routes"][0]["steps"][0]["reagents"] = ["CC=O"]
        state = workspace.load_routes(workspace.new_workspace(TARGET), payload)
        state = workspace.set_model_evidence(state, _model_evidence(state, _model_row("CCO"), _model_row("CC=O")))
        before = copy.deepcopy(state)
        with patch.object(engine, "analyze_routes") as analyze:
            with patch.object(engine, "assess_material") as model_or_rules:
                with patch.object(engine, "predict_byproducts") as predict:
                    with patch.object(engine.requests, "post") as post:
                        changed = workspace.set_policy(state, True, ["O", "[OH2:1]"])
        analyze.assert_not_called()
        model_or_rules.assert_not_called()
        predict.assert_not_called()
        post.assert_not_called()
        self.assertEqual(changed["policy"]["excluded_smiles"], ["O"])
        self.assertEqual(changed["selection"]["routes"][0]["status"], "excluded")
        for key in ("report", "report_sha256", "report_generation"):
            self.assertEqual(changed[key], state[key], key)
        for key in ("model_result", "metadata"):
            self.assertEqual(changed["model_evidence"][key], state["model_evidence"][key])
        self.assertEqual(workspace.context_key(changed), workspace.context_key(state))
        self.assertEqual(state, before)
        changed["model_evidence"]["model_result"]["assessments"][0]["score"] = 0.8
        self.assertEqual(state["model_evidence"]["model_result"]["assessments"][0]["score"], 0.2)

    def test_policy_validation_and_preload_policy_are_atomic(self):
        state = _loaded()
        before = copy.deepcopy(state)
        for strict, exclusions in ((1, []), ("true", []), (False, "CCO"), (False, ("CCO",)),
                                   (False, [None]), (False, ["bad"]), (False, [TARGET])):
            with self.subTest(strict=strict, exclusions=exclusions):
                with self.assertRaises(ValueError):
                    workspace.set_policy(state, strict, exclusions)
                self.assertEqual(state, before)
        initial = workspace.set_policy(workspace.new_workspace(TARGET), True, ["CCO"])
        self.assertIsNone(initial["selection"])
        loaded = workspace.load_routes(initial, _payload())
        self.assertTrue(all(route["status"] == "excluded" for route in loaded["selection"]["routes"]))


class WorkspaceEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(engine.requests, "post", side_effect=AssertionError("No API call"))
        self.post = self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.post.assert_not_called)
        self.inference = patch("admet_service.assess_admet_materials", side_effect=AssertionError("No inference"))
        self.model = self.inference.start()
        self.addCleanup(self.inference.stop)
        self.addCleanup(self.model.assert_not_called)

    def test_listed_material_excludes_the_route_and_keeps_its_reasons(self):
        # Only an ICH M7-listed material removes a route under the default policy.
        payload = _filter_payload("CC=O", "CCC")
        original = copy.deepcopy(payload)
        state = workspace.load_routes(workspace.new_workspace(TARGET), payload)
        self.assertEqual((state["route_id"], state["step_id"]), ("r2", "s1"))
        self.assertEqual([route["id"] for route in workspace.eligible_routes(state)], ["r2"])
        excluded = workspace.excluded_routes(state)
        self.assertEqual([route["id"] for route in excluded], ["r1"])
        self.assertIn("known_concern", {row["code"] for row in excluded[0]["reasons"]})
        self.assertIn("admet_unassessed", {row["code"] for row in excluded[0]["reasons"]})
        self.assertEqual(state["selection"]["routes"][1]["status"], "insufficient_data")
        self.assertEqual(state["original_input"], original)
        self.assertEqual(state["active_input"], original)
        self.assertEqual(payload, original)
        excluded[0]["reasons"].clear()
        self.assertTrue(workspace.excluded_routes(state)[0]["reasons"])
        route_copy = workspace.eligible_routes(state)[0]
        route_copy["steps"].clear()
        self.assertTrue(state["report"]["routes"][1]["steps"])
        self.assertFalse(any(item["canonical_smiles"] == TARGET for route in state["report"]["routes"]
                             for step in route["steps"] for item in step["materials"]))

    def test_rule_only_and_conditional_concerns_are_demoted_not_excluded(self):
        # A structural alert or a conditional warning now lowers a route's rank
        # instead of removing it; the reason itself is still on record.
        for material, reason in (("C1CO1", "structural_alert"), ("CNC", "conditional_warning")):
            with self.subTest(material=material):
                payload = _filter_payload(material, "CCC")
                if material == "CNC":
                    payload["routes"][0]["steps"][0]["reagents"] = ["[Na+].[O-]N=O"]
                original = copy.deepcopy(payload)
                state = workspace.load_routes(workspace.new_workspace(TARGET), payload)
                self.assertEqual((state["route_id"], state["step_id"]), ("r1", "s1"))
                self.assertEqual([route["id"] for route in workspace.eligible_routes(state)],
                                 ["r1", "r2"])
                self.assertEqual(workspace.excluded_routes(state), [])
                concerning, clean = state["selection"]["routes"]
                self.assertEqual(concerning["status"], "needs_review")
                self.assertEqual(clean["status"], "insufficient_data")
                self.assertIn(reason, {row["code"] for row in concerning["reasons"]})
                self.assertIn("admet_unassessed", {row["code"] for row in concerning["reasons"]})
                self.assertEqual(clean["rank_penalty"], 0.0)
                self.assertGreater(concerning["rank_penalty"], clean["rank_penalty"])
                self.assertEqual(state["original_input"], original)
                self.assertEqual(state["active_input"], original)
                self.assertEqual(payload, original)
                route_copy = workspace.eligible_routes(state)[0]
                route_copy["steps"].clear()
                self.assertTrue(state["report"]["routes"][0]["steps"])
                self.assertFalse(any(item["canonical_smiles"] == TARGET for route in state["report"]["routes"]
                                     for step in route["steps"] for item in step["materials"]))

    def test_all_excluded_has_nullable_selection_and_rejects_member_route_selection(self):
        # Two listed materials, so every route is excluded under the default policy.
        payload = _filter_payload("CC=O", "C=O")
        state = workspace.load_routes(workspace.new_workspace(TARGET), payload)
        self.assertIsNone(state["route_id"])
        self.assertIsNone(state["step_id"])
        self.assertIsNone(workspace.selected_step(state))
        self.assertEqual(workspace.eligible_routes(state), [])
        self.assertEqual(len(workspace.excluded_routes(state)), 2)
        self.assertEqual(len(state["report"]["routes"]), 2)
        self.assertEqual(state["original_input"], payload)
        before = copy.deepcopy(state)
        for route_id in ("r1", "r2"):
            with self.assertRaisesRegex(ValueError, "제외"):
                workspace.select_step(state, route_id, "s1")
        with self.assertRaises(ValueError):
            workspace.make_byproduct_request(state, TARGET)
        self.assertEqual(state, before)

    def test_target_is_not_an_impurity_filter_input_and_independent_assessment_remains_distinct(self):
        target = "CC=O"
        payload = _payload(target)
        state = workspace.load_routes(workspace.new_workspace(target), payload)
        self.assertEqual(len(workspace.eligible_routes(state)), 2)
        self.assertEqual(workspace.excluded_routes(state), [])
        self.assertFalse(any(item["canonical_smiles"] == target for route in state["report"]["routes"]
                             for step in route["steps"] for item in step["materials"]))
        standalone = toxicology.assess_material(target)
        self.assertEqual(standalone["status"], "known_concern")
        self.assertEqual(standalone["input_smiles"], target)
        self.assertEqual(state["original_input"], payload)

    def test_model_only_union_demotes_and_keeps_and_comparison_separate(self):
        state = workspace.load_routes(workspace.new_workspace(TARGET), _filter_payload("CCO", "CCC"))
        original = copy.deepcopy(state)
        evidence = _model_evidence(state, _model_row("CCO", 0.8))
        changed = workspace.set_model_evidence(state, evidence)
        # A model-only positive now demotes, so the active route is not removed
        # and the same-material AND comparison stays a separate result.
        self.assertEqual(changed["route_id"], "r1")
        self.assertEqual(changed["selection"]["routes"][0]["status"], "needs_review")
        self.assertGreater(changed["selection"]["routes"][0]["rank_penalty"],
                           changed["selection"]["routes"][1]["rank_penalty"])
        self.assertIn("admet_positive",
                      {row["code"] for row in changed["selection"]["routes"][0]["reasons"]})
        self.assertEqual(changed["model_evidence"]["combined_review"]["routes"][0]["status"], "needs_review")
        self.assertEqual(changed["model_evidence"]["combined_review_policy"], "same_material_AND")
        self.assertEqual(changed["selection"]["policy"]["rule_admet_exclusion"], "rule_OR_current_model")
        self.assertEqual(workspace.model_evidence_status(changed)["status"], "available")
        self.assertEqual(changed["report_generation"], state["report_generation"])
        self.assertEqual(changed["report"], state["report"])
        self.assertEqual(state, original)
        # An explicit prohibition still excludes the active route, and that
        # exclusion still clears stage calculations and stales old requests.
        request = workspace.make_byproduct_request(changed, TARGET)
        prohibited = workspace.set_policy(changed, True, ["CCO"])
        self.assertEqual(prohibited["selection"]["routes"][0]["status"], "excluded")
        self.assertEqual(prohibited["route_id"], "r2")
        self.assertEqual(prohibited["context_revision"], changed["context_revision"] + 1)
        stale = workspace.apply_byproduct_result(prohibited, request, _success("CO"))
        self.assertTrue(stale["byproduct_attempts"][-1]["stale"])
        self.assertFalse(stale["byproduct_attempts"][-1]["applied"])
        self.assertEqual(stale["report"], changed["report"])
        copied = workspace.current_model_evidence(changed)
        copied["model_result"]["assessments"].clear()
        self.assertTrue(changed["model_evidence"]["model_result"]["assessments"])
        self.assertEqual(evidence["model_result"]["assessments"][0]["score"], 0.8)

    def test_negative_model_never_overrides_rule_and_incomplete_rows_never_become_negative(self):
        state = workspace.load_routes(workspace.new_workspace(TARGET), _filter_payload("CC=O", "CCO", "CCC"))
        evidence = _model_evidence(state, _model_row("CC=O", 0.2), _model_row("CCO", status="failed"))
        changed = workspace.set_model_evidence(state, evidence)
        self.assertEqual([route["status"] for route in changed["selection"]["routes"]],
                         ["excluded", "insufficient_data", "insufficient_data"])
        self.assertEqual(changed["route_id"], "r2")
        for decision in changed["selection"]["routes"][1:]:
            codes = {reason["code"] for reason in decision["reasons"]}
            self.assertIn("admet_unassessed", codes)
            self.assertNotIn("admet_negative", codes)
        self.assertEqual(workspace.model_evidence_status(changed)["status"], "available")
        self.assertEqual([route["id"] for route in workspace.eligible_routes(changed)], ["r2", "r3"])

    def test_current_model_demotes_every_route_without_fabricating_alternatives(self):
        # Every route shares the positive material, so all of them are demoted.
        # No route is removed, no replacement route is invented, and the
        # unchanged selection leaves stage calculations valid.
        state = _loaded()
        changed = workspace.set_model_evidence(state, _model_evidence(state, _model_row("CCO", 0.8)))
        self.assertEqual([route["id"] for route in workspace.eligible_routes(changed)], ["r1", "r2"])
        self.assertEqual(workspace.excluded_routes(changed), [])
        self.assertTrue(all(route["status"] == "needs_review"
                            for route in changed["selection"]["routes"]))
        self.assertTrue(all(route["rank_penalty"] > 0
                            for route in changed["selection"]["routes"]))
        self.assertEqual((changed["route_id"], changed["step_id"]),
                         (state["route_id"], state["step_id"]))
        self.assertEqual(len(changed["report"]["routes"]), 2)
        self.assertEqual(changed["context_revision"], state["context_revision"])

    def test_same_current_evidence_reuses_calculations_and_failed_retry_archives_success(self):
        state = workspace.load_routes(workspace.new_workspace(TARGET), _filter_payload("CCO", "CCC"))
        evidence = _model_evidence(state, _model_row("CCO", 0.8), _model_row("CCC"))
        current = workspace.set_model_evidence(state, evidence)
        repeated = workspace.set_model_evidence(current, evidence)
        self.assertEqual(repeated["model_evidence_history"], current["model_evidence_history"])
        self.assertEqual(workspace.context_key(repeated), workspace.context_key(current))
        failure = copy.deepcopy(evidence)
        failure["error"] = "Synthetic latest retry failure"
        failed = workspace.set_model_evidence(repeated, failure)
        self.assertIsNone(failed["model_evidence"])
        self.assertEqual(workspace.model_evidence_status(failed)["status"], "failed")
        self.assertEqual(failed["route_id"], "r1")
        self.assertTrue(all(route["status"] == "insufficient_data" for route in failed["selection"]["routes"]))
        self.assertEqual(failed["model_evidence_history"][-2]["evidence"], repeated["model_evidence"])
        self.assertEqual(failed["model_evidence_history"][-1]["evidence"]["error"], failure["error"])
        self.assertTrue(all(reason["code"] != "admet_negative" for route in failed["selection"]["routes"]
                            for reason in route["reasons"]))

    def test_stale_failed_unbound_and_malformed_envelopes_are_exportable_history_only(self):
        state = _loaded()
        valid = _model_evidence(state, _model_row("CCO", 0.8))
        cases = []
        for field, value, expected in (("report_sha256", "b" * 64, "stale"),
                                       ("report_generation", str(uuid4()), "stale"),
                                       ("error", "Synthetic inference failure", "failed"),
                                       ("schema_version", 2, "invalid")):
            evidence = copy.deepcopy(valid)
            evidence[field] = value
            cases.append((evidence, expected))
        unbound = copy.deepcopy(valid)
        del unbound["report_generation"]
        cases.append((unbound, "stale"))
        for field, value in (("threshold", 0.7), ("source", "untrusted"),
                             ("evaluation_id", "invalid"), ("report_generation", str(uuid4()))):
            evidence = copy.deepcopy(valid)
            evidence["metadata"][field] = value
            cases.append((evidence, "invalid"))
        extra_structure = copy.deepcopy(valid)
        extra_structure["model_result"]["assessments"].append(_model_row("CCC", 0.8))
        cases.append((extra_structure, "invalid"))
        unprovenanced = copy.deepcopy(valid)
        unprovenanced["model_result"]["model"]["weights_sha256"] = {}
        unprovenanced["metadata"]["model"]["weights_sha256"] = {}
        cases.append((unprovenanced, "invalid"))
        conflicting_score = copy.deepcopy(valid)
        conflicting_score["model_result"]["assessments"][0]["score"] = 0.2
        cases.append((conflicting_score, "invalid"))
        nonfinite = copy.deepcopy(valid)
        nonfinite["model_result"]["assessments"][0]["score"] = float("nan")
        cases.append((nonfinite, "invalid"))
        for evidence, expected in cases:
            with self.subTest(expected=expected, evidence=evidence):
                changed = workspace.set_model_evidence(state, evidence)
                self.assertEqual(workspace.model_evidence_status(changed)["status"], expected)
                self.assertIsNone(workspace.current_model_evidence(changed))
                self.assertIsNone(changed["model_evidence"])
                self.assertTrue(changed["model_evidence_history"])
                self.assertEqual(changed["route_id"], state["route_id"])
                self.assertEqual(workspace.context_key(changed), workspace.context_key(state))
                self.assertTrue(all(route["status"] == "insufficient_data" for route in changed["selection"]["routes"]))
                self.assertTrue(all(reason["code"] != "admet_negative" for route in changed["selection"]["routes"]
                                    for reason in route["reasons"]))
                self.assertEqual(json.loads(json.dumps(changed, allow_nan=False)), changed)
                repeated = workspace.set_model_evidence(changed, evidence)
                self.assertEqual(repeated["model_evidence_history"], changed["model_evidence_history"])

    def test_identical_report_regeneration_cannot_reuse_old_positive_evidence(self):
        initial = _loaded()
        evidence = _model_evidence(initial, _model_row("CCO", 0.8))
        excluded = workspace.set_model_evidence(initial, evidence)
        reloaded = workspace.load_routes(excluded, _payload())
        self.assertEqual(reloaded["report_sha256"], initial["report_sha256"])
        self.assertNotEqual(reloaded["report_generation"], initial["report_generation"])
        self.assertEqual(reloaded["route_id"], "r1")
        self.assertIsNone(reloaded["model_evidence"])
        self.assertTrue(reloaded["model_evidence_history"])
        replayed = workspace.set_model_evidence(reloaded, evidence)
        self.assertEqual(workspace.model_evidence_status(replayed)["status"], "stale")
        self.assertEqual(len(workspace.eligible_routes(replayed)), 2)
        self.assertEqual(replayed["original_input"], initial["original_input"])

    def test_policy_exclusion_moves_active_stage_and_preserves_history_then_context_stays_fresh(self):
        state = workspace.load_routes(workspace.new_workspace(TARGET), _filter_payload("CCO", "CCC"))
        state.update(byproduct_attempts=[{"applied": False, "error": "Synthetic history"}])
        request = workspace.make_byproduct_request(state, TARGET)
        changed = workspace.set_policy(state, True, ["CCO"])
        self.assertEqual(changed["route_id"], "r2")
        self.assertEqual(changed["report_generation"], state["report_generation"])
        self.assertEqual(changed["context_revision"], state["context_revision"] + 1)
        self.assertEqual(changed["byproduct_attempts"], state["byproduct_attempts"])
        self.assertEqual(changed["original_input"], state["original_input"])
        all_excluded = workspace.set_policy(changed, True, ["CCO", "CCC"])
        self.assertIsNone(all_excluded["route_id"])
        self.assertIsNone(all_excluded["step_id"])
        self.assertEqual(all_excluded["context_revision"], changed["context_revision"] + 1)
        relaxed = workspace.set_policy(changed, False, [])
        self.assertEqual(relaxed["route_id"], "r2")
        self.assertEqual(workspace.context_key(relaxed), workspace.context_key(changed))
        returned = workspace.select_step(relaxed, "r1", "s1")
        self.assertNotEqual(workspace.context_key(returned), workspace.context_key(state))
        stale = workspace.apply_byproduct_result(returned, request, _success("CO"))
        self.assertTrue(stale["byproduct_attempts"][-1]["stale"])

    def test_byproduct_mutation_reconciles_exclusion_and_archives_model_atomically(self):
        state = workspace.load_routes(workspace.new_workspace(TARGET), _filter_payload("CCO", "CCC"))
        state = workspace.set_model_evidence(state, _model_evidence(state, _model_row("CCO"), _model_row("CCC")))
        original = copy.deepcopy(state)
        request = workspace.make_byproduct_request(state, TARGET)
        changed = workspace.apply_byproduct_result(state, request, _success("CC=O"))
        self.assertTrue(changed["byproduct_attempts"][-1]["applied"])
        # A byproduct concern demotes a route, it does not remove one, so the
        # route stays eligible and the selection stays on it. Before
        # 2026-11-06 the alerting byproduct excluded r1 and the selection was
        # pushed onto r2; the 2026-11-04 measurement showed that rule emptying
        # the whole list, so it was changed to a demotion.
        self.assertEqual(changed["route_id"], "r1")
        self.assertEqual(changed["selection"]["routes"][0]["status"], "needs_review")
        self.assertIn("byproduct_concern",
                      {item["code"] for item in changed["selection"]["routes"][0]["reasons"]})
        self.assertIsNone(changed["model_evidence"])
        self.assertNotEqual(changed["report_generation"], state["report_generation"])
        self.assertEqual(changed["context_revision"], state["context_revision"] + 1)
        history = changed["model_evidence_history"][-1]
        self.assertFalse(history["applicable"])
        self.assertEqual(history["evidence"], state["model_evidence"])
        self.assertEqual(changed["original_input"], state["original_input"])
        self.assertEqual(state, original)

    def test_new_load_reconciles_active_exclusion_without_discarding_attempt_history(self):
        state = _loaded()
        state.update(byproduct_attempts=[{"applied": False, "error": "Synthetic older request"}])
        payload = _filter_payload("CC=O", "CCC")
        changed = workspace.load_routes(state, payload)
        self.assertEqual(changed["route_id"], "r2")
        self.assertEqual(changed["step_id"], "s1")
        self.assertEqual(changed["context_revision"], state["context_revision"] + 1)
        self.assertEqual(changed["byproduct_attempts"], state["byproduct_attempts"])
        self.assertEqual(changed["original_input"], payload)
        self.assertEqual(state["original_input"], _payload())

    def test_view_uses_shared_hash_and_explicit_generation_without_inference(self):
        import route_ames_view as view

        state = _loaded()
        evidence = _model_evidence(state, _model_row("CCO", 0.8))
        with patch.object(view.st, "session_state", {view._SESSION_KEY: evidence}), \
                patch.object(view, "assess_admet_materials") as inference:
            snapshot = view.get_route_ames_evidence(state["report"], report_generation=state["report_generation"])
            self.assertEqual(snapshot["report_sha256"], workspace.report_hash(state["report"]))
            self.assertEqual(snapshot["combined_review_policy"], "same_material_AND")
            self.assertIsNone(view.get_route_ames_evidence(state["report"], report_generation=str(uuid4())))
            self.assertIsNone(view.get_route_ames_evidence(state["report"], report_generation=None))
            inference.assert_not_called()
        changed = workspace.set_model_evidence(state, snapshot)
        # The model positive demotes both routes rather than excluding them.
        self.assertEqual(changed["selection"]["counts"]["excluded"], 0)
        self.assertEqual(changed["selection"]["counts"]["needs_review"], 2)
        self.assertEqual(workspace.model_evidence_status(changed)["status"], "available")
        self.assertNotIn("combined_review", evidence)


class ByproductContextTests(unittest.TestCase):
    def setUp(self):
        self.state = workspace.set_policy(_loaded(), False, [])
        self.post_patcher = patch.object(engine.requests, "post", side_effect=AssertionError("No live request"))
        self.post = self.post_patcher.start()
        self.addCleanup(self.post_patcher.stop)
        self.addCleanup(self.post.assert_not_called)

    def assertSnapshotPreserved(self, updated, previous):
        for key in previous:
            if key != "byproduct_attempts":
                self.assertEqual(updated[key], previous[key], key)
        self.assertEqual(len(updated["byproduct_attempts"]), len(previous["byproduct_attempts"]) + 1)
        self.assertFalse(updated["byproduct_attempts"][-1]["applied"])

    def test_request_freezes_uuid_hash_stage_and_exact_payload_without_calling(self):
        state = self.state
        state["active_input"]["routes"][0]["steps"][0]["reactants"] = [" CCO ", " CC(=O)O\t"]
        before = copy.deepcopy(state)
        request = workspace.make_byproduct_request(state, " " + TARGET + " ")
        self.assertEqual(set(request), {"request_id", "report_sha256", "route_id", "step_id",
                                        "reactants_smiles", "product_smiles", "context_key"})
        self.assertEqual(str(UUID(request["request_id"])), request["request_id"])
        self.assertEqual(request["reactants_smiles"], "CCO.CC(=O)O")
        self.assertEqual(request["product_smiles"], TARGET)
        self.assertEqual(request["report_sha256"], state["report_sha256"])
        self.assertEqual(request["context_key"], workspace.context_key(state))
        self.assertEqual((request["route_id"], request["step_id"]), ("r1", "s1"))
        self.assertNotEqual(request["request_id"], workspace.make_byproduct_request(state, TARGET)["request_id"])
        self.assertEqual(state, before)

    def test_request_requires_selected_step_product_membership_and_valid_reactants(self):
        with self.assertRaises(ValueError):
            workspace.make_byproduct_request(workspace.new_workspace(TARGET), TARGET)
        for product in ("CCO", "", "bad", None, "C" * 4097):
            with self.subTest(product=product):
                with self.assertRaises(ValueError):
                    workspace.make_byproduct_request(self.state, product)
        for reactants in (["not-a-molecule"], []):
            with self.subTest(reactants=reactants):
                state = copy.deepcopy(self.state)
                state["active_input"]["routes"][0]["steps"][0]["reactants"] = reactants
                with self.assertRaises(ValueError):
                    workspace.make_byproduct_request(state, TARGET)
        stale_hash = copy.deepcopy(self.state)
        stale_hash["report"]["disclaimer"] += " changed"
        with self.assertRaisesRegex(ValueError, "해시"):
            workspace.make_byproduct_request(stale_hash, TARGET)

    def test_product_confirmation_supports_an_explicit_intermediate_not_only_final_target(self):
        payload = _payload()
        payload["routes"][0]["steps"][0]["products"] = ["CCO"]
        state = workspace.load_routes(workspace.new_workspace(TARGET), payload)
        request = workspace.make_byproduct_request(state, "[CH3:1][CH2:2][OH:3]")
        self.assertEqual(request["product_smiles"], "[CH3:1][CH2:2][OH:3]")
        with self.assertRaises(ValueError):
            workspace.make_byproduct_request(state, TARGET)

    def test_success_reanalyses_all_routes_and_aggregates_and_preserves_original(self):
        payload = _payload()
        # A prior candidate that only demotes, so the route is still selectable
        # here. An ICH M7-listed one excludes the route whatever the alert
        # setting, which is what the applied result below then checks.
        payload["routes"][0]["steps"][1]["byproducts"] = {
            "status": "success", "candidates": ["CCO"], "source": "Synthetic prior candidate"}
        state = workspace.load_routes(workspace.new_workspace(TARGET), payload)
        state = workspace.set_policy(state, False, ["C1CO1"])
        state = workspace.select_step(state, "r1", "s1")
        # CC=O only enters the report with the applied result below, so an
        # envelope naming it here would cover a material the report lacks.
        state = workspace.set_model_evidence(state, _model_evidence(state, _model_row("CCO")))
        self.assertIsNotNone(state["model_evidence"])
        before = copy.deepcopy(state)
        result = _success("CC=O", "C1CO1", "not-a-molecule")
        request = workspace.make_byproduct_request(state, TARGET)
        with patch.object(engine, "analyze_routes", wraps=engine.analyze_routes) as analyze:
            updated = workspace.apply_byproduct_result(state, request, result)
        analyze.assert_called_once()
        self.assertEqual(len(analyze.call_args.args[0]["routes"]), 2)
        self.assertEqual(updated["report"], engine.analyze_routes(updated["active_input"]))
        self.assertEqual(updated["selection"], workspace.review_route_eligibility(updated["report"], **updated["policy"]))
        self.assertEqual(updated["report_sha256"], workspace.report_hash(updated["report"]))
        self.assertNotEqual(updated["report_sha256"], state["report_sha256"])
        self.assertEqual(updated["context_revision"], state["context_revision"] + 1)
        selected = updated["active_input"]["routes"][0]["steps"][0]
        self.assertEqual(selected["byproducts"]["candidates"], ["CC=O", "C1CO1", "not-a-molecule"])
        self.assertEqual(selected["material_provenance"]["byproducts"],
                         {"status": "proposed", "source": engine.ASKCOS_API})
        report_route = updated["report"]["routes"][0]
        self.assertEqual(report_route["known_concern_count"], 1)
        self.assertEqual(report_route["structural_alert_count"], 1)
        self.assertEqual(report_route["invalid_material_count"], 1)
        self.assertEqual(report_route["material_count"], 5)
        self.assertEqual(report_route["byproduct_coverage"], 1)
        self.assertTrue(any("해석 실패" in missing for missing in report_route["steps"][0]["missing_information"]))
        for material in report_route["steps"][0]["materials"]:
            if material["role"] == "byproducts":
                self.assertEqual(material["evidence_status"], "proposed")
                self.assertEqual(material["origin"], engine.ASKCOS_API)
        self.assertEqual(updated["selection"]["routes"][0]["status"], "excluded")
        self.assertEqual(updated["report"]["routes"][1], state["report"]["routes"][1])
        self.assertEqual(updated["active_input"]["routes"][0]["steps"][1], payload["routes"][0]["steps"][1])
        self.assertEqual(updated["original_input"], payload)
        self.assertEqual(state, before)
        self.assertIsNone(updated["model_evidence"])
        attempt = updated["byproduct_attempts"][-1]
        self.assertTrue(attempt["applied"])
        self.assertIsNone(attempt["error"])
        self.assertIsNone(attempt["application_error"])
        self.assertEqual(attempt["result"], result)
        self.assertEqual(attempt["request"], request)
        self.assertEqual(attempt["historical_model_evidence"], {
            "applicable": False, "report_sha256": state["report_sha256"],
            "report_generation": state["report_generation"],
            "evidence": state["model_evidence"],
        })
        result["candidates"][0]["score"] = 0.99
        self.assertEqual(attempt["result"]["candidates"][0]["score"], 0.25)

    def test_success_replaces_selected_candidates_instead_of_accumulating_them(self):
        request = workspace.make_byproduct_request(self.state, TARGET)
        # An alert demotes the route, so it stays selectable and a second
        # response can replace the first. An ICH M7-listed candidate excludes
        # the route instead, which test_success_reanalyses covers.
        first = workspace.apply_byproduct_result(self.state, request, _success("C1CO1"))
        self.assertEqual(first["report"]["routes"][0]["structural_alert_count"], 1)
        second_request = workspace.make_byproduct_request(first, TARGET)
        replaced = workspace.apply_byproduct_result(first, second_request, _success("O"))
        route = replaced["report"]["routes"][0]
        self.assertEqual(workspace.selected_step(replaced)["byproducts"]["candidates"], ["O"])
        self.assertEqual(route["known_concern_count"], 0)
        self.assertEqual(route["structural_alert_count"], 0)
        self.assertEqual(replaced["original_input"], self.state["original_input"])
        self.assertEqual(len(replaced["byproduct_attempts"]), 2)
        self.assertTrue(all(attempt["applied"] for attempt in replaced["byproduct_attempts"]))

    def test_unchanged_report_keeps_applicable_model_and_stage_calculations(self):
        request = workspace.make_byproduct_request(self.state, TARGET)
        first = workspace.apply_byproduct_result(self.state, request, _success("C1CO1"))
        first = workspace.set_model_evidence(first, _model_evidence(first, _model_row("CCO"), _model_row("C1CO1")))
        request = workspace.make_byproduct_request(first, TARGET)
        repeated = workspace.apply_byproduct_result(first, request, _success("C1CO1"))
        self.assertTrue(repeated["byproduct_attempts"][-1]["applied"])
        for key in ("report", "report_sha256", "model_evidence", "context_revision"):
            self.assertEqual(repeated[key], first[key], key)
        self.assertNotIn("historical_model_evidence", repeated["byproduct_attempts"][-1])

    def test_pending_response_uses_current_policy_without_requiring_a_new_context(self):
        request = workspace.make_byproduct_request(self.state, TARGET)
        changed = workspace.set_policy(self.state, True, ["CC=O"])
        self.assertEqual(workspace.context_key(changed), request["context_key"])
        applied = workspace.apply_byproduct_result(changed, request, _success("CC=O"))
        self.assertTrue(applied["byproduct_attempts"][-1]["applied"])
        self.assertFalse(applied["byproduct_attempts"][-1]["stale"])
        self.assertEqual(applied["policy"], changed["policy"])
        self.assertEqual(applied["selection"]["routes"][0]["status"], "excluded")

    def test_failed_retry_keeps_prior_success_and_separates_latest_failure(self):
        first_request = workspace.make_byproduct_request(self.state, TARGET)
        first = workspace.apply_byproduct_result(self.state, first_request, _success("CC=O"))
        request = workspace.make_byproduct_request(first, TARGET)
        failure = {"status": "failed", "candidates": [], "error": "Timeout: test control"}
        failed = workspace.apply_byproduct_result(first, request, failure)
        self.assertSnapshotPreserved(failed, first)
        self.assertTrue(failed["byproduct_attempts"][0]["applied"])
        self.assertEqual(failed["byproduct_attempts"][-1]["error"], failure["error"])
        retry = workspace.make_byproduct_request(failed, TARGET)
        self.assertNotEqual(retry["request_id"], request["request_id"])
        recovered = workspace.apply_byproduct_result(failed, retry, _success("O"))
        self.assertTrue(recovered["byproduct_attempts"][-1]["applied"])
        self.assertEqual(recovered["byproduct_attempts"][1]["error"], failure["error"])

    def test_empty_success_is_not_absence_and_does_not_erase_prior_candidates(self):
        request = workspace.make_byproduct_request(self.state, TARGET)
        first = workspace.apply_byproduct_result(self.state, request, _success("C1CO1"))
        request = workspace.make_byproduct_request(first, TARGET)
        empty = workspace.apply_byproduct_result(first, request, _success())
        self.assertSnapshotPreserved(empty, first)
        self.assertIn("부재", empty["byproduct_attempts"][-1]["error"])
        self.assertTrue(any("관측" in value and "부재" in value
                            for value in empty["byproduct_attempts"][-1]["limitations"]))
        self.assertEqual(workspace.selected_step(empty)["byproducts"]["candidates"], ["C1CO1"])

    def test_malformed_response_never_partially_applies_or_means_absence(self):
        request = workspace.make_byproduct_request(self.state, TARGET)
        malformed = [None, [], {}, {"status": "success", "candidates": "CC=O"},
                     {"status": "success", "candidates": ["CC=O"]},
                     {"status": "success", "candidates": [None]},
                     {"status": "unsupported", "candidates": []},
                     {"status": "success", "candidates": [], "error": "failure"},
                     {"status": "failed", "candidates": [{"smiles": "CC=O"}]},
                     {"status": "success", "candidates": [{"smiles": "CC=O"}, {}]},
                     {"status": "success", "candidates": [{"smiles": "CC=O", "score": float("nan")}]}]
        malformed.extend({"status": "success", "candidates": [{"smiles": value}]}
                         for value in ("", " \t", None, 1, []))
        malformed.append({"status": "success", "candidates": [{"smiles": "O"}] * 201})
        for result in malformed:
            with self.subTest(result=result):
                updated = workspace.apply_byproduct_result(self.state, request, result)
                self.assertSnapshotPreserved(updated, self.state)
                self.assertIsNotNone(updated["byproduct_attempts"][-1]["application_error"])
                self.assertIsNotNone(updated["byproduct_attempts"][-1]["error"])

    def test_nonjson_evidence_is_diagnosed_bounded_and_remains_strictly_exportable(self):
        request = workspace.make_byproduct_request(self.state, TARGET)
        cyclic = {}
        cyclic["self"] = cyclic
        deeply_nested = []
        for _ in range(2000):
            deeply_nested = [deeply_nested]
        for invalid in ({"score": float("nan")}, {"score": float("inf")},
                        {"not_json": {1}}, cyclic, {"bad_key": {1: "x"}},
                        {"score": float("nan"), "padding": "x" * 5000}):
            with self.subTest(kind=type(invalid).__name__):
                updated = workspace.apply_byproduct_result(self.state, request, invalid)
                self.assertSnapshotPreserved(updated, self.state)
                recorded = updated["byproduct_attempts"][-1]["result"]
                self.assertEqual(recorded["status"], "invalid_evidence")
                self.assertTrue(recorded["diagnostic"])
                self.assertLessEqual(len(recorded["raw_repr"]), 4096)
                self.assertEqual(recorded["raw_repr_truncated"], len(repr(invalid)) > 4096)
                self.assertEqual(json.loads(json.dumps(updated, allow_nan=False)), updated)
        bad_request = {**request, "invalid": float("nan")}
        updated = workspace.apply_byproduct_result(self.state, bad_request, _success("CC=O"))
        self.assertSnapshotPreserved(updated, self.state)
        self.assertEqual(updated["byproduct_attempts"][-1]["request"]["status"], "invalid_evidence")
        json.dumps(updated, allow_nan=False)
        updated = workspace.apply_byproduct_result(self.state, request, deeply_nested)
        self.assertSnapshotPreserved(updated, self.state)
        self.assertEqual(updated["byproduct_attempts"][-1]["result"]["status"], "invalid_evidence")
        json.dumps(updated, allow_nan=False)

    def test_delayed_response_for_changed_route_step_target_or_report_is_history_only(self):
        request = workspace.make_byproduct_request(self.state, TARGET)
        changed_report = workspace.apply_byproduct_result(self.state, request, _success("O"))
        changed_target = workspace.load_routes(workspace.new_workspace("CCO"), _payload("CCO"))
        returned_step = workspace.select_step(workspace.select_step(self.state, "r1", "s2"), "r1", "s1")
        recreated_target = _loaded()
        for current in (workspace.select_step(self.state, "r2", "s1"),
                        workspace.select_step(self.state, "r1", "s2"), changed_report, changed_target,
                        returned_step, recreated_target):
            with self.subTest(context=workspace.context_key(current)):
                before = copy.deepcopy(current)
                with patch.object(engine, "analyze_routes") as analyze:
                    updated = workspace.apply_byproduct_result(current, request, _success("CC=O"))
                    analyze.assert_not_called()
                self.assertSnapshotPreserved(updated, before)
                self.assertTrue(updated["byproduct_attempts"][-1]["stale"])
                self.assertEqual(updated["byproduct_attempts"][-1]["request"], request)
                self.assertEqual(current, before)

    def test_request_validation_rechecks_payload_and_identifiers(self):
        request = workspace.make_byproduct_request(self.state, TARGET)
        invalid = [None, {}, {**request, "request_id": "not-a-uuid"},
                   {**request, "reactants_smiles": "C"}, {**request, "product_smiles": "CCO"}]
        for modified in invalid:
            with self.subTest(request=modified):
                updated = workspace.apply_byproduct_result(self.state, modified, _success("CC=O"))
                self.assertSnapshotPreserved(updated, self.state)
                self.assertIsNotNone(updated["byproduct_attempts"][-1]["application_error"])

    def test_schema_analysis_and_policy_failures_roll_back_every_current_field(self):
        state = copy.deepcopy(self.state)
        state = workspace.set_model_evidence(state, _model_evidence(state, _model_row("CCO")))
        before = copy.deepcopy(state)
        request = workspace.make_byproduct_request(state, TARGET)
        for module, name in ((engine, "validate_routes"), (engine, "analyze_routes"),
                             (workspace, "review_route_eligibility")):
            with self.subTest(failing_stage=name):
                with patch.object(module, name, side_effect=RuntimeError("rollback control")) as failure:
                    updated = workspace.apply_byproduct_result(state, request, _success("CC=O"))
                failure.assert_called_once()
                self.assertSnapshotPreserved(updated, before)
                self.assertIn("rollback control", updated["byproduct_attempts"][-1]["application_error"])
                self.assertNotIn("historical_model_evidence", updated["byproduct_attempts"][-1])
                self.assertEqual(state, before)

    def test_new_report_hash_failure_rolls_back_before_publishing_any_current_values(self):
        state = copy.deepcopy(self.state)
        state = workspace.set_model_evidence(state, _model_evidence(state, _model_row("CCO")))
        request = workspace.make_byproduct_request(state, TARGET)
        original_hash = workspace.report_hash

        def fail_new_report(report):
            if (report.get("schema_version") == "1.0"
                    and report["routes"][0]["steps"][0]["byproducts"]["status"] == "success"):
                raise ValueError("hash rollback control")
            return original_hash(report)

        with patch.object(workspace, "report_hash", side_effect=fail_new_report):
            updated = workspace.apply_byproduct_result(state, request, _success("CC=O"))
        self.assertSnapshotPreserved(updated, state)
        self.assertIn("hash rollback control", updated["byproduct_attempts"][-1]["application_error"])


if __name__ == "__main__":
    unittest.main()
