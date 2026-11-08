"""Shared-workspace workflows with real local chemistry and explicit boundary doubles."""
from copy import deepcopy
import csv
import html
from io import StringIO
import json
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "test"))
from test_app_usability_unit import OfflineAppCase, TABS, fixture_payload
import engine
import workspace as context

SUCCESS = {"status": "success", "candidates": [{"smiles": "CN(C)N=O", "mode": "offline fixture", "provider_score": 0.3}],
           "error": None, "source": engine.ASKCOS_API, "raw_response": {"fixture": "계산 후보·관측 아님"}}
FAILED = {"status": "failed", "candidates": [], "error": "Timeout: offline fixture", "source": engine.ASKCOS_API,
          "raw_response": {"result": {"unexpected": "원문 보존"}}}


class ReviewWorkflowTests(OfflineAppCase):
    def confirm_product(self):
        """A single listed product needs no choice; several still do."""
        products = list(dict.fromkeys(context.selected_step(self.workspace)["products"]))
        product = products[0]
        if len(products) > 1:
            self.app.selectbox(key="byproduct_product").select(product).run()
            self.assert_no_exception()
        self.assertEqual(self.app.session_state["byproduct_product"], product)
        return product

    def send_request(self, result):
        """The button press is the consent: nothing leaves before it."""
        self.predict.side_effect = None
        self.predict.return_value = deepcopy(result)
        before = self.predict.call_count
        self.confirm_product()
        self.assertEqual(self.predict.call_count, before)
        self.app.button(key="byproduct_request").click().run()
        self.assert_no_exception()
        self.assertEqual(self.predict.call_count, before + 1)

    def test_theme_tabs_preserve_selected_context_and_all_tool_drafts(self):
        self.apply_payload(fixture_payload(two_steps=True, two_routes=True))
        self.app.selectbox(key="route_id").select("fixture-b").run()
        self.choose_step("stage-b")
        self.confirm_product()
        request_id = self.app.session_state["_byproduct_request"]["request_id"]
        before = deepcopy(self.workspace)
        binding = context.context_key(before)
        for label in TABS:
            self.open_tab(label)
        for light in (True, False):
            self.app.toggle(key="light_mode").set_value(light).run()
            self.assert_no_exception()
            self.assertEqual(self.workspace, before)
            self.assertEqual(context.context_key(self.workspace), binding)
            self.assertEqual(self.app.selectbox(key="route_id").value, "fixture-b")
            self.assertEqual(self.chosen_step(), "stage-b")
            self.assertEqual(self.app.session_state["_byproduct_request"]["request_id"], request_id)
        self.predict.assert_not_called()
        self.admet.assert_not_called()
        self.generate.assert_not_called()

    def test_step_a_b_a_and_target_a_b_a_cannot_revive_consent(self):
        payload = fixture_payload(two_steps=True)
        self.apply_payload(payload)
        self.confirm_product()
        frozen_request = deepcopy(self.app.session_state["_byproduct_request"])
        initial = context.context_key(self.workspace)
        self.choose_step("stage-b")
        self.choose_step("stage-a")
        self.assert_no_exception()
        self.assertEqual(self.workspace["step_id"], "stage-a")
        self.assertNotEqual(context.context_key(self.workspace), initial)
        # The returned-to step gets a newly frozen request, never the old one.
        self.assertNotEqual(self.app.session_state["_byproduct_request"]["request_id"],
                            frozen_request["request_id"])
        returned = context.context_key(self.workspace)
        self.raw_target("CCO")
        self.raw_target(payload["routes"][0]["target"])
        self.assertIsNone(self.workspace["report"])
        self.assertNotEqual(context.context_key(self.workspace), returned)
        self.predict.assert_not_called()

    def test_no_route_means_no_review_tabs_and_no_route_snapshot(self):
        self.raw_target("CCO")
        self.assertIsNone(self.workspace["report"])
        self.assertEqual([tab.label for tab in self.app.tabs], [])

    def test_byproduct_preview_consent_one_call_atomic_update_and_original_preserved(self):
        self.apply_payload(fixture_payload(two_steps=True, two_routes=True))
        before = deepcopy(self.workspace)
        product = self.confirm_product()
        pending = deepcopy(self.app.session_state["_byproduct_request"])
        self.assertEqual(pending["reactants_smiles"], ".".join(context.selected_step(before)["reactants"]))
        self.assertEqual(pending["product_smiles"], product)
        # The button is enabled because the exact payload is frozen and shown;
        # the press itself is the consent and nothing leaves before it.
        self.assertFalse(self.app.button(key="byproduct_request").disabled)
        self.assertIn(engine.ASKCOS_API, self.text())
        # The byproduct request moved out of its own tab and into a fold on
        # the risk screen; the request itself is unchanged.
        self.open_tab("위험 검토 (ICH M7)")
        self.app.checkbox(key="strict_alerts").uncheck().run()
        self.assertEqual(self.app.session_state["_byproduct_request"], pending)
        self.predict.assert_not_called()
        self.predict.side_effect = None
        self.predict.return_value = deepcopy(SUCCESS)
        self.app.run()
        self.predict.assert_not_called()
        self.app.button(key="byproduct_request").click().run()
        self.assert_no_exception()
        self.predict.assert_called_once_with(pending["reactants_smiles"], product, timeout=60)
        self.assertEqual(self.workspace["original_input"], before["original_input"])
        active = deepcopy(before["active_input"])
        active["routes"][0]["steps"][0]["byproducts"] = self.workspace["active_input"]["routes"][0]["steps"][0]["byproducts"]
        active["routes"][0]["steps"][0]["material_provenance"] = self.workspace["active_input"]["routes"][0]["steps"][0]["material_provenance"]
        self.assertEqual(self.workspace["active_input"], active)
        self.assertEqual(self.workspace["report"], engine.analyze_routes(active))
        self.assertEqual(self.workspace["report_sha256"], context.report_hash(self.workspace["report"]))
        self.assertEqual(self.snapshot["selection"], self.workspace["selection"])
        self.assertEqual(self.snapshot["byproduct_attempts"][0]["result"], SUCCESS)
        self.assertTrue(self.snapshot["byproduct_attempts"][0]["applied"])
        # The consumed request cannot be resent without a new freeze.
        self.assertEqual(self.app.session_state["_byproduct_consumed"], pending["request_id"])
        self.assertNotEqual(self.app.session_state["_byproduct_request"]["request_id"],
                            pending["request_id"])
        self.admet.assert_not_called()

    def test_product_change_clears_consent_and_freezes_a_different_payload(self):
        payload = fixture_payload()
        payload["routes"][0]["steps"][0]["products"].append("CCO")
        self.apply_payload(payload)
        self.confirm_product()
        first = deepcopy(self.app.session_state["_byproduct_request"])
        self.app.selectbox(key="byproduct_product").select("CCO").run()
        self.assert_no_exception()
        second = self.app.session_state["_byproduct_request"]
        self.assertNotEqual(first["request_id"], second["request_id"])
        self.assertEqual(second["product_smiles"], "CCO")
        self.assertEqual(second["reactants_smiles"], first["reactants_smiles"])
        self.predict.assert_not_called()

    def test_retry_consumes_request_and_needs_new_check_including_same_payload(self):
        self.apply_payload(fixture_payload())
        self.send_request(FAILED)
        first = deepcopy(self.workspace["byproduct_attempts"][0])
        pending = deepcopy(self.app.session_state["_byproduct_request"])
        # The sent request is consumed and replaced by a newly frozen one.
        self.assertNotEqual(first["request"]["request_id"], pending["request_id"])
        self.assertEqual(first["request"]["product_smiles"], pending["product_smiles"])
        self.app.run()
        self.assertEqual(self.app.session_state["_byproduct_request"], pending)
        self.predict.assert_called_once()
        self.app.button(key="byproduct_request").click().run()
        self.assert_no_exception()
        self.assertEqual(self.predict.call_count, 2)
        self.assertEqual(len(self.workspace["byproduct_attempts"]), 2)
        self.assertEqual(self.workspace["byproduct_attempts"][1]["request"]["request_id"],
                         pending["request_id"])
        self.assertNotEqual(self.app.session_state["_byproduct_request"]["request_id"],
                            pending["request_id"])

    def test_latest_failed_and_defective_responses_preserve_prior_success_history(self):
        self.apply_payload(fixture_payload())
        self.send_request(SUCCESS)
        successful_report = deepcopy(self.workspace["report"])
        self.send_request(FAILED)
        self.assertEqual(self.workspace["report"], successful_report)
        self.assertFalse(self.workspace["byproduct_attempts"][-1]["applied"])
        self.assertEqual(self.workspace["byproduct_attempts"][-1]["result"]["raw_response"], FAILED["raw_response"])
        self.assertIn("이전 결과는 기록에", self.text())
        self.assertIn("미평가", self.text())
        for malformed in (["not", "an", "object"], {"status": "success", "candidates": [None], "error": None},
                          {"status": "success", "candidates": [], "error": None},
                          {"status": "failed", "candidates": [], "error": "bad", "raw_response": {"not-json"}}):
            with self.subTest(result=repr(malformed)):
                self.send_request(malformed)
                self.assertEqual(self.workspace["report"], successful_report)
                self.assertTrue(self.workspace["byproduct_attempts"][-1]["error"])
                self.assertEqual(self.snapshot["byproduct_attempts"], self.workspace["byproduct_attempts"])
                json.dumps(self.snapshot, allow_nan=False)
        self.assertTrue(self.workspace["byproduct_attempts"][0]["applied"])
        # The no-absence wording moved off the screen into every stored attempt.
        for attempt in self.workspace["byproduct_attempts"]:
            self.assertTrue(any("부산물 부재를 뜻하지 않습니다" in note
                                for note in attempt["limitations"]))

    def test_late_response_uses_authoritative_current_step_not_frozen_workspace(self):
        self.apply_payload(fixture_payload(two_steps=True))
        before = deepcopy(self.workspace["report"])
        self.confirm_product()

        def delayed(reactants, product, timeout):
            current = st.session_state["review_workspace"]
            st.session_state["review_workspace"] = context.select_step(current, current["route_id"], "stage-b")
            return deepcopy(SUCCESS)

        self.predict.side_effect = delayed
        self.app.button(key="byproduct_request").click().run()
        self.assert_no_exception()
        self.assertEqual(self.workspace["step_id"], "stage-b")
        self.assertEqual(self.chosen_step(), "stage-b")
        self.assertEqual(self.workspace["report"], before)
        self.assertTrue(self.workspace["byproduct_attempts"][-1]["stale"])
        self.assertFalse(self.workspace["byproduct_attempts"][-1]["applied"])
        self.predict.assert_called_once()

    def test_detached_response_is_historical_and_cannot_restore_old_report(self):
        self.apply_payload(fixture_payload())
        self.confirm_product()

        def detached(reactants, product, timeout):
            st.session_state["review_workspace"] = None
            return deepcopy(SUCCESS)

        self.predict.side_effect = detached
        self.app.button(key="byproduct_request").click().run()
        self.assert_no_exception()
        self.assertIsNone(self.workspace["report"])
        self.assertIsNone(self.snapshot["report"])
        history = self.snapshot["historical_byproduct_attempts"]
        self.assertEqual(len(history), 1)
        self.assertTrue(history[0]["stale"])
        self.assertFalse(history[0]["applied"])
        self.assertEqual(history[0]["result"], SUCCESS)

    def test_model_failure_is_captured_reused_and_never_becomes_a_negative(self):
        self.enable_model()
        self.admet.side_effect = RuntimeError("offline model unavailable")
        self.apply_payload(fixture_payload())
        self.admet.assert_called_once()
        state = context.model_evidence_status(self.workspace)
        self.assertEqual(state["status"], "failed")
        self.assertIn("offline model unavailable", state["error"])
        self.assertIsNone(self.workspace["model_evidence"])
        self.assertEqual(self.snapshot["model_evidence_state"], state)
        self.assertIsNone(self.snapshot["model_evidence"])
        # A stored failure is reused, not retried, on a policy or theme change.
        self.app.checkbox(key="strict_alerts").uncheck().run()
        self.app.toggle(key="light_mode").set_value(True).run()
        self.assert_no_exception()
        self.admet.assert_called_once()
        # Every route stays unassessed; a failed run never becomes a negative.
        codes = {row["code"] for route in self.workspace["selection"]["routes"]
                 for row in route["reasons"]}
        self.assertIn("admet_unassessed", codes)
        self.assertNotIn("admet_negative", codes)

    def test_unified_download_formats_serialize_the_same_current_snapshot(self):
        downloads = {}
        real_download = st.download_button

        def capture(*args, **kwargs):
            if kwargs.get("key") == "review_download":
                downloads[st.session_state["export_format"]] = kwargs.get("data", args[1] if len(args) > 1 else None)
            return real_download(*args, **kwargs)

        self.start_patch(st, "download_button", side_effect=capture)
        payload = fixture_payload()
        payload["description"] = "<script>alert('not executable')</script>"
        self.apply_payload(payload)
        self.send_request(FAILED)
        self.app.checkbox(key="strict_alerts").check().run()
        expected = deepcopy(self.snapshot)
        self.assertEqual(json.loads(downloads["JSON"]), expected)
        self.assertTrue(self.app.radio(key="export_format").proto.horizontal)
        self.assertEqual(self.app.radio(key="export_format").options, ["JSON", "HTML", "CSV"])
        self.app.radio(key="export_format").set_value("CSV").run()
        self.assertEqual(self.snapshot, expected)
        rows = list(csv.DictReader(StringIO(downloads["CSV"].lstrip("\ufeff"))))
        materials = [material for route in expected["report"]["routes"] for step in route["steps"] for material in step["materials"]]
        self.assertEqual([row["입력 SMILES"] for row in rows], [item["input_smiles"] for item in materials])
        self.assertTrue(all(row["현재 경로 판정"] == expected["selection"]["routes"][0]["status"] for row in rows))
        self.app.radio(key="export_format").set_value("HTML").run()
        self.assert_no_exception()
        self.assertEqual(self.snapshot, expected)
        self.assertIn("Timeout: offline fixture", downloads["HTML"])
        self.assertNotIn("<script>alert", downloads["HTML"])
        self.assertIn(html.escape(payload["description"], quote=True), downloads["HTML"])
        embedded = downloads["HTML"].split("전체 스냅샷 JSON (원본·이력 포함)</summary><pre>", 1)[1].split("</pre>", 1)[0]
        self.assertEqual(json.loads(html.unescape(embedded)), expected)
        self.assertEqual(sum(item.proto.label == "내려받기" for item in self.app.get("download_button")), 1)

    def test_explicit_generation_publishes_only_a_complete_result_and_failure_clears_old(self):
        raw_target = self.workspace["raw_target"]
        generated = fixture_payload(self.workspace["target"])
        self.service.payload = deepcopy(generated)
        self.assertFalse(self.app.button(key="generation_run").disabled)
        self.assertEqual(self.service.started, [])
        self.app.button(key="generation_run").click().run()
        self.assert_no_exception()
        self.assertEqual([item["target"] for item in self.service.started], [raw_target])
        self.assertEqual([item["engine"] for item in self.service.started], ["aizynthfinder"])
        self.assertEqual(self.workspace["report"], engine.analyze_routes(generated))
        self.assertEqual(self.snapshot["source_provenance"]["request"]["engine"], "aizynthfinder")
        self.assertTrue(self.snapshot["generation_attempts"][-1]["applied"])
        # An unusable result publishes nothing and leaves no stale report.
        self.service.payload = {"schema_version": "1.0"}
        self.app.button(key="generation_run").click().run()
        self.assert_no_exception()
        self.assertIsNone(self.workspace["report"])
        self.assertIsNone(self.snapshot["report"])
        attempt = self.snapshot["generation_attempts"][-1]
        self.assertEqual(attempt["status"], "failed")
        self.assertFalse(attempt["applied"])
        self.assertTrue(self.snapshot["historical_reviews"])
        self.predict.assert_not_called()


if __name__ == "__main__":
    import unittest
    unittest.main()
