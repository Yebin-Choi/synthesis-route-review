"""Offline observable UI contracts; actual parsers/rules, mocked external boundaries."""
from copy import deepcopy
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

from PIL import Image
from rdkit.Chem import Draw
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import engine
import presentation
import review_views
import route_ames_view
import workspace as context
from molecule_catalog import MOLECULES

# The cost tab was removed with the pricing screen; three tabs remain.
TABS = ["위험 검토", "독성 예측 툴 비교", "부산물"]
CATALOG_NAMES = {item["name_en"].casefold(): f"{item['name_ko']} ({item['name_en']}), {item['category']}" for item in MOLECULES}


def fixture_payload(target="CCOC(C)=O", *, two_steps=False, two_routes=False):
    """Software-only structures, not a validated synthesis or measured yield."""
    # A structural alert, not an ICH M7-listed material: under the default
    # policy the route carries a concern signal and is still presented.
    first = {"id": "stage-a", "reactants": ["CCO", "CC(=O)O"], "products": [target],
             "reagents": ["C1CO1"], "solvents": [], "conditions": {"status": "unknown"},
             "byproducts": {"status": "not_run", "candidates": []}}
    steps = [first]
    if two_steps:
        steps.append({**deepcopy(first), "id": "stage-b", "reactants": ["CO", "O"], "reagents": []})
    routes = [{"id": "fixture-a", "target": target, "source": "가상 소프트웨어 시험 · 실험 검증 아님", "steps": steps}]
    if two_routes:
        routes.append({**deepcopy(routes[0]), "id": "fixture-b"})
    return {"schema_version": "1.0", "description": "소프트웨어 시험 입력", "routes": routes}


class TabAppTest(AppTest):
    """Include the native Streamlit tab state omitted by the test serializer."""

    def _run(self, widget_state=None, timeout=None):
        if widget_state is not None:
            for container in self.get("tab_container"):
                tab_id = container.proto.tab_container.id
                if container.key == "review_tabs" and not any(state.id == tab_id for state in widget_state.widgets):
                    widget_state.widgets.add(id=tab_id, string_value=self.session_state["review_tabs"])
        return super()._run(widget_state, timeout)


class FixtureGenerationService:
    """Software-only stand-in for the native service: no engine, no subprocess.

    Routes now enter the app only through generation, so the UI tests drive the
    real generation button against this prepared, already-completed result.
    """

    READINESS = {
        "state": "ready", "ready": True, "live_verified": True,
        "dependency_compatibility": "verified", "error": None,
        "configuration_id": "fixture-configuration",
        "configuration": {"fixture": "소프트웨어 시험용 실행 설정 · 실제 환경 아님"},
    }

    def __init__(self):
        self.payload = None
        self.snapshots = {}
        self.started = []

    def readiness(self, engine, llm_selection=None):
        return deepcopy(self.READINESS)

    def start_readiness(self, engine, *, request_id, context_id, llm_selection=None):
        return self._record(engine, None, request_id, context_id,
                            self.READINESS["configuration_id"], "probe", None)

    def start(self, engine, target, *, request_id, context_id, configuration_id, **kwargs):
        self.started.append({"engine": engine, "target": target, "request_id": request_id})
        result = {
            "payload": deepcopy(self.payload),
            "provenance": {"kind": "software_only_fixture",
                           "note": "가상 소프트웨어 시험 · 실제 탐색 근거 아님"},
            "artifacts": [], "item_failures": [],
        }
        return self._record(engine, target, request_id, context_id, configuration_id,
                            "generate", result)

    def _record(self, engine, target, request_id, context_id, configuration_id, operation, result):
        snapshot = {
            "request_id": request_id, "context_id": context_id, "engine": engine,
            "target": target, "operation": operation, "status": "succeeded",
            "configuration_id": configuration_id, "readiness": deepcopy(self.READINESS),
            "result": result, "error": None,
        }
        self.snapshots[request_id] = snapshot
        return deepcopy(snapshot)

    def poll(self, request_id, *, context_id=None):
        snapshot = self.snapshots.get(request_id)
        if snapshot is None:
            return {"request_id": request_id, "context_id": context_id, "status": "failed",
                    "readiness": None, "result": None, "error": {"message": "unknown request"}}
        snapshot = deepcopy(snapshot)
        if context_id is not None and context_id != snapshot["context_id"]:
            snapshot.update(status="stale", result=None, error={"message": "stale"})
        return snapshot

    def cancel(self, request_id):
        snapshot = self.snapshots.get(request_id)
        if snapshot is not None:
            snapshot.update(status="cancelled", result=None, error={"message": "cancelled"})
        return self.poll(request_id)


class OfflineAppCase(unittest.TestCase):
    def setUp(self):
        self.generate = self.start_patch(engine, "generate_routes", side_effect=AssertionError("Unexpected model search"))
        self.predict = self.start_patch(engine, "predict_byproducts", side_effect=AssertionError("Unexpected external prediction"))
        # ADMET-AI now runs on its own once its assets are present, so the
        # default harness keeps them absent and forbids inference. A test that
        # wants model evidence calls enable_model().
        self.admet = self.start_patch(route_ames_view, "assess_admet_materials", side_effect=AssertionError("Unexpected model inference"))
        self.admet_ready = self.start_patch(route_ames_view, "admet_readiness", return_value={"ready": False, "missing": ["weights"]})
        self.network = self.start_patch(engine.requests.sessions.Session, "request", side_effect=AssertionError("Network forbidden"))
        self.socket = self.start_patch(socket, "create_connection", side_effect=AssertionError("Network forbidden"))
        self.service = FixtureGenerationService()
        self.app = TabAppTest(str(ROOT / "app" / "app.py"), default_timeout=30)
        self.app.session_state["_native_service"] = self.service
        self.app.run()
        self.assert_no_exception()

    def tearDown(self):
        self.network.assert_not_called()
        self.socket.assert_not_called()

    def start_patch(self, target, name, **kwargs):
        patcher = patch.object(target, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def assert_no_exception(self):
        self.assertEqual(len(self.app.exception), 0, str(self.app.exception))

    def text(self, element=None):
        element = self.app if element is None else element
        return "\n".join(str(item.value) for kind in ("title", "subheader", "markdown", "caption", "text", "info", "warning", "error", "success", "code") for item in element.get(kind))

    @property
    def workspace(self):
        return self.app.session_state["review_workspace"]

    @property
    def snapshot(self):
        return self.app.session_state["review_export_snapshot"]

    def expander(self, label):
        return next(item for item in self.app.get("expander") if item.label == label)

    def raw_target(self, text):
        self.app.text_input(key="target_smiles").set_value(text).run()
        self.assert_no_exception()

    def choose_step(self, step_id):
        """The step picker lives in the byproduct tab, the one screen it means
        something on. The risk tab lays every step out at once."""
        self.app.radio(key="step_id").set_value(step_id).run()
        self.assert_no_exception()

    def chosen_step(self):
        return self.app.radio(key="step_id").value

    def open_tab(self, label):
        container = next(item for item in self.app.get("tab_container") if item.key == "review_tabs")
        states = self.app._tree.get_widget_states()
        for state in states.widgets:
            if state.id == container.proto.tab_container.id:
                state.string_value = label
                break
        else:
            states.widgets.add(id=container.proto.tab_container.id, string_value=label)
        self.app._run(states)
        self.assert_no_exception()
        self.assertEqual(self.app.session_state["review_tabs"], label)

    def enable_model(self, score=0.2):
        """Make the automatic ADMET-AI run possible, with a fixture result."""
        def inference(smiles, *, threshold=0.5):
            rows = []
            for value in smiles:
                canonical = engine.canonicalize_smiles(value)
                success = canonical is not None
                rows.append({
                    "input_smiles": value, "canonical_smiles": canonical,
                    "status": "success" if success else "invalid",
                    "score": score if success else None,
                    "prediction": (("positive" if score >= threshold else "negative")
                                   if success else "abstain"),
                    "applicability": "unknown",
                    "error": None if success else "소프트웨어 시험용 미평가 · 음성 결과 아님",
                })
            return {
                "model": {"tool": "ADMET-AI", "version": "software-only-fixture",
                          "endpoint": "ames", "weights_sha256": {"fixture.pt": "a" * 64}},
                "threshold": threshold, "assessments": rows,
                "limitations": ["소프트웨어 시험용 모델 근거 · 실제 추론이나 안전성 검증이 아님"],
            }

        self.admet.side_effect = inference
        self.admet_ready.return_value = {"ready": True, "missing": []}

    def apply_payload(self, payload, *, match_target=True):
        """Publish routes the only way the app still accepts them: generation."""
        if match_target:
            self.raw_target(payload["routes"][0]["target"])
        self.service.payload = deepcopy(payload)
        self.app.button(key="generation_run").click().run()
        self.assert_no_exception()

    def fictional_routes(self):
        self.apply_payload(deepcopy(engine.load_example()))


class AppUsabilityTests(OfflineAppCase):
    def test_shared_header_three_tabs_no_legacy_pages_or_automatic_calls(self):
        # No tabs before a report, and no removed entry points: file upload,
        # JSON editing, saved-route loading and the cost tab are all gone.
        self.assertEqual([tab.label for tab in self.app.tabs], [])
        self.assertEqual(len(self.app.sidebar), 0)
        self.assertEqual(self.app.selectbox(key="target_catalog").value, CATALOG_NAMES["propranolol"])
        self.assertIsNone(self.workspace["report"])
        for key in ("load_saved_routes", "route_analyze", "switch_import_target"):
            self.assertFalse(any(item.key == key for item in self.app.button), key)
        self.assertEqual(len(self.app.text_area), 0)
        self.assertEqual(len(self.app.get("file_uploader")), 0)
        self.assertTrue(self.app.button(key="generation_run"))
        self.apply_payload(fixture_payload())
        self.assertEqual([tab.label for tab in self.app.tabs], TABS)
        formats = self.app.radio(key="export_format")
        self.assertEqual(formats.options, ["JSON", "HTML", "CSV"])
        self.assertTrue(formats.proto.horizontal)
        self.assertFalse(any(item.key == "export_format" for item in self.app.selectbox))
        for label in TABS:
            self.open_tab(label)
        self.app.toggle(key="light_mode").set_value(True).run()
        self.assert_no_exception()
        self.generate.assert_not_called()
        self.predict.assert_not_called()
        self.admet.assert_not_called()

    def test_generated_target_to_first_visible_risk_is_one_action(self):
        payload = fixture_payload()
        self.apply_payload(payload)
        self.assertEqual(self.workspace["original_input"], payload)
        self.assertEqual(self.workspace["active_input"], payload)
        self.assertEqual(self.workspace["report"], engine.analyze_routes(payload))
        self.assertEqual(self.workspace["source_provenance"]["kind"], "native_generation")
        self.assertTrue(self.app.tabs[0].get("image"))
        self.assertIn("경로 1", self.text(self.app.tabs[0]))
        self.generate.assert_not_called()
        self.predict.assert_not_called()
        self.admet.assert_not_called()

    def test_unrelated_target_does_not_inherit_generated_routes(self):
        self.fictional_routes()
        self.raw_target("CCO")
        self.assertEqual(self.workspace["target"], "CCO")
        self.assertIsNone(self.workspace["report"])
        self.assertIsNone(self.snapshot["report"])
        self.assertEqual([tab.label for tab in self.app.tabs], [])

    def test_invalid_raw_input_detaches_before_validation_and_recovers_locally(self):
        for bad in ("", "   ", "일반문장", "C1CC", "C(C"):
            with self.subTest(target=bad):
                self.fictional_routes()
                self.raw_target(bad)
                self.assertIsNone(self.workspace)
                self.assertIsNone(self.snapshot["report"])
                self.assertTrue(self.snapshot["target_error"])
                self.assertTrue(self.app.button(key="generation_run").disabled)
                self.raw_target("CCO")
                self.assertEqual(self.workspace["raw_target"], "CCO")
                self.assertIsNone(self.workspace["report"])
        self.predict.assert_not_called()
        self.admet.assert_not_called()

    def test_target_identity_preserves_salts_maps_charge_isotopes_and_stereo(self):
        values = ("CC(=O)[O-].[Na+]", "CC(=O)O", "[NH4+]", "N", "[13CH3]CO", "CCO",
                  "F[C@H](Cl)Br", "F[C@@H](Cl)Br", "[CH3:1][CH2:2][OH:3]")
        identities = []
        for value in values:
            with self.subTest(target=value):
                self.raw_target(value)
                self.assertEqual(self.workspace["raw_target"], value)
                self.assertEqual(self.app.text_input(key="target_smiles").value, value)
                self.assertIsNone(self.workspace["report"])
                identities.append(self.workspace["target"])
        for left, right in ((0, 1), (2, 3), (4, 5), (6, 7)):
            self.assertNotEqual(identities[left], identities[right])
        self.assertEqual(identities[8], "CCO")

    def test_target_length_bound_rejects_without_truncation(self):
        valid = " " * 4093 + "CCO"
        self.raw_target(valid)
        self.assertEqual(self.workspace["raw_target"], valid)
        self.assertEqual(self.workspace["target"], "CCO")
        self.raw_target(valid + " ")
        self.assertIsNone(self.workspace)
        self.assertIn("4096", self.snapshot["target_error"])
        self.assertEqual(self.app.text_input(key="target_smiles").value, valid + " ")

    def test_malformed_generation_result_leaves_no_current_report_or_stale_success(self):
        payload = fixture_payload()
        self.apply_payload(payload)
        self.assertEqual(self.snapshot["original_input"], payload)
        self.apply_payload({"schema_version": "1.0"}, match_target=False)
        self.assertIsNone(self.workspace["report"])
        self.assertIsNone(self.snapshot["report"])
        attempt = self.snapshot["generation_attempts"][-1]
        self.assertEqual(attempt["status"], "failed")
        self.assertFalse(attempt["applied"])
        self.assertIn("확인하지 못했습니다", attempt["error"]["message"])
        # The previous success is archived, never left standing as current.
        self.assertTrue(self.snapshot["historical_reviews"])
        self.assertFalse(self.snapshot["historical_reviews"][-1]["applicable"])
        self.service.payload = deepcopy(payload)
        self.app.button(key="generation_run").click().run()
        self.assert_no_exception()
        self.assertEqual(self.snapshot["report"], engine.analyze_routes(payload))

    def test_mismatched_generated_target_is_not_attached_or_silently_switched(self):
        previous_target = self.workspace["target"]
        previous_raw = self.app.text_input(key="target_smiles").value
        self.apply_payload(fixture_payload("CCO"), match_target=False)
        self.assertEqual(self.workspace["target"], previous_target)
        self.assertIsNone(self.workspace["report"])
        self.assertEqual(self.app.text_input(key="target_smiles").value, previous_raw)
        self.assertEqual(self.snapshot["generation_attempts"][-1]["status"], "failed")

    def test_mixed_targets_are_rejected_not_attached_or_silently_switched(self):
        self.fictional_routes()
        old_target = self.workspace["target"]
        payload = fixture_payload(old_target)
        payload["routes"].append(fixture_payload("CCO")["routes"][0])
        payload["routes"][1]["id"] = "different-target"
        self.apply_payload(payload, match_target=False)
        self.assertEqual(self.workspace["target"], old_target)
        self.assertIsNone(self.workspace["report"])
        self.assertEqual(self.snapshot["generation_attempts"][-1]["status"], "failed")
        self.assertNotIn("_pending_import", self.app.session_state)

    def test_policy_keeps_stable_route_step_ids_without_reanalysis_or_models(self):
        payload = fixture_payload(two_steps=True, two_routes=True)
        self.apply_payload(payload)
        self.app.selectbox(key="route_id").select("fixture-b").run()
        self.choose_step("stage-b")
        before = deepcopy(self.workspace)
        self.app.checkbox(key="strict_alerts").check().run()
        self.assert_no_exception()
        self.assertEqual(self.workspace["report"], before["report"])
        self.assertEqual(context.context_key(self.workspace), context.context_key(before))
        self.assertEqual(self.workspace["route_id"], "fixture-b")
        self.assertEqual(self.workspace["step_id"], "stage-b")
        # Both routes stay selectable; the picker shows a rank label while the
        # saved selection keeps the engine's own ids.
        self.assertEqual(len(self.app.selectbox(key="route_id").options), 2)
        self.assertEqual([item["id"] for item in self.workspace["selection"]["routes"]],
                         ["fixture-a", "fixture-b"])
        self.assertEqual(self.workspace["selection"]["routes"][0]["status"], "needs_review")
        self.assertEqual(self.snapshot["selection"], self.workspace["selection"])
        self.generate.assert_not_called()
        self.predict.assert_not_called()
        self.admet.assert_not_called()

    def test_exclusion_choices_are_unique_all_roles_and_no_target_or_components(self):
        payload = fixture_payload("CCO.[Na+]")
        step = payload["routes"][0]["steps"][0]
        step["reactants"] = ["CCO", "[Na+]", "CO"]
        step["solvents"] = ["[CH3:7][OH:8]", "not-a-structure"]
        step["bases"] = ["[Na+].[OH-]"]
        self.apply_payload(payload)
        choices = review_views.exclusion_choices(self.workspace["report"])
        self.assertNotIn("CCO", choices)
        self.assertNotIn("[Na+]", choices)
        self.assertNotIn(context.validate_target("CCO.[Na+]"), choices)
        self.assertIn("CO", choices)
        self.assertIn("반응물 / 용매", choices["CO"])
        self.assertIn(context.validate_target("[Na+].[OH-]"), choices)
        self.assertNotIn("not-a-structure", choices)
        self.assertEqual(len(choices), len(set(choices)))

    def test_all_excluded_routes_remain_inspectable_and_missing_not_negative(self):
        self.fictional_routes()
        self.app.multiselect(key="excluded_smiles").select("CCO").run()
        self.assert_no_exception()
        self.assertEqual(self.workspace["selection"]["counts"]["excluded"], 3)
        # Nothing is left to select, so every excluded route and its reasons
        # must still be readable in the exclusion section.
        self.assertFalse(any(item.key == "route_id" for item in self.app.selectbox))
        self.assertIn("남은 경로가 없습니다", self.text())
        excluded = context.excluded_routes(self.workspace)
        self.assertEqual(len(excluded), 3)
        text = self.text(self.expander("제외 기준과 빠진 경로"))
        for route, decision in zip(self.workspace["report"]["routes"], excluded):
            self.assertIn("prohibited_material", {row["code"] for row in decision["reasons"]})
            for missing in route["steps"][0]["missing_information"]:
                self.assertIn(missing, " ".join(row["message"] for row in decision["reasons"]))
        self.assertIn("판정 근거는 저장 파일에 함께 들어갑니다", text)

    def test_model_runs_automatically_once_its_assets_are_present(self):
        self.enable_model()
        payload = fixture_payload()
        self.apply_payload(payload)
        self.assertTrue(self.admet.called)
        self.assertEqual(self.admet.call_args.kwargs, {"threshold": 0.5})
        self.assertEqual(context.model_evidence_status(self.workspace)["status"], "available")
        codes = {row["code"] for route in self.workspace["selection"]["routes"]
                 for row in route["reasons"]}
        self.assertIn("admet_negative", codes)
        self.assertNotIn("admet_unassessed", codes)
        self.predict.assert_not_called()

    def test_missing_information_is_not_candidate_under_strict_policy(self):
        self.fictional_routes()
        self.app.selectbox(key="route_id").select("example-information-gap").run()
        self.app.checkbox(key="strict_alerts").check().run()
        self.assert_no_exception()
        decision = next(item for item in self.workspace["selection"]["routes"] if item["id"] == self.workspace["route_id"])
        self.assertEqual(decision["status"], "insufficient_data")
        self.assertIn("정보 부족", self.text(self.app.tabs[0]))
        # The gap is named once, under the route chip. The step block used to
        # repeat it as "부산물 / 안 돌림" on every step of every route, which
        # said only that nobody had pressed the button yet.
        self.assertIn("부산물 미평가", self.text(self.app.tabs[0]))
        self.assertNotIn("안 돌림", self.text(self.app.tabs[0]))

    def test_policy_failure_never_reuses_a_candidate(self):
        self.enable_model()
        payload = fixture_payload()
        step = payload["routes"][0]["steps"][0]
        step["reagents"] = []
        step["conditions"] = {"status": "observed", "source": "소프트웨어 시험", "description": "실험 검증 아님"}
        step["byproducts"] = {"status": "success", "source": "소프트웨어 시험", "candidates": []}
        self.apply_payload(payload)
        self.assertEqual(self.workspace["selection"]["routes"][0]["status"], "candidate")
        before = deepcopy(self.workspace["policy"])
        # A policy evaluation failure is reported, and the previous policy is
        # neither replaced nor re-approved as a new candidate verdict.
        with patch.object(review_views.context, "set_policy",
                          side_effect=ValueError("software-only policy failure")):
            self.app.checkbox(key="strict_alerts").uncheck().run()
            self.assert_no_exception()
            self.assertIn("정책을 적용하지 못했습니다", self.text())
            self.assertEqual(self.workspace["policy"], before)

    def test_conditional_warnings_failures_and_optional_roles_remain_visible(self):
        payload = deepcopy(engine.load_example())
        step = payload["routes"][1]["steps"][0]
        for role, smiles in {"catalysts": "[Pd]", "ligands": "P(c1ccccc1)(c1ccccc1)c1ccccc1", "bases": "[Na+].[OH-]", "coproducts": "O"}.items():
            step[role] = [smiles]
            step.setdefault("material_provenance", {})[role] = {"status": "proposed", "source": "수동 시험 근거 · " + role}
        self.apply_payload(payload)
        self.app.selectbox(key="route_id").select(payload["routes"][1]["id"]).run()
        self.assert_no_exception()
        text = self.text(self.app.tabs[0])
        report_step = self.workspace["report"]["routes"][1]["steps"][0]
        for warning in report_step["conditional_warnings"]:
            self.assertIn(warning["message"], text)
            self.assertIn(warning["basis"], text)
        self.assertIn("테스트용 실패 상태", text)
        self.assertIn("<dt>반응 조건</dt><dd>제안</dd>", text)
        self.assertIn("<dt>근거</dt><dd>제안</dd>", text)
        for role in ("catalysts", "ligands", "bases", "coproducts"):
            self.assertIn("수동 시험 근거 · " + role, text)

    def test_depiction_failure_and_invalid_non_target_preserve_assessment_and_text(self):
        presentation.structure_png.clear()
        self.addCleanup(presentation.structure_png.clear)
        self.start_patch(Draw, "MolToImage", side_effect=OSError("offline drawing failure"))
        payload = fixture_payload()
        # An ICH M7-listed reagent beside an unreadable solvent, so the step
        # carries both a known concern and an invalid material.
        payload["routes"][0]["steps"][0]["reagents"] = ["ClCC1CO1"]
        payload["routes"][0]["steps"][0]["solvents"] = ["not-a-structure"]
        expected = engine.analyze_routes(payload)
        self.apply_payload(payload)
        self.assertEqual(self.snapshot["original_input"], payload)
        self.assertEqual(self.snapshot["report"], expected)
        # The listed reagent excludes this route, so the assessments and the
        # original text must stay readable in the exclusion section.
        self.assertEqual([tab.label for tab in self.app.tabs], [])
        exclusion = self.text(self.expander("제외 기준과 빠진 경로"))
        self.assertIn("구조 그림 없음", self.text())
        self.assertIn("not-a-structure", exclusion)
        materials = self.workspace["report"]["routes"][0]["steps"][0]["materials"]
        self.assertTrue(any(item["status"] == "invalid" for item in materials))
        self.assertTrue(any(item["status"] == "known_concern" for item in materials))
        self.assertIn("구조를 읽지 못함", exclusion)
        self.assertIn(presentation.STATUS["known_concern"], exclusion)

    def test_special_concern_chip_shows_and_no_empty_ttc_row(self):
        """A special-concern precursor is marked, and no card carries an
        empty reference-intake row.

        A reference intake is known for very few structures, so that row sat
        on every card saying "없음". Why a structure has none, including the
        ICH M7 rule that the generic limit does not apply to a special
        concern, stays in the stored record and in the download.
        """
        payload = fixture_payload()
        payload["routes"][0]["steps"][0]["reagents"] = ["CN(C)N=O"]
        self.apply_payload(payload)
        self.assert_no_exception()
        materials = self.workspace["report"]["routes"][0]["steps"][0]["materials"]
        flagged = next(item for item in materials if item["input_smiles"] == "CN(C)N=O")
        self.assertTrue(flagged["special_concern"])
        self.assertIsNone(flagged["intake_ug_day"])
        self.assertEqual(flagged["status"], "structural_alert")
        self.assertIn("No TTC fallback", flagged["intake_basis"])
        self.assertIn("특별 우려 구조", self.text())
        self.assertNotIn("TTC 참고 섭취량", self.text())
        self.admet.assert_not_called()

    def test_generation_is_explicit_and_never_runs_on_its_own(self):
        # The button is the only way routes arrive, and nothing is generated
        # until it is pressed.
        self.assertFalse(self.app.button(key="generation_run").disabled)
        self.assertIsNone(self.workspace["report"])
        self.assertEqual(self.app.session_state["generation_attempts"], [])
        self.assertEqual(self.service.started, [])
        self.app.run()
        self.assert_no_exception()
        self.assertEqual(self.service.started, [])
        self.generate.assert_not_called()

    def test_stored_research_sections_are_collapsed_and_explicit(self):
        self.apply_payload(fixture_payload())
        labels = [item.label for item in self.app.get("expander")]
        self.assertIn("목표 구조", labels)
        self.assertFalse(self.expander("저장").proto.expanded)
        # The legacy stored-condition and stored-benchmark panels are gone.
        for removed in ("저장 조건 근거 자료 · 목표 반응에 자동 적용 안 함",
                        "저장된 연구 비교 · 현재 모델 검사 아님"):
            self.assertNotIn(removed, labels)
        for key in ("open_condition_evidence", "open_stored_benchmark"):
            self.assertFalse(any(item.key == key for item in self.app.checkbox), key)
        self.admet.assert_not_called()


if __name__ == "__main__":
    unittest.main()
