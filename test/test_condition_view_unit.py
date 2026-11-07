"""Offline UI fixtures, not chemical experiments or empirical target validation."""

from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import condition_view
from condition_evidence import get_combinations, load_evidence as load_local_evidence


def fixture_source(source_id, reaction_type, basis):
    return {
        "id": source_id, "reaction_type": reaction_type, "selected_sheet": "fixture_sheet",
        "measurement_basis": basis,
        "measurement_field": "Output" if basis.startswith("reported_") else "Product_Yield_PCT_Area_UV",
        "measurement_unit": "reported_percent" if basis.startswith("reported_") else "UV_area_percent",
        "url": f"https://example.invalid/software-only-fixture/{source_id}.xlsx",
        "sha256": ("a" if source_id == "fixture_doyle" else "b") * 64,
    }


def fixture_component(role, status, *, name=None, smiles=None):
    return {
        "role": role, "status": status, "name": name,
        "input_smiles": smiles, "canonical_smiles": smiles,
        "input_smiles_variants": [smiles] if smiles else [],
        "equivalents": None, "mmol": None,
    }


def fixture_group(source, number, *, components=None):
    group_id = f"{source['id']}:{number:02}"
    return {
        "id": group_id, "source_id": source["id"], "reaction_type": source["reaction_type"],
        "measurement_basis": source["measurement_basis"],
        "components": components if components is not None else [
            fixture_component("ligand", "resolved", name=f"fixture ligand {number:02}", smiles="CCO"),
            # A SMILES-like name is deliberately unresolved; no structure is inferred.
            fixture_component("reagent", "unresolved", name="CCO"),
            fixture_component("catalyst", "missing"),
        ],
        "substrate_contexts": [{
            "components": [fixture_component("reactant_1", "unresolved", name=f"context-only {number:02}")],
            "n": 1, "record_ids": [f"{source['id']}:fixture_sheet:row:{number + 2}"],
        }],
        "n": 1, "record_ids": [f"{source['id']}:fixture_sheet:row:{number + 2}"],
        "measurement": {
            "field": source["measurement_field"], "basis": source["measurement_basis"],
            "unit": source["measurement_unit"], "n_valid": 0 if number == 0 else 1,
            "n_missing": 1 if number == 0 else 0, "median": None if number == 0 else 100 - number,
            "calibrated_yield_established": False,
        },
        "temperature": {"status": "unknown", "value": None, "unit": None},
        "missing_fields": {"temperature": 1},
        "research_threshold": {
            "label": "research_observation_count_not_prediction", "wilson_lower_bound": 0.0,
        },
    }


def fixture_evidence():
    doyle = fixture_source(
        "fixture_doyle", "Buchwald-Hartwig", "reported_Output_percent_basis_unverified"
    )
    other = fixture_source("fixture_uv", "Buchwald-Hartwig", "UV_area_percentage")
    suzuki = fixture_source("fixture_suzuki", "Suzuki-Miyaura", "UV_area_percentage")
    groups = [fixture_group(doyle, number) for number in range(25)]
    groups[-1]["components"][0].update({
        "name": "Late needle ligand", "input_smiles": "N#N", "canonical_smiles": "N#N",
        "input_smiles_variants": ["N#N"],
    })
    groups[-2]["components"][0]["input_smiles_variants"] = ["OCC"]
    groups.extend([
        fixture_group(other, 0),
        fixture_group(suzuki, 0, components=[
            fixture_component("catalyst", "unresolved", name="CCO"),
            fixture_component("ligand", "absent", name="None"),
            fixture_component("solvent", "missing"),
        ]),
    ])
    return {
        "schema": "independent-condition-evidence/1", "provenance": "software_only_fixture",
        "sources": [doyle, other, suzuki], "combinations": groups,
        "limitations": ["Software-only fixture; not a proposed experimental condition set."],
    }


class ConditionViewTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "condition_evidence.json"
        self.kb = fixture_evidence()
        loader = patch.object(condition_view, "load_evidence", side_effect=lambda path: deepcopy(self.kb))
        self.loader = loader.start()
        self.addCleanup(loader.stop)
        assess = patch.object(
            condition_view, "assess_material", side_effect=AssertionError("Unexpected automatic screening")
        )
        self.assess = assess.start()
        self.addCleanup(assess.stop)
        network = patch.object(socket, "create_connection", side_effect=AssertionError("Network is forbidden"))
        self.network = network.start()
        self.addCleanup(network.stop)
        request = patch("requests.sessions.Session.request", side_effect=AssertionError("Network is forbidden"))
        self.request = request.start()
        self.addCleanup(request.stop)

    def tearDown(self):
        self.network.assert_not_called()
        self.request.assert_not_called()

    def open_view(self, *, default_path=False):
        call = "show_condition_evidence()" if default_path else f"show_condition_evidence(path={str(self.path)!r})"
        app = AppTest.from_string(
            "from condition_view import show_condition_evidence\n" + call,
            default_timeout=30,
        ).run()
        self.assert_no_exception(app)
        return app

    def assert_no_exception(self, app):
        self.assertEqual(len(app.exception), 0)

    def text(self, app):
        return "\n".join(
            str(item.value)
            for kind in ("subheader", "markdown", "caption", "info", "warning", "error", "code")
            for item in app.get(kind)
        )

    def table(self, app, column):
        return next(item.value for item in app.dataframe if column in item.value.columns)

    def test_reference_scope_no_automatic_screening_or_route_application(self):
        app = self.open_view()
        text = self.text(app)
        self.assertEqual([item.value for item in app.subheader], ["조건 근거 자료"])
        # The screen carries the short scope caption and the collected source's
        # own limitation lines; the full scope note stays in the saved file.
        self.assertIn("공개 원자료 독립 재수집", text)
        self.assertNotIn(condition_view.SCOPE_NOTE, text)
        self.assertIn("반응 유형은 자료 필터입니다", text)
        self.assertIn("정규화 SMILES가 있는 조건 구성물만 검사합니다", text)
        for note in self.kb["limitations"]:
            self.assertIn(note, text)
        self.assertEqual(app.selectbox(key="condition_evidence_type").options, list(condition_view.REACTION_TYPES))
        self.assertEqual([button.label for button in app.button], ["구성물 검사"])
        self.assess.assert_not_called()

    def test_saved_file_keeps_the_full_scope_note_and_source_limitations(self):
        with patch.object(condition_view.st, "download_button",
                          wraps=condition_view.st.download_button) as download:
            self.open_view()
            payload = json.loads(download.call_args.kwargs["data"])
        self.assertEqual(payload["notice"], condition_view.SCOPE_NOTE)
        self.assertEqual(payload["scope"], "observed_in_original_only_not_proposed_for_target")
        self.assertEqual(payload["limitations"], self.kb["limitations"])

    def test_all_combinations_remain_selectable_beyond_twenty_and_not_median_ranked(self):
        app = self.open_view()
        selector = app.selectbox(key="condition_evidence_combination")
        self.assertEqual(len(selector.options), 26)
        self.assertIn("fixture_doyle", selector.options[0])
        self.assertIn("Output (측정법 미확인)", selector.options[0])
        # ID order starts with an unknown measurement, not the greatest measurement.
        self.assertEqual(app.metric[2].value, "미확인")
        selector.select("fixture_doyle:24").run()
        self.assert_no_exception(app)
        self.assertIn("Late needle ligand", self.table(app, "원자료 이름")["원자료 이름"].tolist())
        self.assertIn("fixture_doyle:fixture_sheet:row:26", self.table(app, "원자료 행 ID")["원자료 행 ID"].tolist())
        # The ordering is stated as a listing order, never as a recommendation.
        self.assertIn("출처·측정 기준·조합 ID 순", self.text(app))
        self.assertNotIn("추천", self.text(app))

    def test_search_matches_late_names_smiles_variants_and_paired_substrate_context(self):
        app = self.open_view()
        for query, expected in (
            ("LATE NEEDLE", "fixture_doyle:24"),
            ("N#N", "fixture_doyle:24"),
            ("OCC", "fixture_doyle:23"),
            ("context-only 22", "fixture_doyle:22"),
        ):
            with self.subTest(query=query):
                app.text_input(key="condition_evidence_search").set_value(query).run()
                self.assert_no_exception(app)
                selector = app.selectbox(key="condition_evidence_combination")
                self.assertEqual(len(selector.options), 1)
                self.assertEqual(selector.value, expected)
        self.assess.assert_not_called()

    def test_empty_search_result_does_not_fall_back_to_another_combination(self):
        app = self.open_view()
        app.text_input(key="condition_evidence_search").set_value("not in any fixture").run()
        self.assert_no_exception(app)
        self.assertEqual(len(app.selectbox), 1)
        self.assertEqual(len(app.get("download_button")), 0)
        # An empty search states only that nothing matched, with the counts
        # visible, and never falls back to another combination.
        self.assertIn("일치하는 조합이 없습니다", self.text(app))
        self.assertIn("검색 일치 0 / 26개", self.text(app))
        self.assertEqual(len(app.metric), 0)
        self.assess.assert_not_called()

    def test_source_and_measurement_basis_stay_separate(self):
        app = self.open_view()
        self.assertIn("Output (측정법 미확인)", app.metric[2].label)
        app.selectbox(key="condition_evidence_combination").select("fixture_uv:00").run()
        self.assert_no_exception(app)
        self.assertIn("UV면적%", app.metric[2].label)
        text = self.text(app)
        self.assertIn("출처 fixture_uv", text)
        self.assertIn("원자료 URL: https://example.invalid/software-only-fixture/fixture_uv.xlsx", text)
        self.assertIn("b" * 64, text)
        self.assertEqual(app.metric[0].value, "1")
        # The measurement is labelled by its own basis and row count, never as a
        # success probability for a target reaction.
        self.assertIn("n은 원자료 행 수입니다", text)
        self.assertIn("단위: UV_area_percent", text)
        self.assertNotIn("성공 확률", text)
        self.assertFalse(any("Wilson" in metric.label for metric in app.metric))

    def test_missing_measurement_temperature_and_unresolved_structures_stay_unknown(self):
        app = self.open_view()
        self.assertEqual(app.metric[1].value, "0")
        self.assertEqual(app.metric[2].value, "미확인")
        self.assertIn("온도: 자료 없음", self.text(app))
        rows = self.table(app, "구조 상태").set_index("역할")
        self.assertEqual(rows.loc["시약", "원자료 이름"], "CCO")
        self.assertEqual(rows.loc["시약", "정규화 SMILES"], "미확인")
        self.assertIn("구조 미해결", rows.loc["시약", "구조 상태"])
        self.assertIn("원자료 누락", rows.loc["촉매", "구조 상태"])
        self.assess.assert_not_called()

    def test_observed_zero_measurement_is_not_replaced_with_unknown(self):
        self.kb["combinations"][0]["measurement"].update({"median": 0.0, "n_valid": 1, "n_missing": 0})
        app = self.open_view()
        self.assertEqual(app.metric[1].value, "1")
        self.assertEqual(app.metric[2].value, "0.0")
        self.assertIn("누락 측정: 0행", self.text(app))

    def test_suzuki_name_only_and_absent_components_are_not_screened_negative(self):
        app = self.open_view()
        app.selectbox(key="condition_evidence_type").select("Suzuki-Miyaura").run()
        self.assert_no_exception(app)
        self.assertTrue(app.button(key="condition_evidence_screen").disabled)
        rows = self.table(app, "구조 상태").set_index("역할")
        self.assertIn("구조 미해결", rows.loc["촉매", "구조 상태"])
        self.assertIn("원자료에 없음", rows.loc["리간드", "구조 상태"])
        self.assertIn("원자료 누락", rows.loc["용매", "구조 상태"])
        self.assertFalse(any("선별 결과" in item.value.columns for item in app.dataframe))
        self.assess.assert_not_called()

    def test_user_screening_only_passes_existing_canonical_smiles(self):
        app = self.open_view()
        self.assess.side_effect = None
        self.assess.return_value = {"status": "no_alert_detected", "alerts": []}
        app.button(key="condition_evidence_screen").click().run()
        self.assert_no_exception(app)
        self.assess.assert_called_once_with("CCO")
        rows = self.table(app, "선별 결과").set_index("역할")
        # Only the screened structure gets a screening verdict. Unresolved and
        # missing components keep their own status and an unassessed alert cell,
        # so they can never be read as a negative screen.
        self.assertEqual(rows.loc["리간드", "선별 결과"], "해당 없음")
        self.assertEqual(rows.loc["리간드", "경고구조"], "해당 없음")
        self.assertEqual(rows.loc["시약", "선별 결과"], "구조 미해결")
        self.assertEqual(rows.loc["촉매", "선별 결과"], "원자료 누락")
        self.assertEqual(rows.loc["시약", "경고구조"], "미평가")
        self.assertEqual(rows.loc["촉매", "경고구조"], "미평가")

    def test_resolved_label_without_canonical_structure_is_still_not_assessed(self):
        self.kb["combinations"][0]["components"].append(
            fixture_component("base", "resolved", name="structure absent")
        )
        app = self.open_view()
        self.assertEqual(
            self.table(app, "구조 상태").set_index("역할").loc["염기", "구조 상태"],
            "구조 미확인",
        )
        self.assess.side_effect = None
        self.assess.return_value = {"status": "no_alert_detected", "alerts": []}
        app.button(key="condition_evidence_screen").click().run()
        self.assert_no_exception(app)
        self.assess.assert_called_once_with("CCO")
        screened = self.table(app, "선별 결과").set_index("역할")
        self.assertEqual(screened.loc["염기", "선별 결과"], "구조 미확인")
        self.assertEqual(screened.loc["염기", "경고구조"], "미평가")

    def test_screening_failure_is_explicit_and_not_a_negative_result(self):
        app = self.open_view()
        self.assess.side_effect = RuntimeError("software-only screening failure")
        app.button(key="condition_evidence_screen").click().run()
        self.assert_no_exception(app)
        rows = self.table(app, "선별 결과").set_index("역할")
        self.assertEqual(rows.loc["리간드", "선별 결과"], "선별 실패")
        self.assertEqual(rows.loc["리간드", "경고구조"], "미평가")
        self.assertIn("software-only screening failure", self.text(app))

    def test_audit_counts_are_not_invented_when_audit_is_missing_or_mismatched(self):
        app = self.open_view()
        self.assertIn("감사 요약 파일 없음", self.text(app))
        self.kb["summary"] = {"records_retained": 27, "records_quarantined": 3}
        audit = {key: deepcopy(self.kb[key]) for key in ("schema", "provenance", "sources", "summary")}
        self.path.with_name("audit.json").write_text(json.dumps(audit), encoding="utf-8")
        app.run()
        self.assert_no_exception(app)
        self.assertIn("원자료 기록 27행 · 격리 3행", self.text(app))
        audit["sources"][0]["sha256"] = "c" * 64
        self.path.with_name("audit.json").write_text(json.dumps(audit), encoding="utf-8")
        app.run()
        self.assert_no_exception(app)
        self.assertIn("감사 요약을 확인하지 못했습니다", self.text(app))
        self.assertNotIn("원자료 기록 27행 · 격리 3행", self.text(app))

    def test_download_contains_only_selected_source_separated_original_evidence(self):
        before = deepcopy(self.kb)
        with patch.object(condition_view.st, "download_button", wraps=condition_view.st.download_button) as download:
            app = self.open_view()
            app.selectbox(key="condition_evidence_combination").select("fixture_uv:00").run()
            self.assert_no_exception(app)
            payload = json.loads(download.call_args.kwargs["data"])
        self.assertEqual(payload["scope"], "observed_in_original_only_not_proposed_for_target")
        self.assertEqual(payload["combination"]["id"], "fixture_uv:00")
        self.assertEqual(payload["combination"]["source"]["id"], "fixture_uv")
        self.assertEqual(payload["combination"]["measurement_basis"], "UV_area_percentage")
        self.assertEqual(self.kb, before)
        self.assess.assert_not_called()

    def test_missing_corrupt_and_nonfinite_evidence_show_errors_without_fake_fallback(self):
        for error in (
            FileNotFoundError("software-only missing file"),
            json.JSONDecodeError("software-only corrupt JSON", "{", 1),
            ValueError("software-only invalid schema"),
        ):
            with self.subTest(error=type(error).__name__):
                self.loader.side_effect = error
                app = self.open_view()
                self.assertIn("조건 근거 자료를 불러오지 못했습니다", self.text(app))
                self.assertEqual(len(app.selectbox), 0)
                self.assertEqual(len(app.get("download_button")), 0)
        self.loader.side_effect = lambda path: deepcopy(self.kb)
        self.kb["combinations"][0]["measurement"]["median"] = float("nan")
        app = self.open_view()
        self.assertIn("선택한 원자료 근거를 표시하지 못했습니다", self.text(app))
        self.assertEqual(len(app.get("download_button")), 0)
        self.assess.assert_not_called()

    def test_malformed_component_array_fails_explicitly(self):
        self.kb["combinations"][0]["components"] = "not a component array"
        app = self.open_view()
        self.assertIn("조건 근거 조합을 읽지 못했습니다", self.text(app))
        self.assertEqual(len(app.get("download_button")), 0)
        self.assess.assert_not_called()

    @unittest.skipUnless(condition_view.DEFAULT_PATH.is_file(), "Independently recollected local artifact unavailable")
    def test_available_actual_artifact_loads_offline_with_all_groups_and_unknowns_preserved(self):
        self.loader.side_effect = load_local_evidence
        kb = load_local_evidence()
        app = self.open_view(default_path=True)
        self.loader.assert_called_with(condition_view.DEFAULT_PATH)
        self.assertIn(f"불러온 출처 {len(kb['sources']):,}개", self.text(app))
        audit_path = condition_view.DEFAULT_PATH.with_name("audit.json")
        if audit_path.is_file():
            summary = json.loads(audit_path.read_text(encoding="utf-8"))["summary"]
            self.assertIn(
                f"원자료 기록 {summary['records_retained']:,}행 · 격리 {summary['records_quarantined']:,}행",
                self.text(app),
            )
        for reaction_type in condition_view.REACTION_TYPES:
            app.selectbox(key="condition_evidence_type").select(reaction_type).run()
            self.assert_no_exception(app)
            groups = get_combinations(kb, reaction_type, limit=len(kb["combinations"]))
            self.assertEqual(len(app.selectbox(key="condition_evidence_combination").options), len(groups))
            self.assertGreater(len(groups), 20)
            self.assertIn("온도: 자료 없음", self.text(app))
            self.assertIn(groups[0]["source"]["url"], self.text(app))
            if reaction_type == "Suzuki-Miyaura":
                self.assertTrue(app.button(key="condition_evidence_screen").disabled)
                rows = self.table(app, "구조 상태")
                self.assertNotIn("구조 해석됨", list(rows["구조 상태"]))
        self.assess.assert_not_called()


if __name__ == "__main__":
    unittest.main()
