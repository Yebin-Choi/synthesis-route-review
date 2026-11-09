"""Mocked AMES UI fixtures, not model benchmarks or empirical safety evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import socket
import sys
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import engine
import route_ames_view as view
from toxicology import canonicalize_smiles
from workspace import report_hash


TARGET = "CCOC(C)=O"
SESSION_KEY = "route_ames_review_evidence"
GENERATION = "11111111-1111-4111-8111-111111111111"


def fixture_payload(*smiles):
    return {
        "schema_version": "1.0",
        "routes": [
            {
                "id": f"fixture-route-{index}", "target": TARGET,
                "source": "소프트웨어 시험용 · 실제 합성 근거 아님",
                "steps": [{
                    "id": "fixture-step-1", "reactants": [material], "products": [TARGET],
                    "reagents": [], "solvents": [],
                    "conditions": {
                        "status": "observed", "source": "Software-only fixture",
                        "description": "Synthetic branch fixture, not a laboratory observation.",
                    },
                    "byproducts": {
                        "status": "success", "candidates": [],
                        "source": "Synthetic completed-attempt label, not a real prediction.",
                    },
                }],
            }
            for index, material in enumerate(smiles, 1)
        ],
    }


def fixture_report(*smiles):
    return engine.analyze_routes(fixture_payload(*smiles))


def fixture_row(smiles, score=0.2, *, status=None):
    canonical = canonicalize_smiles(smiles)
    if status is None:
        status = "success" if canonical is not None else "invalid"
    return {
        "input_smiles": smiles, "canonical_smiles": canonical, "status": status,
        "score": score if status == "success" else None,
        "prediction": ("positive" if score >= 0.5 else "negative") if status == "success" else "abstain",
        "applicability": "unknown",
        "error": None if status == "success" else "Synthetic AMES abstention; not a negative result",
    }


def fixture_result(*rows):
    return {
        "model": {
            "tool": "ADMET-AI", "version": "synthetic-ui-fixture", "endpoint": "ames",
            "weights_sha256": {"fixture.pt": "a" * 64},
        },
        "threshold": 0.5, "assessments": list(rows),
        "limitations": ["소프트웨어 시험용 모델 근거 · 실제 추론이나 안전성 검증이 아님"],
    }


class EvidenceSnapshotTests(unittest.TestCase):
    def test_snapshot_preserves_failure_and_does_not_mutate_session(self):
        report = fixture_report("CCO")
        entry = {"schema_version": 1, "report_sha256": report_hash(report),
                 "report_generation": GENERATION,
                 "model_result": None, "error": "failed attempt"}
        with patch.object(view.st, "session_state", {SESSION_KEY: entry}):
            snapshot = view.get_route_ames_evidence(report, report_generation=GENERATION)
        self.assertEqual(snapshot["error"], "failed attempt")
        self.assertIsNone(snapshot["combined_review"])
        snapshot["error"] = "changed export"
        self.assertEqual(entry["error"], "failed attempt")

    def test_snapshot_rejects_changed_report_without_inference(self):
        with patch.object(view.st, "session_state", {
            SESSION_KEY: {"report_sha256": "different", "model_result": None},
        }), patch.object(view, "assess_admet_materials") as inference:
            self.assertIsNone(view.get_route_ames_evidence(
                fixture_report("CCO"), report_generation=GENERATION,
            ))
            inference.assert_not_called()

    def test_identical_report_from_another_generation_is_not_reused(self):
        report = fixture_report("CCO")
        entry = {"schema_version": 1, "report_sha256": report_hash(report),
                 "report_generation": GENERATION,
                 "model_result": None, "error": "old attempt"}
        with patch.object(view.st, "session_state", {SESSION_KEY: entry}), \
                patch.object(view, "assess_admet_materials") as inference:
            self.assertIsNone(view.get_route_ames_evidence(
                report, report_generation="22222222-2222-4222-8222-222222222222",
            ))
            inference.assert_not_called()


class RouteAmesViewTests(unittest.TestCase):
    def setUp(self):
        self.report = fixture_report("CCO", "CCC", "not-a-molecule", "C1CO1")
        self.result = fixture_result(
            fixture_row("CCO"), fixture_row("CCC", 0.8),
            fixture_row("not-a-molecule"), fixture_row("C1CO1", 0.8),
        )
        readiness = patch.object(view, "admet_readiness", return_value={"ready": True, "missing": []})
        self.readiness = readiness.start()
        self.addCleanup(readiness.stop)
        assessor = patch.object(
            view, "assess_admet_materials", side_effect=lambda *args, **kwargs: deepcopy(self.result)
        )
        self.assess = assessor.start()
        self.addCleanup(assessor.stop)
        reviewer = patch.object(view, "review_ames_routes", wraps=view.review_ames_routes)
        self.review = reviewer.start()
        self.addCleanup(reviewer.stop)
        download = patch.object(view.st, "download_button", wraps=view.st.download_button)
        self.download = download.start()
        self.addCleanup(download.stop)
        self.forbidden_calls = []
        for target in (
            "requests.sessions.Session.request", "subprocess.run",
            "engine.generate_routes", "engine.predict_byproducts",
        ):
            forbidden = patch(target, side_effect=AssertionError("External calls are forbidden"))
            self.forbidden_calls.append(forbidden.start())
            self.addCleanup(forbidden.stop)
        network = patch.object(socket, "create_connection", side_effect=AssertionError("Network is forbidden"))
        self.forbidden_calls.append(network.start())
        self.addCleanup(network.stop)

    def tearDown(self):
        for forbidden in self.forbidden_calls:
            forbidden.assert_not_called()

    def open_view(self, report=None, *, route_id=None):
        app = AppTest.from_string(
            "import streamlit as st\n"
            "from route_ames_view import show_route_ames_review\n"
            "st.session_state['fixture_snapshot'] = show_route_ames_review(st.session_state['fixture_report'], "
            "report_generation=st.session_state['fixture_generation'], "
            "excluded_smiles=st.session_state.get('fixture_excluded_smiles', []), "
            "route_id=st.session_state.get('fixture_route_id'))",
            default_timeout=30,
        )
        app.session_state["fixture_report"] = deepcopy(self.report if report is None else report)
        app.session_state["fixture_generation"] = GENERATION
        if route_id is not None:
            app.session_state["fixture_route_id"] = route_id
        app.run()
        self.assert_no_exception(app)
        self.app = app
        return app

    def assert_no_exception(self, app):
        self.assertEqual(len(app.exception), 0)

    def text(self, app):
        return "\n".join(
            str(item.value)
            for kind in ("subheader", "markdown", "caption", "info", "warning", "error", "code", "json")
            for item in app.get(kind)
        )

    def table(self, app, column):
        return next(item.value for item in app.dataframe if column in item.value.columns)

    def verdicts(self, app, header):
        """Rows of the chip table with this first column, as plain cell text."""
        for block in (str(item.value) for item in app.markdown):
            if f"<th>{header}</th>" not in block:
                continue
            body = block.split("<tbody>")[1].split("</tbody>")[0]
            rows = []
            for row in body.split("</tr>"):
                if "<td" not in row:
                    continue
                rows.append([" ".join(re.sub(r"<[^>]+>", " ", cell).split())
                             for cell in row.split("</td>") if "<td" in cell])
            return rows
        raise AssertionError(f"칩 표를 찾지 못했습니다: {header}")

    def route_folds(self, app):
        """(route id, status) per route, read off the per-route folds.

        The flat reason table became one fold per route, so the route id and
        its verdict now live in the fold's own label.
        """
        statuses = set(view.STATUS_LABELS.values())
        folds = []
        for item in app.expander:
            parts = [part.strip() for part in str(item.label).split("·")]
            if len(parts) >= 2 and parts[1] in statuses:
                folds.append((parts[0], parts[1]))
        return folds

    def metrics(self, app):
        return {item.label: item.value for item in app.metric}

    def raw(self, app):
        """The collapsed original record, which still holds the reason codes."""
        return next(json.loads(item.value) for item in app.get("json")
                    if "reason_codes" in str(item.value))

    def run_model(self, app):
        next(button for button in app.button if button.key.startswith("route_ames_run_")).click().run()
        self.assert_no_exception(app)
        return app

    def snapshot(self):
        return self.app.session_state["fixture_snapshot"]

    def assert_visible_alert(self, app, kind, phrase):
        self.assertTrue(any(phrase in str(item.value) for item in app.get(kind)))
        for expander in app.expander:
            self.assertFalse(any(phrase in str(item.value) for item in expander.get(kind)))

    def test_initial_render_is_pending_separate_and_has_no_model_or_threshold_control(self):
        app = self.open_view()
        self.assertEqual([item.value for item in app.subheader], ["ADMET-AI 교차 검사"])
        text = self.text(app)
        for phrase in (
            "별개입니다", "동일 물질의 규칙 우려 AND", "목표물 자체는 불순물 평가에서 제외",
            "구조가 알려지지 않은 부산물은 미평가", "모든 가능한 생성물을 포괄하지 않음",
            "로컬 실제 ADMET-AI 모델 추론", "benchmark CSV 조회가 아님", "외부 API 호출 없음",
            "0.5는 고정 기준선", "최적 임계값", "보정된 독성 확률이 아님",
            "결과는 현재 세션에만 한 번 보관합니다", "아직 검사하지 않았습니다",
        ):
            self.assertIn(phrase, text)
        # Before a run there is no verdict at all, so nothing can read as negative.
        self.assertNotIn("음성", text)
        self.assertEqual([button.label for button in app.button], ["모델 검사"])
        self.assertFalse(app.button[0].disabled)
        self.assertEqual(len(app.slider), 0)
        self.assertEqual(len(app.number_input), 0)
        self.assertEqual(len(app.dataframe), 0)
        self.assess.assert_not_called()
        self.review.assert_not_called()
        self.download.assert_not_called()

    def test_missing_assets_disable_action_and_show_preparation_without_assessing(self):
        for asset, label in (
            ("interpreter", "Python 실행 파일"), ("worker", "AMES 작업 스크립트"),
            ("package", "ADMET-AI 패키지"), ("weights", "로컬 .pt 가중치"),
        ):
            with self.subTest(asset=asset):
                self.readiness.return_value = {"ready": False, "missing": [asset]}
                app = self.open_view()
                self.assertTrue(app.button[0].disabled)
                self.assertEqual(app.button[0].label, "모델 검사")
                self.assert_visible_alert(app, "warning", f"검사 준비 필요: {label}")
                self.assertIn("파일 확인만 한 상태입니다", self.text(app))
                # Nothing was run, so nothing may read as a negative result.
                self.assertNotIn("음성", self.text(app))
                app.run()
                self.assert_no_exception(app)
                self.assertNotIn(SESSION_KEY, app.session_state)
                self.assertEqual(app.session_state["fixture_report"], self.report)
                self.assertEqual(len(app.dataframe), 0)
                self.assess.assert_not_called()
                self.review.assert_not_called()
                self.download.assert_not_called()

    def test_assets_disappearing_after_render_reject_a_stale_button_click(self):
        app = self.open_view()
        self.assertFalse(app.button[0].disabled)
        self.readiness.return_value = {"ready": False, "missing": ["weights"]}
        self.run_model(app)
        self.assertTrue(app.button[0].disabled)
        self.assertNotIn(SESSION_KEY, app.session_state)
        self.assess.assert_not_called()
        self.review.assert_not_called()

    def test_missing_assets_preserve_previous_success_and_restore_explicit_retry(self):
        app = self.run_model(self.open_view())
        before = deepcopy(app.session_state[SESSION_KEY])
        snapshot_before = deepcopy(self.snapshot())
        self.readiness.return_value = {"ready": False, "missing": ["package", "weights"]}
        app.run()
        self.assert_no_exception(app)
        self.assertTrue(app.button[0].disabled)
        self.assertEqual(app.button[0].label, "모델 검사 다시")
        self.assertEqual(app.session_state[SESSION_KEY], before)
        self.assertEqual(self.snapshot(), snapshot_before)
        self.assertEqual(len(self.route_folds(app)), 4)
        self.assess.assert_called_once()
        self.readiness.return_value = {"ready": True, "missing": []}
        app.run()
        self.assert_no_exception(app)
        self.assertFalse(app.button[0].disabled)
        self.assertEqual(app.session_state[SESSION_KEY], before)
        self.assess.assert_called_once()
        self.run_model(app)
        self.assertEqual(self.assess.call_count, 2)
        self.assertNotEqual(app.session_state[SESSION_KEY]["metadata"]["evaluation_id"],
                            before["metadata"]["evaluation_id"])

    def test_essential_scope_caption_and_primary_action_stay_visible_before_and_after_run(self):
        app = self.open_view()
        # The long scope wording moved into the details expander; the short scope
        # caption stays outside it, before and after a run.
        caption = "기재된 비목표 구조만 검사합니다"
        for evaluated in (False, True):
            with self.subTest(evaluated=evaluated):
                if evaluated:
                    self.run_model(app)
                self.assertEqual([item.value for item in app.caption].count(caption), 1)
                for expander in app.expander:
                    self.assertNotIn(caption, [item.value for item in expander.caption])
                self.assertIn(view.SCOPE_LABELS["target"], self.text(app.expander[0]))
                self.assertEqual(app.button[0].proto.type, "primary")
                self.assertEqual(app.button[0].label,
                                 "모델 검사 다시" if evaluated else "모델 검사")
        self.assess.assert_called_once()

    def test_scope_cache_and_policy_details_are_grouped_without_losing_evidence(self):
        app = self.open_view()
        self.assertEqual([item.label for item in app.expander], ["검사 범위와 해석 한계"])
        details = app.expander[0]
        self.assertFalse(details.proto.expanded)
        for label in view.SCOPE_LABELS.values():
            self.assertIn(label, self.text(details))
            self.assertEqual(self.text(app).count(label), 1)
        for phrase in (
            "기본 경로 제시의 규칙 OR 현재 모델 양성 제외와 별개입니다",
            "결과는 현재 세션에만 한 번 보관합니다",
            "같은 정규화 구조의 입력 별칭은 한 평가로 합쳐질 수 있습니다",
            "모든 경로를 입력 순서대로 유지하며 모든 사유를 표시합니다",
        ):
            self.assertIn(phrase, self.text(details))
        self.assess.assert_not_called()
        self.run_model(app)
        details = next(item for item in app.expander if item.label == "검사 범위와 해석 한계")
        self.assertFalse(details.proto.expanded)
        review = self.snapshot()["combined_review"]
        self.assertEqual(json.loads(details.get("json")[0].value), review["policy"])
        for note in review["limitations"]:
            self.assertIn(note, self.text(details))
        for label in view.SCOPE_LABELS.values():
            self.assertIn(label, self.text(details))
        self.assertNotIn("공동 연구 정책·평가 한계", [item.label for item in app.expander])
        self.assess.assert_called_once()

    def test_button_submits_unique_original_inputs_from_all_steps_including_invalid_strings(self):
        payload = fixture_payload("OCC", "CCO", "not-a-molecule", "CC(")
        second_step = deepcopy(payload["routes"][0]["steps"][0])
        second_step.update(id="fixture-step-2", reactants=["CCO", "CCC", "CC("])
        payload["routes"][0]["steps"].append(second_step)
        self.report = engine.analyze_routes(payload)
        alias = fixture_row("OCC")
        alias["input_smiles_list"] = ["OCC", "CCO"]
        self.result = fixture_result(alias, fixture_row("CCC"), fixture_row("CC("), fixture_row("not-a-molecule"))
        app = self.open_view()
        self.run_model(app)
        self.assess.assert_called_once_with(
            ["OCC", "CCO", "CCC", "CC(", "not-a-molecule"], threshold=0.5
        )
        self.assertNotIn(TARGET, self.assess.call_args.args[0])
        rows = self.verdicts(app, "구조(SMILES)")
        self.assertEqual(len(rows), 4)
        # The merged-alias column is gone. Its place now says what the
        # structure is used as and where in the route it turns up.
        self.assertEqual(rows[0][1], "반응물")
        self.assertEqual(rows[0][2], "경로 1 1단계 / 경로 1 2단계 / 경로 2 1단계")
        self.assertEqual(app.button[0].label, "모델 검사 다시")
        self.review.assert_called_once()

    def test_rerun_and_exclusion_changes_only_recompute_review_with_fixed_evidence(self):
        app = self.run_model(self.open_view())
        before = deepcopy(app.session_state[SESSION_KEY])
        self.result["assessments"][0]["score"] = 0.99
        app.run()
        self.assert_no_exception(app)
        app.session_state["fixture_excluded_smiles"] = ["OCC"]
        app.run()
        self.assert_no_exception(app)
        self.assess.assert_called_once()
        self.assertEqual(app.session_state[SESSION_KEY], before)
        self.assertEqual(self.route_folds(app)[0][1], "연구 정책으로 제외")
        self.review.assert_called_with(self.report, before["model_result"], excluded_smiles=["OCC"])
        self.assertEqual(self.snapshot()["combined_review"]["policy"]["excluded_smiles"], ["CCO"])
        self.assertEqual(app.session_state["fixture_report"], self.report)

    def test_successful_retry_has_new_evaluation_identity_and_invalid_schema_is_not_reused(self):
        app = self.run_model(self.open_view())
        first = deepcopy(app.session_state[SESSION_KEY]["metadata"])
        self.run_model(app)
        second = app.session_state[SESSION_KEY]["metadata"]
        self.assertNotEqual(first["evaluation_id"], second["evaluation_id"])
        self.assertTrue(second["completed_at_utc"].endswith("+00:00"))
        self.assertEqual(first["report_sha256"], second["report_sha256"])
        entry = dict(app.session_state[SESSION_KEY])
        entry["schema_version"] = -1
        app.session_state[SESSION_KEY] = entry
        app.run()
        self.assertNotIn(SESSION_KEY, app.session_state)
        self.assertIn("아직 검사하지 않았습니다", self.text(app))
        self.assertEqual(self.assess.call_count, 2)

    def test_stable_json_key_order_does_not_invalidate_same_report(self):
        app = self.run_model(self.open_view())
        before = deepcopy(app.session_state[SESSION_KEY])
        app.session_state["fixture_report"] = dict(reversed(list(self.report.items())))
        app.run()
        self.assert_no_exception(app)
        self.assess.assert_called_once()
        self.assertEqual(app.session_state[SESSION_KEY], before)
        self.assertEqual(len(self.route_folds(app)), 4)

    def test_any_full_report_change_hides_stale_results_until_explicit_new_run(self):
        app = self.run_model(self.open_view())
        old_digest = app.session_state[SESSION_KEY]["report_sha256"]
        changed = deepcopy(self.report)
        changed["routes"][0]["source"] = "다른 출처 · 구조는 동일한 소프트웨어 시험용 보고서"
        app.session_state["fixture_report"] = changed
        self.download.reset_mock()
        app.run()
        self.assert_no_exception(app)
        self.assess.assert_called_once()
        self.assertEqual(len(app.dataframe), 0)
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertIn("아직 검사하지 않았습니다", self.text(app))
        self.assertNotIn(SESSION_KEY, app.session_state)
        self.download.assert_not_called()
        self.run_model(app)
        self.assertEqual(self.assess.call_count, 2)
        self.assertNotEqual(app.session_state[SESSION_KEY]["report_sha256"], old_digest)
        self.assertEqual(self.snapshot()["original_report"], changed)

    def test_all_route_ids_statuses_counts_and_model_provenance_are_visible(self):
        app = self.run_model(self.open_view())
        routes = self.route_folds(app)
        self.assertEqual([row[0] for row in routes],
                         [route["id"] for route in self.report["routes"]])
        # C1CO1 is a structural alert plus a model positive: both demote, so the
        # route is kept for expert review instead of being removed.
        self.assertEqual([row[1] for row in routes],
                         ["후보", "전문가 검토 필요", "정보 부족", "전문가 검토 필요"])
        # The counts are the same four numbers, now as a metric strip. The
        # structure counts and the route counts are separate strips, so the
        # reader cannot mistake one for the other.
        numbers = self.metrics(app)
        self.assertEqual(numbers["검사한 구조"], "4")
        self.assertEqual(numbers["예측 결과 없음"], "1")
        self.assertEqual(numbers["연구 정책으로 제외"], "0")
        self.assertEqual(numbers["전문가 검토 필요"], "2")
        self.assertEqual(numbers["정보 부족"], "1")
        self.assertEqual(numbers["후보"], "1")
        # Provenance in the QMRF field order, with what we did not measure
        # stated rather than omitted.
        for phrase in ("판정 대상", "Ames 변이원성", "synthetic-ui-fixture",
                       "판정 기준선", "적용 범위 평가", "이 검사에서는 하지 않음",
                       "보정된 독성 확률이 아닙니다"):
            self.assertIn(phrase, self.text(app))
        self.assertIn("a" * 64, self.text(app))
        self.assertTrue(any("원본 기록" in item.label for item in app.expander))
        # The reason codes left the page but not the record.
        self.assertTrue(self.raw(app)["reason_codes"])
        # Each structure carries a score beside its verdict.
        self.assertTrue(all("0." in row[3] for row in self.verdicts(app, "구조(SMILES)")
                            if "상충된 결과" not in row[3]))

    def test_joint_positive_demotes_and_keeps_missing_and_lower_priority_reasons(self):
        payload = fixture_payload("C1CO1")
        step = payload["routes"][0]["steps"][0]
        step["reagents"] = ["not-a-molecule", "CNC", "[Na+].[O-]N=O"]
        step["conditions"] = {"status": "unknown"}
        step["byproducts"] = {"status": "failed", "candidates": []}
        self.report = engine.analyze_routes(payload)
        self.result = fixture_result(fixture_row("C1CO1", 0.8))
        app = self.run_model(self.open_view())
        # A structural alert AND a model positive both demote under the default
        # policy, so the route stays for review with every reason on record.
        self.assertEqual(self.route_folds(app)[0][1], "전문가 검토 필요")
        # The screen shows the reasons in words; the internal codes moved into
        # the collapsed record. Both must still be complete, because a dropped
        # reason is a reason the reviewer never sees.
        codes = [row["code"] for row in self.raw(app)["reason_codes"]]
        self.assertTrue({
            "structural_alert", "admet_positive", "admet_unassessed",
            "invalid_material", "missing_information", "conditional_warning", "conditions_not_observed",
            "byproducts_incomplete",
        } <= set(codes))
        self.assertEqual(codes.count("admet_unassessed"), 3)
        # Every reason still reaches the screen; it is bulleted inside the
        # route's own fold now instead of being a row in a flat table. A
        # dropped reason is a reason the reviewer never sees.
        expected = self.snapshot()["combined_review"]["routes"][0]["reasons"]
        shown = self.text(app)
        quiet = {"admet_negative", "candidate_within_scope"}
        # A gap repeats on every step and every route, so it reads as a short
        # label with the steps beside it; its full sentence stays in the
        # saved record and in the scope block above. A finding keeps its
        # whole sentence, because that is what the reviewer acts on.
        gaps = {"missing_information", "conditions_not_observed",
                "byproducts_incomplete"}
        for reason in expected:
            code = reason["code"]
            if code in quiet:
                continue
            with self.subTest(code=code):
                if code in gaps:
                    self.assertIn(view.ROUTINE_CODES[code], shown)
                else:
                    self.assertIn(reason["message"], shown)
        self.assertEqual([fold[0] for fold in self.route_folds(app)],
                         [self.report["routes"][0]["id"]])
        self.assertEqual(set(row["route"] for row in self.raw(app)["reason_codes"]),
                         {self.report["routes"][0]["id"]})

    def test_partial_failures_and_original_error_rows_are_retained_on_rerun(self):
        self.result["assessments"][1] = fixture_row("CCC", status="failed")
        app = self.run_model(self.open_view())
        row = next(row for row in self.verdicts(app, "구조(SMILES)") if row[0] == "CCC")
        self.assertIn("상충된 결과", row[3])
        self.assertNotIn("음성", row[3])
        self.assertIn("Synthetic AMES abstention", row[3])
        self.assertEqual(self.metrics(app)["예측 결과 없음"], "2")
        self.assert_visible_alert(app, "warning", "예측 결과 없음은 음성이 아닙니다")
        self.assertEqual(self.snapshot()["model_result"], self.result)
        app.run()
        self.assert_no_exception(app)
        self.assess.assert_called_once()
        self.assertEqual(app.session_state[SESSION_KEY]["model_result"], self.result)

    def test_all_none_or_missing_assessments_are_unassessed_never_negative_candidates(self):
        self.report = fixture_report("CCO", "CCC", "not-a-molecule")
        for rows in (
            [fixture_row("CCO", status="failed"), fixture_row("CCC", status="failed"),
             fixture_row("not-a-molecule")],
            [],
        ):
            with self.subTest(returned_rows=len(rows)):
                self.result = fixture_result(*rows)
                self.result["model"].update(version=None, weights_sha256={})
                self.assess.reset_mock()
                app = self.run_model(self.open_view())
                self.assertEqual([row[1] for row in self.route_folds(app)],
                                 ["정보 부족"] * 3)
                self.assertEqual(self.snapshot()["combined_review"]["counts"]["candidate"], 0)
                self.assertIn("음성으로 대체하지 않습니다", self.text(app))
                # The version is missing and the screen says so rather than
                # leaving the field out, which is what QMRF asks for.
                self.assertIn("버전", self.text(app))
                self.assertIn("미확인", self.text(app))
                self.assertEqual(self.metrics(app)["예측 결과 없음"], str(len(rows)))
                self.assert_visible_alert(app, "warning", "점수가 나온 구조가 없습니다")
                self.assert_visible_alert(app, "warning", "모델 가중치 해시가 없습니다")
                if rows:
                    for row in self.verdicts(app, "구조(SMILES)"):
                        self.assertIn("상충된 결과", row[3])
                        self.assertNotIn("음성", row[3])
                else:
                    self.assertIn("모델 결과가 없습니다", self.text(app))
                self.assess.assert_called_once()

    def test_call_exception_is_visible_without_fake_counts_and_explicit_retry_uses_same_call(self):
        self.assess.side_effect = RuntimeError("Synthetic local backend failure")
        app = self.run_model(self.open_view())
        self.assertIn("Synthetic local backend failure", self.text(app))
        # The failure is shown as a failure: no verdict, no counts, no negative.
        self.assertNotIn("음성", self.text(app))
        self.assert_visible_alert(app, "error", "Synthetic local backend failure")
        self.assertEqual(len(app.dataframe), 0)
        self.assertNotIn("경로 수:", self.text(app))
        self.assertEqual(app.button[0].label, "모델 검사 다시")
        self.review.assert_not_called()
        self.download.assert_not_called()
        failed_call = self.assess.call_args
        failed_entry = deepcopy(app.session_state[SESSION_KEY])
        self.readiness.return_value = {"ready": False, "missing": ["interpreter"]}
        app.run()
        self.assert_no_exception(app)
        self.assertTrue(app.button[0].disabled)
        self.assertEqual(app.session_state[SESSION_KEY], failed_entry)
        self.assert_visible_alert(app, "error", "Synthetic local backend failure")
        self.assess.assert_called_once()
        self.readiness.return_value = {"ready": True, "missing": []}
        app.run()
        self.assert_no_exception(app)
        self.assertFalse(app.button[0].disabled)
        self.assert_visible_alert(app, "error", "Synthetic local backend failure")
        self.assess.assert_called_once()
        self.assess.side_effect = lambda *args, **kwargs: deepcopy(self.result)
        self.run_model(app)
        self.assertEqual(self.assess.call_count, 2)
        self.assertEqual(self.assess.call_args, failed_call)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(self.route_folds(app)), 4)

    def test_failed_retry_keeps_previous_partial_evidence_and_labels_its_counts(self):
        self.result["assessments"][1] = fixture_row("CCC", status="failed")
        app = self.run_model(self.open_view())
        before = deepcopy(app.session_state[SESSION_KEY])
        self.assess.side_effect = RuntimeError("Synthetic retry failure")
        self.run_model(app)
        entry = app.session_state[SESSION_KEY]
        self.assertEqual(entry["model_result"], before["model_result"])
        self.assertEqual(entry["metadata"], before["metadata"])
        # The retained counts are labelled as the previous run, not this attempt.
        self.assertIn("아래는 이전 실행 결과입니다", self.text(app))
        self.assertEqual(self.metrics(app)["예측 결과 없음"], "2")
        self.assert_visible_alert(app, "error", "Synthetic retry failure")
        self.assert_visible_alert(app, "warning", "아래는 이전 실행 결과입니다")
        self.assertIn("Synthetic retry failure", self.snapshot()["latest_attempt_error"])
        self.assertEqual(self.snapshot()["model_result"], before["model_result"])
        self.assertEqual(self.assess.call_count, 2)

    def test_export_snapshot_preserves_report_and_fixed_evidence_with_current_policy(self):
        report_before, result_before = deepcopy(self.report), deepcopy(self.result)
        app = self.run_model(self.open_view())
        app.session_state["fixture_excluded_smiles"] = ["OCC"]
        app.run()
        self.assert_no_exception(app)
        payload = self.snapshot()
        self.assertEqual(payload["original_report"], report_before)
        self.assertEqual(payload["model_result"], result_before)
        self.assertEqual(payload["combined_review"]["policy"]["excluded_smiles"], ["CCO"])
        self.assertEqual(payload["scope_labels"], view.SCOPE_LABELS)
        self.assertEqual(payload["policy_status_labels"], view.STATUS_LABELS)
        expected_digest = hashlib.sha256(json.dumps(
            report_before, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        metadata = payload["evidence_metadata"]
        self.assertEqual(payload["schema_version"], 1)
        self.assertTrue(metadata["evaluation_id"])
        self.assertTrue(metadata["completed_at_utc"].endswith("+00:00"))
        self.assertEqual(metadata["report_sha256"], expected_digest)
        self.assertEqual(metadata["report_generation"], GENERATION)
        self.assertEqual(metadata["source"], "local_ADMET-AI_model_inference")
        self.assertEqual(metadata["model"], result_before["model"])
        self.assertEqual(metadata["threshold"], 0.5)
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self.assertIn("소프트웨어 시험용", text)
        self.assertNotIn("\\u", text)
        json.loads(text, parse_constant=lambda value: self.fail(f"Non-finite JSON constant: {value}"))
        self.download.assert_not_called()
        self.assertEqual(app.session_state["fixture_report"], report_before)
        self.assertEqual(self.result, result_before)
        self.assess.assert_called_once()

    def test_invalid_reports_and_nonfinite_extra_fields_block_inference_explicitly(self):
        nonfinite = deepcopy(self.report)
        nonfinite["synthetic_extra_score"] = float("nan")
        for report in ({}, {"schema_version": "1.0", "routes": []}, fixture_payload("CCO"), nonfinite):
            with self.subTest(report=report):
                app = self.open_view(report)
                self.assertIn("입력 보고서를 확인하지 못했습니다", self.text(app))
                self.assertEqual(len(app.button), 0)
                self.assertEqual(len(app.dataframe), 0)
        self.assess.assert_not_called()
        self.download.assert_not_called()

    def test_invalid_model_evidence_or_prohibited_target_policy_has_no_fake_review(self):
        self.result["assessments"][0]["score"] = float("nan")
        app = self.run_model(self.open_view())
        self.assertIn("모델 근거를 확인하지 못했습니다", self.text(app))
        self.assertEqual(len(app.dataframe), 0)
        self.download.assert_not_called()
        self.assess.assert_called_once()
        self.result["assessments"][0] = fixture_row("CCO")
        self.run_model(app)
        self.download.reset_mock()
        app.session_state["fixture_excluded_smiles"] = [TARGET]
        app.run()
        self.assert_no_exception(app)
        self.assertIn("목표물 target", self.text(app))
        self.assertEqual(len(app.dataframe), 0)
        self.download.assert_not_called()
        self.assertEqual(self.assess.call_count, 2)


class SelectedRouteTableTests(unittest.TestCase):
    """The table follows the route the reader picked, and names each material.

    A reader looking at route 1 was shown every route's materials at once,
    with no column saying which route or step a structure came from. The
    default is now the selected route; the whole report is one toggle away.
    """

    setUp = RouteAmesViewTests.setUp
    tearDown = RouteAmesViewTests.tearDown
    open_view = RouteAmesViewTests.open_view
    run_model = RouteAmesViewTests.run_model
    assert_no_exception = RouteAmesViewTests.assert_no_exception
    verdicts = RouteAmesViewTests.verdicts
    metrics = RouteAmesViewTests.metrics
    text = RouteAmesViewTests.text

    def test_default_shows_only_the_selected_route_and_the_toggle_shows_them_all(self):
        route_id = self.report["routes"][1]["id"]
        app = self.run_model(self.open_view(route_id=route_id))
        rows = self.verdicts(app, "구조(SMILES)")
        self.assertEqual([row[0] for row in rows], ["CCC"])
        self.assertEqual(rows[0][1], "반응물")
        self.assertEqual(rows[0][2], "1단계")
        self.assertEqual(self.metrics(app)["검사한 구조"], "1")
        scope = next(item for item in app.radio if item.label == "보기 범위")
        self.assertEqual(scope.options, ["선택한 경로", "전체 경로 보기"])
        scope.set_value("전체 경로 보기").run()
        self.assert_no_exception(app)
        rows = self.verdicts(app, "구조(SMILES)")
        self.assertEqual(len(rows), 4)
        # Across the report a step number alone is ambiguous, so the route
        # is named with it.
        self.assertEqual(rows[0][2], "경로 1 1단계")
        self.assertEqual(self.metrics(app)["검사한 구조"], "4")
        self.assess.assert_called_once()

    def test_byproduct_candidates_are_named_apart_from_the_written_materials(self):
        payload = fixture_payload("CCO")
        step = payload["routes"][0]["steps"][0]
        step["reagents"] = ["CNC"]
        step["byproducts"] = {"status": "success", "candidates": ["C1CO1"],
                              "source": "Synthetic fixture candidate, not a prediction."}
        self.report = engine.analyze_routes(payload)
        self.result = fixture_result(fixture_row("CCO"), fixture_row("CNC"),
                                     fixture_row("C1CO1", 0.8))
        app = self.run_model(self.open_view(route_id=self.report["routes"][0]["id"]))
        kinds = {row[0]: row[1] for row in self.verdicts(app, "구조(SMILES)")}
        self.assertEqual(kinds["CCO"], "반응물")
        self.assertEqual(kinds["CNC"], "시약")
        self.assertEqual(kinds["C1CO1"], "부산물 후보")
        # One route, so there is nothing to choose between.
        self.assertEqual([item.label for item in app.radio], [])
        self.assertIn("경로 1 · 물질별 검사 결과", self.text(app))

    def test_a_structure_used_twice_lists_both_steps_and_both_roles(self):
        payload = fixture_payload("CCO")
        second = deepcopy(payload["routes"][0]["steps"][0])
        second.update(id="fixture-step-2", reactants=["CCC"], reagents=["CCO"])
        payload["routes"][0]["steps"].append(second)
        self.report = engine.analyze_routes(payload)
        self.result = fixture_result(fixture_row("CCO"), fixture_row("CCC"))
        app = self.run_model(self.open_view(route_id=self.report["routes"][0]["id"]))
        row = next(row for row in self.verdicts(app, "구조(SMILES)") if row[0] == "CCO")
        self.assertEqual(row[1], "반응물 / 시약")
        self.assertEqual(row[2], "1·2단계")


if __name__ == "__main__":
    unittest.main()
