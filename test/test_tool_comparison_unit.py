"""Synthetic fixtures for the tool comparison screen, not evidence.

Every structure, call and score here is software test material. Nothing in
this file is an AMES measurement or a toxicity determination.

The screen's job is to show what each tool said without deciding who is right,
so these tests check that nothing is quietly resolved: a disagreement stays a
disagreement, a tool that did not answer keeps its reason, and a tool that
could not run does not take the others down with it.
"""

from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import screening_judges
import tool_comparison_view as view


def _report(*materials):
    return {"routes": [{"steps": [{"materials": [
        {"canonical_smiles": item, "input_smiles": item} for item in materials]}]}]}


def _call(smiles, call, *, score=None, error=None):
    return {"input_smiles": smiles, "canonical_smiles": smiles, "call": call,
            "score": score, "kind": None, "error": error}


class MaterialListingTests(unittest.TestCase):
    def test_materials_are_listed_once_in_the_order_they_first_appear(self):
        report = {"routes": [
            {"steps": [{"materials": [{"canonical_smiles": "CCO", "input_smiles": "CCO"},
                                      {"canonical_smiles": "CC=O", "input_smiles": "CC=O"}]}]},
            {"steps": [{"materials": [{"canonical_smiles": "CCO", "input_smiles": "CCO"}]}]},
        ]}
        self.assertEqual(view.route_materials(report), ["CCO", "CC=O"])

    def test_a_report_without_routes_lists_nothing(self):
        self.assertEqual(view.route_materials({}), [])


class ComparisonTests(unittest.TestCase):
    def test_a_disagreement_is_reported_rather_than_resolved(self):
        results = {
            "rules": {"calls": [_call("CC=O", "1")]},
            "ames_xgb": {"calls": [_call("CC=O", "0", score=0.07)]},
        }
        row = view.build_comparison(["CC=O"], results)[0]
        self.assertTrue(row["_disagree"])
        self.assertEqual(row["ICH M7"], "양성")
        self.assertEqual(row[view.TOOL_PERFORMANCE["ames_xgb"]["label"]], "음성 (0.07)")

    def test_tools_that_agree_are_not_marked_as_disagreeing(self):
        results = {
            "rules": {"calls": [_call("C1CO1", "1")]},
            "ames_xgb": {"calls": [_call("C1CO1", "1", score=0.9)]},
        }
        self.assertFalse(view.build_comparison(["C1CO1"], results)[0]["_disagree"])

    def test_an_abstention_shows_its_reason_instead_of_a_blank(self):
        results = {"toxtree": {"calls": [
            _call("c1ccccc1N", "abstain", error="결정 트리를 적용하지 못했습니다")]}}
        row = view.build_comparison(["c1ccccc1N"], results)[0]
        self.assertIn("예측 결과 없음", row["Toxtree 3.1.0"])
        self.assertIn("결정 트리", row["Toxtree 3.1.0"])
        self.assertEqual(row["_answered"], 0)

    def test_an_abstention_does_not_count_as_a_disagreement(self):
        results = {
            "rules": {"calls": [_call("X", "1")]},
            "toxtree": {"calls": [_call("X", "abstain", error="오류")]},
        }
        self.assertFalse(view.build_comparison(["X"], results)[0]["_disagree"])

    def test_a_tool_that_returned_nothing_is_marked_rather_than_dropped(self):
        results = {"rules": {"calls": []}}
        row = view.build_comparison(["CCO"], results)[0]
        self.assertEqual(row["ICH M7"], "실행 안 됨")


class ToolRunningTests(unittest.TestCase):
    def test_one_failing_tool_does_not_stop_the_others_from_reporting(self):
        original = dict(screening_judges.JUDGES)
        screening_judges.JUDGES["rules"] = dict(
            original["rules"],
            run=lambda materials, **kwargs: (_ for _ in ()).throw(
                RuntimeError("규칙 실행 실패")))
        try:
            results = view.run_tools(["CCO"], ["rules", "ames_xgb"])
        finally:
            screening_judges.JUDGES.clear()
            screening_judges.JUDGES.update(original)
        self.assertIn("규칙 실행 실패", results["rules"]["tool_error"])
        self.assertNotIn("tool_error", results["ames_xgb"])


class LabelContractTests(unittest.TestCase):
    """A tool's label is also a dict key, so two tools may not share one.

    `build_comparison` files each call under the tool's label, and the matrix,
    the CSV and the score chart all read it back. Two tools with one label
    would silently drop a column. Nothing enforced this before.
    """

    def test_every_tool_label_is_distinct(self):
        labels = [info["label"] for info in view.TOOL_PERFORMANCE.values()]
        self.assertEqual(len(labels), len(set(labels)))

    def test_no_label_collides_with_a_fixed_column(self):
        labels = {info["label"] for info in view.TOOL_PERFORMANCE.values()}
        self.assertFalse(labels & {"물질", "엇갈림", "모델 적용 범위"})


class PerformanceTableTests(unittest.TestCase):
    def test_every_integrated_judge_has_a_measured_row(self):
        for judge_id in screening_judges.implemented_judge_ids():
            with self.subTest(judge=judge_id):
                self.assertIn(judge_id, view.TOOL_PERFORMANCE)

    def test_every_tool_is_measured_on_the_same_set_so_the_rows_compare(self):
        rows = view.performance_rows(list(view.TOOL_PERFORMANCE))
        sets = {row["측정 자료"] for row in rows}
        # One set for all five. The table was previously built from two
        # different evaluation files, which made the rows look like a ranking
        # when they were not comparable.
        self.assertEqual(len(sets), 1)
        self.assertTrue(all(row["측정 자료"] for row in rows))

    def test_the_table_records_which_set_the_figures_came_from(self):
        for row in view.performance_rows(list(view.TOOL_PERFORMANCE)):
            with self.subTest(tool=row["도구"]):
                self.assertIn("1,201", row["측정 자료"])

    def test_both_ich_m7_methodologies_are_covered_by_the_tool_list(self):
        """The two the guideline names, in the words its Korean text uses.

        식약처 「의약품 불순물 유전독성 평가 가이드라인」: "서로 보완적인
        2 종류의 (Q)SAR 예측 방법을 이용한다. 첫째는 전문적 경험 규칙에
        기반한 (expert rule-based) 방법, 둘째는 통계에 기반한 방법이다."
        """
        methods = {info["method"] for info in view.TOOL_PERFORMANCE.values()}
        self.assertEqual(methods, {"전문가 규칙 기반", "통계 기반"})


class AucTests(unittest.TestCase):
    """AUC is measured on the same structures as the rest of the table.

    It was added because the header chip used to show balanced accuracy with
    the 0.50 decision line beside it, which is a different quantity and read
    as a baseline.
    """

    SCORING = ("ames_xgb", "ames_desc", "admet", "epa_test")

    def test_every_tool_that_returns_a_score_has_an_auc(self):
        for tool_id in self.SCORING:
            with self.subTest(tool=tool_id):
                value = view.TOOL_PERFORMANCE[tool_id].get("auc")
                self.assertIsNotNone(value)
                self.assertTrue(0.5 <= value <= 1.0, value)

    def test_a_rule_screen_has_no_auc_and_the_table_says_why(self):
        """An alert is not a score, so there is no curve to take an area of."""
        for tool_id in ("rules", "toxtree", "qsar_toolbox"):
            with self.subTest(tool=tool_id):
                self.assertIsNone(view.TOOL_PERFORMANCE[tool_id].get("auc"))
        rows = {row["도구"]: row["AUC"] for row in view.performance_rows(["rules", "ames_xgb"])}
        self.assertEqual(rows[view.TOOL_PERFORMANCE["rules"]["label"]], "점수를 내지 않음")
        self.assertEqual(rows[view.TOOL_PERFORMANCE["ames_xgb"]["label"]], "0.902")

    def test_a_tool_that_draws_a_line_is_not_assumed_to_have_an_auc(self):
        """The two are independent: EPA TEST judges on its own line and still
        returns a score, while a rule screen does neither."""
        for tool_id, has_line, has_auc in (("epa_test", False, True),
                                           ("ames_xgb", True, True),
                                           ("toxtree", False, False)):
            with self.subTest(tool=tool_id):
                info = view.TOOL_PERFORMANCE[tool_id]
                self.assertEqual(info["line"] is not None, has_line)
                self.assertEqual(info.get("auc") is not None, has_auc)

    def test_the_two_in_house_models_are_named_by_their_features(self):
        self.assertIn("fingerprint", view.TOOL_PERFORMANCE["ames_xgb"]["label"])
        self.assertIn("descriptor", view.TOOL_PERFORMANCE["ames_desc"]["label"])


if __name__ == "__main__":
    unittest.main()


class ProfessionalLayoutTests(unittest.TestCase):
    """The rendering helpers are pure, so the layout can be checked here."""

    def _rows(self):
        results = {
            "rules": {"calls": [_call("CC=O", "1"), _call("CCO", "0")]},
            "ames_xgb": {"calls": [_call("CC=O", "0", score=0.07),
                                   _call("CCO", "0", score=0.02)]},
        }
        return view.build_comparison(["CC=O", "CCO"], results)

    def test_the_matrix_marks_the_row_where_the_tools_disagree(self):
        html = view._matrix_html(self._rows(), ["rules", "ames_xgb"])
        self.assertEqual(html.count("uk-mark"), 1)
        self.assertIn("uk-pos", html)
        self.assertIn("uk-neg", html)

    def test_a_verdict_is_a_chip_and_an_abstention_carries_its_reason(self):
        chip = view._chip(_call("X", "abstain", error="적용 범위 밖"))
        self.assertIn("uk-abs", chip)
        self.assertIn("적용 범위 밖", chip)
        self.assertIn("예측 결과 없음", chip)

    def test_a_tool_that_returned_nothing_renders_without_a_verdict_colour(self):
        chip = view._chip(None)
        self.assertIn("uk-none", chip)
        self.assertNotIn("uk-pos", chip)
        self.assertNotIn("uk-neg", chip)

    def test_the_exported_file_keeps_the_disagreement_flag(self):
        text = view.comparison_csv(self._rows(), ["rules", "ames_xgb"])
        lines = [line for line in text.splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        self.assertIn("엇갈림", lines[0])
        self.assertIn("예", lines[1])
        self.assertIn("아니오", lines[2])


class ScorePlotTests(unittest.TestCase):
    """The score-and-line plot, which only draws numbers our line judges."""

    def _rows(self, **results):
        return view.build_comparison(["CC=O"], results)

    def test_a_score_becomes_one_point_per_tool_with_its_call(self):
        rows = self._rows(
            ames_xgb={"calls": [_call("CC=O", "1", score=0.62)]},
            ames_desc={"calls": [_call("CC=O", "0", score=0.41)]})
        points = view.score_points(rows, ["ames_xgb", "ames_desc"])
        self.assertEqual([point["점수"] for point in points], [0.62, 0.41])
        self.assertEqual([point["판정"] for point in points], ["양성", "음성"])

    def test_a_rule_tool_has_no_number_to_plot(self):
        rows = self._rows(rules={"calls": [_call("CC=O", "1")]})
        self.assertEqual(view.score_points(rows, ["rules"]), [])

    def test_epa_test_is_left_out_because_our_line_did_not_judge_it(self):
        """It publishes its own call, so its number is not ours to cut."""
        rows = self._rows(epa_test={"calls": [_call("CC=O", "1", score=0.71)]})
        self.assertIsNone(view.measured_line("epa_test"))
        self.assertEqual(view.score_points(rows, ["epa_test"]), [])

    def test_an_abstention_leaves_no_point_rather_than_a_zero(self):
        rows = self._rows(ames_xgb={"calls": [_call("CC=O", "abstain",
                                                    error="적용 범위 밖")]})
        self.assertEqual(view.score_points(rows, ["ames_xgb"]), [])

    def test_the_point_keeps_the_whole_structure_for_its_tooltip(self):
        long_smiles = "C" * 40
        rows = view.build_comparison(
            [long_smiles],
            {"ames_xgb": {"calls": [_call(long_smiles, "1", score=0.8)]}})
        point = view.score_points(rows, ["ames_xgb"])[0]
        self.assertEqual(point["전체 구조"], long_smiles)
        self.assertLess(len(point["물질"]), len(long_smiles))
        self.assertTrue(point["물질"].endswith("…"))


class ApplicabilityTests(unittest.TestCase):
    """Placing a material in a measured band, not guessing its danger."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
        import applicability
        self.applicability = applicability
        if not applicability.available():
            self.skipTest("학습 지문 캐시가 없습니다")

    def test_a_structure_the_model_was_trained_on_lands_in_the_top_band(self):
        placed = self.applicability.band_for("CC=O")
        self.assertAlmostEqual(placed["similarity"], 1.0, places=3)
        self.assertEqual(placed["label"], "매우 높음")
        self.assertGreater(placed["measured_auc"], 0.95)

    def test_an_unreadable_structure_reports_unknown_rather_than_a_band(self):
        placed = self.applicability.band_for("not-a-molecule")
        self.assertIsNone(placed["similarity"])
        self.assertIsNone(placed["measured_auc"])
        self.assertEqual(placed["label"], "확인 불가")

    def test_every_band_edge_maps_to_a_measured_figure(self):
        for low, high, auc, label in self.applicability.BANDS:
            with self.subTest(band=(low, high)):
                self.assertLess(low, high)
                self.assertTrue(0.5 < auc <= 1.0)
                self.assertTrue(label)

    def test_the_summary_counts_materials_the_model_is_weak_on(self):
        summary = self.applicability.summarise(["CC=O", "not-a-molecule"])
        self.assertEqual(summary["materials"], 2)
        self.assertEqual(summary["counts"]["확인 불가"], 1)

    def test_the_band_cell_marks_a_weak_band_differently_from_a_strong_one(self):
        strong = view._band_cell({"similarity": 0.9, "measured_auc": 0.965,
                                  "label": "매우 높음"})
        weak = view._band_cell({"similarity": 0.2, "measured_auc": 0.823,
                                "label": "매우 낮음"})
        self.assertIn("0.965", strong)
        self.assertIn("0.823", weak)
        self.assertNotEqual(strong.split('class="')[1][:20],
                            weak.split('class="')[1][:20])

    def test_a_material_without_a_band_still_renders(self):
        self.assertIn("확인 불가", view._band_cell(None))


class ExplanationTests(unittest.TestCase):
    """The explanation must lead with the block split, not the feature list."""

    def _entry(self, fingerprint, named):
        return {"status": "success", "canonical_smiles": "X",
                "baseline_log_odds": -0.87,
                "fingerprint_block_contribution": fingerprint,
                "named_feature_contribution": named,
                "pushed_positive": [{"feature": "fr_nitro", "contribution": 0.4}],
                "pushed_negative": [{"feature": "MolWt", "contribution": -0.2}]}

    def test_a_fingerprint_driven_call_says_the_descriptor_list_is_a_sideshow(self):
        lines = view.explanation_lines(self._entry(3.07, 0.006))
        joined = " ".join(lines)
        self.assertIn("100%", joined)
        self.assertIn("곁가지", joined)

    def test_a_balanced_call_does_not_warn_about_the_descriptor_list(self):
        lines = view.explanation_lines(self._entry(0.5, 0.5))
        self.assertNotIn("곁가지", " ".join(lines))

    def test_both_directions_are_listed_when_present(self):
        joined = " ".join(view.explanation_lines(self._entry(1.0, 0.5)))
        self.assertIn("fr_nitro", joined)
        self.assertIn("MolWt", joined)

    def test_a_failed_explanation_says_so_instead_of_inventing_one(self):
        self.assertEqual(view.explanation_lines(None),
                         ["설명을 계산하지 못했습니다."])
        self.assertEqual(view.explanation_lines({"status": "failed"}),
                         ["설명을 계산하지 못했습니다."])


class MeasuredCutoffTest(unittest.TestCase):
    """The decision line and its trade-off table must match what was measured.

    Benchmark log 6-20, on the 933 independent structures. These are recorded
    measurements, so a change here is a change to a claim on screen.
    """

    def test_the_line_is_the_conventional_one(self):
        # The whole point of carrying the imbalance in the training weights is
        # that the line does not have to be moved off the published default.
        self.assertEqual(view.MEASURED_CUTOFF, 0.50)

    def test_the_line_costs_almost_nothing_against_the_balance_point(self):
        rows = {row[0]: row for row in view.CUTOFF_TRADE_OFF}
        best = max(view.CUTOFF_TRADE_OFF, key=lambda row: row[3])[3]
        self.assertLess(best - rows[view.MEASURED_CUTOFF][3], 0.01)

    def test_the_line_finds_most_of_the_positives(self):
        # 0.698 before the refit, 0.774 after (log 6-20). Below 0.75 the claim
        # that the weighting removed the need to move the line is no longer true.
        rows = {row[0]: row for row in view.CUTOFF_TRADE_OFF}
        self.assertGreater(rows[view.MEASURED_CUTOFF][1], 0.75)

    def test_sensitivity_falls_and_specificity_rises_with_the_line(self):
        rows = sorted(view.CUTOFF_TRADE_OFF)
        for earlier, later in zip(rows, rows[1:]):
            self.assertGreater(earlier[1], later[1])
            self.assertLess(earlier[2], later[2])


class PerToolLineTest(unittest.TestCase):
    """Every score-returning tool runs at 0.50; a rule tool carries no line.

    They share that line by measurement, not by convenience: the in-house
    models carry the collection's 30.2% active rate in their training weights
    (benchmark log 6-20), so 0.50 sits on the flat top of their trade-off
    curve instead of a third of the way down one side.
    """

    def test_rule_tools_have_no_line(self):
        for tool_id in ("rules", "toxtree", "epa_test"):
            with self.subTest(tool_id):
                self.assertIsNone(view.measured_line(tool_id))

    def test_score_tools_all_run_at_the_conventional_line(self):
        for tool_id in ("ames_xgb", "ames_desc", "admet"):
            with self.subTest(tool_id):
                self.assertEqual(view.measured_line(tool_id), 0.50)

    def test_unknown_tool_has_no_line_rather_than_a_default(self):
        self.assertIsNone(view.measured_line("nothing_registered"))

    def test_performance_table_states_each_line(self):
        rows = view.performance_rows(["ames_xgb", "admet", "toxtree"])
        lines = [row["판정 기준선"] for row in rows]
        self.assertEqual(lines, ["0.50", "0.50", "없음"])

    def test_recorded_figures_match_the_line_they_were_measured_at(self):
        # The weighted refit measured on the independent rows, log 6-20.
        ours = view.TOOL_PERFORMANCE["ames_xgb"]
        self.assertAlmostEqual(ours["balanced"], 0.847, places=3)
        self.assertAlmostEqual(ours["sensitivity"], 0.774, places=3)

    def test_every_tool_in_the_table_is_a_registered_judge(self):
        registered = set(screening_judges.judge_ids())
        self.assertTrue(set(view.TOOL_PERFORMANCE) <= registered)

    def test_explainable_model_answers_more_often_than_the_one_in_use(self):
        # It reads every structure; the fingerprint model abstains on 22%.
        self.assertGreater(view.TOOL_PERFORMANCE["ames_desc"]["answered"],
                           view.TOOL_PERFORMANCE["ames_xgb"]["answered"])


class RuleNegativeLabelTest(unittest.TestCase):
    """A rule screen raising nothing is not a negative prediction.

    The ISS Ames profiler derived from Toxtree's mutagenicity module says so
    outright, and our own figure agrees: the rule screen's sensitivity is
    0.552, so nearly half the real positives raise no alert. The call stays a
    negative so the measured sensitivity and specificity keep their meaning;
    only the word on screen changes.
    """

    def test_a_rule_tool_says_no_alert_where_a_model_says_negative(self):
        self.assertEqual(view.call_label("0", "rules"), "경고구조 없음")
        self.assertEqual(view.call_label("0", "toxtree"), "경고구조 없음")
        self.assertEqual(view.call_label("0", "ames_xgb"), "음성")
        self.assertEqual(view.call_label("0", "admet"), "음성")

    def test_a_positive_and_an_abstention_are_unaffected_by_the_tool(self):
        for tool in ("rules", "ames_xgb"):
            with self.subTest(tool):
                self.assertEqual(view.call_label("1", tool), "양성")
                self.assertEqual(view.call_label("abstain", tool), "예측 결과 없음")

    def test_the_word_reaches_both_the_chip_and_the_downloadable_table(self):
        rows = view.build_comparison(
            ["CCO"], {"rules": {"judge": "rules", "calls": [_call("CCO", "0")],
                                "evidence": {}}})
        self.assertEqual(rows[0]["ICH M7"], "경고구조 없음")
        self.assertIn("경고구조 없음", view._matrix_html(rows, ["rules"]))
        self.assertIn("경고구조 없음", view.comparison_csv(rows, ["rules"]))


class ToolKeepsItsOwnLineTest(unittest.TestCase):
    """A tool we have no measured line for must keep its own.

    EPA TEST publishes a call beside its consensus value, and the figures in
    the performance table were taken from that call. For a while the screen
    threw our own line at it instead, which reversed the tool on about half
    the structures while still showing figures measured the other way.
    """

    def _lines_used(self, tool_ids, **kwargs):
        seen = {}

        def record(tool_id, materials, **options):
            seen[tool_id] = options.get("threshold", "도구 자체")
            return {"judge": tool_id, "calls": [], "evidence": {}}

        with mock.patch.object(view.screening_judges, "run_judge",
                               side_effect=record):
            view.run_tools(["CCO"], tool_ids, **kwargs)
        return seen

    def test_a_tool_without_a_measured_line_is_not_given_ours(self):
        used = self._lines_used(["epa_test", "ames_xgb", "rules"])
        self.assertEqual(used["epa_test"], "도구 자체")
        self.assertEqual(used["rules"], "도구 자체")
        self.assertEqual(used["ames_xgb"], 0.50)

    def test_an_explicit_override_still_reaches_every_scored_tool(self):
        used = self._lines_used(["epa_test", "ames_xgb"], threshold=0.4)
        self.assertEqual(used["epa_test"], 0.4)
        self.assertEqual(used["ames_xgb"], 0.4)

    def test_the_table_says_whose_line_each_row_was_measured_at(self):
        rows = {row["도구"]: row["판정 기준선"]
                for row in view.performance_rows(
                    ["rules", "epa_test", "ames_xgb", "admet"])}
        self.assertEqual(rows["EPA TEST 5.1.2"], "도구 자체 판정")
        self.assertEqual(rows["ICH M7"], "없음")
        self.assertEqual(rows[view.TOOL_PERFORMANCE["ames_xgb"]["label"]], "0.50")
        self.assertEqual(rows["ADMET-AI 2.0.1"], "0.50")


class MeasuredIntervalTests(unittest.TestCase):
    """Each figure's measured range, and what the table does without one."""

    def test_every_tool_with_a_balanced_figure_has_a_measured_range(self):
        for tool_id, info in view.TOOL_PERFORMANCE.items():
            with self.subTest(tool=tool_id):
                bounds = info.get("balanced_interval")
                self.assertIsNotNone(bounds, f"{tool_id}에 구간이 없습니다")
                low, high = bounds
                self.assertLess(low, info["balanced"])
                self.assertGreater(high, info["balanced"])

    def test_a_score_tool_has_a_range_for_its_auc_too(self):
        for tool_id, info in view.TOOL_PERFORMANCE.items():
            if info.get("auc") is None:
                continue
            with self.subTest(tool=tool_id):
                low, high = info["auc_interval"]
                self.assertLess(low, info["auc"])
                self.assertGreater(high, info["auc"])

    def test_the_table_prints_both_ranges(self):
        row = view.performance_rows(["ames_xgb"])[0]
        self.assertEqual(row["균형 정확도 95% 구간"], "0.821 - 0.871")
        self.assertEqual(row["AUC 95% 구간"], "0.879 - 0.924")

    def test_a_rule_tool_says_it_has_no_score_rather_than_a_blank_range(self):
        row = view.performance_rows(["rules"])[0]
        self.assertEqual(row["AUC 95% 구간"], "점수를 내지 않음")
        self.assertEqual(row["균형 정확도 95% 구간"], "0.633 - 0.689")

    def test_overlapping_ranges_are_visible_where_the_order_implies_a_ranking(self):
        # ADMET-AI is listed above Toxtree and reads higher, but the two
        # ranges overlap, so the table has to let a reader see that.
        admet = view.TOOL_PERFORMANCE["admet"]["balanced_interval"]
        toxtree = view.TOOL_PERFORMANCE["toxtree"]["balanced_interval"]
        self.assertLess(admet[0], toxtree[1])
        # The fingerprint model's range does not overlap Toxtree's, so that
        # gap is a real one.
        self.assertGreater(view.TOOL_PERFORMANCE["ames_xgb"]["balanced_interval"][0],
                           toxtree[1])


class RouteScopedTableTests(unittest.TestCase):
    """The tool table on the risk screen belongs to the selected route."""

    def _report(self):
        return {"routes": [
            {"id": "r1", "steps": [
                {"materials": [{"canonical_smiles": "CCO", "input_smiles": "CCO"},
                               {"canonical_smiles": "CC=O", "input_smiles": "CC=O"}]},
                {"materials": [{"canonical_smiles": "CCO", "input_smiles": "CCO"}]}]},
            {"id": "r2", "steps": [
                {"materials": [{"canonical_smiles": "C1CO1", "input_smiles": "C1CO1"}]}]},
        ]}

    def test_only_the_selected_route_is_listed_and_repeats_are_listed_once(self):
        report = self._report()
        self.assertEqual(view.materials_of_route(report, "r1"), ["CCO", "CC=O"])
        self.assertEqual(view.materials_of_route(report, "r2"), ["C1CO1"])

    def test_the_whole_report_still_lists_every_route(self):
        # The comparison tab keeps its own wider list; the two must not be
        # the same call.
        self.assertEqual(view.route_materials(self._report()),
                         ["CCO", "CC=O", "C1CO1"])

    def test_an_unknown_or_missing_route_lists_nothing(self):
        self.assertEqual(view.materials_of_route(self._report(), "nope"), [])
        self.assertEqual(view.materials_of_route({}, "r1"), [])

    def test_each_section_has_its_own_picker_and_run_button(self):
        # The risk screen takes one tool; the comparison screen takes many.
        # Merging them made the second question vanish into the first.
        source = Path(view.__file__).read_text(encoding="utf-8")
        for key in ("risk_tool", "risk_tool_run", "comparison_tools",
                    "comparison_run"):
            self.assertEqual(source.count('key="%s"' % key), 1)
        self.assertIn('key="risk_tool", label_visibility', source)
        self.assertIn("st.selectbox(", source.split("def show_route_tool_table")[1])
        self.assertIn("st.multiselect(", source.split("def show_tool_comparison")[1])

    def test_a_stored_run_is_dropped_when_the_route_changes(self):
        report = self._report()
        stored = {"results": {}, "materials": ["CCO", "CC=O"],
                  "tools": ["rules"], "route_id": "r1"}
        with mock.patch.object(view.st, "session_state", {"k": stored}):
            self.assertIs(view.stored_tool_result(
                {"report": report, "route_id": "r1"}, state_key="k",
                tools=["rules"]), stored)
            self.assertIsNone(view.stored_tool_result(
                {"report": report, "route_id": "r2"}, state_key="k",
                tools=["rules"]))

    def test_a_stored_run_is_dropped_when_the_tool_set_changes(self):
        report = self._report()
        stored = {"results": {}, "materials": ["CCO", "CC=O"],
                  "tools": ["rules"], "route_id": "r1"}
        with mock.patch.object(view.st, "session_state", {"k": stored}):
            self.assertIsNone(view.stored_tool_result(
                {"report": report, "route_id": "r1"}, state_key="k",
                tools=["rules", "ames_xgb"]))

    def test_a_stored_run_is_dropped_when_the_route_gained_a_material(self):
        # A byproduct sweep adds candidates to the report, so answers from
        # before it no longer cover the route.
        report = self._report()
        stored = {"results": {}, "materials": ["CCO"], "tools": ["rules"],
                  "route_id": "r1"}
        with mock.patch.object(view.st, "session_state", {"k": stored}):
            self.assertIsNone(view.stored_tool_result(
                {"report": report, "route_id": "r1"}, state_key="k",
                tools=["rules"]))

    def test_the_two_sections_keep_their_answers_apart(self):
        source = Path(view.__file__).read_text(encoding="utf-8")
        self.assertIn('state_key="_risk_tool_result"', source)
        self.assertIn('state_key="_comparison_result"', source)
