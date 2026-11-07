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
        self.assertEqual(row["자체 규칙 (ICH M7)"], "양성")
        self.assertEqual(row["우리 모델 (XGBoost)"], "음성 (0.07)")

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
        self.assertIn("판정 못함", row["Toxtree 3.1.0"])
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
        self.assertEqual(row["자체 규칙 (ICH M7)"], "실행 안 됨")


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
        methods = {info["method"] for info in view.TOOL_PERFORMANCE.values()}
        self.assertEqual(methods, {"규칙", "통계"})


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
        self.assertIn("판정 못함", chip)

    def test_a_tool_that_returned_nothing_renders_without_a_verdict_colour(self):
        chip = view._chip(None)
        self.assertIn("uk-none", chip)
        self.assertNotIn("uk-pos", chip)
        self.assertNotIn("uk-neg", chip)

    def test_the_summary_counts_answers_and_positives_per_tool(self):
        summary = view.agreement_summary(self._rows(), ["rules", "ames_xgb"])
        rules = next(row for row in summary if row["도구"].startswith("자체 규칙"))
        model = next(row for row in summary if row["도구"].startswith("우리 모델"))
        self.assertEqual(rules["판정한 물질"], 2)
        self.assertEqual(rules["양성으로 본 물질"], 1)
        self.assertEqual(model["양성으로 본 물질"], 0)

    def test_an_abstention_is_not_counted_as_an_answer_in_the_summary(self):
        results = {"toxtree": {"calls": [_call("X", "abstain", error="오류")]}}
        rows = view.build_comparison(["X"], results)
        summary = view.agreement_summary(rows, ["toxtree"])[0]
        self.assertEqual(summary["판정한 물질"], 0)
        self.assertEqual(summary["답하지 못한 물질"], 1)

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

    def test_a_nearby_line_reports_the_measured_row(self):
        self.assertEqual(view.cutoff_row(0.25)[0], 0.25)
        self.assertEqual(view.cutoff_row(0.50)[0], 0.50)

    def test_an_unmeasured_line_reports_nothing_rather_than_guessing(self):
        self.assertIsNone(view.cutoff_row(0.95))
        self.assertIsNone(view.cutoff_row(0.15))


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
        self.assertEqual(lines, ["0.50", "0.50", "해당 없음 (규칙)"])

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
        self.assertEqual(view.call_label("0", "rules"), "경보 없음")
        self.assertEqual(view.call_label("0", "toxtree"), "경보 없음")
        self.assertEqual(view.call_label("0", "ames_xgb"), "음성")
        self.assertEqual(view.call_label("0", "admet"), "음성")

    def test_a_positive_and_an_abstention_are_unaffected_by_the_tool(self):
        for tool in ("rules", "ames_xgb"):
            with self.subTest(tool):
                self.assertEqual(view.call_label("1", tool), "양성")
                self.assertEqual(view.call_label("abstain", tool), "판정 못함")

    def test_the_word_reaches_both_the_chip_and_the_downloadable_table(self):
        rows = view.build_comparison(
            ["CCO"], {"rules": {"judge": "rules", "calls": [_call("CCO", "0")],
                                "evidence": {}}})
        self.assertEqual(rows[0]["자체 규칙 (ICH M7)"], "경보 없음")
        self.assertIn("경보 없음", view._matrix_html(rows, ["rules"]))
        self.assertIn("경보 없음", view.comparison_csv(rows, ["rules"]))


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
        self.assertEqual(rows["자체 규칙 (ICH M7)"], "해당 없음 (규칙)")
        self.assertEqual(rows["우리 모델 (XGBoost)"], "0.50")
        self.assertEqual(rows["ADMET-AI 2.0.1"], "0.50")
