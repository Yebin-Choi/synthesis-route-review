"""Synthetic arithmetic/identity checks, not empirical AMES validation.

Parent verification: python -m unittest discover -s test -p 'test_ames_thresholds_unit.py'
The real PNG command additionally needs the existing ADMET environment's matplotlib.
"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "scripts"))
import analyze_ames_thresholds as thresholds


_DATA_COLUMNS = ("id", "smiles", "label", "source", "split")
_SCORE_COLUMNS = (
    "id", "smiles", "tool", "model", "version", "endpoint", "prediction",
    "applicability", "source", "score",
)
_SOURCE = "synthetic software fixture; not experimental evidence"


def material(index, label, score, **extra):
    return dict({"id": f"m{index}", "smiles": "C" * (index + 1), "label": label, "score": score}, **extra)


def dataset():
    return [
        {"id": "m0", "smiles": "CCO", "label": "1", "source": _SOURCE, "split": "test"},
        {"id": "m1", "smiles": "CCN", "label": "0", "source": _SOURCE, "split": "test"},
    ]


def prediction(row, score="0.8", **extra):
    result = {
        "id": row["id"], "smiles": row["smiles"], "tool": "fixture",
        "model": "fixed-score-fixture", "version": "1", "endpoint": "ames",
        "prediction": "1", "applicability": "unknown", "source": _SOURCE, "score": score,
    }
    result.update(extra)
    return result


class ThresholdArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            material(0, 1, 0), material(1, 1, 0.5), material(2, 1, 1),
            material(3, 0, 0), material(4, 0, 0.5), material(5, 0, 1),
        ]

    def test_endpoints_and_ties_use_greater_than_or_equal(self):
        report = thresholds.analyze(self.rows)
        sweep = report["threshold_sweep"]
        self.assertEqual([row["threshold"] for row in sweep], [index / 100 for index in range(101)])
        self.assertEqual(sum(row["threshold"] == 0.5 for row in sweep), 1)
        for index, expected in ((0, (3, 0, 0, 3)), (50, (2, 1, 1, 2)), (100, (1, 2, 2, 1))):
            with self.subTest(threshold=sweep[index]["threshold"]):
                self.assertEqual(tuple(sweep[index][key] for key in ("tp", "fn", "tn", "fp")), expected)
                self.assertEqual(sweep[index]["assessed"], 6)
                self.assertEqual(sweep[index]["coverage"], 1)
        self.assertEqual(report["baseline_metrics"], sweep[50])
        self.assertEqual(sweep[49]["tp"], 2)
        self.assertEqual(sweep[50]["tp"], 2)
        self.assertEqual(sweep[51]["tp"], 1)
        self.assertIsNone(sweep[0]["npv"])
        self.assertAlmostEqual(sweep[50]["sensitivity"], 2 / 3)
        self.assertAlmostEqual(sweep[50]["specificity"], 1 / 3)
        self.assertEqual(sweep[50]["ppv"], 0.5)
        self.assertEqual(sweep[50]["npv"], 0.5)
        self.assertEqual(sweep[50]["balanced_accuracy"], 0.5)

    def test_distinct_class_and_predictive_value_denominators(self):
        labels = (1, 1, 1, 0, 0, 0, 0, 0)
        scores = (0.9, 0.8, 0.1, 0.9, 0.7, 0.2, 0.3, 0.4)
        report = thresholds.analyze([material(index, label, score) for index, (label, score) in enumerate(zip(labels, scores))])
        metric = report["baseline_metrics"]
        self.assertEqual((metric["tp"], metric["fn"], metric["tn"], metric["fp"]), (2, 1, 3, 2))
        self.assertEqual((metric["positives"], metric["negatives"]), (3, 5))
        self.assertAlmostEqual(metric["sensitivity"], 2 / 3)
        self.assertAlmostEqual(metric["specificity"], 3 / 5)
        self.assertEqual(metric["ppv"], 1 / 2)
        self.assertEqual(metric["npv"], 3 / 4)
        self.assertAlmostEqual(metric["balanced_accuracy"], ((2 / 3) + (3 / 5)) / 2)

    def test_absent_observed_classes_produce_null_not_perfect_metrics(self):
        for label in (0, 1):
            with self.subTest(label=label):
                metric = thresholds.analyze([material(0, label, 0), material(1, label, 1)])["baseline_metrics"]
                self.assertIsNone(metric["sensitivity"] if label == 0 else metric["specificity"])
                self.assertIsNone(metric["balanced_accuracy"])
                if label == 0:
                    self.assertIsNone(metric["end_to_end_positive_detection_fraction"])
                    self.assertEqual((metric["ppv"], metric["npv"]), (0, 1))
                else:
                    self.assertEqual((metric["ppv"], metric["npv"]), (1, 0))

    def test_absent_predicted_classes_produce_null_predictive_values(self):
        negative = thresholds.analyze([material(0, 1, 0), material(1, 0, 0)])["baseline_metrics"]
        positive = thresholds.analyze([material(0, 1, 1), material(1, 0, 1)])["baseline_metrics"]
        self.assertIsNone(negative["ppv"])
        self.assertEqual(negative["npv"], 0.5)
        self.assertIsNone(positive["npv"])
        self.assertEqual(positive["ppv"], 0.5)

    def test_invalid_scores_remain_abstentions_and_positive_denominators(self):
        invalid = ("", None, "NaN", math.inf, -math.inf, "not-a-score", -0.01, 1.01, True)
        rows = [material(index, 1, score) for index, score in enumerate(invalid)]
        rows.append(material(9, 1, 0.9))
        report = thresholds.analyze(rows)
        metric = report["baseline_metrics"]
        self.assertEqual((metric["total"], metric["positives"], metric["assessed"], metric["abstentions"]), (10, 10, 1, 9))
        self.assertEqual((metric["tp"], metric["fn"], metric["tn"], metric["fp"]), (1, 0, 0, 0))
        self.assertEqual(metric["sensitivity"], 1)
        self.assertEqual(metric["coverage"], 0.1)
        self.assertEqual(metric["end_to_end_positive_detection_fraction"], 0.1)
        self.assertEqual(report["summary"]["abstentions_by_label"], {"0": 0, "1": 9})
        self.assertEqual(report["summary"]["score_status_counts"], {
            "valid": 1, "missing_score": 2, "non_numeric_score": 2,
            "non_finite_score": 3, "out_of_range_score": 2,
        })
        self.assertEqual(len(report["abstention_records"]), 9)
        self.assertEqual(report["histogram"]["groups"]["1"]["observed_materials"], 10)
        self.assertEqual(report["histogram"]["groups"]["1"]["assessed_scores"], 1)
        self.assertTrue(all(row["abstentions"] == 9 for row in report["threshold_sweep"]))
        self.assertEqual(report["threshold_sweep"][-1]["end_to_end_positive_detection_fraction"], 0)
        json.dumps(report, allow_nan=False)

    def test_saved_failures_and_domain_exclusions_override_finite_scores(self):
        rows = [
            material(0, 1, 0.9, prediction="abstain"),
            material(1, 0, 0.2, prediction="error"),
            material(2, 1, 0.9, applicability="out_of_domain"),
            material(3, 0, 0.2, prediction="unexpected"),
            material(4, 1, 0.7, applicability="unsupported"),
            material(5, 0, 0.1, prediction="0", applicability="unknown"),
            material(6, 1, 0.8, prediction="0", applicability="in_domain"),
        ]
        report = thresholds.analyze(rows)
        metric = report["baseline_metrics"]
        self.assertEqual((metric["tp"], metric["fn"], metric["tn"], metric["fp"]), (1, 0, 1, 0))
        self.assertEqual(metric["coverage"], 2 / 7)
        self.assertEqual(metric["end_to_end_positive_detection_fraction"], 1 / 4)
        self.assertEqual(report["summary"]["abstentions_by_label"], {"0": 2, "1": 3})
        self.assertEqual(report["summary"]["score_status_counts"]["valid"], 7)
        self.assertEqual(report["summary"]["unknown_applicability_assessed"], 1)
        self.assertEqual(report["summary"]["abstention_reason_counts"], {
            "invalid_applicability": 1, "invalid_saved_prediction": 1,
            "out_of_domain": 1, "saved_prediction_abstention": 2,
        })
        self.assertEqual(report["histogram"]["groups"]["1"]["assessed_scores"], 1)
        self.assertEqual(report["histogram"]["groups"]["0"]["assessed_scores"], 1)

    def test_overlapping_failure_reasons_do_not_double_count_materials(self):
        report = thresholds.analyze([material(0, 1, None, prediction="abstain")])
        self.assertEqual(report["summary"]["abstentions"], 1)
        self.assertEqual(report["summary"]["abstention_reason_counts"], {
            "missing_score": 1, "saved_prediction_abstention": 1,
        })
        for metric in report["threshold_sweep"]:
            self.assertEqual(sum(metric[key] for key in ("tp", "fn", "tn", "fp")), 0)
            self.assertEqual(metric["coverage"], 0)
            self.assertEqual(metric["end_to_end_positive_detection_fraction"], 0)
            for key in ("sensitivity", "specificity", "ppv", "npv", "balanced_accuracy"):
                self.assertIsNone(metric[key], key)
        self.assertIsNone(report["histogram"]["groups"]["1"]["mean"])
        self.assertIsNone(report["histogram"]["groups"]["0"]["min"])

    def test_histogram_edges_include_exact_zero_one_and_internal_ties(self):
        report = thresholds.analyze([
            material(0, 0, 0), material(1, 0, 0.05),
            material(2, 1, 0.5), material(3, 1, 0.999), material(4, 1, 1),
        ])
        negative = report["histogram"]["groups"]["0"]
        positive = report["histogram"]["groups"]["1"]
        self.assertEqual(negative["histogram_counts"][:2], [1, 1])
        self.assertEqual(sum(negative["histogram_counts"]), 2)
        self.assertEqual(positive["histogram_counts"][10], 1)
        self.assertEqual(positive["histogram_counts"][-1], 2)
        self.assertEqual(sum(positive["histogram_counts"]), 3)
        self.assertEqual((negative["min"], negative["max"], negative["median"]), (0, 0.05, 0.025))
        self.assertEqual((positive["min"], positive["max"], positive["median"]), (0.5, 1, 0.999))

    def test_analysis_does_not_mutate_inputs(self):
        original = deepcopy(self.rows)
        thresholds.analyze(iter(self.rows))
        self.assertEqual(self.rows, original)

    def test_empty_or_invalid_labels_are_not_silently_excluded(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            thresholds.analyze([])
        for invalid in (None, "", "positive", 0.5, True):
            with self.subTest(label=invalid), self.assertRaisesRegex(ValueError, "label must be 0 or 1"):
                thresholds.analyze([material(0, invalid, 0.9)])


class ThresholdIdentityTests(unittest.TestCase):
    def setUp(self):
        self.data = dataset()
        self.predictions = [prediction(self.data[0], "0.1"), prediction(self.data[1], "0.9")]

    def test_join_uses_exact_ids_not_row_order_or_prediction_labels(self):
        self.predictions[0].update(label="0", prediction="1")
        self.predictions[1].update(label="1", prediction="0")
        joined = thresholds.join_rows(self.data, reversed(self.predictions))
        self.assertEqual([(row["id"], row["label"], row["score"]) for row in joined], [("m0", 1, "0.1"), ("m1", 0, "0.9")])
        report = thresholds.analyze(joined)
        self.assertEqual((report["baseline_metrics"]["fn"], report["baseline_metrics"]["fp"]), (1, 1))
        self.assertEqual(report["baseline_metrics"]["balanced_accuracy"], 0)
        self.assertGreater(max(row["balanced_accuracy"] for row in report["threshold_sweep"]), 0)
        self.assertEqual(report["baseline_metrics"]["threshold"], 0.5)
        self.assertNotIn("selected_threshold", report)
        self.assertNotIn("optimal_threshold", report)
        changed_labels = [dict(row, label=str(1 - int(row["label"]))) for row in self.data]
        other = thresholds.analyze(thresholds.join_rows(changed_labels, self.predictions))
        self.assertEqual([row["threshold"] for row in other["threshold_sweep"]], [row["threshold"] for row in report["threshold_sweep"]])
        self.assertEqual(other["baseline_metrics"]["threshold"], 0.5)

    def test_equivalent_smiles_and_atom_maps_use_shared_canonicalization(self):
        for smiles in ("OCC", "[CH3:7][CH2:2][OH:4]"):
            with self.subTest(smiles=smiles):
                predictions = [dict(self.predictions[0], smiles=smiles), self.predictions[1]]
                joined = thresholds.join_rows(self.data, predictions)
                self.assertEqual(joined[0]["canonical_smiles"], "CCO")

    def test_identity_keys_are_not_trimmed_case_folded_or_structure_substituted(self):
        for invalid in ("M0", "m0 ", "alternate-id"):
            with self.subTest(material_id=invalid), self.assertRaisesRegex(ValueError, "unmatched prediction id"):
                thresholds.join_rows(self.data, [dict(self.predictions[0], id=invalid), self.predictions[1]])

    def test_structure_mismatch_and_missing_or_extra_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "structure mismatch"):
            thresholds.join_rows(self.data, [dict(self.predictions[0], smiles="CCN"), self.predictions[1]])
        with self.assertRaisesRegex(ValueError, "Missing prediction ids"):
            thresholds.join_rows(self.data, self.predictions[:1])
        with self.assertRaisesRegex(ValueError, "unmatched prediction id"):
            thresholds.join_rows(self.data, self.predictions + [dict(self.predictions[0], id="extra")])

    def test_duplicate_or_conflicting_prediction_records_fail(self):
        for score in ("0.1", "0.9"):
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "duplicate prediction id"):
                thresholds.join_rows(self.data, self.predictions + [dict(self.predictions[0], score=score)])

    def test_duplicate_and_conflicting_labeled_ids_fail(self):
        for change, message in (({}, "duplicate labeled id"), ({"label": "0"}, "conflicting labeled rows"), ({"smiles": "CCC"}, "conflicting labeled rows")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                thresholds.join_rows(self.data + [dict(self.data[0], **change)], self.predictions)

    def test_duplicate_and_conflicting_canonical_structures_fail(self):
        for label, message in (("1", "duplicate canonical structure"), ("0", "conflicting labels")):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, message):
                thresholds.join_rows(self.data + [dict(self.data[0], id="other-id", smiles="OCC", label=label)], self.predictions)
        with self.assertRaisesRegex(ValueError, "duplicate canonical structure"):
            thresholds.analyze([material(0, 1, 0.2), dict(material(1, 1, 0.9), smiles="C")])

    def test_invalid_structures_fail_not_score_abstentions(self):
        with self.assertRaisesRegex(ValueError, "invalid SMILES"):
            thresholds.join_rows([dict(self.data[0], smiles="not-smiles")], self.predictions[:1])
        with self.assertRaisesRegex(ValueError, "invalid prediction SMILES"):
            thresholds.join_rows(self.data, [dict(self.predictions[0], smiles="not-smiles"), self.predictions[1]])

    def test_mixed_models_and_wrong_endpoints_are_not_pooled(self):
        with self.assertRaisesRegex(ValueError, "one tool/model/version"):
            thresholds.join_rows(self.data, [self.predictions[0], dict(self.predictions[1], version="2")])
        with self.assertRaisesRegex(ValueError, "endpoint='ames'"):
            thresholds.join_rows(self.data, [self.predictions[0], dict(self.predictions[1], endpoint="other")])


class ThresholdCsvTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.data = dataset()
        self.predictions = [prediction(self.data[0], "", prediction="abstain"), prediction(self.data[1], "0.9")]

    def _csv(self, name, columns, rows):
        path = self.directory / name
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _inputs(self):
        return (self._csv("labels.csv", _DATA_COLUMNS, self.data),
                self._csv("scores.csv", _SCORE_COLUMNS, reversed(self.predictions)))

    def test_csv_schemas_keep_failed_rows_in_full_population(self):
        data, scores = self._inputs()
        report = thresholds.analyze(thresholds.load_joined_rows(data, scores))
        self.assertEqual(report["summary"]["total"], 2)
        self.assertEqual(report["summary"]["abstentions_by_label"], {"0": 0, "1": 1})
        self.assertEqual(report["baseline_metrics"]["end_to_end_positive_detection_fraction"], 0)
        self.assertIsNone(report["baseline_metrics"]["sensitivity"])

    def test_missing_duplicate_or_ragged_csv_columns_fail(self):
        data, _ = self._inputs()
        invalid = (
            "id,smiles\nm0,CCO\n",
            ",id,smiles,score\n,m0,CCO,0.9\n",
            "id,id,smiles,score\nm0,m0,CCO,0.9\n",
            ",".join(_SCORE_COLUMNS) + "\nm0,CCO\n",
            ",".join(_SCORE_COLUMNS) + "\n" + ",".join(str(self.predictions[0][key]) for key in _SCORE_COLUMNS) + ",extra\n",
        )
        for index, text in enumerate(invalid):
            with self.subTest(index=index):
                path = self.directory / f"invalid-{index}.csv"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    thresholds.load_joined_rows(data, path)

    def test_cli_json_and_csv_preserve_nulls_counts_and_input_receipts(self):
        data, scores = self._inputs()
        output = self.directory / "metrics.json"
        sweep = self.directory / "sweep.csv"
        histogram = self.directory / "scores.png"
        argv = ["--data", str(data), "--predictions", str(scores), "--output", str(output), "--sweep-output", str(sweep), "--histogram-output", str(histogram)]
        # Real matplotlib rendering is left to the parent-run CLI; unit tests need no GUI dependency.
        with patch.object(thresholds, "_write_histogram") as plot, redirect_stdout(io.StringIO()):
            result = thresholds.main(argv)
        self.assertEqual(result, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["inputs"]["score_column"], "score")
        self.assertFalse(report["inputs"]["fresh_inference"])
        self.assertEqual(len(report["inputs"]["saved_predictions"]["sha256"]), 64)
        self.assertEqual(report["baseline_metrics"]["abstentions_positive"], 1)
        self.assertIsNone(report["baseline_metrics"]["sensitivity"])
        with sweep.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 101)
        self.assertEqual((rows[0]["threshold"], rows[50]["threshold"], rows[-1]["threshold"]), ("0.0", "0.5", "1.0"))
        self.assertEqual(rows[50]["sensitivity"], "")
        self.assertEqual(rows[50]["end_to_end_positive_detection_fraction"], "0.0")
        self.assertEqual(rows[50]["abstentions_positive"], "1")
        plotted_report, plotted_path = plot.call_args.args
        self.assertEqual(plotted_path, histogram)
        self.assertEqual(plotted_report["histogram"]["groups"]["1"]["assessed_scores"], 0)
        self.assertIn("NOT independent", " ".join(report["limitations"]))
        self.assertIn("No ICH M7 numerical score cutoff", " ".join(report["limitations"]))

    def test_cli_output_collisions_fail_before_overwriting_or_plotting(self):
        data, scores = self._inputs()
        original_data = data.read_bytes()
        original_scores = scores.read_bytes()
        output = self.directory / "metrics.json"
        sweep = self.directory / "sweep.csv"
        histogram = self.directory / "scores.png"
        for json_path, csv_path, png_path in ((data, sweep, histogram), (scores, sweep, histogram), (output, output, histogram), (output, sweep, sweep)):
            with self.subTest(paths=(json_path, csv_path, png_path)):
                argv = ["--data", str(data), "--predictions", str(scores), "--output", str(json_path), "--sweep-output", str(csv_path), "--histogram-output", str(png_path)]
                with patch.object(thresholds, "_write_histogram") as plot, redirect_stderr(io.StringIO()) as error:
                    self.assertEqual(thresholds.main(argv), 2)
                plot.assert_not_called()
                self.assertIn("Output paths must differ", error.getvalue())
        self.assertEqual(data.read_bytes(), original_data)
        self.assertEqual(scores.read_bytes(), original_scores)


if __name__ == "__main__":
    unittest.main()
