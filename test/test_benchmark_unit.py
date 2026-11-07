"""Software-only fixture labels test arithmetic, not empirical AMES performance.

Parent verification: python -m unittest discover -s test -p 'test_benchmark_unit.py'
"""

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import benchmark


_DATA_COLUMNS = ("id", "smiles", "label", "source", "split")
_PREDICTION_COLUMNS = (
    "id", "smiles", "tool", "model", "version", "endpoint", "prediction",
    "applicability", "source",
)
_FIXTURE_SOURCE = "software arithmetic fixture; not experimental evidence"


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        smiles = ("CCO", "CCCO", "CCCCO", "CCCCCO", "CC", "CCC", "CCCC", "CCCCC")
        self.rows = [
            self._material(f"m{index}", structure, int(index < 4))
            for index, structure in enumerate(smiles)
        ]

    def _material(self, material_id, smiles, label, split="test"):
        return {"id": material_id, "smiles": smiles, "label": label,
                "source": _FIXTURE_SOURCE, "split": split}

    def _csv(self, name, columns, rows):
        path = self.directory / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _data(self, rows=None):
        return self._csv("data.csv", _DATA_COLUMNS, self.rows if rows is None else rows)

    def _prediction(self, material, value, model="fixture-model", version="fixture-v1",
                    tool="unrecognized-tool", applicability="in_domain"):
        return {
            "id": material["id"], "smiles": material["smiles"], "tool": tool,
            "model": model, "version": version, "endpoint": "ames",
            "prediction": "abstain" if value is None else value,
            "applicability": applicability, "source": _FIXTURE_SOURCE,
        }

    def _predictions(self, rows):
        return self._csv("predictions.csv", _PREDICTION_COLUMNS, rows)

    def _fixed_rules(self, values=None):
        """Inject fixed software outputs; labels never determine these outputs."""
        values = {} if values is None else values

        def assess(smiles):
            value = values.get(smiles, 0)
            status = {0: "no_alert_detected", 1: "structural_alert", None: "invalid"}[value]
            return {"status": status, "name": None, "alerts": []}

        return patch.object(benchmark.toxicology, "assess_material", side_effect=assess)

    def _external_id(self, report, model="fixture-model", version="fixture-v1"):
        return next(system["id"] for system in report["systems"]
                    if system["model"] == model and system["version"] == version)

    def test_balanced_fixture_metrics_include_both_classes_of_abstention(self):
        values = [1, 0, None, 1, 0, 1, None, 0]
        predictions = self._predictions([
            self._prediction(row, value) for row, value in zip(self.rows, values)
        ])
        with self._fixed_rules():
            report = benchmark.evaluate_dataset(self._data(), predictions)
        system_id = self._external_id(report)
        metrics = report["results"]["test"]["systems"][system_id]
        self.assertEqual((metrics["total"], metrics["positives"], metrics["negatives"]), (8, 4, 4))
        self.assertEqual((metrics["assessed"], metrics["abstentions"]), (6, 2))
        self.assertEqual(metrics["coverage"], 0.75)
        self.assertEqual(metrics["abstentions_by_label"], {"0": 1, "1": 1})
        self.assertEqual((metrics["tp"], metrics["fn"], metrics["tn"], metrics["fp"]), (2, 1, 2, 1))
        for key in ("sensitivity", "specificity", "balanced_accuracy", "precision", "accuracy"):
            self.assertAlmostEqual(metrics[key], 2 / 3, msg=key)
        self.assertEqual(metrics["end_to_end_positive_detection_fraction"], 0.5)
        common = report["results"]["test"]["common_intersection"]
        self.assertEqual(common["material_ids"], ["m0", "m1", "m3", "m4", "m5", "m7"])
        self.assertEqual(common["total"], 6)
        self.assertEqual(common["excluded_materials"], 2)
        for source_metrics in common["systems"].values():
            self.assertEqual(source_metrics["total"], 6)
            self.assertEqual(source_metrics["coverage"], 1)
        self.assertEqual(common["ensembles"]["strict_majority"]["total"], 6)
        self.assertEqual(common["ensembles"]["strict_majority"]["assessed"], 3)
        self.assertEqual(report["results"]["test"]["systems"]["rules"]["total"], 8)

    def test_balanced_accuracy_weights_unequal_classes_equally(self):
        rows = [dict(row, label=int(index < 2)) for index, row in enumerate(self.rows)]
        values = [1, 0, 0, 0, 0, 0, 0, 0]
        predictions = self._predictions([
            self._prediction(row, value) for row, value in zip(rows, values)
        ])
        with self._fixed_rules():
            report = benchmark.evaluate_dataset(self._data(rows), predictions)
        metrics = report["results"]["test"]["systems"][self._external_id(report)]
        self.assertEqual(metrics["sensitivity"], 0.5)
        self.assertEqual(metrics["specificity"], 1)
        self.assertEqual(metrics["balanced_accuracy"], 0.75)
        self.assertEqual(metrics["accuracy"], 0.875)

    def test_specificity_and_precision_use_their_distinct_denominators(self):
        rows = [dict(row, label=int(index < 3)) for index, row in enumerate(self.rows)]
        values = [1, 1, 0, 0, 0, 0, 1, 1]
        predictions = self._predictions([
            self._prediction(row, value) for row, value in zip(rows, values)
        ])
        with self._fixed_rules():
            report = benchmark.evaluate_dataset(self._data(rows), predictions)
        metrics = report["results"]["test"]["systems"][self._external_id(report)]
        self.assertEqual((metrics["tp"], metrics["fn"], metrics["tn"], metrics["fp"]), (2, 1, 3, 2))
        self.assertAlmostEqual(metrics["sensitivity"], 2 / 3)
        self.assertEqual(metrics["specificity"], 3 / 5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["accuracy"], 5 / 8)
        self.assertAlmostEqual(metrics["balanced_accuracy"], ((2 / 3) + (3 / 5)) / 2)

    def test_all_abstentions_and_empty_split_have_undefined_denominators(self):
        predictions = self._predictions([self._prediction(row, None) for row in self.rows])
        with self._fixed_rules():
            report = benchmark.evaluate_dataset(self._data(), predictions)
        system_id = self._external_id(report)
        metrics = report["results"]["test"]["systems"][system_id]
        self.assertEqual(metrics["assessed"], 0)
        self.assertEqual(metrics["coverage"], 0)
        self.assertEqual(metrics["abstentions_by_label"], {"0": 4, "1": 4})
        self.assertEqual(metrics["end_to_end_positive_detection_fraction"], 0)
        for key in ("sensitivity", "specificity", "balanced_accuracy", "precision", "accuracy"):
            self.assertIsNone(metrics[key], key)
        empty = report["results"]["development"]["systems"][system_id]
        self.assertEqual(empty["total"], 0)
        self.assertIsNone(empty["coverage"])
        self.assertIsNone(empty["end_to_end_positive_detection_fraction"])
        self.assertEqual(report["results"]["test"]["common_intersection"]["total"], 0)

    def test_no_positive_class_or_positive_prediction_is_not_perfect_sensitivity(self):
        rows = [dict(row, label=0) for row in self.rows]
        with self._fixed_rules():
            report = benchmark.evaluate_dataset(self._data(rows))
        metrics = report["results"]["test"]["systems"]["rules"]
        self.assertEqual(metrics["tn"], 8)
        self.assertEqual(metrics["specificity"], 1)
        self.assertEqual(metrics["accuracy"], 1)
        for key in ("sensitivity", "balanced_accuracy", "precision", "end_to_end_positive_detection_fraction"):
            self.assertIsNone(metrics[key], key)

    def test_missing_rows_are_abstentions_per_model_not_negative_predictions(self):
        rows = self.rows[:2]
        predictions = self._predictions([
            self._prediction(rows[0], 1, model="full"),
            self._prediction(rows[1], 0, model="full"),
            self._prediction(rows[0], 1, model="partial"),
        ])
        with self._fixed_rules():
            report = benchmark.evaluate_dataset(self._data(rows), predictions)
        full = self._external_id(report, "full")
        partial = self._external_id(report, "partial")
        partial_metrics = report["results"]["test"]["systems"][partial]
        self.assertEqual(partial_metrics["coverage"], 0.5)
        self.assertEqual(partial_metrics["fn"], 0)
        self.assertEqual(partial_metrics["abstentions_by_label"]["1"], 1)
        self.assertEqual(partial_metrics["applicability_counts"]["missing"], 1)
        self.assertEqual(partial_metrics["abstention_reasons"], {"missing_prediction": 1})
        self.assertIsNone(report["materials"][1]["predictions"][partial]["prediction"])
        self.assertEqual(report["results"]["test"]["systems"][full]["assessed"], 2)
        self.assertIsNone(report["results"]["test"]["systems"][full]["specificity"])
        self.assertIsNone(report["results"]["test"]["systems"][full]["balanced_accuracy"])
        common = report["results"]["test"]["common_intersection"]
        self.assertEqual(common["material_ids"], [rows[0]["id"]])
        self.assertEqual(common["systems"][full]["accuracy"], 1)
        self.assertEqual(common["systems"][partial]["accuracy"], 1)

    def test_out_of_domain_abstains_and_unknown_domain_remains_visible(self):
        predictions = self._predictions([
            self._prediction(self.rows[0], 1, applicability="unknown"),
            self._prediction(self.rows[1], 1, applicability="out_of_domain"),
            self._prediction(self.rows[4], None, applicability="unknown"),
            self._prediction(self.rows[5], 0),
        ])
        with self._fixed_rules():
            report = benchmark.evaluate_dataset(self._data(), predictions)
        system_id = self._external_id(report)
        metrics = report["results"]["test"]["systems"][system_id]
        self.assertEqual(metrics["assessed"], 2)
        self.assertEqual(metrics["applicability_counts"], {
            "in_domain": 1, "out_of_domain": 1, "unknown": 2, "missing": 4,
        })
        self.assertEqual(metrics["unknown_applicability_assessed"], 1)
        self.assertEqual(metrics["abstention_reasons"], {
            "explicit_abstention": 1, "missing_prediction": 4, "out_of_domain": 1,
        })
        entry = report["materials"][1]["predictions"][system_id]
        self.assertEqual(entry["reported_prediction"], 1)
        self.assertIsNone(entry["prediction"])
        self.assertEqual(entry["source"], _FIXTURE_SOURCE)

    def test_two_source_ensembles_ties_and_incomplete_votes(self):
        rows = self.rows[:7]
        rules_values = [0, 0, 1, 0, 1, None, None]
        external_values = [1, 0, 1, "missing", "missing", 0, 1]
        predictions = self._predictions([
            self._prediction(row, value) for row, value in zip(rows, external_values)
            if value != "missing"
        ])
        with self._fixed_rules(dict(zip((row["smiles"] for row in rows), rules_values))):
            report = benchmark.evaluate_dataset(self._data(rows), predictions)
        actual = [material["ensembles"] for material in report["materials"]]
        expected = [
            (1, None, None), (0, 0, 0), (1, 1, 1),
            (None, None, None), (1, None, None), (None, None, None), (1, None, None),
        ]
        for entry, values in zip(actual, expected):
            self.assertEqual(tuple(entry[name] for name in (
                "any_positive", "strict_majority", "unanimity")), values)
        self.assertEqual(report["results"]["test"]["ensembles"]["strict_majority"]["assessed"], 2)

    def test_three_source_majority_requires_all_values_even_with_two_positive_votes(self):
        rows = self.rows[:2]
        predictions = self._predictions([
            self._prediction(rows[0], 0, model="a"),
            self._prediction(rows[1], 1, model="a"),
            self._prediction(rows[0], 1, model="b"),
        ])
        with self._fixed_rules({rows[0]["smiles"]: 0, rows[1]["smiles"]: 1}):
            report = benchmark.evaluate_dataset(self._data(rows), predictions)
        self.assertEqual(report["materials"][0]["ensembles"], {
            "any_positive": 1, "strict_majority": 0, "unanimity": None,
        })
        self.assertEqual(report["materials"][1]["ensembles"], {
            "any_positive": 1, "strict_majority": None, "unanimity": None,
        })
        self.assertEqual(len(report["ensemble_system_ids"]), 3)

    def test_local_rules_mapping_and_known_material_warning_use_actual_assessment(self):
        rows = [
            self._material("known", "ClCC1CO1", 1),
            self._material("alert", "C1CO1", 1),
            self._material("unflagged", "CCO", 0),
        ]
        report = benchmark.evaluate_dataset(self._data(rows))
        self.assertEqual([material["rules_status"] for material in report["materials"]],
                         ["known_concern", "structural_alert", "no_alert_detected"])
        self.assertEqual([material["predictions"]["rules"]["prediction"] for material in report["materials"]],
                         [1, 1, 0])
        self.assertEqual(report["rules_benchmark_mapping"], {
            "known_concern": 1, "structural_alert": 1, "no_alert_detected": 0, "invalid": None,
        })
        self.assertEqual(report["dataset"]["splits"]["test"]["known_material_count"], 1)
        self.assertIn("not independent", report["systems"][0]["training_overlap"])
        self.assertTrue(report["systems"][0]["version"].startswith("sha256:"))
        self.assertIn("not evidence of safety", " ".join(report["notes"]))

    def test_invalid_rules_assessment_abstains_instead_of_inventing_zero(self):
        with self._fixed_rules({row["smiles"]: None for row in self.rows}):
            report = benchmark.evaluate_dataset(self._data())
        metrics = report["results"]["test"]["systems"]["rules"]
        self.assertEqual(metrics["assessed"], 0)
        self.assertEqual(metrics["abstention_reasons"], {"invalid_assessment": 8})
        self.assertTrue(all(material["ensembles"]["any_positive"] is None
                            for material in report["materials"]))

    def test_development_and_test_metrics_are_never_pooled(self):
        rows = [dict(row, label=index % 2, split="development" if index < 2 else "test")
                for index, row in enumerate(self.rows[:4])]
        predictions = self._predictions([
            self._prediction(row, value) for row, value in zip(rows, [0, 1, 1, 0])
        ])
        with self._fixed_rules():
            report = benchmark.evaluate_dataset(self._data(rows), predictions)
        system_id = self._external_id(report)
        self.assertEqual(set(report["results"]), {"development", "test"})
        self.assertEqual(report["results"]["development"]["systems"][system_id]["accuracy"], 1)
        self.assertEqual(report["results"]["test"]["systems"][system_id]["accuracy"], 0)
        self.assertEqual(report["dataset"]["splits"]["development"]["source_counts"], {_FIXTURE_SOURCE: 2})

    def test_changing_labels_does_not_change_rules_or_votes(self):
        data = self._data()
        original = benchmark.evaluate_dataset(data)
        reversed_labels = [dict(row, label=1 - row["label"]) for row in self.rows]
        changed = benchmark.evaluate_dataset(self._data(reversed_labels))
        for first, second in zip(original["materials"], changed["materials"]):
            self.assertEqual(first["predictions"], second["predictions"])
            self.assertEqual(first["ensembles"], second["ensembles"])
        self.assertEqual(original["systems"], changed["systems"])

    def test_atom_maps_and_smiles_order_match_both_id_and_structure(self):
        rows = [self._material("a", "CCO", 0)]
        prediction = self._prediction(rows[0], 1)
        prediction["smiles"] = "[OH:9][CH2:8][CH3:7]"
        report = benchmark.evaluate_dataset(self._data(rows), self._predictions([prediction]))
        system_id = self._external_id(report)
        self.assertEqual(report["materials"][0]["canonical_smiles"], "CCO")
        self.assertEqual(report["materials"][0]["predictions"][system_id]["prediction"], 1)

    def test_normalized_duplicates_are_excluded_with_no_alias_prediction_substitution(self):
        rows = [
            self._material("first", "CCO", 0),
            self._material("alias", "[CH3:1][CH2:2][OH:3]", 0),
            self._material("first", "OCC", 0),
        ]
        predictions = self._predictions([self._prediction(rows[1], 0)])
        report = benchmark.evaluate_dataset(self._data(rows), predictions)
        self.assertEqual(report["dataset"]["input_rows"], 3)
        self.assertEqual(report["dataset"]["retained_materials"], 1)
        self.assertEqual(report["dataset"]["exclusion_count"], 2)
        self.assertEqual([entry["retained_id"] for entry in report["dataset"]["exclusions"]], ["first", "first"])
        self.assertEqual(report["external_input"]["exclusion_count"], 1)
        system_id = self._external_id(report)
        self.assertIsNone(report["materials"][0]["predictions"][system_id]["prediction"])
        self.assertEqual(report["results"]["test"]["systems"][system_id]["applicability_counts"]["missing"], 1)

    def test_same_molecule_with_conflicting_labels_is_rejected_after_map_removal(self):
        rows = [self._material("a", "CCO", 0),
                self._material("b", "[CH3:4][CH2:3][OH:2]", 1)]
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            benchmark.evaluate_dataset(self._data(rows))

    def test_same_molecule_across_splits_is_rejected_after_map_removal(self):
        rows = [self._material("a", "CCO", 0, "development"),
                self._material("b", "[CH3:4][CH2:3][OH:2]", 0, "test")]
        with self.assertRaisesRegex(ValueError, "cross-split leakage"):
            benchmark.evaluate_dataset(self._data(rows))

    def test_same_id_cannot_describe_different_structures(self):
        rows = [self._material("a", "CCO", 0), self._material("a", "CC", 0)]
        with self.assertRaisesRegex(ValueError, "different structures"):
            benchmark.evaluate_dataset(self._data(rows))

    def test_prediction_structure_mismatch_and_unknown_id_are_rejected(self):
        data = self._data()
        for update, error in (({"smiles": "CO"}, "structure mismatch"),
                              ({"id": "absent"}, "unknown material id"),
                              ({"smiles": "CC("}, "invalid prediction SMILES")):
            with self.subTest(update=update):
                prediction = dict(self._prediction(self.rows[0], 1), **update)
                with self.assertRaisesRegex(ValueError, error):
                    benchmark.evaluate_dataset(data, self._predictions([prediction]))

    def test_tool_models_and_versions_are_distinct_and_provenance_is_preserved(self):
        rows = [self.rows[0]]
        predictions = self._predictions([
            self._prediction(rows[0], 1, model="a", version="v1", tool="new research tool"),
            self._prediction(rows[0], 0, model="a", version="v2", tool="new research tool"),
            self._prediction(rows[0], None, model="b", version="v1", tool="new research tool"),
        ])
        report = benchmark.evaluate_dataset(self._data(rows), predictions)
        systems = report["systems"][1:]
        self.assertEqual(len(systems), 3)
        self.assertEqual(len({system["id"] for system in systems}), 3)
        self.assertEqual({(system["model"], system["version"]) for system in systems},
                         {("a", "v1"), ("a", "v2"), ("b", "v1")})
        for system in systems:
            self.assertEqual(system["tool"], "new research tool")
            self.assertEqual(system["endpoint"], "ames")
            self.assertEqual(system["source_counts"], {_FIXTURE_SOURCE: 1})
            self.assertEqual(system["training_overlap"], "unknown")

    def test_duplicate_tool_model_version_prediction_is_rejected_even_if_identical(self):
        prediction = self._prediction(self.rows[0], 1)
        for second in (dict(prediction), dict(prediction, prediction=0)):
            with self.subTest(second=second), self.assertRaisesRegex(ValueError, "duplicate tool-model-version"):
                benchmark.evaluate_dataset(self._data(), self._predictions([prediction, second]))

    def test_prediction_field_validation_keeps_other_endpoints_out(self):
        data = self._data()
        for field, value, error in (
            ("model", " ", "model must be nonempty"),
            ("version", "", "version must be nonempty"),
            ("tool", "", "tool must be nonempty"),
            ("source", "", "source must be nonempty"),
            ("endpoint", "genotoxicity", "only endpoint='ames'"),
            ("endpoint", "carcinogenicity", "only endpoint='ames'"),
            ("endpoint", "AMES", "only endpoint='ames'"),
            ("prediction", "0.5", "prediction must be"),
            ("prediction", "", "prediction must be nonempty"),
            ("applicability", "maybe", "invalid applicability"),
        ):
            with self.subTest(field=field, value=value):
                prediction = dict(self._prediction(self.rows[0], 1), **{field: value})
                with self.assertRaisesRegex(ValueError, error):
                    benchmark.evaluate_dataset(data, self._predictions([prediction]))

    def test_dataset_field_validation_and_empty_dataset(self):
        for field, value, error in (
            ("id", "", "id must be nonempty"),
            ("source", " \t", "source must be nonempty"),
            ("smiles", "CC(", "invalid SMILES"),
            ("smiles", "CCO trailing-name", "invalid SMILES"),
            ("label", "2", "label must be 0 or 1"),
            ("label", "1.0", "label must be 0 or 1"),
            ("split", "training", "split must be development or test"),
        ):
            with self.subTest(field=field, value=value):
                row = dict(self.rows[0], **{field: value})
                with self.assertRaisesRegex(ValueError, error):
                    benchmark.evaluate_dataset(self._data([row]))
        with self.assertRaisesRegex(ValueError, "at least one valid material"):
            benchmark.evaluate_dataset(self._data([]))

    def test_missing_duplicate_and_malformed_csv_columns_are_rejected(self):
        path = self.directory / "malformed.csv"
        for text, error in (
            ("id,smiles,label,split\na,CCO,0,test\n", "missing columns: source"),
            ("id,smiles,label,source,split,id\na,CCO,0,fixture,test,a\n", "duplicate CSV"),
            ("id,smiles,label,source,split\na,CCO,0,fixture,test,extra\n", "row length"),
            ("id,smiles,label,source,split\na,CCO,0,fixture\n", "row length"),
            ("", "header is required"),
        ):
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    benchmark.evaluate_dataset(path)
        data = self._data()
        path.write_text("id,smiles,tool,model,endpoint,prediction,applicability,source\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing columns: version"):
            benchmark.evaluate_dataset(data, path)

    def test_utf8_bom_and_header_only_predictions_are_accepted(self):
        data = self._data()
        text = data.read_text(encoding="utf-8")
        data.write_text(text, encoding="utf-8-sig")
        predictions = self._predictions([])
        report = benchmark.evaluate_dataset(data, predictions)
        self.assertEqual(len(report["systems"]), 1)
        self.assertEqual(report["external_input"]["input_rows"], 0)
        self.assertEqual(report["ensemble_system_ids"], ["rules"])

    def test_cli_writes_actual_json_and_reusable_rules_prediction_csv(self):
        data = self._data()
        predictions = self._predictions([self._prediction(self.rows[0], 1, applicability="unknown")])
        output = self.directory / "report.json"
        exported = self.directory / "rules.csv"
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = benchmark.main([
                "--data", str(data), "--predictions", str(predictions),
                "--output", str(output), "--export-rules", str(exported),
            ])
        self.assertEqual(exit_code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["dataset"]["retained_materials"], 8)
        self.assertEqual(report["endpoint"], "ames")
        with exported.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertTrue(set(_PREDICTION_COLUMNS).issubset(reader.fieldnames))
            exported_rows = list(reader)
        self.assertEqual(len(exported_rows), 8)
        self.assertEqual(exported_rows[0]["model"], "assess_material")
        self.assertEqual(exported_rows[0]["version"], report["systems"][0]["version"])
        self.assertEqual(exported_rows[0]["prediction"], "0")
        self.assertEqual(exported_rows[0]["applicability"], "unknown")
        self.assertIn("unknown_AD=", stdout.getvalue())
        self.assertIn("common assessed intersection=1", stdout.getvalue())
        reimported = benchmark.evaluate_dataset(data, exported)
        self.assertEqual(len(reimported["systems"]), 2)
        self.assertEqual(reimported["external_input"]["input_rows"], 8)

    def test_cli_reports_bad_inputs_without_writing_a_report(self):
        data = self._data()
        bad = dict(self._prediction(self.rows[0], 1), smiles="CO")
        predictions = self._predictions([bad])
        output = self.directory / "must-not-exist.json"
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            exit_code = benchmark.main([
                "--data", str(data), "--predictions", str(predictions), "--output", str(output),
            ])
        self.assertEqual(exit_code, 2)
        self.assertFalse(output.exists())
        self.assertIn("structure mismatch", stderr.getvalue())

    def test_cli_rejects_output_collisions_without_overwriting_inputs(self):
        data = self._data()
        original = data.read_bytes()
        output = self.directory / "report.json"
        for arguments in (
            ["--data", str(data), "--output", str(data)],
            ["--data", str(data), "--output", str(output), "--export-rules", str(output)],
        ):
            with self.subTest(arguments=arguments), patch("sys.stderr", io.StringIO()):
                self.assertEqual(benchmark.main(arguments), 2)
        self.assertEqual(data.read_bytes(), original)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
