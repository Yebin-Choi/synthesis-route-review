"""The explainable Ames model's envelope, with the backend replaced.

Every structure and score here is software test material, never an AMES
measurement and never a toxicity determination.

What these tests are for: this model exists so a prediction can be described,
so the reasons must travel with the score; and a failure must stay an
abstention, because turning one into a negative prediction would be a safety
claim nobody made.
"""

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import ames_descriptor_service as service


def _worker(payload, *, missing_output=False, exit_error=None):
    """Stand in for the subprocess, writing what the real worker would write."""

    def run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        if missing_output:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if exit_error is not None:
            output.write_text(json.dumps({"error": exit_error, "assessments": []}),
                              encoding="utf-8")
            return subprocess.CompletedProcess(command, 1, b"", b"")
        requested = json.loads(
            Path(command[command.index("--input") + 1]).read_text(encoding="utf-8"))
        rows = [payload[item] | {"canonical_smiles": item}
                for item in requested["canonical_smiles"]]
        output.write_text(json.dumps(
            {"model": {"tool": "in-house Ames descriptors", "version": None,
                       "endpoint": "ames", "columns": 215,
                       "weights_sha256": {"model.joblib": "a" * 64}},
             "assessments": rows}, ensure_ascii=False), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    return run


SUCCESS = {"status": "success", "score": 0.81, "error": None,
           "baseline_log_odds": -0.9,
           "pushed_positive": [{"feature": "fr_epoxide", "contribution": 1.85}],
           "pushed_negative": [{"feature": "SlogP_VSA6", "contribution": -0.42}]}
NEGATIVE = {"status": "success", "score": 0.02, "error": None,
            "baseline_log_odds": -0.9, "pushed_positive": [],
            "pushed_negative": [{"feature": "AvgIpc", "contribution": -0.29}]}


class EnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.present = mock.patch.object(
            service.Path, "is_file", lambda self: True)

    def _assess(self, materials, payload, **kwargs):
        with mock.patch.object(service, "AMES_PYTHON") as python, \
             mock.patch.object(service, "MODEL_DIR") as model_dir, \
             mock.patch.object(service.subprocess, "run",
                               side_effect=_worker(payload, **kwargs)):
            python.is_file.return_value = True
            model_dir.__truediv__ = lambda self, name: mock.Mock(
                is_file=mock.Mock(return_value=True))
            return service.assess_ames_descriptor_materials(materials)

    def test_named_reasons_travel_with_the_score(self):
        report = self._assess(["C1CO1"], {"C1CO1": SUCCESS})
        row = report["assessments"][0]
        self.assertEqual(row["status"], "success")
        self.assertAlmostEqual(row["score"], 0.81)
        self.assertEqual(row["pushed_positive"][0]["feature"], "fr_epoxide")
        self.assertEqual(row["pushed_negative"][0]["feature"], "SlogP_VSA6")

    def test_default_line_is_the_conventional_one(self):
        report = self._assess(["CCO"], {"CCO": NEGATIVE})
        self.assertEqual(report["threshold"], 0.50)

    def test_invalid_structure_is_an_abstention_not_a_negative(self):
        report = self._assess(["not-a-structure"], {})
        row = report["assessments"][0]
        self.assertEqual(row["status"], "invalid")
        self.assertIsNone(row["score"])

    def test_missing_output_is_an_abstention_with_a_reason(self):
        report = self._assess(["CCO"], {"CCO": NEGATIVE}, missing_output=True)
        row = report["assessments"][0]
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["score"])
        self.assertTrue(row["error"])

    def test_backend_error_is_reported_never_scored(self):
        report = self._assess(["CCO"], {"CCO": NEGATIVE},
                              exit_error="backend_missing:xgboost")
        row = report["assessments"][0]
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["score"])
        self.assertIn("dependency", row["error"])

    def test_duplicate_structures_are_scored_once_and_aliased(self):
        report = self._assess(["CCO", "OCC"], {"CCO": NEGATIVE})
        self.assertEqual(len(report["assessments"]), 1)
        self.assertEqual(report["assessments"][0]["input_smiles_list"],
                         ["CCO", "OCC"])

    def test_provenance_of_the_weights_is_carried(self):
        report = self._assess(["C1CO1"], {"C1CO1": SUCCESS})
        self.assertIn("model.joblib", report["model"]["weights_sha256"])

    def test_limitations_say_the_score_is_not_a_probability(self):
        report = self._assess(["CCO"], {"CCO": NEGATIVE})
        joined = " ".join(report["limitations"])
        self.assertIn("not a chance", joined)
        self.assertIn("89.8%", joined)

    def test_limitations_say_why_the_line_is_the_conventional_one(self):
        """The line must be traceable to a measurement, not to convention alone.

        0.50 is where published work puts it, so a reader has no way to tell a
        measured choice from a default. The envelope says what was measured:
        the imbalance sits in the training weights, which is what puts 0.50
        beside this model's own balance point instead of far from it.
        """
        report = self._assess(["CCO"], {"CCO": NEGATIVE})
        joined = " ".join(report["limitations"])
        self.assertIn("0.50", joined)
        self.assertIn("training weights", joined)
        self.assertIn("balance point", joined)

    def test_limitations_refuse_to_resolve_a_disagreement(self):
        report = self._assess(["CCO"], {"CCO": NEGATIVE})
        self.assertTrue(any("not resolved" in item
                            for item in report["limitations"]))


class ArgumentTests(unittest.TestCase):
    def test_threshold_must_be_a_number_in_range(self):
        for bad in ("0.3", True, -0.1, 1.5):
            with self.subTest(bad), self.assertRaises(ValueError):
                service.assess_ames_descriptor_materials(["CCO"], threshold=bad)

    def test_timeout_must_be_positive(self):
        with self.assertRaises(ValueError):
            service.assess_ames_descriptor_materials(["CCO"], timeout=0)

    def test_readiness_names_what_is_missing_without_running_anything(self):
        state = service.descriptor_model_readiness()
        self.assertIn("ready", state)
        self.assertIsInstance(state["missing"], list)


if __name__ == "__main__":
    unittest.main()
