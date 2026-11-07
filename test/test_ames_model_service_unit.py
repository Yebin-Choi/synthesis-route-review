"""Synthetic fixtures for the in-house Ames model judge, not evidence.

Every structure and score here is software test material. Nothing in this file
is an AMES measurement, a toxicity determination or a regulatory judgement.

The point of these tests is the one property the screening logic depends on:
a failure of any kind leaves an abstention, never a negative prediction. A
negative that was really a crash would let a route through on no evidence.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import ames_model_service
import screening_judges


class ReadinessTests(unittest.TestCase):
    def test_readiness_names_each_missing_asset_without_running_the_model(self):
        missing = Path("C:/does-not-exist/nowhere")
        originals = (ames_model_service.AMES_PYTHON, ames_model_service.MODEL_DIR,
                     ames_model_service.AMES_WORKER)
        ames_model_service.AMES_PYTHON = missing / "python.exe"
        ames_model_service.AMES_WORKER = missing / "worker.py"
        ames_model_service.MODEL_DIR = missing
        try:
            report = ames_model_service.ames_model_readiness()
        finally:
            (ames_model_service.AMES_PYTHON, ames_model_service.MODEL_DIR,
             ames_model_service.AMES_WORKER) = originals
        self.assertFalse(report["ready"])
        self.assertEqual(set(report["missing"]),
                         {"interpreter", "worker", "weights", "spec", "manifest"})

    def test_the_shipped_installation_reports_its_own_paths(self):
        report = ames_model_service.ames_model_readiness()
        self.assertIn("ames-env", report["interpreter"])
        self.assertIn("ames_xgboost", report["model_dir"])


class AbstentionTests(unittest.TestCase):
    def test_an_unreadable_structure_abstains_and_is_never_negative(self):
        report = ames_model_service.assess_ames_model_materials(["not a structure"])
        row = report["assessments"][0]
        self.assertEqual(row["status"], "invalid")
        self.assertIsNone(row["score"])
        self.assertIsNotNone(row["error"])

    def test_a_missing_interpreter_fails_rather_than_predicting_negative(self):
        original = ames_model_service.AMES_PYTHON
        ames_model_service.AMES_PYTHON = Path("C:/does-not-exist/python.exe")
        try:
            report = ames_model_service.assess_ames_model_materials(["CCO"])
        finally:
            ames_model_service.AMES_PYTHON = original
        row = report["assessments"][0]
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["score"])
        self.assertIn("설치", row["error"])

    def test_duplicate_structures_are_scored_once_and_keep_their_spellings(self):
        original = ames_model_service.AMES_PYTHON
        ames_model_service.AMES_PYTHON = Path("C:/does-not-exist/python.exe")
        try:
            report = ames_model_service.assess_ames_model_materials(["CCO", "OCC"])
        finally:
            ames_model_service.AMES_PYTHON = original
        self.assertEqual(len(report["assessments"]), 1)
        self.assertEqual(report["assessments"][0].get("input_smiles_list"), ["CCO", "OCC"])

    def test_a_threshold_outside_zero_to_one_is_refused(self):
        for value in (-0.1, 1.1, "0.5"):
            with self.subTest(threshold=value):
                with self.assertRaises(ValueError):
                    ames_model_service.assess_ames_model_materials(["CCO"], threshold=value)


class JudgeTests(unittest.TestCase):
    """The judge turns scores into calls; the threshold is the only line."""

    def _fake(self, rows, threshold=0.5):
        def run(materials, *, threshold=threshold):
            return {"model": {"tool": "in-house Ames XGBoost", "version": "test",
                              "endpoint": "ames", "weights_sha256": {}},
                    "threshold": threshold, "limitations": [],
                    "assessments": list(rows)}
        return run

    def _with_fake(self, rows, threshold=0.5):
        original = ames_model_service.assess_ames_model_materials
        ames_model_service.assess_ames_model_materials = self._fake(rows, threshold)
        try:
            return screening_judges.run_judge("ames_xgb", [row["input_smiles"]
                                                           for row in rows],
                                              threshold=threshold)
        finally:
            ames_model_service.assess_ames_model_materials = original

    def test_the_threshold_is_the_only_line_between_positive_and_negative(self):
        rows = [
            {"input_smiles": "A", "canonical_smiles": "A", "status": "success",
             "score": 0.49, "error": None},
            {"input_smiles": "B", "canonical_smiles": "B", "status": "success",
             "score": 0.50, "error": None},
            {"input_smiles": "C", "canonical_smiles": "C", "status": "success",
             "score": 0.51, "error": None},
        ]
        result = self._with_fake(rows)
        self.assertEqual([call["call"] for call in result["calls"]], ["0", "1", "1"])
        self.assertEqual(result["judge"], "ames_xgb")

    def test_a_failed_row_abstains_instead_of_becoming_a_negative_call(self):
        rows = [{"input_smiles": "A", "canonical_smiles": "A", "status": "failed",
                 "score": None, "error": "backend down"}]
        result = self._with_fake(rows)
        self.assertEqual(result["calls"][0]["call"], "abstain")
        self.assertIsNone(result["calls"][0]["score"])
        self.assertEqual(result["calls"][0]["error"], "backend down")

    def test_the_judge_reports_which_weights_produced_the_scores(self):
        rows = [{"input_smiles": "A", "canonical_smiles": "A", "status": "success",
                 "score": 0.9, "error": None}]
        evidence = self._with_fake(rows)["evidence"]
        self.assertEqual(evidence["tool"], "in-house Ames XGBoost")
        self.assertEqual(evidence["endpoint"], "ames")
        self.assertEqual(evidence["threshold"], 0.5)
        self.assertIn("weights_sha256", evidence)

    def test_the_judge_is_registered_as_an_integrated_model_judge(self):
        entry = screening_judges.get_judge("ames_xgb")
        self.assertTrue(entry["implemented"])
        self.assertTrue(entry["uses_threshold"])
        self.assertEqual(entry["kind"], "model")


if __name__ == "__main__":
    unittest.main()
