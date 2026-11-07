"""Synthetic subprocess/index/provenance checks; never run a real ADMET model."""

from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import admet_service as service
import admet_worker as worker
from toxicology import canonicalize_smiles


MODEL = {
    "tool": "ADMET-AI", "version": "synthetic-fixture", "endpoint": "ames",
    "weights_sha256": {"classification/model_0.pt": hashlib.sha256(b"synthetic weights").hexdigest()},
}
_DEFAULT = object()


def prediction(smiles, score, **changes):
    row = {"canonical_smiles": smiles, "status": "success", "score": score, "error": None}
    row.update(changes)
    return row


def response(rows, **changes):
    payload = {"model": deepcopy(MODEL), "assessments": rows, "error": None}
    payload.update(changes)
    return payload


class FileRunner:
    """Emulate only the worker's temporary-file subprocess protocol."""

    def __init__(self, payload=_DEFAULT, *, raw=None, returncode=0, stderr="", error=None, write_output=True):
        self.payload = payload
        self.raw = raw
        self.returncode = returncode
        self.stderr = stderr
        self.error = error
        self.write_output = write_output
        self.requests = []
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        input_path = Path(command[command.index("--input") + 1])
        output_path = Path(command[command.index("--output") + 1])
        request = json.loads(input_path.read_text(encoding="utf-8"))
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.write_output:
            if isinstance(self.raw, bytes):
                output_path.write_bytes(self.raw)
            elif self.raw is not None:
                output_path.write_text(self.raw, encoding="utf-8")
            else:
                payload = self.payload
                if payload is _DEFAULT:
                    payload = response([prediction(value, 0.25) for value in request["canonical_smiles"]])
                output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=self.returncode, stdout="", stderr=self.stderr)


class FakeFrame:
    def __init__(self, items, columns=("AMES",)):
        self.columns = list(columns)
        self._items = items

    def __getitem__(self, key):
        if key != "AMES":
            raise AssertionError("Only the AMES endpoint may be read")
        return SimpleNamespace(items=lambda: iter(self._items))


class AdmetReadinessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.python = root / "admet-env" / "Scripts" / "python.exe"
        self.worker = root / "app" / "admet_worker.py"
        self.package = root / "admet-env" / "Lib" / "site-packages" / "admet_ai"
        self.models = self.package / "resources" / "models"
        for name, value in (("ADMET_PYTHON", self.python), ("ADMET_WORKER", self.worker)):
            replacement = patch.object(service, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        self.forbidden_calls = []
        for owner, name in (
            (service.subprocess, "run"), (service.subprocess, "Popen"),
            (service, "assess_admet_materials"), (service, "canonicalize_smiles"),
            (worker, "run_predictions"),
        ):
            forbidden = patch.object(owner, name, side_effect=AssertionError("Readiness is file inspection only"))
            self.forbidden_calls.append(forbidden.start())
            self.addCleanup(forbidden.stop)

    def tearDown(self):
        for forbidden in self.forbidden_calls:
            forbidden.assert_not_called()

    def write_asset_files(self, *, interpreter=True, worker_script=True, package=True, weights=True):
        files = []
        if interpreter:
            files.append(self.python)
        if worker_script:
            files.append(self.worker)
        if package:
            files.extend((self.package / "__init__.py", self.package / "constants.py"))
        if weights:
            files.append(self.models / "classification" / "model_0.pt")
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"raise AssertionError('File-presence fixture must never be executed')\n")
        return files

    def test_missing_interpreter_is_reported_before_a_model_attempt(self):
        self.write_asset_files(interpreter=False)
        self.assertEqual(service.admet_readiness(), {"ready": False, "missing": ["interpreter"]})

    def test_missing_worker_is_reported_before_a_model_attempt(self):
        self.write_asset_files(worker_script=False)
        self.assertEqual(service.admet_readiness(), {"ready": False, "missing": ["worker"]})

    def test_missing_package_and_weights_are_reported_without_importing_them(self):
        self.write_asset_files(package=False, weights=False)
        self.assertEqual(service.admet_readiness(), {"ready": False, "missing": ["package", "weights"]})

    def test_incomplete_package_is_not_ready_even_with_weights(self):
        self.write_asset_files()
        for name in ("__init__.py", "constants.py"):
            with self.subTest(name=name):
                path = self.package / name
                content = path.read_bytes()
                path.unlink()
                try:
                    self.assertEqual(service.admet_readiness(), {"ready": False, "missing": ["package"]})
                finally:
                    path.write_bytes(content)

    def test_only_local_pt_files_in_the_default_models_directory_count_as_weights(self):
        self.write_asset_files(weights=False)
        self.models.mkdir(parents=True)
        (self.models / "metadata.json").write_text("{}", encoding="utf-8")
        (self.models / "model.PT").write_bytes(b"wrong suffix for the worker")
        (self.models / "directory.pt").mkdir()
        (self.package / "other-model.pt").write_bytes(b"outside the default resource directory")
        self.assertEqual(service.admet_readiness(), {"ready": False, "missing": ["weights"]})

    def test_unreadable_weight_directory_is_not_ready(self):
        self.write_asset_files()
        with patch.object(Path, "rglob", side_effect=PermissionError("unreadable fixture directory")):
            self.assertEqual(service.admet_readiness(), {"ready": False, "missing": ["weights"]})

    def test_available_files_are_checked_without_imports_execution_or_mutation(self):
        files = self.write_asset_files()
        before = {path: path.read_bytes() for path in files}
        with patch.dict(sys.modules, {"admet_ai": None, "admet_ai.constants": None}), patch.object(
            Path, "open", side_effect=AssertionError("Readiness must not read or change file contents"),
        ) as opened:
            self.assertEqual(service.admet_readiness(), {"ready": True, "missing": []})
        opened.assert_not_called()
        self.assertEqual({path: path.read_bytes() for path in files}, before)


class AdmetServiceTests(unittest.TestCase):
    def assess(self, smiles, runner=None, **kwargs):
        if runner is None:
            runner = FileRunner()
        with patch.object(service, "ADMET_PYTHON", Path(sys.executable)), patch.object(
            service.subprocess, "run", side_effect=runner,
        ):
            return service.assess_admet_materials(smiles, **kwargs)

    def assert_abstention(self, row, status="failed"):
        self.assertEqual(row["status"], status)
        self.assertEqual(row["prediction"], "abstain")
        self.assertIsNone(row["score"])
        self.assertEqual(row["applicability"], "unknown")
        self.assertTrue(row["error"])

    def test_shared_canonicalization_deduplicates_maps_and_preserves_aliases(self):
        aliases = [" CCO ", "OCC", "[CH3:1][CH2:2][OH:3]", "CCO"]
        runner = FileRunner()
        result = self.assess(aliases + ["not-smiles", "CCO trailing-name", None], runner)
        self.assertEqual(runner.requests, [{"canonical_smiles": [canonicalize_smiles("CCO")]}])
        self.assertEqual(len(result["assessments"]), 4)
        row = result["assessments"][0]
        self.assertEqual(row["input_smiles"], aliases[0])
        self.assertEqual(row["input_smiles_list"], aliases)
        self.assertEqual(row["status"], "success")
        self.assertEqual([row["input_smiles"] for row in result["assessments"][1:]],
                         ["not-smiles", "CCO trailing-name", "None"])
        for row in result["assessments"][1:]:
            self.assert_abstention(row, "invalid")
            self.assertIsNone(row["canonical_smiles"])

    def test_chemically_distinct_forms_are_not_merged(self):
        values = ["ClC[C@H]1CO1", "ClC[C@@H]1CO1", "CCO.[Na+]", "CCO", "[13CH3]CO",
                  "[O-]CC1CO1", "OCC1CO1", "CC=O", "C=CO"]
        runner = FileRunner()
        result = self.assess(values, runner)
        self.assertEqual(len(result["assessments"]), len(values))
        self.assertEqual(runner.requests[0]["canonical_smiles"], [canonicalize_smiles(value) for value in values])

    def test_distinct_invalid_entries_are_retained_without_backend(self):
        values = ["", " ", ".", "CC(", "CCO\x00junk", "CCO name", "", None, 12, b"CCO"]
        runner = FileRunner()
        result = self.assess(values, runner)
        self.assertEqual(runner.calls, [])
        self.assertEqual(len(result["assessments"]), len(values) - 1)
        for row in result["assessments"]:
            self.assert_abstention(row, "invalid")
            self.assertIsInstance(row["input_smiles"], str)
        self.assertEqual(result["model"]["weights_sha256"], {})

    def test_repeated_invalid_strings_are_exact_identity_deduplicated_in_mixed_batch(self):
        runner = FileRunner()
        result = self.assess(["CC(", "CCO", "CC(", " CC(", "OCC", " CC("], runner)
        self.assertEqual(runner.requests, [{"canonical_smiles": ["CCO"]}])
        invalid = [row for row in result["assessments"] if row["status"] == "invalid"]
        self.assertEqual([row["input_smiles"] for row in invalid], ["CC(", " CC("])
        for row in invalid:
            self.assert_abstention(row, "invalid")
        self.assertEqual(len(result["assessments"]), 3)

    def test_mixed_invalid_source_types_share_exact_text_identity_and_keep_valid_aliases(self):
        runner = FileRunner()
        values = ["None", None, "CCO", 12, "12", None, "None", "CC(", "OCC", "CC(", " None "]
        result = self.assess(values, runner)
        self.assertEqual(runner.requests, [{"canonical_smiles": ["CCO"]}])
        self.assertEqual(len(result["assessments"]), 5)
        invalid = [row for row in result["assessments"] if row["status"] == "invalid"]
        self.assertEqual([(row["input_smiles"], row["input_types"]) for row in invalid],
                         [("None", ["str", "NoneType"]), ("12", ["int", "str"]),
                          ("CC(", ["str"]), (" None ", ["str"])])
        for row in invalid:
            self.assert_abstention(row, "invalid")
            self.assertIsNone(row["canonical_smiles"])
        valid = result["assessments"][1]
        self.assertEqual(valid["status"], "success")
        self.assertEqual(valid["input_smiles"], "CCO")
        self.assertEqual(valid["input_smiles_list"], ["CCO", "OCC"])
        self.assertNotIn("input_types", valid)

    def test_repeated_nonstrings_and_strings_merge_without_backend_and_keep_distinct_invalid_text(self):
        runner = FileRunner()
        values = [None, None, "None", 12, "12", b"CCO", "b'CCO'", "", " ", "", ".", "CC(", " CC("]
        result = self.assess(values, runner)
        self.assertEqual(runner.calls, [])
        self.assertEqual([(row["input_smiles"], row["input_types"]) for row in result["assessments"]],
                         [("None", ["NoneType", "str"]), ("12", ["int", "str"]),
                          ("b'CCO'", ["bytes", "str"]), ("", ["str"]), (" ", ["str"]),
                          (".", ["str"]), ("CC(", ["str"]), (" CC(", ["str"])])
        for row in result["assessments"]:
            self.assert_abstention(row, "invalid")
            self.assertIsNone(row["canonical_smiles"])
        self.assertEqual(result["model"]["weights_sha256"], {})

    def test_nonstring_rendering_valid_smiles_is_still_invalid_before_text_grouping(self):
        class SmilesLike:
            def __str__(self):
                return "CCO"

        runner = FileRunner()
        result = self.assess([SmilesLike(), SmilesLike()], runner)
        self.assertEqual(runner.calls, [])
        self.assertEqual(len(result["assessments"]), 1)
        row = result["assessments"][0]
        self.assertEqual(row["input_smiles"], "CCO")
        self.assertEqual(row["input_types"], ["SmilesLike"])
        self.assertIsNone(row["canonical_smiles"])
        self.assert_abstention(row, "invalid")

    def test_single_string_is_not_split_into_characters_and_empty_batch_does_not_run(self):
        runner = FileRunner()
        result = self.assess("CCO", runner)
        self.assertEqual(runner.requests, [{"canonical_smiles": ["CCO"]}])
        self.assertEqual(len(result["assessments"]), 1)
        empty_runner = FileRunner()
        self.assertEqual(self.assess([], empty_runner)["assessments"], [])
        self.assertEqual(empty_runner.calls, [])

    def test_distinct_valid_limit_is_an_error_not_truncation(self):
        runner = FileRunner()
        with self.assertRaisesRegex(ValueError, "256 distinct valid"):
            self.assess([f"[{index}CH4]" for index in range(1, 258)], runner)
        self.assertEqual(runner.calls, [])

    def test_limit_counts_canonical_structures_not_duplicate_batch_entries(self):
        runner = FileRunner()
        result = self.assess([f"[CH3:{index}][OH:{index + 1000}]" for index in range(1, 301)], runner)
        self.assertEqual(runner.requests, [{"canonical_smiles": ["CO"]}])
        self.assertEqual(len(result["assessments"]), 1)
        self.assertEqual(len(result["assessments"][0]["input_smiles_list"]), 300)

    def test_exact_limit_runs_all_materials(self):
        runner = FileRunner()
        result = self.assess([f"[{index}CH4]" for index in range(1, 257)], runner)
        self.assertEqual(len(runner.requests[0]["canonical_smiles"]), 256)
        self.assertEqual(sum(row["status"] == "success" for row in result["assessments"]), 256)

    def test_threshold_endpoints_and_half_ties_use_greater_than_or_equal(self):
        rows = [prediction("C", 0), prediction("CC", 0.5), prediction("CCC", 1)]
        for threshold, expected in ((0, ["positive"] * 3),
                                    (0.5, ["negative", "positive", "positive"]),
                                    (1, ["negative", "negative", "positive"])):
            with self.subTest(threshold=threshold):
                result = self.assess(["C", "CC", "CCC"], FileRunner(response(rows)), threshold=threshold)
                self.assertEqual(result["threshold"], float(threshold))
                self.assertEqual([row["prediction"] for row in result["assessments"]], expected)
                self.assertEqual([row["score"] for row in result["assessments"]], [0, 0.5, 1])

    def test_invalid_thresholds_and_timeouts_are_rejected_before_subprocess(self):
        invalid = (True, False, None, math.nan, math.inf, -math.inf, "bad")
        for key, values in (("threshold", invalid + (-0.01, 1.01)), ("timeout", invalid + (0, -1))):
            for value in values:
                with self.subTest(key=key, value=value):
                    runner = FileRunner()
                    with self.assertRaises(ValueError):
                        self.assess(["CCO"], runner, **{key: value})
                    self.assertEqual(runner.calls, [])

    def test_output_reordering_does_not_change_material_identity(self):
        result = self.assess(["CCO", "CCN"], FileRunner(response([
            prediction("CCN", 0.9), prediction("CCO", 0.1),
        ])))
        self.assertEqual([(row["canonical_smiles"], row["score"], row["prediction"])
                          for row in result["assessments"]], [("CCO", 0.1, "negative"), ("CCN", 0.9, "positive")])

    def test_missing_identity_abstains_only_for_missing_material(self):
        result = self.assess(["CCO", "CCN"], FileRunner(response([prediction("CCN", 0.2)])))
        self.assert_abstention(result["assessments"][0])
        self.assertIn("exact requested structure identity", result["assessments"][0]["error"])
        self.assertEqual(result["assessments"][1]["status"], "success")

    def test_duplicate_response_identity_is_never_averaged(self):
        result = self.assess(["CCO", "CCN"], FileRunner(response([
            prediction("CCO", 0.1), prediction("CCO", 0.9), prediction("CCN", 0.2),
        ])))
        self.assert_abstention(result["assessments"][0])
        self.assertIn("duplicate", result["assessments"][0]["error"])
        self.assertEqual(result["assessments"][1]["score"], 0.2)

    def test_different_or_unexpected_identity_invalidates_batch_not_positionally_matched(self):
        for rows in ([prediction("OCC", 0.1)],
                     [prediction("CCO", 0.1), prediction("N", 0.1)],
                     [{"canonical_smiles": None, "status": "success", "score": 0.1}], [None]):
            with self.subTest(rows=rows):
                result = self.assess(["CCO"], FileRunner(response(rows)))
                self.assert_abstention(result["assessments"][0])
                self.assertIn("identity", result["assessments"][0]["error"])

    def test_invalid_scores_cannot_become_negative_predictions(self):
        for score in (None, True, False, "0.2", math.nan, math.inf, -math.inf, -0.01, 1.01, 10 ** 400):
            with self.subTest(score=score):
                result = self.assess(["CCO"], FileRunner(response([prediction("CCO", score)])))
                self.assert_abstention(result["assessments"][0])
                self.assertIn("score", result["assessments"][0]["error"])

    def test_failed_or_contradictory_backend_row_with_a_score_remains_abstention(self):
        for status in ("failed", "success"):
            with self.subTest(status=status):
                result = self.assess(["CCO"], FileRunner(response([
                    prediction("CCO", 0.1, status=status, error="missing_index"),
                ])))
                self.assert_abstention(result["assessments"][0])

    def test_missing_interpreter_does_not_run_or_change_invalid_materials(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            service, "ADMET_PYTHON", Path(directory) / "missing.exe",
        ), patch.object(service.subprocess, "run") as run:
            result = service.assess_admet_materials(["CCO", "bad-smiles"])
        run.assert_not_called()
        self.assert_abstention(result["assessments"][0])
        self.assertIn("interpreter", result["assessments"][0]["error"])
        self.assert_abstention(result["assessments"][1], "invalid")

    def test_timeout_and_startup_failure_are_visible_abstentions(self):
        errors = (subprocess.TimeoutExpired(["worker"], 2, stderr="PRIVATE_TOKEN=value"),
                  FileNotFoundError("PRIVATE_PATH"), PermissionError("PRIVATE_PATH"))
        for error in errors:
            with self.subTest(error=type(error).__name__):
                result = self.assess(["CCO"], FileRunner(error=error), timeout=2)
                row = result["assessments"][0]
                self.assert_abstention(row)
                self.assertNotIn("PRIVATE", row["error"])
                self.assertLess(len(row["error"]), 300)

    def test_process_failure_has_bounded_private_diagnostic(self):
        stderr = "ModuleNotFoundError SECRET_TOKEN=value C:\\private\\name " + "x" * 10000
        result = self.assess(["CCO"], FileRunner(returncode=17, stderr=stderr))
        row = result["assessments"][0]
        self.assert_abstention(row)
        self.assertIn("code 17", row["error"])
        self.assertIn("missing backend dependency", row["error"])
        self.assertNotIn("SECRET_TOKEN", row["error"])
        self.assertNotIn("private", row["error"])
        self.assertLess(len(row["error"]), 300)

    def test_bad_missing_and_non_object_json_results_are_abstentions(self):
        runners = [FileRunner(raw="not-json"), FileRunner(raw=b"\xff"),
                   FileRunner(write_output=False), FileRunner(payload=[]),
                   FileRunner(response(None)), FileRunner(raw="[" * 2000 + "]" * 2000)]
        for runner in runners:
            with self.subTest(raw=runner.raw, write_output=runner.write_output):
                result = self.assess(["CCO"], runner)
                self.assert_abstention(result["assessments"][0])

    def test_model_provenance_is_retained_but_cannot_be_faked_by_wrong_identity(self):
        result = self.assess(["CCO"], FileRunner(response([prediction("CCO", 0.2)])))
        self.assertEqual(result["model"], MODEL)
        mutations = ({"tool": "different"}, {"endpoint": "LD50"}, {"version": True},
                     {"weights_sha256": None}, {"weights_sha256": {"model.pt": "not-a-hash"}},
                     {"weights_sha256": {"C:/private/model.pt": "a" * 64}},
                     {"weights_sha256": {"../model.pt": "a" * 64}}, {"weights_sha256": {}})
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                model = dict(deepcopy(MODEL), **mutation)
                result = self.assess(["CCO"], FileRunner(response([prediction("CCO", 0.2)], model=model)))
                self.assert_abstention(result["assessments"][0])

    def test_unknown_version_is_allowed_with_actual_weight_provenance(self):
        model = dict(deepcopy(MODEL), version=None)
        result = self.assess(["CCO"], FileRunner(response([prediction("CCO", 0.2)], model=model)))
        self.assertEqual(result["assessments"][0]["status"], "success")
        self.assertIsNone(result["model"]["version"])
        self.assertEqual(result["model"]["weights_sha256"], MODEL["weights_sha256"])

    def test_backend_error_keeps_model_provenance_without_echoing_worker_messages(self):
        for code in ("backend_missing", "missing_endpoint", "SECRET_TOKEN=value"):
            with self.subTest(code=code):
                result = self.assess(["CCO"], FileRunner(response([], error=code)))
                self.assertEqual(result["model"], MODEL)
                self.assert_abstention(result["assessments"][0])
                self.assertNotIn("SECRET_TOKEN", result["assessments"][0]["error"])

    def test_command_is_an_argument_list_and_all_run_files_are_temporary(self):
        runner = FileRunner()
        untrusted = "CCO;$(echo PRIVATE)"
        result = self.assess(["CCO", untrusted], runner, timeout=13)
        command, options = runner.calls[0]
        self.assertIsInstance(command, list)
        self.assertFalse(options["shell"])
        self.assertEqual(options["timeout"], 13)
        self.assertIn("-B", command)
        self.assertNotIn(untrusted, " ".join(command))
        self.assertNotIn(untrusted, json.dumps(runner.requests))
        run_dir = Path(options["cwd"])
        self.assertEqual(Path(command[command.index("--input") + 1]).parent, run_dir)
        self.assertEqual(Path(command[command.index("--output") + 1]).parent, run_dir)
        self.assertFalse(run_dir.exists())
        for key in ("TMP", "TEMP", "TMPDIR", "MPLCONFIGDIR", "TORCH_HOME", "XDG_CACHE_HOME"):
            self.assertTrue(Path(options["env"][key]).is_relative_to(run_dir))
        self.assert_abstention(result["assessments"][1], "invalid")

    def test_public_contract_for_success_failed_and_invalid_rows(self):
        result = self.assess(["CCO", "CCN", "bad-smiles"], FileRunner(response([prediction("CCO", 0.2)])))
        self.assertEqual(set(result), {"model", "threshold", "assessments", "limitations"})
        keys = {"canonical_smiles", "input_smiles", "status", "score", "prediction", "applicability", "error"}
        for row in result["assessments"]:
            self.assertEqual(set(row), keys | {"input_types"} if row["status"] == "invalid" else keys)
            self.assertEqual(row["applicability"], "unknown")
        self.assertEqual([row["status"] for row in result["assessments"]], ["success", "failed", "invalid"])
        self.assertEqual(result["assessments"][2]["input_types"], ["str"])
        wording = " ".join(result["limitations"])
        self.assertIn("not clinically calibrated probabilities", wording)
        self.assertIn("not a safety determination or an ICH M7 classification", wording)


class AdmetWorkerTests(unittest.TestCase):
    def test_reordered_model_index_aligns_exactly_and_only_ames_is_reported(self):
        rows = worker._align_predictions(["CCO", "CCN"], FakeFrame(
            [("CCN", 0.8), ("CCO", 0.1)], columns=("LD50", "AMES"),
        ))
        self.assertEqual(rows, [prediction("CCO", 0.1), prediction("CCN", 0.8)])

    def test_missing_duplicate_and_empty_model_indices_never_use_positions_or_means(self):
        cases = (([("CCN", 0.2)], "missing_index"),
                 ([("CCO", 0.1), ("CCO", 0.9), ("CCN", 0.2)], "duplicate_index"))
        for items, error in cases:
            with self.subTest(error=error):
                rows = worker._align_predictions(["CCO", "CCN"], FakeFrame(items))
                self.assertEqual(rows[0], {"canonical_smiles": "CCO", "status": "failed", "score": None, "error": error})
                self.assertEqual(rows[1], prediction("CCN", 0.2))
        rows = worker._align_predictions(["CCO"], FakeFrame([]))
        self.assertEqual(rows[0]["error"], "missing_index")

    def test_unexpected_index_or_missing_or_duplicate_endpoint_is_rejected(self):
        cases = ((FakeFrame([("OCC", 0.1)]), "unexpected_index"),
                 (FakeFrame([(0, 0.1)]), "unexpected_index"),
                 (FakeFrame([("CCO", 0.1)], columns=("LD50",)), "missing_endpoint"),
                 (FakeFrame([("CCO", 0.1)], columns=("AMES", "AMES")), "duplicate_endpoint"))
        for frame, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(worker._WorkerError, error):
                    worker._align_predictions(["CCO"], frame)

    def test_invalid_model_scores_fail_only_relevant_material(self):
        for score in (None, True, "0.2", math.nan, math.inf, -math.inf, -0.01, 1.01, 10 ** 400):
            with self.subTest(score=score):
                rows = worker._align_predictions(["CCO", "CCN"], FakeFrame([("CCO", score), ("CCN", 0.2)]))
                self.assertEqual(rows[0]["status"], "failed")
                self.assertIsNone(rows[0]["score"])
                self.assertEqual(rows[0]["error"], "invalid_score")
                self.assertEqual(rows[1], prediction("CCN", 0.2))

    def test_score_endpoints_and_half_tie_remain_finite_scores(self):
        rows = worker._align_predictions(["C", "CC", "CCC"], FakeFrame([("C", 0), ("CC", 0.5), ("CCC", 1)]))
        self.assertEqual(rows, [prediction("C", 0), prediction("CC", 0.5), prediction("CCC", 1)])

    def test_worker_rejects_duplicate_batch_or_bad_request_before_loading_model(self):
        payloads = (None, [], {}, {"canonical_smiles": []}, {"canonical_smiles": "CCO"},
                    {"canonical_smiles": [None]}, {"canonical_smiles": [""]},
                    {"canonical_smiles": ["CCO", "CCO"]}, {"canonical_smiles": ["C"] * 257},
                    {"canonical_smiles": ["CCO"], "untrusted": "extra"})
        for payload in payloads:
            with self.subTest(payload=payload), patch.object(worker, "_model_provenance") as provenance:
                with redirect_stderr(io.StringIO()):
                    result = worker.run_predictions(payload)
                self.assertIn(result["error"], {"invalid_request", "duplicate_request"})
                provenance.assert_not_called()

    def test_provenance_hashes_actual_resource_bytes_and_uses_distribution_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "classification" / "model_0.pt"
            weights.parent.mkdir()
            weights.write_bytes(b"first synthetic weights")
            metadata = root / "metadata.json"
            metadata.write_bytes(b'{"fixture":true}')
            with patch.object(worker.importlib.metadata, "version", return_value="fixture-version") as version:
                model = worker._model_provenance(root)
                weights.write_bytes(b"changed synthetic weights")
                changed = worker._model_provenance(root)
            version.assert_called_with("admet-ai")
            self.assertEqual(model["version"], "fixture-version")
            self.assertEqual(model["weights_sha256"], {
                "classification/model_0.pt": hashlib.sha256(b"first synthetic weights").hexdigest(),
                "metadata.json": hashlib.sha256(b'{"fixture":true}').hexdigest(),
            })
            self.assertNotEqual(model["weights_sha256"]["classification/model_0.pt"],
                                changed["weights_sha256"]["classification/model_0.pt"])
            self.assertEqual(model["weights_sha256"]["metadata.json"], changed["weights_sha256"]["metadata.json"])

    def test_absent_model_resources_fail_and_missing_distribution_version_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(worker._WorkerError, "model_resources_missing"):
                worker._model_provenance(root)
            (root / "model.pt").write_bytes(b"synthetic weights")
            with patch.object(worker.importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
                model = worker._model_provenance(root)
            self.assertIsNone(model["version"])
            self.assertEqual(model["weights_sha256"]["model.pt"], hashlib.sha256(b"synthetic weights").hexdigest())

    def fake_backend(self, frame):
        package = ModuleType("admet_ai")
        package.ADMETModel = Mock()
        package.ADMETModel.return_value.predict.return_value = frame
        constants = ModuleType("admet_ai.constants")
        constants.DEFAULT_MODELS_DIR = Path("synthetic-installed-models")
        return package, constants

    def test_worker_calls_installed_model_with_fixed_settings_and_canonical_list(self):
        package, constants = self.fake_backend(FakeFrame([("CCN", 0.8), ("CCO", 0.1)]))
        with patch.dict(sys.modules, {"admet_ai": package, "admet_ai.constants": constants}), patch.object(
            worker, "_model_provenance", return_value=deepcopy(MODEL),
        ) as provenance:
            result = worker.run_predictions({"canonical_smiles": ["CCO", "CCN"]})
        package.ADMETModel.assert_called_once_with(include_physchem=False, drugbank_path=None, num_workers=0)
        package.ADMETModel.return_value.predict.assert_called_once_with(["CCO", "CCN"])
        provenance.assert_called_once_with(constants.DEFAULT_MODELS_DIR)
        self.assertEqual(result, response([prediction("CCO", 0.1), prediction("CCN", 0.8)]))

    def test_worker_dependency_and_prediction_failures_are_private_and_keep_abstentions(self):
        with patch.dict(sys.modules, {"admet_ai": None, "admet_ai.constants": None}), redirect_stderr(io.StringIO()):
            missing = worker.run_predictions({"canonical_smiles": ["CCO"]})
        self.assertEqual(missing["error"], "backend_missing")
        self.assertEqual(missing["assessments"][0]["status"], "failed")
        package, constants = self.fake_backend(FakeFrame([]))
        package.ADMETModel.return_value.predict.side_effect = RuntimeError("SECRET_TOKEN=value")
        stderr = io.StringIO()
        with patch.dict(sys.modules, {"admet_ai": package, "admet_ai.constants": constants}), patch.object(
            worker, "_model_provenance", return_value=deepcopy(MODEL),
        ), redirect_stderr(stderr):
            failed = worker.run_predictions({"canonical_smiles": ["CCO"]})
        self.assertEqual(failed["error"], "backend_error")
        self.assertEqual(failed["model"], MODEL)
        self.assertIsNone(failed["assessments"][0]["score"])
        self.assertNotIn("SECRET_TOKEN", json.dumps(failed) + stderr.getvalue())

    def test_worker_endpoint_error_reaches_json_with_provenance(self):
        package, constants = self.fake_backend(FakeFrame([("CCO", 0.1)], columns=("LD50",)))
        with patch.dict(sys.modules, {"admet_ai": package, "admet_ai.constants": constants}), patch.object(
            worker, "_model_provenance", return_value=deepcopy(MODEL),
        ), redirect_stderr(io.StringIO()):
            result = worker.run_predictions({"canonical_smiles": ["CCO"]})
        self.assertEqual(result["model"], MODEL)
        self.assertEqual(result["error"], "missing_endpoint")
        self.assertEqual(result["assessments"][0]["error"], "missing_endpoint")

    def test_worker_cli_reads_and_writes_only_requested_json_files(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "output.json"
            request = {"canonical_smiles": ["CCO"]}
            payload = response([prediction("CCO", 0.2)])
            input_path.write_text(json.dumps(request), encoding="utf-8")
            with patch.object(worker, "run_predictions", return_value=payload) as run:
                status = worker.main(["--input", str(input_path), "--output", str(output_path)])
            self.assertEqual(status, 0)
            run.assert_called_once_with(request)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), payload)

    def test_worker_cli_bad_json_becomes_explicit_request_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "output.json"
            input_path.write_text("PRIVATE invalid JSON", encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                worker.main(["--input", str(input_path), "--output", str(output_path)])
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["error"], "invalid_request")
            self.assertNotIn("PRIVATE", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
