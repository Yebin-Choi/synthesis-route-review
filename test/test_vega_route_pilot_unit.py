"""Software-only synthetic fixtures; no Java, prediction or validation evidence."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "evaluation" / "scripts")]
import run_vega_route_pilot as runner


# Structure identities mirror the saved route union; scores and rules below are
# deliberately synthetic software fixtures, not recovered chemical evidence.
ROUTE_IDENTITIES = (
    "CC(C)N1CC(O)C1", "Oc1cccc2ccccc12", "CC(C)N",
    "c1ccc2c(OCC3CO3)cccc2c1", "ClCC1CO1", "C=CCCl",
    "O=C(OO)c1cccc(Cl)c1", "CC(C)NCC(O)CCl", "c1ccc2ccccc2c1",
    "CC(C)NCC(O)CO", "OCC1CO1", "C=CCO", "CC(C)NCC1CO1",
    "CC(C)N(Cc1ccccc1)CC(O)COc1cccc2ccccc12", "CC(C)NCc1ccccc1",
)


def source_fixture():
    assessments = []
    materials = []
    for index, raw in enumerate(ROUTE_IDENTITIES):
        assessments.append({
            "input_smiles": raw, "canonical_smiles": runner.canonicalize_smiles(raw),
            "status": "failed" if index == 14 else "success",
            "score": None if index == 14 else (0.9 if index % 2 else 0.1),
            "fixture_only": True,
        })
        materials.append({
            "input_smiles": raw, "role": "reactant",
            "status": "structural_alert" if index % 3 else "no_alert_detected",
            "fixture_only": True,
        })
    return {
        "scope": "Synthetic software fixture only; no model has been executed.",
        "analysis_report": {"routes": [{
            "id": "fixture-route", "source": "synthetic",
            "steps": [{"id": "fixture-step", "materials": materials}],
        }]},
        "admet": {"threshold": 0.5, "assessments": assessments},
    }


def result_fixture():
    return runner._blank_result({
        "id": "fixture-row", "canonical_smiles": "CCO", "raw_inputs": ["CCO"],
        "planned_action": "execute_model", "recorded_admet": [],
        "source_roles": [{"kind": "sentinel", "name": "synthetic"}],
        "score_rule_evidence": {},
    })


def adi_fixture(value=1.0, assessment_class="GOOD reliability"):
    adi = Mock(spec=["GetIndexName", "GetIndexValue", "GetAssessment",
                     "GetAssessmentClassAsString"])
    adi.GetIndexName.return_value = "AD index"
    adi.GetIndexValue.return_value = value
    adi.GetAssessment.return_value = "Synthetic domain text only"
    adi.GetAssessmentClassAsString.return_value = assessment_class
    return adi


def output_fixture(value=0.0, status=1, *, returned_id="fixture-row",
                   returned_smiles="OCC", experimental=False):
    output = Mock(spec=[
        "getStatus", "getResults", "getMainResultValue", "getAssessment",
        "getAssessmentVerbose", "getErrMessage", "getMoleculeSMILES",
        "getMoleculeId", "HasExperimental", "getADIndexSize", "getADI",
        "getExperimentalFormatted", "getExperimental",
    ])
    output.getStatus.return_value = status
    label = {-1: "Possible NON-Mutagenic", 0: "NON-Mutagenic", 1: "Mutagenic"}.get(
        value, "Synthetic unknown class")
    output.getResults.return_value = [label, "0", "0"]
    output.getMainResultValue.return_value = value
    output.getAssessment.return_value = "NON-Mutagenic (EXPERIMENTAL value)"
    output.getAssessmentVerbose.return_value = "Synthetic default assessment only"
    output.getErrMessage.return_value = ""
    output.getMoleculeSMILES.return_value = returned_smiles
    output.getMoleculeId.return_value = returned_id
    output.HasExperimental.return_value = experimental
    output.getExperimentalFormatted.return_value = "NON-Mutagenic"
    output.getExperimental.return_value = 0.0
    output.getADIndexSize.return_value = 4
    output.getADI.return_value = adi_fixture()
    return output


def extract_fixture(output):
    row = result_fixture()
    row.update(parsed_valid=True, java_conversion_attempted=True, java_model_called=True)
    runner.extract_output(row, output)
    return row


def gateway_fixture():
    gateway = Mock()
    model = gateway.jvm.insilico.mutagenicity_sarpy.ismMutagenicitySarpy.return_value
    info = model.getInfo.return_value
    info.getName.return_value = "Synthetic model fixture only"
    info.getVersion.return_value = "1.0.8"
    info.getKey.return_value = "MUTA_SARPY"
    info.getClassValues.return_value = "Synthetic class metadata only"
    info.getQMRF.return_value = ""
    info.getSummary.return_value = "No real model execution"
    model.GetResultsName.return_value = ["Predicted Mutagen activity"]
    output_class = gateway.jvm.insilico.core.model.InsilicoModelOutput
    for name, value in runner.STATUS_CONSTANTS.items():
        setattr(output_class, name, value)

    def convert(smiles):
        molecule = Mock(spec=["SetId", "IsValid", "GetErrors"])
        molecule.fixture_smiles = smiles
        molecule.SetId.side_effect = lambda value: setattr(molecule, "fixture_id", value)
        molecule.IsValid.return_value = True
        return molecule

    def execute(molecule):
        return output_fixture(returned_id=molecule.fixture_id,
                              returned_smiles=molecule.fixture_smiles)

    gateway.jvm.insilico.core.molecule.conversion.SmilesMolecule.Convert.side_effect = convert
    model.Execute.side_effect = execute
    return gateway, model


class VegaRoutePilotManifestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source.json"
        self.jar = self.root / "synthetic-not-a-real.jar"
        self.manifest_path = self.root / "manifest.json"
        self.output_path = self.root / "results.json"
        self.payload = source_fixture()
        self.save_source()
        self.jar.write_bytes(b"Synthetic JAR fixture; never launched")
        provenance = {}
        for name in runner.PROVENANCE:
            path = self.root / f"{name}.fixture"
            path.write_text(f"Synthetic {name} provenance only", encoding="utf-8")
            provenance[name] = path
        patcher = patch.object(runner, "PROVENANCE", provenance)
        patcher.start()
        self.addCleanup(patcher.stop)

    def save_source(self):
        self.source.write_text(json.dumps(self.payload, allow_nan=False), encoding="utf-8")

    def prepare(self):
        return runner.prepare_manifest(self.manifest_path, self.output_path,
                                       source_path=self.source, jar_path=self.jar)

    def save_manifest(self, manifest):
        self.manifest_path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")

    def resign(self, manifest):
        manifest["input_sha256"] = runner.json_sha256(manifest["inputs"])
        manifest["manifest_id"] = f"vega-route-pilot-{manifest['input_sha256'][:16]}"
        manifest["manifest_sha256"] = runner.manifest_sha256(manifest)

    def assert_run_refused(self, exception, message):
        with patch.object(runner, "open_gateway") as launch:
            with self.assertRaisesRegex(exception, message):
                runner.run_pilot(self.manifest_path)
        launch.assert_not_called()
        self.assertFalse(self.output_path.exists())

    def test_prepare_keeps_all_fifteen_route_identities_sentinels_and_invalid_control(self):
        with patch.object(runner, "open_gateway") as launch, patch.dict(
            sys.modules, {"py4j": None, "py4j.java_gateway": None}
        ):
            manifest = self.prepare()
        launch.assert_not_called()
        self.assertFalse(self.output_path.exists())
        self.assertEqual(manifest["planned_status"], "planned_not_executed")
        route_rows = [row for row in manifest["inputs"] if row["recorded_admet"]]
        self.assertEqual(len(route_rows), 15)
        self.assertEqual({row["canonical_smiles"] for row in route_rows}, {
            record["canonical_smiles"] for record in self.payload["admet"]["assessments"]})
        self.assertEqual([row["recorded_admet"][0]["record"] for row in route_rows],
                         self.payload["admet"]["assessments"])
        self.assertTrue(any(not row["score_rule_evidence"]["available"] for row in route_rows))
        self.assertEqual({role["name"] for row in manifest["inputs"]
                          for role in row["source_roles"] if role["kind"] == "sentinel"},
                         {"ethanol", "epichlorohydrin", "NDMA"})
        epichlorohydrin = next(row for row in route_rows
                              if row["canonical_smiles"] == "ClCC1CO1")
        self.assertEqual({role["kind"] for role in epichlorohydrin["source_roles"]},
                         {"route_material", "sentinel"})
        invalid = [row for row in manifest["inputs"] if row["canonical_smiles"] is None]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]["raw_inputs"], ["CC("])
        self.assertEqual(invalid[0]["planned_action"], "skip_invalid_control")
        denominators = manifest["denominators"]
        self.assertEqual(denominators["planned_records"], 18)
        self.assertEqual(denominators["valid_canonical_identities"], 17)
        self.assertEqual(denominators["recorded_admet_canonical_identities"], 15)
        self.assertEqual(denominators["sentinel_roles"], 3)
        self.assertEqual(denominators["invalid_control_records"], 1)
        self.assertLessEqual(denominators["distinct_raw_identities"], 20)
        self.assertEqual(runner._load_json(self.manifest_path), manifest)
        runner.verify_manifest(manifest)

    def test_raw_aliases_are_retained_without_duplicate_calls_and_cap_is_enforced(self):
        for raw in ("NC(C)C", "[CH3:1]C(C)N"):
            self.payload["admet"]["assessments"].append({
                "input_smiles": raw, "canonical_smiles": "CC(C)N",
                "status": "success", "score": 0.9,
            })
        inputs = runner.collect_inputs(self.payload)
        denominators = runner._planned_denominators(inputs)
        self.assertEqual(denominators["distinct_raw_identities"], 20)
        self.assertEqual(denominators["recorded_admet_canonical_identities"], 15)
        isopropylamine = next(row for row in inputs if row["canonical_smiles"] == "CC(C)N")
        self.assertEqual(isopropylamine["raw_inputs"], ["CC(C)N", "NC(C)C", "[CH3:1]C(C)N"])
        self.assertEqual(len(isopropylamine["recorded_admet"]), 3)
        self.payload["admet"]["assessments"].append({
            "input_smiles": "[CH3:2]C(C)N", "canonical_smiles": "CC(C)N",
            "status": "success", "score": 0.9,
        })
        with self.assertRaisesRegex(ValueError, "exceeds 20"):
            runner.collect_inputs(self.payload)

    def test_missing_or_mismatched_recorded_identity_cannot_silently_reduce_coverage(self):
        missing = deepcopy(self.payload)
        missing["admet"]["assessments"].pop()
        with self.assertRaisesRegex(ValueError, "Expected all 15"):
            runner.collect_inputs(missing)
        mismatched = deepcopy(self.payload)
        mismatched["admet"]["assessments"][0]["canonical_smiles"] = "CCO"
        with self.assertRaisesRegex(ValueError, "mismatched recorded ADMET identity"):
            runner.collect_inputs(mismatched)

    def test_prepare_never_overwrites_existing_manifest_or_result(self):
        for existing in (self.manifest_path, self.output_path):
            with self.subTest(existing=existing.name):
                existing.write_bytes(b"Original evidence must survive")
                with patch.object(runner, "open_gateway") as launch:
                    with self.assertRaises(FileExistsError):
                        self.prepare()
                launch.assert_not_called()
                self.assertEqual(existing.read_bytes(), b"Original evidence must survive")
                other = self.output_path if existing == self.manifest_path else self.manifest_path
                self.assertFalse(other.exists())
                existing.unlink()

    def test_manifest_and_result_must_have_distinct_paths(self):
        with self.assertRaisesRegex(ValueError, "paths must differ"):
            runner.prepare_manifest(self.manifest_path, self.manifest_path,
                                    source_path=self.source, jar_path=self.jar)
        self.assertFalse(self.manifest_path.exists())

    def test_run_never_overwrites_existing_result_or_manifest(self):
        self.prepare()
        original_manifest = self.manifest_path.read_bytes()
        self.output_path.write_bytes(b"Original result evidence")
        with patch.object(runner, "open_gateway") as launch:
            for destination in (None, self.manifest_path, self.source, self.jar):
                with self.subTest(destination=destination), self.assertRaises(FileExistsError):
                    runner.run_pilot(self.manifest_path, destination)
        launch.assert_not_called()
        self.assertEqual(self.output_path.read_bytes(), b"Original result evidence")
        self.assertEqual(self.manifest_path.read_bytes(), original_manifest)
        self.assertEqual(runner._load_json(self.source), self.payload)
        self.assertEqual(self.jar.read_bytes(), b"Synthetic JAR fixture; never launched")

    def test_exclusive_admission_rejects_result_created_after_existence_check(self):
        self.prepare()
        original = b'{"external_result": "Must not be overwritten"}'
        open_path = Path.open

        def race_admission(path, mode="r", *args, **kwargs):
            if path == self.output_path and mode == "x":
                with open_path(path, "wb") as handle:
                    handle.write(original)
            return open_path(path, mode, *args, **kwargs)

        with patch.object(Path, "open", autospec=True, side_effect=race_admission), patch.object(
            runner, "open_gateway"
        ) as launch, patch.object(runner, "_persist") as persist:
            with self.assertRaises(FileExistsError):
                runner.run_pilot(self.manifest_path)
        launch.assert_not_called()
        persist.assert_not_called()
        self.assertEqual(self.output_path.read_bytes(), original)

    def test_exclusive_destination_handle_is_closed_before_every_atomic_replacement(self):
        self.prepare()
        gateway, model = gateway_fixture()
        open_path = Path.open
        replace = runner.os.replace
        admitted_handles = []

        def capture_admission(path, mode="r", *args, **kwargs):
            handle = open_path(path, mode, *args, **kwargs)
            if path == self.output_path:
                self.assertEqual(mode, "x")
                admitted_handles.append(handle)
            return handle

        def replace_closed_destination(source, destination):
            self.assertEqual(destination, self.output_path)
            self.assertEqual(len(admitted_handles), 1)
            self.assertTrue(admitted_handles[0].closed)
            return replace(source, destination)

        with patch.object(Path, "open", autospec=True, side_effect=capture_admission), patch.object(
            runner.os, "replace", side_effect=replace_closed_destination
        ) as publish, patch.object(runner, "open_gateway", return_value=gateway):
            report = runner.run_pilot(self.manifest_path)
        self.assertEqual(report["run_status"], "completed")
        self.assertEqual(model.Execute.call_count, 17)
        self.assertGreater(publish.call_count, 17)
        self.assertEqual(runner._load_json(self.output_path), report)

    def test_unsigned_manifest_tampering_is_refused_before_java_or_output_creation(self):
        manifest = self.prepare()
        manifest["inputs"][0]["canonical_smiles"] = "CCO"
        self.save_manifest(manifest)
        self.assert_run_refused(ValueError, "Manifest digest mismatch")

    def test_rehashed_input_tampering_still_cannot_change_frozen_source_coverage(self):
        original = self.prepare()
        mutations = (
            lambda value: value["inputs"].pop(0),
            lambda value: value["inputs"][0]["source_roles"].clear(),
            lambda value: value["inputs"][0].update(planned_action="skip_invalid_control"),
            lambda value: value["inputs"][-1]["raw_inputs"].append("CCO"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                manifest = deepcopy(original)
                mutate(manifest)
                self.resign(manifest)
                self.save_manifest(manifest)
                self.assert_run_refused(ValueError, "differ from frozen source union")

    def test_rehashed_policy_or_denominator_tampering_is_refused(self):
        original = self.prepare()
        for field in ("raw_prediction_mapping", "denominators"):
            with self.subTest(field=field):
                manifest = deepcopy(original)
                if field == "raw_prediction_mapping":
                    manifest[field]["main_result_classes"]["-1"] = "negative"
                else:
                    manifest[field]["recorded_admet_canonical_identities"] = 14
                self.resign(manifest)
                self.save_manifest(manifest)
                self.assert_run_refused(ValueError, "policy or identity|denominator mismatch")

    def test_source_jar_and_every_provenance_change_are_refused_before_java(self):
        self.prepare()
        for path in (self.source, self.jar, *runner.PROVENANCE.values()):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"\nSynthetic tampering")
                self.assert_run_refused(ValueError, "Source/JAR digest mismatch")
                path.write_bytes(original)

    def test_nonfinite_or_duplicate_json_is_refused(self):
        for text in ('{"value": NaN}', '{"value": Infinity}', '{"value": 1, "value": 2}'):
            with self.subTest(text=text):
                self.manifest_path.write_text(text, encoding="utf-8")
                self.assert_run_refused(ValueError, "Nonfinite JSON constant|Duplicate JSON key")

    def test_hashing_streams_more_than_one_chunk_without_python_311_file_digest(self):
        data = bytes(range(256)) * 8193
        path = self.root / "multi-chunk.bin"
        path.write_bytes(data)
        with patch.object(runner.hashlib, "file_digest", create=True,
                          side_effect=AssertionError("Python 3.11-only API must not be used")):
            self.assertEqual(runner.file_sha256(path), hashlib.sha256(data).hexdigest())
        with self.assertRaises(FileNotFoundError):
            runner.file_sha256(self.root / "missing.bin")
        with patch.object(Path, "open", side_effect=OSError("Synthetic read failure")):
            with self.assertRaisesRegex(OSError, "Synthetic read failure"):
                runner.file_sha256(path)

    def test_completed_mock_run_skips_invalid_control_and_keeps_inputs_unchanged(self):
        manifest = self.prepare()
        originals = {path: path.read_bytes() for path in (
            self.source, self.jar, self.manifest_path, *runner.PROVENANCE.values())}
        gateway, model = gateway_fixture()
        with patch.object(runner, "open_gateway", return_value=gateway) as launch:
            report = runner.run_pilot(self.manifest_path)
        launch.assert_called_once_with(str(self.jar))
        gateway.shutdown.assert_called_once()
        converter = gateway.jvm.insilico.core.molecule.conversion.SmilesMolecule.Convert
        self.assertEqual(converter.call_count, 17)
        self.assertEqual(model.Execute.call_count, 17)
        self.assertNotIn("CC(", [call.args[0] for call in converter.call_args_list])
        invalid = report["results"][-1]
        self.assertEqual(invalid["execution_status"], "skipped_invalid_control")
        self.assertFalse(invalid["java_conversion_attempted"])
        self.assertFalse(invalid["java_model_called"])
        self.assertEqual(invalid["conservative_assessment"], "abstain")
        self.assertEqual(report["run_status"], "completed")
        self.assertEqual(report["denominators"], manifest["denominators"])
        self.assertEqual(report["counts"]["returned_output_records"], 17)
        self.assertEqual(report["counts"]["conservative_assessment_counts"],
                         {"positive": 0, "negative": 17, "abstain": 1})
        self.assertEqual(runner._load_json(self.output_path), report)
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_row_failure_preserves_prior_raw_output_and_continues_remaining_inputs(self):
        self.prepare()
        gateway, model = gateway_fixture()
        execute = model.Execute.side_effect
        calls = 0

        def fail_second(molecule):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Synthetic single-row failure")
            return execute(molecule)

        model.Execute.side_effect = fail_second
        with patch.object(runner, "open_gateway", return_value=gateway):
            report = runner.run_pilot(self.manifest_path)
        self.assertEqual(report["run_status"], "completed")
        self.assertEqual(report["results"][0]["main_value_raw"], 0.0)
        failed = report["results"][1]
        self.assertEqual(failed["execution_status"], "row_failed")
        self.assertEqual(failed["conservative_assessment"], "abstain")
        self.assertEqual(failed["errors"][0]["message"], "Synthetic single-row failure")
        self.assertEqual(report["results"][2]["execution_status"], "output_received")
        self.assertEqual(report["counts"]["java_model_calls"], 17)
        self.assertEqual(report["counts"]["returned_output_records"], 16)
        self.assertEqual(report["counts"]["rows_with_errors"], 1)
        self.assertEqual(runner._load_json(self.output_path), report)

    def test_interrupt_preserves_completed_output_and_checkpoint_before_current_call(self):
        self.prepare()
        gateway, model = gateway_fixture()
        execute = model.Execute.side_effect
        calls = 0

        def interrupt_second(molecule):
            nonlocal calls
            calls += 1
            if calls == 2:
                checkpoint = runner._load_json(self.output_path)
                self.assertEqual(checkpoint["results"][0]["execution_status"], "output_received")
                self.assertEqual(checkpoint["results"][0]["results_raw"], ["NON-Mutagenic", "0", "0"])
                self.assertEqual(checkpoint["results"][1]["execution_status"], "running")
                self.assertEqual(checkpoint["results"][1]["conservative_assessment"], "abstain")
                raise KeyboardInterrupt("Synthetic bounded-process interruption")
            return execute(molecule)

        model.Execute.side_effect = interrupt_second
        with patch.object(runner, "open_gateway", return_value=gateway):
            report = runner.run_pilot(self.manifest_path)
        self.assertEqual(report["run_status"], "failed")
        self.assertEqual(report["errors"][0]["type"], "KeyboardInterrupt")
        self.assertEqual(report["results"][0]["conservative_assessment"], "negative")
        for row in report["results"][1:-1]:
            self.assertEqual(row["execution_status"], "interrupted_or_not_executed")
            self.assertEqual(row["conservative_assessment"], "abstain")
        self.assertEqual(report["results"][-1]["execution_status"], "skipped_invalid_control")
        self.assertEqual(report["counts"]["java_model_calls"], 2)
        self.assertEqual(report["counts"]["returned_output_records"], 1)
        self.assertEqual(runner._load_json(self.output_path), report)
        gateway.shutdown.assert_called_once()

    def test_initialization_failure_saves_all_abstentions_without_replacing_evidence(self):
        self.prepare()
        with patch.object(runner, "open_gateway", side_effect=OSError("Synthetic launch failure")):
            report = runner.run_pilot(self.manifest_path)
        self.assertEqual(report["run_status"], "failed")
        self.assertEqual(report["errors"][0]["message"], "Synthetic launch failure")
        self.assertEqual(report["counts"]["java_model_calls"], 0)
        self.assertEqual(report["counts"]["conservative_assessment_counts"]["abstain"], 18)
        self.assertEqual(runner._load_json(self.output_path), report)
        original = self.output_path.read_bytes()
        with patch.object(runner, "open_gateway") as launch:
            with self.assertRaises(FileExistsError):
                runner.run_pilot(self.manifest_path)
        launch.assert_not_called()
        self.assertEqual(self.output_path.read_bytes(), original)

    def test_runtime_model_or_status_identity_mismatch_prevents_every_execution(self):
        self.prepare()
        for mismatch in ("model", "status"):
            with self.subTest(mismatch=mismatch):
                gateway, model = gateway_fixture()
                if mismatch == "model":
                    model.getInfo.return_value.getVersion.return_value = "synthetic-unknown"
                else:
                    gateway.jvm.insilico.core.model.InsilicoModelOutput.OUTPUT_OK = 99
                destination = self.root / f"{mismatch}-failed.json"
                with patch.object(runner, "open_gateway", return_value=gateway):
                    report = runner.run_pilot(self.manifest_path, destination)
                self.assertEqual(report["run_status"], "failed")
                self.assertEqual(report["counts"]["java_model_calls"], 0)
                model.Execute.assert_not_called()
                self.assertEqual(runner._load_json(destination), report)
                gateway.shutdown.assert_called_once()

    def test_shutdown_failure_keeps_all_collected_raw_records(self):
        self.prepare()
        gateway, model = gateway_fixture()
        gateway.shutdown.side_effect = OSError("Synthetic shutdown failure")
        with patch.object(runner, "open_gateway", return_value=gateway):
            report = runner.run_pilot(self.manifest_path)
        self.assertEqual(model.Execute.call_count, 17)
        self.assertEqual(report["run_status"], "failed")
        self.assertEqual(report["errors"][-1]["stage"], "gateway_shutdown")
        self.assertEqual(report["counts"]["returned_output_records"], 17)
        self.assertEqual(runner._load_json(self.output_path), report)

    def test_cli_prepare_and_run_are_distinct_and_report_saved_phase_outcomes(self):
        manifest = self.prepare()
        with patch.object(runner, "prepare_manifest", return_value=manifest) as prepare, patch.object(
            runner, "run_pilot"
        ) as run, patch("sys.stdout", new_callable=io.StringIO) as stdout:
            code = runner.main(["--prepare", "--manifest", str(self.manifest_path),
                                "--output", str(self.output_path)])
        self.assertEqual(code, 0)
        prepare.assert_called_once_with(self.manifest_path, self.output_path)
        run.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue())["planned_status"], "planned_not_executed")
        for status, expected_code in (("completed", 0), ("failed", 1)):
            with self.subTest(status=status), patch.object(runner, "prepare_manifest") as prepare, patch.object(
                runner, "run_pilot", return_value={
                    "output_path": str(self.output_path), "run_status": status,
                    "denominators": {}, "counts": {},
                }
            ) as run, patch("sys.stdout", new_callable=io.StringIO) as stdout:
                code = runner.main(["--run", "--manifest", str(self.manifest_path)])
            self.assertEqual(code, expected_code)
            prepare.assert_not_called()
            run.assert_called_once_with(self.manifest_path, None)
            self.assertEqual(json.loads(stdout.getvalue())["run_status"], status)


class VegaRoutePilotCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.destination = self.root / "results.json"
        completed = extract_fixture(output_fixture(1.0, experimental=True))
        self.report = {"run_status": "running", "results": [completed]}
        self.destination.write_text(json.dumps(self.report, allow_nan=False), encoding="utf-8")
        self.original = self.destination.read_bytes()
        self.report["results"].append(result_fixture())

    def assert_previous_checkpoint_intact(self):
        self.assertEqual(self.destination.read_bytes(), self.original)
        saved = runner._load_json(self.destination)
        self.assertEqual(len(saved["results"]), 1)
        self.assertEqual(saved["results"][0]["results_raw"], ["Mutagenic", "0", "0"])
        self.assertEqual(saved["results"][0]["experimental_value_raw"], "NON-Mutagenic")

    def test_complete_sibling_checkpoint_is_synced_and_closed_before_replacement(self):
        create_temporary = runner.tempfile.NamedTemporaryFile
        replace = runner.os.replace
        created_handles = []

        def capture_temporary(*args, **kwargs):
            handle = create_temporary(*args, **kwargs)
            created_handles.append(handle)
            return handle

        def replace_complete_checkpoint(source, destination):
            self.assertEqual(Path(source).parent, self.destination.parent)
            self.assertEqual(destination, self.destination)
            self.assertTrue(created_handles[0].closed)
            sync.assert_called_once()
            self.assert_previous_checkpoint_intact()
            self.assertEqual(runner._load_json(source), self.report)
            return replace(source, destination)

        with patch.object(runner.tempfile, "NamedTemporaryFile", side_effect=capture_temporary), patch.object(
            runner.os, "fsync", wraps=runner.os.fsync
        ) as sync, patch.object(runner.os, "replace", side_effect=replace_complete_checkpoint) as publish:
            runner._persist(self.destination, self.report)
        publish.assert_called_once()
        self.assertEqual(runner._load_json(self.destination), self.report)
        self.assertEqual(set(self.root.iterdir()), {self.destination})

    def test_partial_write_failure_preserves_previous_json_and_removes_temporary_file(self):
        create_temporary = runner.tempfile.NamedTemporaryFile
        failure = OSError("Synthetic partial checkpoint write failure")

        def partial_write_temporary(*args, **kwargs):
            handle = create_temporary(*args, **kwargs)

            def fail_write(text):
                handle.file.write(text[:len(text) // 2])
                raise failure

            handle.write = Mock(side_effect=fail_write)
            return handle

        with patch.object(runner.tempfile, "NamedTemporaryFile", side_effect=partial_write_temporary), patch.object(
            runner.os, "replace"
        ) as publish:
            with self.assertRaises(OSError) as caught:
                runner._persist(self.destination, self.report)
        self.assertIs(caught.exception, failure)
        publish.assert_not_called()
        self.assert_previous_checkpoint_intact()
        self.assertEqual(set(self.root.iterdir()), {self.destination})

    def test_flush_or_sync_failure_preserves_previous_json_without_replacement(self):
        create_temporary = runner.tempfile.NamedTemporaryFile
        for stage in ("flush", "fsync"):
            with self.subTest(stage=stage):
                failure = OSError(f"Synthetic checkpoint {stage} failure")

                def failing_flush_temporary(*args, **kwargs):
                    handle = create_temporary(*args, **kwargs)
                    if stage == "flush":
                        handle.flush = Mock(side_effect=failure)
                    return handle

                sync_effect = failure if stage == "fsync" else None
                with patch.object(runner.tempfile, "NamedTemporaryFile", side_effect=failing_flush_temporary), patch.object(
                    runner.os, "fsync", side_effect=sync_effect
                ), patch.object(runner.os, "replace") as publish:
                    with self.assertRaises(OSError) as caught:
                        runner._persist(self.destination, self.report)
                self.assertIs(caught.exception, failure)
                publish.assert_not_called()
                self.assert_previous_checkpoint_intact()
                self.assertEqual(set(self.root.iterdir()), {self.destination})

    def test_interruption_or_failure_before_replace_preserves_previous_checkpoint(self):
        for failure in (KeyboardInterrupt("Synthetic pre-replace interruption"),
                        OSError("Synthetic checkpoint replacement failure")):
            with self.subTest(failure=type(failure).__name__), patch.object(
                runner.os, "replace", side_effect=failure
            ) as publish:
                with self.assertRaises(type(failure)) as caught:
                    runner._persist(self.destination, self.report)
            self.assertIs(caught.exception, failure)
            publish.assert_called_once()
            self.assert_previous_checkpoint_intact()
            self.assertEqual(set(self.root.iterdir()), {self.destination})

    def test_temporary_cleanup_failure_does_not_mask_original_failure(self):
        failure = OSError("Synthetic original replacement failure")
        with patch.object(runner.os, "replace", side_effect=failure), patch.object(
            Path, "unlink", side_effect=OSError("Synthetic temporary cleanup failure")
        ) as cleanup:
            with self.assertRaises(OSError) as caught:
                runner._persist(self.destination, self.report)
        self.assertIs(caught.exception, failure)
        cleanup.assert_called_once_with(missing_ok=True)
        self.assert_previous_checkpoint_intact()

    def test_serialization_failure_leaves_previous_checkpoint_without_creating_temporary(self):
        self.report["unserializable_fixture"] = float("nan")
        with patch.object(runner.tempfile, "NamedTemporaryFile") as create, patch.object(
            runner.os, "replace"
        ) as publish:
            with self.assertRaises(ValueError):
                runner._persist(self.destination, self.report)
        create.assert_not_called()
        publish.assert_not_called()
        self.assert_previous_checkpoint_intact()
        self.assertEqual(set(self.root.iterdir()), {self.destination})


class VegaRoutePilotAssessmentTests(unittest.TestCase):
    def test_possible_non_raw_minus_one_never_becomes_experimental_negative(self):
        row = extract_fixture(output_fixture(-1.0, experimental=True))
        self.assertEqual(row["main_value_raw"], -1.0)
        self.assertEqual(row["results_raw"], ["Possible NON-Mutagenic", "0", "0"])
        self.assertEqual(row["experimental_value_raw"], "NON-Mutagenic")
        self.assertEqual(row["experimental_numeric_raw"], 0.0)
        self.assertEqual(row["default_assessment"], "NON-Mutagenic (EXPERIMENTAL value)")
        self.assertEqual(row["raw_prediction"], "abstain")
        self.assertEqual(row["conservative_assessment"], "abstain")
        self.assertEqual(row["raw_prediction_reason"], "possible_non_mutagenic_is_abstention")
        self.assertFalse(row["domain_admitted"])

    def test_positive_raw_prediction_is_not_replaced_by_experimental_non(self):
        row = extract_fixture(output_fixture(1.0, experimental=True))
        self.assertEqual(row["raw_prediction"], "positive")
        self.assertEqual(row["conservative_assessment"], "positive")
        self.assertEqual(row["experimental_numeric_raw"], 0.0)
        self.assertEqual(row["raw_class_label"], "Mutagenic")
        self.assertEqual(row["adi"]["assessment_class"], "GOOD reliability")

    def test_equivalent_returned_smiles_pass_identity_but_mismatches_abstain(self):
        matched = extract_fixture(output_fixture(1.0, returned_smiles="OCC"))
        self.assertTrue(matched["identity_verified"])
        self.assertEqual(matched["returned_canonical_smiles"], "CCO")
        for changes in (
            {"returned_id": "wrong-row"}, {"returned_id": None},
            {"returned_smiles": "CCN"}, {"returned_smiles": None},
            {"returned_smiles": "CCO.[Na+]"}, {"returned_smiles": "C[13CH2]O"},
        ):
            with self.subTest(changes=changes):
                row = extract_fixture(output_fixture(1.0, **changes))
                self.assertFalse(row["identity_verified"])
                self.assertEqual(row["raw_prediction"], "abstain")
                self.assertEqual(row["conservative_assessment"], "abstain")
                self.assertEqual(row["main_value_raw"], 1.0)

    def test_stereoisomer_mismatch_does_not_merge_identity(self):
        row = result_fixture()
        row.update(canonical_smiles=runner.canonicalize_smiles("C[C@H](O)Cl"), parsed_valid=True)
        runner.extract_output(row, output_fixture(1.0, returned_smiles="C[C@@H](O)Cl"))
        self.assertFalse(row["identity_verified"])
        self.assertEqual(row["conservative_assessment"], "abstain")

    def test_failed_unknown_and_wrong_type_statuses_abstain_even_with_good_adi(self):
        for status in (0, -1, 99, None, True, "1", 1.0):
            with self.subTest(status=status):
                row = extract_fixture(output_fixture(1.0, status))
                self.assertFalse(row["successful_known_status"])
                self.assertEqual(row["raw_prediction"], "abstain")
                self.assertEqual(row["conservative_assessment"], "abstain")
                self.assertIn("failed_missing_or_unknown_output_status", row["assessment_reasons"])

    def test_missing_or_inapplicable_domain_status_retains_raw_but_abstains(self):
        for status in (2, 3):
            with self.subTest(status=status):
                row = extract_fixture(output_fixture(1.0, status))
                self.assertTrue(row["successful_known_status"])
                self.assertEqual(row["raw_prediction"], "positive")
                self.assertEqual(row["conservative_assessment"], "abstain")
                self.assertFalse(row["domain_admitted"])
                self.assertIn("status_reports_missing_or_inapplicable_domain", row["assessment_reasons"])

    def test_nonfinite_missing_and_unknown_raw_classes_never_use_experimental_fallback(self):
        for value in (float("nan"), float("inf"), float("-inf"), None, True, "0", 2.0):
            with self.subTest(value=value):
                row = extract_fixture(output_fixture(value, experimental=True))
                self.assertEqual(row["raw_prediction"], "abstain")
                self.assertEqual(row["conservative_assessment"], "abstain")
                self.assertEqual(row["main_value_raw"], runner._number_raw(value))
                self.assertEqual(row["experimental_numeric_raw"], 0.0)
                json.dumps(row, allow_nan=False)

    def test_missing_bad_or_out_of_domain_adi_retains_raw_but_abstains(self):
        domains = [None]
        domains.extend(adi_fixture(value=value) for value in (
            None, float("nan"), float("inf"), float("-inf"), True, "1.0"))
        domains.extend(adi_fixture(assessment_class=value) for value in (
            None, "", "MODERATE reliability", "LOW reliability", "GOOD reliability ",
            "Unknown applicability domain"))
        for index, domain in enumerate(domains):
            with self.subTest(domain=index):
                output = output_fixture(0.0)
                output.getADI.return_value = domain
                row = extract_fixture(output)
                self.assertEqual(row["raw_prediction"], "negative")
                self.assertEqual(row["conservative_assessment"], "abstain")
                self.assertFalse(row["domain_admitted"])
                json.dumps(row, allow_nan=False)

    def test_no_unobserved_numeric_domain_cutoff_is_invented(self):
        output = output_fixture(0.0)
        output.getADI.return_value = adi_fixture(value=0.37)
        row = extract_fixture(output)
        self.assertTrue(row["domain_admitted"])
        self.assertEqual(row["conservative_assessment"], "negative")

    def test_failed_adi_getter_preserves_other_raw_fields_and_abstains(self):
        output = output_fixture(1.0, experimental=True)
        output.getADI.side_effect = RuntimeError("Synthetic domain extraction failure")
        row = extract_fixture(output)
        self.assertEqual(row["execution_status"], "output_extraction_failed")
        self.assertEqual(row["results_raw"], ["Mutagenic", "0", "0"])
        self.assertEqual(row["experimental_value_raw"], "NON-Mutagenic")
        self.assertEqual(row["raw_prediction"], "positive")
        self.assertEqual(row["conservative_assessment"], "abstain")
        self.assertEqual(row["errors"][0]["stage"], "adi_object")

    def test_partial_output_extraction_failure_never_admits_conservative_assessment(self):
        output = output_fixture(1.0)
        output.getAssessmentVerbose.side_effect = RuntimeError("Synthetic verbose getter failure")
        row = extract_fixture(output)
        self.assertEqual(row["main_value_raw"], 1.0)
        self.assertEqual(row["raw_prediction"], "positive")
        self.assertEqual(row["conservative_assessment"], "abstain")
        self.assertIn("incomplete_or_failed_output_extraction", row["assessment_reasons"])

    def test_reported_model_error_abstains_without_losing_raw_results(self):
        output = output_fixture(1.0)
        output.getErrMessage.return_value = "Synthetic model error"
        row = extract_fixture(output)
        self.assertEqual(row["main_value_raw"], 1.0)
        self.assertEqual(row["results_raw"], ["Mutagenic", "0", "0"])
        self.assertEqual(row["raw_prediction"], "abstain")
        self.assertEqual(row["conservative_assessment"], "abstain")
        self.assertIn("reported_model_error", row["assessment_reasons"])

    def test_invalid_control_is_never_converted_or_executed_even_when_called_directly(self):
        invalid = runner._blank_result({
            "id": "fixture-invalid", "canonical_smiles": None,
            "raw_inputs": ["CC("], "planned_action": "skip_invalid_control",
            "recorded_admet": [], "source_roles": [], "score_rule_evidence": {},
        })
        gateway, model = gateway_fixture()
        runner._execute_row(invalid, gateway, model)
        gateway.jvm.insilico.core.molecule.conversion.SmilesMolecule.Convert.assert_not_called()
        model.Execute.assert_not_called()
        self.assertFalse(invalid["java_conversion_attempted"])
        self.assertFalse(invalid["java_model_called"])
        self.assertEqual(invalid["conservative_assessment"], "abstain")
        self.assertEqual(invalid["assessment_reasons"], ["invalid_control_never_sent_to_java"])

    def test_java_conversion_failure_is_preserved_and_never_executes_model(self):
        gateway, model = gateway_fixture()
        molecule = Mock()
        molecule.IsValid.return_value = False
        molecule.GetErrors.return_value.GetMessages.return_value = "Synthetic conversion failure"
        gateway.jvm.insilico.core.molecule.conversion.SmilesMolecule.Convert.side_effect = None
        gateway.jvm.insilico.core.molecule.conversion.SmilesMolecule.Convert.return_value = molecule
        row = result_fixture()
        runner._execute_row(row, gateway, model)
        self.assertEqual(row["execution_status"], "conversion_failed")
        self.assertEqual(row["conversion_error_raw"], "Synthetic conversion failure")
        self.assertEqual(row["raw_prediction"], "abstain")
        self.assertEqual(row["conservative_assessment"], "abstain")
        model.Execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
