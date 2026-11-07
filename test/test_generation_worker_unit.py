"""Original-source adaptation/provenance tests with offline upstream doubles only."""
from copy import deepcopy
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "synthex" / "src"), str(ROOT / "test")]
import env_support

# SynthEx needs pydantic, which lives in the SynthEx runtime rather than in
# whatever interpreter runs the suite. Skipping the module says so; letting
# the import fail makes the whole file uncollectable and the suite report an
# error that looks like a broken test.
if not env_support.importable("pydantic"):
    raise unittest.SkipTest(
        "pydantic이 없어 SynthEx를 불러올 수 없습니다. SynthEx는 "
        "generation.local.json에 설정된 자체 런타임에서 돌아갑니다.")

import engine
import generation_service as service
import generation_worker as worker
from synthex.artifacts import ArtifactStore, Failure, RunManifest, digest_json, now_utc
from synthex.completion import LeafSearchResult, complete_route
from synthex.config import ManuscriptConfig
from synthex.routejson import Route
import synthex.artifacts
import synthex.cli
import synthex.pipeline

SENTINEL = "offline-personal-key-sentinel-0123456789"


def molecule(smiles, children=None):
    return {"type": "mol", "smiles": smiles, "children": children or []}


def reaction_tree(target="CC", precursor="C"):
    return molecule(target, [{"type": "reaction", "metadata": {"original": "preserved"},
                              "children": [molecule(precursor), molecule(precursor)]}])


class OfflineCompleter:
    def __init__(self, failed=False):
        self.failed = failed
        self.calls = []

    def in_stock(self, smiles):
        return smiles == "C"

    def search(self, smiles):
        self.calls.append(smiles)
        return LeafSearchResult(smiles=smiles, status="failed" if self.failed else "solved",
                                route_tree=None if self.failed else reaction_tree(smiles),
                                backend="explicit-offline-test", backend_version="test-only",
                                error="offline completion failure" if self.failed else None)


class WorkerFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.models = self.root / "test" / "models"
        self.models.mkdir(parents=True)
        names = (*worker.MODEL_NAMES, "ringbreaker_model.onnx", "ringbreaker_templates.csv.gz")
        for name in names:
            (self.models / name).write_bytes(("offline fixture, never a loaded model: " + name).encode())
        self.planner = self.root / "planner.json"
        self.planner.write_text(json.dumps({
            "expansion": {"uspto": [str(self.models / worker.MODEL_NAMES[0]), str(self.models / worker.MODEL_NAMES[1])],
                          "ringbreaker": [str(self.models / "ringbreaker_model.onnx"), str(self.models / "ringbreaker_templates.csv.gz")]},
            "filter": {"uspto": str(self.models / worker.MODEL_NAMES[2])},
            "stock": {"zinc": str(self.models / worker.MODEL_NAMES[3])}}), encoding="utf-8")
        self.asset_manifest = self.root / "assets.json"
        self.asset_manifest.write_text(json.dumps({"format_version": "external-assets-1.0", "assets": [
            {"identifier": name, "path": str(self.models / name), "sha256": worker._file_digest(self.models / name)}
            for name in names]}), encoding="utf-8")
        self.config = ManuscriptConfig(completion={"aizynthfinder_config": self.planner}, external_asset_manifest=self.asset_manifest)
        self.config_path = self.root / "synthex.json"
        self.config_path.write_text(self.config.model_dump_json(), encoding="utf-8")
        self.request = {"schema_version": "1.0", "operation": "generate", "engine": "synthex",
                        "request_id": "req1", "context_id": "ctx1", "target": "CCCC",
                        "configuration_id": None,
                        "settings": {"fingerprint": "a" * 64, "config": str(self.config_path),
                                     "config_sha256": worker._digest(self.config_path.read_bytes())}}
        source = self.root / "synthex" / "src" / "synthex"
        source.mkdir(parents=True)
        for name in ("cli.py", "pipeline.py", "completion.py"):
            (source / name).write_bytes((ROOT / "synthex" / "src" / "synthex" / name).read_bytes())
        replacement = patch.object(worker, "ROOT", self.root)
        replacement.start()
        self.addCleanup(replacement.stop)
        probe = {"files": [self.planner], "config": self.config,
                 "asset_sha256": worker._asset_digests([self.planner])}
        self.request["configuration_id"] = worker._digest(service._json_bytes(worker._configuration(self.request, probe)))
        self.route = Route.from_op_plan({"target": "[CH3:1][CH2:2][CH2:3][CH3:4]", "steps": [
            {"id": "S1", "desc": "offline graph edit", "ops": [{"op": "break_bond", "map_a": 2, "map_b": 3}]}]})

    def write_run(self, *, stages=worker.STAGES, status="complete", failed_index=None,
                  review_failed_index=None, completion_failed=False, root=None):
        directory = self.root if root is None else Path(root)
        store = worker._utf8_lf_store(ArtifactStore, directory / "run")
        config = self.config.model_dump(mode="json")
        input_value = {"target": self.request["target"], "target_id": self.request["request_id"], "constraint": None}
        manifest = RunManifest(
            run_id=digest_json({"input": input_value, "config": config})[:16],
            target=input_value["target"], target_id=input_value["target_id"],
            input_sha256=digest_json(input_value), config=config, config_sha256=digest_json(config),
            code_revision="offline-test-not-live-evidence", dependency_versions={},
            provider="gemini", model=self.config.llm.model, prompt_versions={}, prompt_sha256={},
            external_assets={"aizynthfinder_config": worker._file_digest(self.planner),
                             "asset_manifest": worker._file_digest(self.asset_manifest),
                             **{f'asset:{asset["identifier"]}': asset["sha256"]
                                for asset in json.loads(self.asset_manifest.read_text())["assets"]}},
            created_at=now_utc(), updated_at=now_utc(), status=status,
            current_stage="complete" if status == "complete" else (stages[-1] if stages else "created"))
        if status == "failed":
            manifest.failure = Failure(stage=manifest.current_stage, category="OfflineTestError", message=SENTINEL)
        strategies = ["offline strategy one", "offline strategy two", "offline strategy three"]
        route_json = self.route.model_dump(mode="json")
        built = [{"status": "ok", "strategy": strategy, "route": deepcopy(route_json), "termination": "model_finished"}
                 for strategy in strategies]
        reviewed = [{"status": "ok", "strategy": strategy, "route": deepcopy(route_json),
                     "final_critique": {"is_satisfactory": True, "problematic_steps": []}, "iterations": []}
                    for strategy in strategies]
        analyses = [{"status": "ok", "feasibility_category": "offline", "summary": "test-only", "risks": [],
                     "key_steps": [], "details": {}} for _ in strategies]
        completer = OfflineCompleter(failed=completion_failed)
        stitched_route = complete_route(self.route, completer)
        self.assertEqual(completer.calls, ["CC"])
        stitched = [stitched_route.model_dump(mode="json") | {"status": "ok"} for _ in strategies]
        if failed_index is not None:
            failure = {"status": "failed", "strategy": strategies[failed_index],
                       "failure": {"category": "OfflineTestError", "message": "test failure " + SENTINEL}}
            for collection in (built, reviewed, analyses, stitched):
                collection[failed_index] = deepcopy(failure)
        if review_failed_index is not None:
            reviewed[review_failed_index].update(status="review_failed", final_critique=None,
                failure={"category": "OfflineTestError", "message": "test review failure " + SENTINEL})
        payloads = {"strategies": {"strategies": strategies, "origin": "generated"}, "routes": built,
                    "reviewed_routes": reviewed, "analyses": analyses, "stitched_routes": stitched}
        parent = None
        for stage in stages:
            reference = store.write_checkpoint(stage, payloads[stage], parent)
            manifest.artifacts[stage] = reference
            manifest.completed_stages.append(stage)
            manifest.stage_configs[stage] = deepcopy(config)
            manifest.stage_config_sha256[stage] = digest_json(config)
            parent = reference.sha256
        store.save_manifest(manifest)
        return store, manifest

    def run_cli_double(self, **options):
        self.arguments = []
        def offline_cli(arguments):
            self.arguments.append(list(arguments))
            runtime = Path(arguments[arguments.index("--config") + 1])
            resolved = json.loads(runtime.read_text(encoding="utf-8"))
            self.assertEqual(resolved, self.config.model_dump(mode="json"))
            self.write_run(**options)
            return 0
        with patch.object(worker, "_probe", return_value={"files": [self.planner], "config": self.config}), \
                patch.object(synthex.cli, "main", side_effect=offline_cli), \
                patch.dict(worker.os.environ, {"GEMINI_API_KEY": SENTINEL, "GOOGLE_GENAI_USE_VERTEXAI": "false"}, clear=True):
            return worker.run_request(deepcopy(self.request), self.root)


class OriginalEngineTests(unittest.TestCase):
    def test_original_aizynth_call_retains_every_tree_metadata_and_exact_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            here = Path(directory) / "app"
            here.mkdir()
            models = here.parent / "test" / "models"
            models.mkdir(parents=True)
            for name in worker.MODEL_NAMES:
                (models / name).write_bytes(b"test-only")
            captured = {}
            class NativeTree:
                is_solved = True
                def __init__(self, ordinal):
                    self.ordinal = ordinal
                def to_dict(self, *, include_metadata=False):
                    self_assert.assertTrue(include_metadata)
                    value = reaction_tree()
                    value["children"][0]["metadata"]["ordinal"] = self.ordinal
                    return value
            self_assert = self
            class Finder:
                def __init__(self, configdict):
                    captured["config"] = configdict
                    self.stock = SimpleNamespace(select_first=lambda: captured.update(stock=True))
                    self.expansion_policy = SimpleNamespace(select_first=lambda: captured.update(expansion=True))
                    self.filter_policy = SimpleNamespace(select=lambda value: captured.update(filter=value))
                    self.config = SimpleNamespace(search=SimpleNamespace())
                    self.routes = SimpleNamespace(reaction_trees=[NativeTree(index) for index in (1, 2, 3)])
                    captured["finder"] = self
                def tree_search(self):
                    captured["search"] = True
                def build_routes(self):
                    self_assert.assertTrue(captured["search"])
                    captured["build"] = True
            package = ModuleType("aizynthfinder")
            package.__path__ = []
            module = ModuleType("aizynthfinder.aizynthfinder")
            module.AiZynthFinder = Finder
            with patch.dict(sys.modules, {"aizynthfinder": package, "aizynthfinder.aizynthfinder": module}), patch.object(engine, "HERE", here):
                value = engine.generate_routes("CC", time_limit=30, iteration_limit=30)
            self.assertTrue(captured["build"])
            self.assertEqual(captured["finder"].config.search.time_limit, 30)
            self.assertEqual(captured["finder"].config.search.iteration_limit, 30)
            self.assertEqual(captured["filter"], "uspto")
            self.assertEqual(len(value["routes"]), 3)
            self.assertEqual(len(value["native_trees"]), 3)
            self.assertEqual([route["upstream"]["ordinal"] for route in value["routes"]], [1, 2, 3])
            self.assertEqual(value["routes"][2]["upstream"]["tree"]["children"][0]["metadata"]["ordinal"], 3)

    def test_native_tree_conversion_does_not_drop_stitched_chemistry_or_change_analysis_schema(self):
        native = reaction_tree("CCC", "CC")
        native["children"][0]["children"][0] = reaction_tree("CC")
        payload = engine.from_native_tree(native, "tree-1", "CCC", "offline source")
        self.assertEqual(len(payload["routes"][0]["steps"]), 2)
        self.assertEqual(payload["routes"][0]["upstream"]["tree"], native)
        report = engine.analyze_routes(payload)
        self.assertEqual(set(report), {"schema_version", "disclaimer", "routes", "comparison_note"})
        self.assertEqual(report["routes"][0]["id"], "tree-1")

    def test_tree_target_shape_and_step_bounds_are_rejected_not_silently_omitted(self):
        bad = reaction_tree()
        bad["children"][0]["children"] = []
        for native, target in ((reaction_tree(), "CO"), (bad, "CC"), (molecule("CC", [molecule("C")]), "CC")):
            with self.subTest(target=target), self.assertRaises(ValueError):
                engine.from_native_tree(native, "bad", target, "offline")
        native = molecule("CC")
        current = native
        for _ in range(101):
            child = molecule("CC")
            current["children"] = [{"type": "reaction", "children": [child]}]
            current = child
        with self.assertRaises(ValueError):
            engine.from_native_tree(native, "too-many", "CC", "offline")


class PlatformArtifactIoTests(unittest.TestCase):
    def test_original_store_hashes_exact_utf8_lf_bytes_and_still_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            store = worker._utf8_lf_store(ArtifactStore, directory)
            payload = {"message": "학술제 단계", "items": [1, 2]}
            self.assertIs(type(store), ArtifactStore)
            for method in ("write_checkpoint", "read_checkpoint", "save_manifest", "load_manifest", "validate"):
                self.assertIs(getattr(store, method).__func__, getattr(ArtifactStore, method))
            reference = store.write_checkpoint("unicode", payload)
            path = store.root / reference.path
            expected = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            self.assertEqual(path.read_bytes(), expected)
            self.assertNotIn(b"\r\n", path.read_bytes())
            self.assertEqual(reference.sha256, hashlib.sha256(expected).hexdigest())
            self.assertEqual(store.read_checkpoint(reference), payload)
            self.assertEqual(store.write_checkpoint("unicode", payload), reference)
            # A byte-level CRLF change remains corruption; the adapter must not
            # normalize bytes or substitute an expected digest during reads.
            path.write_bytes(expected.replace(b"\n", b"\r\n"))
            with self.assertRaisesRegex(ValueError, "checkpoint digest mismatch"):
                store.read_checkpoint(reference)
            with self.assertRaisesRegex(ValueError, "immutable checkpoint"):
                store.write_checkpoint("unicode", payload)

    def test_store_factory_scopes_both_original_import_seams_and_restores_on_error(self):
        original_store = synthex.artifacts.ArtifactStore
        original_pipeline_store = synthex.pipeline.ArtifactStore
        original_path_writer = Path.write_text
        source = Path(synthex.artifacts.__file__)
        original_source = source.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "offline boundary failure"):
                with worker._synthex_artifact_io() as factory:
                    self.assertIs(synthex.artifacts.ArtifactStore, factory)
                    self.assertIs(synthex.pipeline.ArtifactStore, factory)
                    self.assertIs(Path.write_text, original_path_writer)
                    for index, constructor in enumerate((synthex.artifacts.ArtifactStore, synthex.pipeline.ArtifactStore)):
                        store = constructor(Path(directory) / str(index))
                        reference = store.write_checkpoint("unicode", {"단계": index})
                        self.assertIs(type(store), original_store)
                        self.assertEqual(store.read_checkpoint(reference), {"단계": index})
                    raise RuntimeError("offline boundary failure")
        self.assertIs(synthex.artifacts.ArtifactStore, original_store)
        self.assertIs(synthex.pipeline.ArtifactStore, original_pipeline_store)
        self.assertIs(Path.write_text, original_path_writer)
        self.assertEqual(source.read_bytes(), original_source)


class FullWorkflowContractTests(WorkerFixture):
    def test_real_original_cli_pipeline_preserves_all_checkpoint_bytes_with_offline_llm(self):
        from synthex.agents import Analyst, Critic, Editor, RouteBuilder, StrategyGenerator
        from synthex.llm import ReplayLLMClient
        from synthex.pipeline import PipelineComponents, SynthexPipeline

        strategies = ["분할 전략 하나", "분할 전략 둘", "분할 전략 셋"]
        responses = [{"text": json.dumps({"strategies": strategies}, ensure_ascii=False)}]
        for _ in strategies:
            responses.extend([
                {"text": json.dumps({"done": False, "step": {
                    "id": "S1", "description": "오프라인 분할", "kind": "key",
                    "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}]}})},
                {"text": '{"done": true}'},
            ])
        responses.extend({"text": json.dumps({"is_satisfactory": True,
                          "overall_feedback": "오프라인 검토", "problematic_steps": []})} for _ in strategies)
        responses.extend({"text": json.dumps({"feasibility_category": "offline-only", "summary": "시험 근거",
                          "key_steps": [], "risks": [], "details": {}})} for _ in strategies)
        client = ReplayLLMClient(responses)
        completer = OfflineCompleter()
        pipeline = SynthexPipeline(self.config, PipelineComponents(
            llm=client, strategy_generator=StrategyGenerator(), route_builder=RouteBuilder(),
            critic=Critic(), editor=Editor(), analyst=Analyst(), leaf_completer=completer))
        original_store = synthex.artifacts.ArtifactStore
        with patch.object(worker, "_probe", return_value={"files": [self.planner], "config": self.config}), \
                patch.object(synthex.cli, "_load_live_pipeline", return_value=pipeline), \
                patch.dict(worker.os.environ, {"GEMINI_API_KEY": SENTINEL, "GOOGLE_GENAI_USE_VERTEXAI": "false"}, clear=True):
            response = worker.run_request(deepcopy(self.request), self.root)
        self.assertEqual(response["status"], "succeeded", response)
        self.assertIs(synthex.artifacts.ArtifactStore, original_store)
        self.assertIs(synthex.pipeline.ArtifactStore, original_store)
        self.assertEqual(len(client.requests), 13)
        self.assertEqual(completer.calls, ["CC", "CC", "CC"])
        self.assertEqual([len(route["steps"]) for route in response["result"]["payload"]["routes"]], [3, 3, 3])
        store = worker._utf8_lf_store(ArtifactStore, self.root / "run")
        manifest = store.validate()
        self.assertEqual(manifest.completed_stages, list(worker.STAGES))
        self.assertEqual(manifest.config["agents"]["edit_iteration_limit"], 6)
        for reference in manifest.artifacts.values():
            data = (store.root / reference.path).read_bytes()
            self.assertNotIn(b"\r\n", data)
            self.assertEqual(reference.sha256, hashlib.sha256(data).hexdigest())
        self.assertEqual(response["result"]["provenance"]["platform_io_adapter"], worker.SYNTHEX_ARTIFACT_IO)

    def test_original_full_cli_no_show_or_reduced_overrides_and_all_final_routes(self):
        response = self.run_cli_double()
        self.assertEqual(response["status"], "succeeded", response)
        self.assertEqual(len(self.arguments), 1)
        arguments = self.arguments[0]
        self.assertEqual(arguments[:2], ["plan", "CCCC"])
        for prohibited in ("show", "stage", "--env-file", "--resume", "--edit-rounds", "--strategy-count", "--route-steps"):
            self.assertNotIn(prohibited, arguments)
        self.assertNotIn(SENTINEL, json.dumps(arguments))
        payload = response["result"]["payload"]
        self.assertEqual([route["id"] for route in payload["routes"]], ["synthex-req1-0001", "synthex-req1-0002", "synthex-req1-0003"])
        # Original complete_route adds two leaf reactions to the single strategic
        # reaction. CLI `show` would expose only the one strategic reaction.
        self.assertEqual([len(route["steps"]) for route in payload["routes"]], [3, 3, 3])
        for route in payload["routes"]:
            self.assertIn("completions", route["upstream"]["stitched"])
            self.assertEqual(route["upstream"]["checkpoint_path"], "stitched_routes.json")
        manifest = response["result"]["provenance"]["manifest"]
        self.assertEqual(manifest["completed_stages"], list(worker.STAGES))
        self.assertEqual(manifest["config"]["agents"]["edit_iteration_limit"], 6)
        self.assertTrue(manifest["config"]["completion"]["enabled"])
        self.assertFalse(manifest["config"]["llm"]["use_vertex_ai"])
        self.assertNotIn(SENTINEL, json.dumps(response))

    def test_per_item_failure_keeps_other_routes_original_ordinals_and_safe_failure_evidence(self):
        response = self.run_cli_double(failed_index=1)
        self.assertEqual(response["status"], "partial", response)
        result = response["result"]
        self.assertEqual([route["id"] for route in result["payload"]["routes"]], ["synthex-req1-0001", "synthex-req1-0003"])
        self.assertEqual({row["stage"] for row in result["item_failures"]}, {"routes", "reviewed_routes", "analyses", "stitched_routes"})
        self.assertNotIn(SENTINEL, json.dumps(response))
        artifact = next(row for row in result["artifacts"] if row["path"] == "run/routes.json")
        self.assertTrue(artifact["redacted"])
        self.assertNotEqual(artifact["original_sha256"], artifact["export_sha256"])
        self.assertIn(SENTINEL, (self.root / "run" / "routes.json").read_text())

    def test_review_failure_and_leaf_completion_failure_are_not_complete_success(self):
        response = self.run_cli_double(review_failed_index=0, completion_failed=True)
        self.assertEqual(response["status"], "partial", response)
        self.assertEqual(len(response["result"]["payload"]["routes"]), 3)
        statuses = {row["status"] for row in response["result"]["item_failures"]}
        self.assertTrue({"review_failed", "failed", "unsolved"} <= statuses)
        self.assertFalse(response["readiness"]["live_verified"])

    def test_exit_zero_with_pre_stitch_checkpoints_never_becomes_full_success_or_fallback(self):
        response = self.run_cli_double(stages=worker.STAGES[:-1], status="partial")
        self.assertEqual(response["status"], "partial", response)
        self.assertIsNone(response["result"]["payload"])
        self.assertEqual(response["result"]["provenance"]["cli_exit_code"], 0)
        self.assertTrue(any(row["status"] == "no_completed_routes" for row in response["result"]["item_failures"]))
        self.assertIn("run/reviewed_routes.json", {row["path"] for row in response["result"]["artifacts"]})

    def test_failed_manifest_preserves_failure_without_fabricated_routes(self):
        response = self.run_cli_double(stages=(), status="failed")
        self.assertEqual(response["status"], "partial", response)
        self.assertIsNone(response["result"]["payload"])
        self.assertNotIn(SENTINEL, json.dumps(response))
        self.assertIn("[REDACTED]", json.dumps(response))

    def test_complete_claim_with_missing_stitch_is_rejected(self):
        response = self.run_cli_double(stages=worker.STAGES[:-1], status="complete")
        self.assertEqual(response["status"], "failed", response)
        self.assertEqual(response["error"]["code"], "invalid_output")
        self.assertIsNone(response["result"])


class ManifestValidationTests(WorkerFixture):
    def reject(self, mutation, expected):
        store, manifest = self.write_run()
        mutation(store, manifest)
        with self.assertRaises(service._Failure) as caught:
            worker._synthex_routes(self.request, self.root, self.config)
        self.assertEqual(caught.exception.code, expected)

    def test_checkpoint_content_digest_tampering(self):
        def mutate(store, manifest):
            path = store.root / "stitched_routes.json"
            path.write_text(path.read_text() + "\n")
        self.reject(mutate, "digest_mismatch")

    def test_checkpoint_traversal_is_checked_before_upstream_store_reads(self):
        def mutate(store, manifest):
            manifest.artifacts["stitched_routes"].path = "../outside.json"
            store.save_manifest(manifest)
        self.reject(mutate, "path_escape")

    def test_target_input_config_stage_and_parent_digest_bindings(self):
        mutations = [
            (lambda store, manifest: setattr(manifest, "target", "CC"), "invalid_output"),
            (lambda store, manifest: setattr(manifest, "input_sha256", "0" * 64), "invalid_output"),
            (lambda store, manifest: setattr(manifest, "config_sha256", "0" * 64), "digest_mismatch"),
            (lambda store, manifest: manifest.stage_config_sha256.update(routes="0" * 64), "digest_mismatch"),
            (lambda store, manifest: setattr(manifest.artifacts["routes"], "parent_sha256", "0" * 64), "digest_mismatch"),
        ]
        for mutate, expected in mutations:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory(dir=self.root) as directory:
                root = Path(directory)
                store, manifest = self.write_run(root=root)
                mutate(store, manifest)
                store.save_manifest(manifest)
                with self.assertRaises(service._Failure) as caught:
                    worker._synthex_routes(self.request, root, self.config)
                self.assertEqual(caught.exception.code, expected)

    def test_malformed_stitched_item_or_alignment_is_rejected_even_with_valid_digests(self):
        # Rewrite a complete test run with a new, valid digest chain so schema
        # and target validation, not only file hashing, must catch the mutation.
        store, manifest = self.write_run()
        item = store.read_checkpoint(manifest.artifacts["stitched_routes"])
        item[0]["stitched_tree"]["smiles"] = "CC"
        path = store.root / "stitched_routes.json"
        path.write_text(json.dumps(item), encoding="utf-8")
        manifest.artifacts["stitched_routes"].sha256 = worker._file_digest(path)
        store.save_manifest(manifest)
        with self.assertRaises(ValueError):
            worker._synthex_routes(self.request, self.root, self.config)


class ConfigurationAndProtocolTests(WorkerFixture):
    def test_original_full_config_requires_personal_mode_and_original_six_round_settings(self):
        for change in ({"llm": {"use_vertex_ai": True}}, {"agents": {"edit_iteration_limit": 0}},
                       {"completion": {"enabled": False}}, {"completion": {"budget": {"time_limit": 1}}},
                       {"llm": {"max_attempts": 1}}, {"completion": {"expansion_policies": ["uspto"]}}):
            with self.subTest(change=change):
                value = self.config.model_dump(mode="json")
                for section, fields in change.items():
                    value[section].update(fields)
                self.config_path.write_text(json.dumps(value), encoding="utf-8")
                settings = dict(self.request["settings"], config_sha256=worker._digest(self.config_path.read_bytes()))
                with self.assertRaises(service._Failure) as caught:
                    worker._full_config(settings)
                self.assertEqual(caught.exception.code, "incompatible_backend")

    def test_missing_ringbreaker_or_manifest_never_passes_full_readiness(self):
        value = json.loads(self.planner.read_text(encoding="utf-8"))
        del value["expansion"]["ringbreaker"]
        self.planner.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaises(service._Failure) as caught:
            worker._full_config(self.request["settings"])
        self.assertEqual(caught.exception.code, "assets_missing")
        value = self.config.model_dump(mode="json")
        value["external_asset_manifest"] = None
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        settings = dict(self.request["settings"], config_sha256=worker._digest(self.config_path.read_bytes()))
        with self.assertRaises(service._Failure) as caught:
            worker._full_config(settings)
        self.assertEqual(caught.exception.code, "assets_missing")

    def test_ordinary_uspto_assets_cannot_be_relabelled_as_ringbreaker(self):
        value = json.loads(self.planner.read_text(encoding="utf-8"))
        value["expansion"]["ringbreaker"] = list(value["expansion"]["uspto"])
        self.planner.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaises(service._Failure) as caught:
            worker._full_config(self.request["settings"])
        self.assertEqual(caught.exception.code, "assets_missing")

    def test_companion_asset_manifest_digest_mismatch_stops_readiness(self):
        manifest = json.loads(self.asset_manifest.read_text(encoding="utf-8"))
        manifest["assets"][0]["sha256"] = "0" * 64
        self.asset_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(service._Failure) as caught:
            worker._full_config(self.request["settings"])
        self.assertEqual(caught.exception.code, "digest_mismatch")

    def test_config_uses_only_explicit_existing_local_assets_and_no_environment_substitution(self):
        config, files = worker._full_config(self.request["settings"])
        self.assertEqual(config.agents.edit_iteration_limit, 6)
        self.assertIn(self.models / worker.MODEL_NAMES[0], files)
        value = self.config.model_dump(mode="json")
        value["completion"]["aizynthfinder_config"] = "${COMPANY_CONFIG}"
        self.config_path.write_text(json.dumps(value), encoding="utf-8")
        settings = dict(self.request["settings"], config_sha256=worker._digest(self.config_path.read_bytes()))
        with self.assertRaises(service._Failure) as caught:
            worker._full_config(settings)
        self.assertEqual(caught.exception.code, "config_invalid")

    def test_unknown_dynamic_or_remote_planner_assets_are_not_loaded(self):
        for source in ("https://example.invalid/model.onnx", "custom.Provider"):
            value = {"expansion": {"x": [source, str(self.models / worker.MODEL_NAMES[1])]},
                     "filter": {"x": str(self.models / worker.MODEL_NAMES[2])},
                     "stock": {"x": str(self.models / worker.MODEL_NAMES[3])}}
            self.planner.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(service._Failure) as caught:
                worker._planner_assets(self.planner)
            self.assertEqual(caught.exception.code, "config_invalid")

    def test_probe_missing_package_and_assets_are_distinct_and_never_run(self):
        request = deepcopy(self.request)
        request.update(operation="probe", target=None, configuration_id=None)
        for failure, code in ((ModuleNotFoundError(SENTINEL), "package_missing"), (service._Failure("assets_missing"), "assets_missing")):
            with patch.dict(worker.os.environ, {}, clear=True), patch.object(worker, "_probe", side_effect=failure), \
                    patch.object(worker, "_synthex", side_effect=AssertionError("must not run")):
                value = worker.run_request(request, self.root)
            self.assertEqual(value["error"]["code"], code)
            self.assertEqual(value["readiness"]["state"], code)
            self.assertNotIn(SENTINEL, json.dumps(value))
        with patch.dict(worker.os.environ, {}, clear=True), patch.object(worker, "_probe", return_value={"files": [], "config": None}), \
                patch.object(worker, "_synthex", side_effect=AssertionError("must not run")):
            value = worker.run_request(request, self.root)
        self.assertEqual(value["status"], "succeeded")
        self.assertEqual(value["readiness"]["state"], "live_unverified")
        self.assertTrue(value["readiness"]["ready"])
        self.assertFalse(value["readiness"]["live_verified"])

    def test_serialized_secret_extra_fields_and_keys_in_wrong_children_are_rejected(self):
        request = deepcopy(self.request)
        request["gemini_api_key"] = SENTINEL
        with patch.dict(worker.os.environ, {}, clear=True):
            value = worker.run_request(request, self.root)
        self.assertEqual(value["error"]["code"], "invalid_request")
        self.assertNotIn(SENTINEL, json.dumps(value))
        request = deepcopy(self.request)
        request.update(operation="probe", target=None, configuration_id=None)
        with patch.dict(worker.os.environ, {"GEMINI_API_KEY": SENTINEL}, clear=True):
            value = worker.run_request(request, self.root)
        self.assertEqual(value["error"]["code"], "unexpected_key")
        request.update(operation="generate", target="CCCC", engine="aizynthfinder", configuration_id="b" * 64,
                       settings={"fingerprint": "a" * 64})
        with patch.dict(worker.os.environ, {"GEMINI_API_KEY": SENTINEL}, clear=True):
            value = worker.run_request(request, self.root)
        self.assertEqual(value["error"]["code"], "unexpected_key")

    def test_asset_change_after_consent_rejects_before_original_cli_or_api_use(self):
        self.planner.write_text(self.planner.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with patch.dict(worker.os.environ, {"GEMINI_API_KEY": SENTINEL}, clear=True), \
                patch.object(worker, "_probe", return_value={"files": [self.planner], "config": self.config}), \
                patch.object(worker, "_synthex", side_effect=AssertionError("must not execute stale consent")):
            value = worker.run_request(self.request, self.root)
        self.assertEqual(value["error"]["code"], "config_changed")
        self.assertIsNone(value["result"])

    def test_empty_native_search_is_not_impossibility_or_a_saved_route_fallback(self):
        request = deepcopy(self.request)
        request.update(engine="aizynthfinder", settings={"fingerprint": "a" * 64})
        probe = {"files": [], "config": None, "asset_sha256": {}}
        request["configuration_id"] = worker._digest(service._json_bytes(worker._configuration(request, probe)))
        with patch.dict(worker.os.environ, {}, clear=True), patch.object(worker, "_probe", return_value=probe), \
                patch.object(engine, "generate_routes", side_effect=engine.NoRoutesError("offline empty search")) as generate:
            value = worker.run_request(request, self.root)
        generate.assert_called_once_with("CCCC", time_limit=30, iteration_limit=30)
        self.assertEqual(value["error"]["code"], "no_routes")
        self.assertIn("does not establish", value["error"]["message"])
        self.assertTrue(value["readiness"]["ready"])
        self.assertIsNone(value["result"])

    def test_artifact_redaction_keeps_original_versus_derived_digest_honesty(self):
        encoded = base64.b64encode(SENTINEL.encode()).decode()
        content = json.dumps({"diagnostic": SENTINEL, "encoded": encoded, "api_key": SENTINEL})
        path = self.root / "evidence.json"
        path.write_text(content, encoding="utf-8")
        artifacts = worker._export_artifacts(self.root, SENTINEL)
        exported = next(row for row in artifacts if row["path"] == "evidence.json")
        self.assertNotIn(SENTINEL, exported["content"])
        self.assertNotIn(encoded, exported["content"])
        self.assertEqual(exported["original_sha256"], hashlib.sha256(content.encode()).hexdigest())
        self.assertEqual(exported["export_sha256"], hashlib.sha256(exported["content"].encode()).hexdigest())
        self.assertNotEqual(exported["original_sha256"], exported["export_sha256"])
        self.assertEqual(path.read_text(), content)
        self.assertEqual(service._redact_text(exported["content"], SENTINEL), exported["content"])

    def test_secret_only_or_unknown_binary_artifacts_are_rejected_not_exported(self):
        path = self.root / "unsafe"
        for data in (SENTINEL.encode(), b"\xff\xfe\x00"):
            with self.subTest(kind=len(data)):
                path.write_bytes(data)
                with self.assertRaises(service._Failure) as caught:
                    worker._export_artifacts(self.root, SENTINEL)
                self.assertEqual(caught.exception.code, "secret_output")

    def test_protocol_main_requires_owned_start_gate_and_contains_output_paths(self):
        input_path = self.root / "request.json"
        output_path = self.root / "output.json"
        input_path.write_text(json.dumps(self.request), encoding="utf-8")
        with patch.object(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"x"))), \
                patch.object(worker, "run_request", side_effect=AssertionError("gate must stop it")):
            self.assertEqual(worker.main(["--input", str(input_path), "--output", str(output_path)]), 2)
        with patch.object(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"\n"))), \
                patch.object(worker, "run_request", side_effect=AssertionError("path must stop it")):
            self.assertEqual(worker.main(["--input", str(input_path), "--output", str(self.root.parent / "escape.json")]), 2)
        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
