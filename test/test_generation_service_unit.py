"""Software-only native-generation lifecycle/security tests; no models or API calls.

Subprocess tests execute the real worker protocol with explicit offline backend
substitutes installed only by their temporary test launcher. They are not native
chemistry or live Gemini success evidence.
"""
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import env_support
import generation_service as service

SENTINEL = "offline-personal-key-sentinel-0123456789"


def tree(target="CCO"):
    return {"type": "mol", "smiles": target, "children": [
        {"type": "reaction", "metadata": {"original": "retained"}, "children": [
            {"type": "mol", "smiles": "CC", "children": []},
            {"type": "mol", "smiles": "O", "children": []}]}]}


def routes():
    return {"schema_version": "1.0", "native_trees": [tree(), tree()], "routes": [
        {"id": f"original-{index}", "target": "CCO", "source": "explicit offline test double",
         "upstream": {"ordinal": index, "is_solved": True, "tree": tree()},
         "steps": [{"id": "1", "reactants": ["CC", "O"], "products": ["CCO"],
                    "reagents": [], "solvents": [], "conditions": {"status": "unknown"},
                    "byproducts": {"status": "not_run", "candidates": []}}]}
        for index in (1, 2)]}


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    event = threading.Event()
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        event.wait(0.02)
    raise AssertionError("Offline worker condition did not complete within the test bound")


def alive(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.WaitForSingleObject.restype = ctypes.c_ulong
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 0x102
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    proc = Path(f"/proc/{pid}/stat")
    if proc.is_file() and proc.read_text().split(") ", 1)[1].startswith("Z"):
        return False
    return True


class ServiceFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "generation.local.json"
        self.synthex_config = self.root / "synthex.json"
        self.synthex_config.write_text("{}", encoding="utf-8")
        self.config.write_text(json.dumps({"schema_version": "1.0",
            "aizynthfinder": {"python": sys.executable},
            "synthex": {"python": sys.executable, "config": str(self.synthex_config)}}), encoding="utf-8")
        models = self.root / "test" / "models"
        models.mkdir(parents=True)
        for name in service.MODEL_NAMES:
            (models / name).write_bytes(b"offline file-presence fixture, never loaded")
        self.root_patch = patch.object(service, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.client = service.GenerationService(self.config)
        self.addCleanup(self.stop_service)

    def stop_service(self):
        self.client.close()
        for job in list(self.client._requests.values()):
            if not job.done.wait(10):
                raise AssertionError("Owned test worker was not reaped")

    def prepare(self, engine):
        ready = self.client.readiness(engine)
        if not ready["ready"]:
            request_id = f"fixture-probe-{len(self.client._requests)}"
            self.client.start_readiness(engine, request_id=request_id, context_id="fixture-readiness")
            probe = self.finish(request_id)
            self.assertEqual(probe["status"], "succeeded", probe)
            ready = probe["readiness"]
        return ready["configuration_id"]

    def start(self, engine, target, **kwargs):
        configuration_id = self.prepare(engine)
        return self.client.start(engine, target, configuration_id=configuration_id, **kwargs)

    def launcher(self, mode="success"):
        script = self.root / f"offline_{mode}.py"
        pidfile = self.root / "descendant.json"
        marker = self.root / "started.marker"
        release = self.root / "release.marker"
        # Only this test-local launcher substitutes the backend. Production has
        # no fake-package switch, replay fixture, saved-data or fallback path.
        script.write_text(f'''import json, os, subprocess, sys, time
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, {str(ROOT / "app")!r})
import generation_worker as worker
import engine
MODE = {mode!r}
PAYLOAD = json.loads({json.dumps(routes())!r})
PIDFILE = Path({str(pidfile)!r})
MARKER = Path({str(marker)!r})
RELEASE = Path({str(release)!r})
def offline_generate(target, time_limit=30, iteration_limit=30):
    assert (time_limit, iteration_limit) == (30, 30)
    if MODE in {{"hang", "config_change"}}:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        PIDFILE.write_text(json.dumps({{"root": os.getpid(), "child": child.pid}}))
        MARKER.write_text("started")
        while MODE == "hang" or not RELEASE.exists():
            time.sleep(0.02)
        child.terminate()
        child.wait()
    if MODE in {{"stdout_overflow", "stderr_overflow"}}:
        stream = sys.stdout if MODE == "stdout_overflow" else sys.stderr
        stream.buffer.write(b"x" * (1024 * 1024 + 1))
        stream.flush()
    if MODE == "artifact_bytes":
        with open("too_large.bin", "wb") as handle:
            handle.truncate(32 * 1024 * 1024 + 1)
        time.sleep(5)
    if MODE == "artifact_files":
        for index in range(101):
            Path("f" + str(index)).write_text("x")
        time.sleep(5)
    return deepcopy(PAYLOAD)
engine.generate_routes = offline_generate
worker._probe = lambda request: {{"files": [], "config": None}}
def offline_synthex(request, root, probe, secret):
    if MODE == "secret_exception":
        print(secret, file=sys.stderr)
        raise RuntimeError(secret)
    if MODE == "secret_artifact":
        worker._write_json(root / "evidence.json", {{"diagnostic": secret, "api_key": secret}})
        print(secret, file=sys.stderr)
    payload = deepcopy(PAYLOAD)
    for index, route in enumerate(payload["routes"], 1):
        worker._tag_route(route, request, index)
    return {{"payload": payload, "provenance": {{"execution": "offline test boundary only", "live_verified": False}}, "item_failures": []}}
worker._synthex = offline_synthex
original_run = worker.run_request
def intercepted(request, root):
    value = original_run(request, root)
    if request["operation"] == "probe":
        return value
    if MODE == "stale":
        value["context_id"] = "a-different-context"
    if MODE == "digest":
        value["result"]["artifacts"][0]["export_sha256"] = "0" * 64
    if MODE == "wrong_target":
        value["result"]["payload"]["routes"][0]["target"] = "C"
    if MODE == "duplicate_ids":
        value["result"]["payload"]["routes"][1]["id"] = value["result"]["payload"]["routes"][0]["id"]
    if MODE == "path_escape":
        value["result"]["artifacts"][0]["path"] = "../outside.json"
    return value
worker.run_request = intercepted
if MODE == "startup_exit":
    raise SystemExit(3)
raise SystemExit(worker.main())
''', encoding="utf-8")
        replacement = patch.object(service, "WORKER", script)
        replacement.start()
        self.addCleanup(replacement.stop)
        return marker, release, pidfile

    def finish(self, request_id="r1"):
        return wait_until(lambda: (value if (value := self.client.poll(request_id))["status"] in service.TERMINAL else None))


class ReadinessAndConfigurationTests(ServiceFixture):
    def test_no_setup_is_actionable_and_never_uses_ui_interpreter_or_saved_data(self):
        self.client.config_path = self.root / "missing.json"
        with patch.object(service.subprocess, "Popen", side_effect=AssertionError("must not execute")):
            value = self.client.readiness("aizynthfinder")
        self.assertEqual(value["state"], "not_configured")
        self.assertFalse(value["ready"])
        self.assertIn("explicit interpreter", value["error"]["message"])

    def test_file_presence_is_not_ready_and_readiness_never_executes(self):
        with patch.object(service.subprocess, "Popen", side_effect=AssertionError("must not execute")):
            value = self.client.readiness("aizynthfinder")
        self.assertEqual(value["state"], "live_unverified")
        self.assertFalse(value["ready"])
        self.assertEqual(value["dependency_compatibility"], "unverified")

    def test_absolute_explicit_interpreter_and_strict_nonsecret_config(self):
        for entry in ({"python": "python"}, {"python": sys.executable, "api_key": SENTINEL}):
            with self.subTest(entry=list(entry)):
                self.config.write_text(json.dumps({"schema_version": "1.0", "aizynthfinder": entry}), encoding="utf-8")
                self.assertEqual(self.client.readiness("aizynthfinder")["error"]["code"], "config_invalid")

    def test_missing_interpreter_and_assets_are_distinct(self):
        value = json.loads(self.config.read_text())
        value["aizynthfinder"]["python"] = str(self.root / "missing_python.exe")
        self.config.write_text(json.dumps(value), encoding="utf-8")
        self.assertEqual(self.client.readiness("aizynthfinder")["error"]["code"], "interpreter_missing")
        value["aizynthfinder"]["python"] = sys.executable
        self.config.write_text(json.dumps(value), encoding="utf-8")
        (self.root / "test" / "models" / service.MODEL_NAMES[0]).unlink()
        self.assertEqual(self.client.readiness("aizynthfinder")["state"], "assets_missing")

    def test_duplicate_json_keys_and_environment_substitution_are_rejected(self):
        self.config.write_text('{"schema_version":"1.0","schema_version":"1.0"}', encoding="utf-8")
        self.assertEqual(self.client.readiness("synthex")["error"]["code"], "config_invalid")
        self.config.write_text(json.dumps({"schema_version": "1.0", "synthex": {
            "python": sys.executable, "config": "${COMPANY_CONFIG}"}}), encoding="utf-8")
        self.assertEqual(self.client.readiness("synthex")["error"]["code"], "config_invalid")

    def test_runtime_environment_blocks_inherited_credentials_proxy_debug_and_home(self):
        inherited = {"GEMINI_API_KEY": "inherited", "GOOGLE_API_KEY": "inherited", "GOOGLE_APPLICATION_CREDENTIALS": "company.json",
                     "GOOGLE_CLOUD_PROJECT": "company", "GOOGLE_GENAI_USE_VERTEXAI": "true", "HTTPS_PROXY": "proxy",
                     "AWS_SECRET_ACCESS_KEY": "company", "PYTHONPATH": "untrusted", "DEBUG": "true", "HOME": "company-home"}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, inherited):
            env = service._child_environment(sys.executable, directory, SENTINEL)
            self.assertEqual(env["GEMINI_API_KEY"], SENTINEL)
            self.assertEqual(env["GOOGLE_GENAI_USE_VERTEXAI"], "false")
            self.assertTrue(Path(env["HOME"]).is_relative_to(Path(directory)))
            for key in set(inherited) - {"GEMINI_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "HOME"}:
                self.assertNotIn(key, env)
            self.assertEqual(os.environ["GEMINI_API_KEY"], "inherited")
        with tempfile.TemporaryDirectory() as directory:
            self.assertNotIn("GEMINI_API_KEY", service._child_environment(sys.executable, directory))


@env_support.requires_isolated_worker()
class RealWorkerProtocolTests(ServiceFixture):
    def test_start_poll_all_routes_provenance_stale_and_independent_snapshots(self):
        self.launcher()
        started = self.start("aizynthfinder", "CCO", request_id="r1", context_id="context1")
        self.assertIn(started["status"], {"queued", "running"})
        finished = self.finish()
        self.assertEqual(finished["status"], "succeeded", finished)
        payload = finished["result"]["payload"]
        self.assertEqual([route["id"] for route in payload["routes"]], ["aizynthfinder-r1-0001", "aizynthfinder-r1-0002"])
        self.assertEqual([route["upstream"]["original_id"] for route in payload["routes"]], ["original-1", "original-2"])
        self.assertEqual(payload["routes"][0]["upstream"]["tree"], tree())
        self.assertEqual(finished["result"]["provenance"]["original_call"], "engine.generate_routes")
        stale = self.client.poll("r1", context_id="context2")
        self.assertEqual(stale["status"], "stale")
        self.assertIsNone(stale["result"])
        payload["routes"].clear()
        self.assertEqual(len(self.client.poll("r1")["result"]["payload"]["routes"]), 2)
        with self.assertRaises(ValueError):
            self.start("aizynthfinder", "CCO", request_id="r1", context_id="context1")

    def test_probe_and_execution_use_the_same_configured_interpreter_and_no_shell(self):
        self.launcher()
        calls = []
        original = subprocess.Popen
        def capture(command, **kwargs):
            calls.append((list(command), dict(kwargs, env=dict(kwargs["env"]))))
            return original(command, **kwargs)
        with patch.object(service.subprocess, "Popen", side_effect=capture):
            self.client.start_readiness("aizynthfinder", request_id="probe1", context_id="ctx")
            probe = self.finish("probe1")
            self.assertEqual(probe["status"], "succeeded", probe)
            self.assertTrue(probe["readiness"]["ready"])
            self.assertFalse(probe["readiness"]["live_verified"])
            self.client.start("aizynthfinder", "CCO", request_id="r1", context_id="ctx",
                              configuration_id=probe["readiness"]["configuration_id"])
            self.assertEqual(self.finish()["status"], "succeeded")
        self.assertEqual([command[0] for command, _ in calls], [sys.executable, sys.executable])
        for command, kwargs in calls:
            self.assertFalse(kwargs["shell"])
            self.assertIn("-I", command)
            self.assertEqual(command[command.index("-X") + 1], "utf8")
            self.assertNotIn("GEMINI_API_KEY", kwargs["env"])

    def test_key_is_separate_child_only_redacted_and_hashes_remain_honest(self):
        self.launcher("secret_artifact")
        calls = []
        original = subprocess.Popen
        def capture(command, **kwargs):
            request_path = Path(command[command.index("--input") + 1])
            calls.append((list(command), dict(kwargs["env"]), request_path.read_text(encoding="utf-8")))
            return original(command, **kwargs)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "inherited-company-key", "GOOGLE_APPLICATION_CREDENTIALS": "company.json"}), \
                patch.object(service.subprocess, "Popen", side_effect=capture):
            self.start("synthex", "CCO", request_id="r1", context_id="ctx", gemini_api_key=SENTINEL)
            finished = self.finish()
            self.assertEqual(os.environ["GEMINI_API_KEY"], "inherited-company-key")
        self.assertEqual(finished["status"], "succeeded", finished)
        self.assertNotIn(SENTINEL, json.dumps(finished))
        self.assertNotIn(hashlib.sha256(SENTINEL.encode()).hexdigest(), json.dumps(finished))
        self.assertNotIn("GEMINI_API_KEY", calls[0][1])
        command, env, request = calls[-1]
        self.assertNotIn(SENTINEL, json.dumps(command))
        self.assertNotIn(SENTINEL, request)
        self.assertEqual(env["GEMINI_API_KEY"], SENTINEL)
        self.assertNotIn("GOOGLE_APPLICATION_CREDENTIALS", env)
        artifact = next(row for row in finished["result"]["artifacts"] if row["path"] == "evidence.json")
        self.assertTrue(artifact["redacted"])
        self.assertNotEqual(artifact["original_sha256"], artifact["export_sha256"])
        self.assertEqual(artifact["export_sha256"], hashlib.sha256(artifact["content"].encode()).hexdigest())
        self.assertIn("[REDACTED]", artifact["content"])

    def test_secret_sdk_exception_and_stderr_are_classified_not_saved_or_returned(self):
        self.launcher("secret_exception")
        self.start("synthex", "CCO", request_id="r1", context_id="ctx", gemini_api_key=SENTINEL)
        finished = self.finish()
        self.assertEqual(finished["status"], "failed", finished)
        self.assertEqual(finished["error"]["code"], "backend_failed")
        self.assertIsNone(finished["result"])
        self.assertNotIn(SENTINEL, json.dumps(finished))

    def test_key_missing_unexpected_and_key_in_serializable_fields_never_spawn(self):
        with patch.object(service.subprocess, "Popen", side_effect=AssertionError("must not start")):
            self.assertEqual(self.client.start("synthex", "CCO", request_id="r1", context_id="c", configuration_id=None)["error"]["code"], "missing_key")
            self.assertEqual(self.client.start("aizynthfinder", "CCO", request_id="r2", context_id="c", configuration_id=None, gemini_api_key=SENTINEL)["error"]["code"], "unexpected_key")
            with self.assertRaises(ValueError) as caught:
                self.client.start("synthex", SENTINEL, request_id="r3", context_id="c", configuration_id=None, gemini_api_key=SENTINEL)
            self.assertNotIn(SENTINEL, str(caught.exception))
            google_shaped = "AIza" + "f" * 35
            with self.assertRaises(ValueError):
                self.client.start("synthex", google_shaped, request_id="r4", context_id="c",
                                  configuration_id=None, gemini_api_key=google_shaped)

    def test_existing_interpreter_with_startup_failure_is_not_ready(self):
        self.launcher("startup_exit")
        self.client.start_readiness("synthex", request_id="r1", context_id="ctx")
        value = self.finish()
        self.assertEqual(value["status"], "failed", value)
        self.assertEqual(value["error"]["code"], "startup_failed")
        self.assertFalse(value["readiness"]["ready"])

    def test_malformed_stale_target_duplicate_digest_and_path_outputs_are_rejected(self):
        for mode, expected in (("stale", "invalid_output"), ("wrong_target", "invalid_output"),
                               ("duplicate_ids", "invalid_output"), ("digest", "digest_mismatch"), ("path_escape", "path_escape")):
            with self.subTest(mode=mode):
                self.launcher(mode)
                self.start("aizynthfinder", "CCO", request_id=mode, context_id="ctx")
                finished = self.finish(mode)
                self.assertEqual(finished["status"], "failed", finished)
                self.assertEqual(finished["error"]["code"], expected)
                self.assertIsNone(finished["result"])

    def test_config_change_during_execution_is_not_applied(self):
        marker, release, _ = self.launcher("config_change")
        self.start("aizynthfinder", "CCO", request_id="r1", context_id="ctx")
        wait_until(marker.exists)
        self.config.write_text(self.config.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        release.write_text("release", encoding="utf-8")
        finished = self.finish()
        self.assertEqual(finished["error"]["code"], "config_changed", finished)
        self.assertIsNone(finished["result"])

    def test_diagnostic_and_artifact_limits_reject_not_truncate_success(self):
        for mode, code in (("stdout_overflow", "output_limit"), ("stderr_overflow", "output_limit"),
                           ("artifact_files", "artifact_limit"), ("artifact_bytes", "artifact_limit")):
            with self.subTest(mode=mode):
                self.launcher(mode)
                self.start("aizynthfinder", "CCO", request_id=mode, context_id="ctx")
                finished = self.finish(mode)
                self.assertEqual(finished["status"], "bounds_exceeded", finished)
                self.assertEqual(finished["error"]["code"], code)
                self.assertIsNone(finished["result"])

    def test_cancel_is_nonblocking_reaps_owned_descendants_and_not_other_processes(self):
        marker, _, pidfile = self.launcher("hang")
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        def cleanup_unrelated():
            if unrelated.poll() is None:
                unrelated.kill()
            unrelated.wait(timeout=5)
        self.addCleanup(cleanup_unrelated)
        self.start("aizynthfinder", "CCO", request_id="r1", context_id="ctx")
        wait_until(marker.exists)
        pids = json.loads(pidfile.read_text())
        self.assertTrue(alive(pids["child"]))
        started = time.monotonic()
        self.assertEqual(self.client.cancel("r1")["status"], "cancelling")
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(self.finish()["status"], "cancelled")
        wait_until(lambda: not alive(pids["child"]))
        self.assertFalse(alive(pids["root"]))
        self.assertIsNone(unrelated.poll())
        unrelated.terminate()
        unrelated.wait(timeout=5)

    def test_timeout_and_close_reap_the_owned_worker_tree(self):
        marker, _, pidfile = self.launcher("hang")
        with patch.dict(service.WALL_SECONDS, {"aizynthfinder": 2}):
            self.start("aizynthfinder", "CCO", request_id="r1", context_id="ctx")
            wait_until(marker.exists)
            pids = json.loads(pidfile.read_text())
            finished = self.finish()
        self.assertEqual(finished["status"], "timed_out", finished)
        wait_until(lambda: not alive(pids["child"]))
        self.client.close()
        value = self.client.start("aizynthfinder", "CCO", request_id="r2", context_id="ctx", configuration_id=None)
        self.assertEqual(value["error"]["code"], "service_closed")

    def test_configuration_change_between_consent_and_start_rejects_before_spawn(self):
        self.launcher()
        consented = self.prepare("synthex")
        self.synthex_config.write_text('{"llm":{"model":"changed-after-consent"}}', encoding="utf-8")
        with patch.object(service.subprocess, "Popen", side_effect=AssertionError("changed config must not spawn")):
            value = self.client.start("synthex", "CCO", request_id="r1", context_id="ctx",
                                      configuration_id=consented, gemini_api_key=SENTINEL)
        self.assertEqual(value["error"]["code"], "config_changed")
        self.assertIsNone(value["result"])


class ArtifactBoundaryTests(unittest.TestCase):
    def test_exact_bounds_and_hardlink_or_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data"
            with path.open("wb") as stream:
                stream.truncate(service.MAX_ARTIFACT_BYTES)
            self.assertEqual(service._inventory(root), [path])
            with path.open("ab") as stream:
                stream.write(b"x")
            with self.assertRaises(service._Failure) as caught:
                service._inventory(root)
            self.assertEqual(caught.exception.code, "artifact_limit")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(service.MAX_ARTIFACT_FILES):
                (root / str(index)).write_text("x")
            self.assertEqual(len(service._inventory(root)), service.MAX_ARTIFACT_FILES)
            (root / "extra").write_text("x")
            with self.assertRaises(service._Failure):
                service._inventory(root)

    def test_path_traversal_absolute_drive_and_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "okay.json").write_text("{}")
            for value in ("../outside.json", "/outside.json", "C:/outside.json", "sub\\outside.json"):
                with self.subTest(path=value), self.assertRaises(service._Failure):
                    service._safe_relative(root, value)
            linked = root / "hardlink.json"
            os.link(root / "okay.json", linked)
            with self.assertRaises(service._Failure) as caught:
                service._inventory(root)
            self.assertEqual(caught.exception.code, "path_escape")


if __name__ == "__main__":
    unittest.main()
