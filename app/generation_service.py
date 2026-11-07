"""Session-owned, nonblocking boundary around the original native generators.

Only explicit local configuration is consulted. This module does not import a
scientific backend, discover credentials, install packages, or execute on import.
A service belongs to one UI session; keep it out of shared caches and exports.

Local JSON: {"schema_version":"1.0", "aizynthfinder":{"python":"ABSOLUTE PATH"},
            "synthex":{"python":"ABSOLUTE PATH", "config":"ABSOLUTE PATH"}}.
Finish start_readiness/poll before consent, bind its configuration_id, then pass
that ID to start. Keys are separate arguments and are never configuration data.
"""
from __future__ import annotations

import atexit
import base64
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile
import threading
import time
from urllib.parse import quote, quote_plus

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = ROOT / "generation.local.json"
WORKER = Path(__file__).with_name("generation_worker.py")
ENGINES = {"aizynthfinder", "synthex"}
# Nonsecret provider selection. The key itself is never part of it and never
# travels with the request; only these names carry a key into a child process.
PROVIDERS = {"gemini", "openai_compatible"}
KEY_VARIABLES = {"gemini": "GEMINI_API_KEY", "openai_compatible": "SYNTHEX_LLM_API_KEY"}
KEY_VARIABLE_NAMES = frozenset(KEY_VARIABLES.values())
WALL_SECONDS = {"aizynthfinder": 120, "synthex": 900}
READINESS_SECONDS = 30
MAX_STREAM_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_FILES = 100
MODEL_NAMES = ("uspto_model.onnx", "uspto_templates.csv.gz", "uspto_filter_model.onnx", "zinc_stock.hdf5")
TERMINAL = {"succeeded", "partial", "failed", "cancelled", "timed_out", "bounds_exceeded"}
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ERRORS = {
    "not_configured": "Set an explicit interpreter in generation.local.json; no default environment or saved routes are used.",
    "config_invalid": "The nonsecret native-generation configuration is invalid; use explicit absolute local paths and no credentials or environment substitutions.",
    "config_changed": "The configured inputs changed during this request. Start a new request after reviewing the configuration.",
    "interpreter_missing": "The explicitly configured Python interpreter is missing. Prepare its isolated dependencies outside the UI.",
    "worker_missing": "The native-generation worker source is missing.",
    "package_missing": "The configured interpreter cannot import a required original backend or dependency. Prepare compatible packages outside the UI.",
    "assets_missing": "Required local models or stocks are missing. Full SynthEx also requires distinct ring-breaker model/templates and a companion asset manifest with matching SHA-256 digests. No download or reduced workflow was attempted.",
    "incompatible_backend": "The configured dependency or full-workflow settings do not match the original backend contract.",
    "startup_failed": "The configured interpreter could not start the isolated worker. File presence alone does not establish runtime readiness.",
    "backend_failed": "The original backend did not complete the request. Raw SDK diagnostics are withheld for privacy; no automatic retry or fallback was performed.",
    "no_routes": "No route containing reaction steps was found within the original search budget. The target may already be in stock; this does not establish synthesis impossibility.",
    "target_invalid": "A valid complete target SMILES of at most 4096 characters is required.",
    "invalid_request": "The generation request does not satisfy the worker protocol.",
    "missing_key": "SynthEx requires a separately supplied personal API key for the selected provider and explicit UI consent for this request.",
    "unexpected_key": "An API key must never be supplied to an AiZynthFinder or readiness worker.",
    "invalid_output": "The original output is missing, malformed, incomplete, or bound to a different request or target.",
    "digest_mismatch": "Original checkpoint, configuration, or exported artifact digests did not validate.",
    "path_escape": "An output path, link, or filesystem entry escaped the owned request boundary.",
    "secret_output": "A secret-bearing or unsafe output could not be safely validated. It was not exposed or exported.",
    "output_limit": "A diagnostic stream exceeded 1 MiB. The owned process tree was stopped; output was not silently truncated into a success.",
    "artifact_limit": "The owned request exceeded 32 MiB or 100 artifact files. The owned process tree was stopped.",
    "timeout": "The original backend exceeded its declared wall-clock limit. The owned process tree was stopped.",
    "cancelled": "This request was cancelled; already transmitted API requests or charges cannot be undone.",
    "cleanup_failed": "Owned worker resources could not be completely reaped or removed; no result was applied.",
    "partial_output": "The original workflow returned partial, failed, unreviewed, or unsolved items. They remain explicit in the retained evidence.",
    "service_closed": "This session's native-generation service is closed.",
    "request_unknown": "No request with this ID belongs to this session.",
    "stale": "The result belongs to an earlier request context and cannot be applied.",
}


class _Failure(ValueError):
    def __init__(self, code):
        self.code = code if code in _ERRORS else "backend_failed"
        super().__init__(self.code)


def _error(code):
    code = code if isinstance(code, str) and code in _ERRORS else "backend_failed"
    return {"code": code, "message": _ERRORS[code]}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _Failure("invalid_output")
        result[key] = value
    return result


def _read_json(path, limit=MAX_ARTIFACT_BYTES):
    try:
        if path.stat().st_size > limit:
            raise _Failure("artifact_limit")
        data = path.read_bytes()
        if len(data) > limit:
            raise _Failure("artifact_limit")
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(_Failure("invalid_output")))
    except _Failure:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        raise _Failure("invalid_output") from None


def _secret_values(secret):
    if not secret:
        return ()
    return {secret, quote(secret, safe=""), quote_plus(secret),
            json.dumps(secret, ensure_ascii=True)[1:-1],
            base64.b64encode(secret.encode("utf-8")).decode("ascii")}


def _contains_secret(text, secret):
    return any(value in text for value in _secret_values(secret))


def _redact_text(text, secret=None):
    """Create derived exports, never modify digest-linked upstream checkpoints."""
    for value in sorted(_secret_values(secret), key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    text = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED]", text)
    text = re.sub(r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password)\b[\"']?\s*[:=]\s*[\"']?)([^\s\"'&,}]+)",
                  r"\1[REDACTED]", text)
    return re.sub(r"(?i)\bBearer\s+[A-Za-z0-9_.~+/=-]+", "Bearer [REDACTED]", text)


def _base_url_allowed(value):
    """Accept HTTPS endpoints and a loopback HTTP endpoint for a local model."""
    if not isinstance(value, str) or not 8 <= len(value) <= 512:
        return False
    if "@" in value or any(character.isspace() or character in "\"'\\<>" for character in value):
        return False
    if value.startswith("https://"):
        return len(value) > len("https://")
    if not value.startswith("http://"):
        return False
    host = value[len("http://"):].split("/")[0].rsplit(":", 1)[0].casefold()
    return host in {"localhost", "127.0.0.1", "[::1]"}


def _llm_selection(value):
    """Validate the nonsecret provider/endpoint/model choice supplied by the UI."""
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {"provider", "base_url", "model"}
            or value["provider"] not in PROVIDERS
            or not isinstance(value["model"], str) or not 1 <= len(value["model"]) <= 256
            or any(character.isspace() for character in value["model"])
            or _redact_text(value["model"]) != value["model"]):
        raise _Failure("config_invalid")
    base_url = value["base_url"]
    if value["provider"] == "gemini":
        if base_url is not None:
            raise _Failure("config_invalid")
    elif not _base_url_allowed(base_url) or _redact_text(base_url) != base_url:
        raise _Failure("config_invalid")
    return {"provider": value["provider"], "base_url": base_url, "model": value["model"]}


def _safe_relative(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise _Failure("path_escape")
    relative = Path(name)
    if relative.is_absolute() or any(part in {"..", "."} for part in name.split("/")):
        raise _Failure("path_escape")
    path = root / relative
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            raise _Failure("path_escape")
        for part in (path, *path.parents):
            if part == root.parent:
                break
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise _Failure("path_escape")
        if not path.is_file() or path.stat().st_nlink != 1:
            raise _Failure("path_escape")
    except _Failure:
        raise
    except OSError:
        raise _Failure("invalid_output") from None
    return path


def _inventory(root):
    """Monitor all owned files, including caches and temporary checkpoint writes."""
    count = total = 0
    files = []
    try:
        for directory, dirs, names in os.walk(root, followlinks=False):
            for name in [*dirs, *names]:
                path = Path(directory) / name
                try:
                    info = path.lstat()
                except FileNotFoundError:  # Atomic manifest replacement is normal.
                    continue
                if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
                        or not path.resolve().is_relative_to(root.resolve())):
                    raise _Failure("path_escape")
                if stat.S_ISDIR(info.st_mode):
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise _Failure("path_escape")
                count += 1
                total += info.st_size
                if count > MAX_ARTIFACT_FILES or total > MAX_ARTIFACT_BYTES:
                    raise _Failure("artifact_limit")
                files.append(path)
    except _Failure:
        raise
    except OSError:
        raise _Failure("invalid_output") from None
    return sorted(files)


def _child_environment(python, directory, secret=None, key_variable="GEMINI_API_KEY"):
    """Allowlist runtime values; never copy the parent's credential-bearing env.

    ``key_variable`` selects which single credential name the chosen provider
    reads. Exactly one is ever set, and only for this one child process.
    """
    directory = Path(directory)
    home = directory / "home"
    home.mkdir()
    runtime = directory / "runtime"
    runtime.mkdir()
    env = {}
    for name in ("SystemRoot", "WINDIR", "SYSTEMDRIVE"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    prefix = Path(python).parent
    if prefix.name.lower() in {"scripts", "bin"}:
        prefix = prefix.parent
    search = [Path(python).parent, prefix, prefix / "Library" / "bin",
              prefix / "Library" / "usr" / "bin", prefix / "Library" / "mingw-w64" / "bin",
              prefix / "Scripts"]
    if os.name == "nt":
        system = Path(env.get("SystemRoot", "C:/Windows"))
        search.extend((system / "System32", system))
    else:
        search.extend((Path("/usr/bin"), Path("/bin")))
    env.update(PATH=os.pathsep.join(str(path) for path in search),
               HOME=str(home), USERPROFILE=str(home), APPDATA=str(home / "AppData" / "Roaming"),
               LOCALAPPDATA=str(home / "AppData" / "Local"),
               TMP=str(runtime), TEMP=str(runtime), TMPDIR=str(runtime),
               XDG_CONFIG_HOME=str(home / "config"), XDG_CACHE_HOME=str(home / "cache"),
               XDG_DATA_HOME=str(home / "data"), CLOUDSDK_CONFIG=str(home / "gcloud"),
               MPLCONFIGDIR=str(runtime / "matplotlib"), TORCH_HOME=str(runtime / "torch"),
               HF_HOME=str(runtime / "huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               NUMBA_CACHE_DIR=str(runtime / "numba"), PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               GOOGLE_GENAI_USE_VERTEXAI="false", GIT_CONFIG_NOSYSTEM="1",
               GIT_CONFIG_GLOBAL=os.devnull)
    if secret is not None:
        if key_variable not in KEY_VARIABLE_NAMES:
            raise _Failure("invalid_request")
        env[key_variable] = secret
    return env


class _WindowsJob:
    """Kernel-owned descendant lifetime, including descendants after root exit."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Io(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Io),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.handle:
            raise _Failure("startup_failed")
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise _Failure("startup_failed")

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise _Failure("startup_failed")

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            if not self.kernel.CloseHandle(handle):
                raise _Failure("cleanup_failed")


def _stop_tree(process, owned_job):
    try:
        if owned_job is not None:
            owned_job.close()
        elif os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):
        raise _Failure("cleanup_failed") from None


def _settings(config_path, engine):
    if engine not in ENGINES:
        raise _Failure("invalid_request")
    path = Path(config_path)
    if str(path).startswith(("\\\\", "//")):
        raise _Failure("config_invalid")
    if not path.is_file():
        raise _Failure("not_configured")
    try:
        value = _read_json(path, 64 * 1024)
        if (not isinstance(value, dict) or value.get("schema_version") != "1.0"
                or set(value) - {"schema_version", *ENGINES}):
            raise _Failure("config_invalid")
        for name in ENGINES & set(value):
            entry = value[name]
            keys = {"python"} if name == "aizynthfinder" else {"python", "config"}
            if not isinstance(entry, dict) or set(entry) != keys:
                raise _Failure("config_invalid")
            for item in entry.values():
                if (not isinstance(item, str) or not item or item.startswith(("\\\\", "//"))
                        or not Path(item).is_absolute() or "${" in item):
                    raise _Failure("config_invalid")
        if engine not in value:
            raise _Failure("not_configured")
        settings = dict(value[engine])
        python = Path(settings["python"])
        if not python.is_file():
            raise _Failure("interpreter_missing")
        if not WORKER.is_file():
            raise _Failure("worker_missing")
        info = python.stat()
        identity = {"local_config": _digest(path.read_bytes()), "python": str(python.resolve()),
                    "python_mtime_ns": info.st_mtime_ns, "python_size": info.st_size,
                    "python_sha256": _digest(python.read_bytes())}
        if engine == "synthex":
            source = Path(settings["config"])
            if not source.is_file():
                raise _Failure("assets_missing")
            if source.stat().st_size > 1024 * 1024:
                raise _Failure("config_invalid")
            data = source.read_bytes()
            if b"${" in data:
                raise _Failure("config_invalid")
            settings["config_sha256"] = _digest(data)
            identity["synthex_config"] = settings["config_sha256"]
        else:
            missing = [name for name in MODEL_NAMES if not (ROOT / "test" / "models" / name).is_file()]
            if missing:
                raise _Failure("assets_missing")
        settings["fingerprint"] = _digest(_json_bytes(identity))
        return settings
    except _Failure as exc:
        if exc.code in {"invalid_output", "artifact_limit"}:
            raise _Failure("config_invalid") from None
        raise
    except (OSError, ValueError, TypeError):
        raise _Failure("config_invalid") from None


def _readiness(state="live_unverified", ready=False, code=None, configuration=None):
    return {"state": state, "ready": ready, "live_verified": False,
            "dependency_compatibility": "unverified", "error": _error(code) if code else None,
            "configuration_id": _digest(_json_bytes(configuration)) if configuration is not None else None,
            "configuration": configuration}


def _readiness_failure(code):
    state = code if code in {"not_configured", "package_missing", "assets_missing"} else "failed"
    return _readiness(state, code=code)


@dataclass
class _Request:
    request: dict
    settings: dict
    snapshot: dict
    selection: dict | None = None
    key_variable: str = "GEMINI_API_KEY"
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    done: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)


# One readiness cache per configuration file, for the life of the process.
# Service objects are built per browser session; the probe they run is about
# the machine, not the session.
_SHARED_READINESS = {}
_SHARED_READINESS_LOCK = threading.RLock()


def _shared_readiness(config_path):
    with _SHARED_READINESS_LOCK:
        return _SHARED_READINESS.setdefault(str(Path(config_path)), {})


class GenerationService:
    """One session's explicit start/poll/cancel lifecycle; no automatic retries.

    Request/context IDs are nonsecret opaque identifiers, never key hashes.
    `readiness` only inspects configuration/files and cached probe evidence.
    `start_readiness` performs an asynchronous, key-free interpreter probe.
    `start` independently checks readiness inside the same selected interpreter.
    """
    def __init__(self, config_path=None):
        self.config_path = Path(config_path) if config_path is not None else LOCAL_CONFIG
        self._requests = {}
        # Shared with every other service on the same configuration file in
        # this process, so a reader who reloads the page does not wait out a
        # fresh interpreter probe. What the key already carries makes this
        # safe: the configuration file's digest and the interpreter's path,
        # size, mtime and digest, plus the provider selection. Change any of
        # them and it is a different key, not a stale hit. Only a completed
        # probe is written, so a failure is never cached.
        self._readiness_cache = _shared_readiness(self.config_path)
        self._lock = threading.RLock()
        self._closed = False
        atexit.register(self._shutdown)

    def readiness(self, engine, llm_selection=None):
        try:
            settings = _settings(self.config_path, engine)
            selection = _llm_selection(llm_selection)
        except _Failure as exc:
            return _readiness_failure(exc.code)
        with self._lock:
            return deepcopy(self._readiness_cache.get(
                self._cache_key(engine, settings["fingerprint"], selection), _readiness()))

    @staticmethod
    def _cache_key(engine, fingerprint, selection):
        """A provider/model change is a different configuration, not a cache hit."""
        return engine, fingerprint, _digest(_json_bytes(selection))

    def start_readiness(self, engine, *, request_id, context_id, llm_selection=None):
        return self._start(engine, None, request_id, context_id, None, "probe", None, llm_selection)

    def start(self, engine, target, *, request_id, context_id, configuration_id,
              gemini_api_key=None, api_key=None, llm_selection=None):
        if gemini_api_key is not None and api_key is not None:
            raise ValueError(_ERRORS["invalid_request"])
        secret = api_key if gemini_api_key is None else gemini_api_key
        return self._start(engine, target, request_id, context_id, secret, "generate",
                           configuration_id, llm_selection)

    def _start(self, engine, target, request_id, context_id, secret, operation, configuration_id,
               llm_selection=None):
        if (not isinstance(request_id, str) or not _ID.fullmatch(request_id)
                or not isinstance(context_id, str) or not _ID.fullmatch(context_id)):
            raise ValueError(_ERRORS["invalid_request"])
        selected = list(llm_selection.values()) if isinstance(llm_selection, dict) else []
        if isinstance(secret, str) and secret and any(
                isinstance(value, str) and _contains_secret(value, secret)
                for value in (engine, target, request_id, context_id, configuration_id, *selected)):
            raise ValueError(_ERRORS["invalid_request"])
        with self._lock:
            if request_id in self._requests:
                raise ValueError("A request ID can only be used once in this session.")
            request = {"schema_version": "1.0", "operation": operation, "engine": engine,
                       "request_id": request_id, "context_id": context_id, "target": target,
                       "configuration_id": configuration_id}
            snapshot = {"request_id": request_id, "context_id": context_id, "engine": engine,
                        "target": target, "operation": operation, "status": "queued",
                        "configuration_id": configuration_id,
                        "readiness": _readiness(), "result": None, "error": None}
            job = _Request(request, {}, snapshot)
            self._requests[request_id] = job
            try:
                if self._closed:
                    raise _Failure("service_closed")
                if operation == "generate" and (not isinstance(target, str) or not target.strip() or len(target) > 4096):
                    raise _Failure("target_invalid")
                if engine == "aizynthfinder" or operation == "probe":
                    if secret is not None:
                        raise _Failure("unexpected_key")
                elif engine == "synthex":
                    if (not isinstance(secret, str) or not 8 <= len(secret) <= 4096
                            or secret == "[REDACTED]"
                            or any(character in secret for character in "\r\n\0")):
                        raise _Failure("missing_key")
                selection = _llm_selection(llm_selection)
                if selection is not None and engine != "synthex":
                    raise _Failure("invalid_request")
                settings = _settings(self.config_path, engine)
                if operation == "generate":
                    prepared = self._readiness_cache.get(
                        self._cache_key(engine, settings["fingerprint"], selection), {})
                    if (not isinstance(configuration_id, str) or not _DIGEST.fullmatch(configuration_id)
                            or not prepared.get("ready") or prepared.get("configuration_id") != configuration_id):
                        raise _Failure("config_changed")
                request["settings"] = {key: value for key, value in settings.items() if key != "python"}
                # Omitted entirely for the unchanged Gemini-from-config path, so
                # that request stays byte-identical to the one shipped before.
                if selection is not None:
                    request["llm"] = deepcopy(selection)
                job.key_variable = KEY_VARIABLES[(selection or {}).get("provider", "gemini")]
                job.selection = selection
                if secret and _contains_secret(_json_bytes(request).decode("utf-8"), secret):
                    request.pop("settings", None)
                    raise _Failure("config_invalid")
                job.settings = settings
            except _Failure as exc:
                snapshot.update(status="failed", readiness=_readiness_failure(exc.code), error=_error(exc.code))
                job.done.set()
                return deepcopy(snapshot)
            job.thread = threading.Thread(target=self._run, args=(job, secret), daemon=True,
                                          name="native-generation")
            job.thread.start()
            return deepcopy(snapshot)

    def poll(self, request_id, *, context_id=None):
        with self._lock:
            job = self._requests.get(request_id)
            if job is None:
                return {"request_id": request_id, "context_id": context_id, "status": "failed",
                        "readiness": None, "result": None, "error": _error("request_unknown")}
            snapshot = deepcopy(job.snapshot)
            if context_id is not None and context_id != snapshot["context_id"]:
                snapshot.update(status="stale", result=None, error=_error("stale"))
            return snapshot

    def cancel(self, request_id):
        with self._lock:
            job = self._requests.get(request_id)
            if job is None:
                return self.poll(request_id)
            job.cancel_event.set()
            if job.snapshot["status"] in TERMINAL:
                job.snapshot.update(status="cancelled", result=None, error=_error("cancelled"))
            else:
                job.snapshot.update(status="cancelling", result=None, error=_error("cancelled"))
            return deepcopy(job.snapshot)

    def close(self):
        with self._lock:
            self._closed = True
            for request_id in self._requests:
                self.cancel(request_id)
        atexit.unregister(self._shutdown)

    def _shutdown(self):
        self.close()
        for job in list(self._requests.values()):
            job.done.wait(6)

    def _run(self, job, secret):
        process = owned_job = directory = None
        tree_reaped = False
        streams = []
        env = {}
        result = None
        failure = None
        try:
            if job.cancel_event.is_set():
                raise _Failure("cancelled")
            directory = tempfile.TemporaryDirectory(prefix="native_generation_")
            root = Path(directory.name)
            input_path, output_path = root / "request.json", root / "output.json"
            input_path.write_bytes(_json_bytes(job.request))
            env = _child_environment(job.settings["python"], root, secret, job.key_variable)
            command = [job.settings["python"], "-I", "-B", "-X", "utf8", str(WORKER),
                       "--input", str(input_path), "--output", str(output_path)]
            if os.name == "nt":
                owned_job = _WindowsJob()
            kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            started = time.monotonic()
            process = subprocess.Popen(command, shell=False, cwd=root, env=env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                                       close_fds=True, **kwargs)
            if owned_job is not None:
                owned_job.assign(process)
            overflow = threading.Event()
            counts = [0, 0]

            def drain(pipe, index):
                try:
                    while True:
                        data = pipe.read(65536)
                        if not data:
                            return
                        counts[index] += len(data)
                        if counts[index] > MAX_STREAM_BYTES:
                            overflow.set()
                        # Deliberately discard raw diagnostics in memory, never log/save them.
                finally:
                    pipe.close()

            for index, pipe in enumerate((process.stdout, process.stderr)):
                thread = threading.Thread(target=drain, args=(pipe, index), daemon=True)
                thread.start()
                streams.append(thread)
            # The worker cannot import/run a backend until it belongs to our Job/session.
            try:
                process.stdin.write(b"\n")
                process.stdin.close()
            except (OSError, BrokenPipeError):
                raise _Failure("startup_failed") from None
            with self._lock:
                if not job.cancel_event.is_set():
                    job.snapshot["status"] = "running"
            limit = READINESS_SECONDS if job.request["operation"] == "probe" else WALL_SECONDS[job.request["engine"]]
            while True:
                if job.cancel_event.is_set():
                    raise _Failure("cancelled")
                if overflow.is_set():
                    raise _Failure("output_limit")
                _inventory(root)
                if time.monotonic() - started >= limit:
                    raise _Failure("timeout")
                if process.poll() is not None:
                    break
                job.cancel_event.wait(0.05)
            _stop_tree(process, owned_job)
            owned_job = None
            tree_reaped = True
            for thread in streams:
                thread.join(timeout=5)
                if thread.is_alive():
                    raise _Failure("cleanup_failed")
            if overflow.is_set():
                raise _Failure("output_limit")
            _inventory(root)
            if process.returncode != 0:
                raise _Failure("startup_failed")
            current = _settings(self.config_path, job.request["engine"])
            if current["fingerprint"] != job.settings["fingerprint"]:
                raise _Failure("config_changed")
            result = self._validate_response(job, _read_json(output_path), root, secret)
        except _Failure as exc:
            failure = exc.code
        except (OSError, ValueError, TypeError, RecursionError, subprocess.SubprocessError):
            failure = "startup_failed"
        except Exception:
            failure = "backend_failed"
        finally:
            if process is not None and not tree_reaped:
                try:
                    _stop_tree(process, owned_job)
                    owned_job = None
                except _Failure:
                    failure = "cleanup_failed"
            elif owned_job is not None:
                try:
                    owned_job.close()
                except _Failure:
                    failure = "cleanup_failed"
            for thread in streams:
                thread.join(timeout=5)
                if thread.is_alive():
                    failure = "cleanup_failed"
            env.clear()
            secret = None
            if directory is not None:
                try:
                    directory.cleanup()
                except OSError:
                    failure = "cleanup_failed"
            with self._lock:
                if job.cancel_event.is_set() and failure != "cleanup_failed":
                    failure = "cancelled"
                if failure:
                    status = {"cancelled": "cancelled", "timeout": "timed_out",
                              "output_limit": "bounds_exceeded", "artifact_limit": "bounds_exceeded"}.get(failure, "failed")
                    job.snapshot.update(status=status, result=None, error=_error(failure),
                                        readiness=_readiness_failure(failure))
                else:
                    job.snapshot.update(status=result["status"], result=result["result"],
                                        error=result["error"], readiness=result["readiness"],
                                        configuration_id=result["readiness"]["configuration_id"])
                    self._readiness_cache[self._cache_key(
                        job.request["engine"], job.settings["fingerprint"], job.selection)] = result["readiness"]
                job.done.set()

    def _validate_response(self, job, response, root, secret):
        if (not isinstance(response, dict) or set(response) != {"schema_version", "request_sha256", "request_id", "context_id", "engine", "operation", "target", "status", "readiness", "result", "error"}
                or response["schema_version"] != "1.0"
                or response["request_sha256"] != _digest(_json_bytes(job.request))
                or any(response[key] != job.request[key] for key in ("request_id", "context_id", "engine", "operation", "target"))
                or response["status"] not in {"succeeded", "partial", "failed"}):
            raise _Failure("invalid_output")
        text = _json_bytes(response).decode("utf-8")
        if _redact_text(text, secret) != text:
            raise _Failure("secret_output")
        readiness = response["readiness"]
        if (not isinstance(readiness, dict) or set(readiness) != {"state", "ready", "live_verified", "dependency_compatibility", "error", "configuration_id", "configuration"}
                or readiness["state"] not in {"ready", "live_unverified", "failed", "package_missing", "assets_missing"}
                or type(readiness["ready"]) is not bool or type(readiness["live_verified"]) is not bool
                or readiness["dependency_compatibility"] not in {"unverified", "executed"}):
            raise _Failure("invalid_output")
        configuration = readiness["configuration"]
        if readiness["ready"]:
            if (not isinstance(configuration, dict)
                    or configuration.get("interpreter_config") != job.settings["fingerprint"]
                    or readiness["configuration_id"] != _digest(_json_bytes(configuration))
                    or (job.request["operation"] == "generate" and readiness["configuration_id"] != job.request["configuration_id"])):
                raise _Failure("config_changed")
        elif configuration is not None or readiness["configuration_id"] is not None:
            raise _Failure("invalid_output")
        if readiness["error"] is not None:
            readiness["error"] = _error(readiness["error"].get("code") if isinstance(readiness["error"], dict) else None)
        error = response["error"]
        if error is not None:
            response["error"] = _error(error.get("code") if isinstance(error, dict) else None)
        if response["status"] == "failed":
            if response["result"] is not None or response["error"] is None:
                raise _Failure("invalid_output")
            return response
        if job.request["operation"] == "probe":
            if response["result"] is not None or response["status"] != "succeeded" or response["error"] is not None or not readiness["ready"]:
                raise _Failure("invalid_output")
            return response
        result = response["result"]
        if (not isinstance(result, dict) or set(result) != {"payload", "provenance", "artifacts", "item_failures"}
                or not isinstance(result["provenance"], dict) or not isinstance(result["item_failures"], list)
                or not isinstance(result["artifacts"], list)):
            raise _Failure("invalid_output")
        seen = set()
        for artifact in result["artifacts"]:
            if (not isinstance(artifact, dict) or set(artifact) != {"path", "original_sha256", "export_sha256", "redacted", "content"}
                    or not isinstance(artifact["path"], str)
                    or not isinstance(artifact["content"], str) or type(artifact["redacted"]) is not bool
                    or not isinstance(artifact["original_sha256"], str) or not _DIGEST.fullmatch(artifact["original_sha256"])
                    or artifact["path"] in seen):
                raise _Failure("invalid_output")
            seen.add(artifact["path"])
            path = _safe_relative(root, artifact["path"])
            original = path.read_bytes()
            exported = artifact["content"].encode("utf-8")
            if (_digest(original) != artifact["original_sha256"] or _digest(exported) != artifact["export_sha256"]
                    or artifact["redacted"] != (original != exported)):
                raise _Failure("digest_mismatch")
            if _redact_text(artifact["content"], secret) != artifact["content"]:
                # Known redaction markers are idempotent; fresh credentials are not.
                raise _Failure("secret_output")
        expected_files = {path.relative_to(root).as_posix() for path in _inventory(root)
                          if not (path.parent == root and path.name in {"request.json", "output.json"})}
        if seen != expected_files:
            raise _Failure("invalid_output")
        payload = result["payload"]
        if payload is not None:
            if __package__:
                from .engine import validate_routes, validate_target_smiles
            else:
                from engine import validate_routes, validate_target_smiles
            try:
                validate_routes(payload)
                identity = validate_target_smiles(job.request["target"])
                for route in payload["routes"]:
                    ordinal = route.get("upstream", {}).get("ordinal")
                    if (type(ordinal) is not int or ordinal < 1
                            or route["id"] != f'{job.request["engine"]}-{job.request["request_id"]}-{ordinal:04d}'
                            or validate_target_smiles(route["target"]) != identity):
                        raise _Failure("invalid_output")
            except (ValueError, TypeError, KeyError, AttributeError):
                raise _Failure("invalid_output") from None
        if response["status"] == "succeeded" and (payload is None or result["item_failures"] or response["error"] is not None):
            raise _Failure("invalid_output")
        if response["status"] == "partial" and response["error"] is None:
            raise _Failure("invalid_output")
        return response
