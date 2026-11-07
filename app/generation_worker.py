"""Thin isolated adapter for original engine.generate_routes and SynthEx CLI.

The parent provides a key-free request file and a separate, allowlisted child
environment. No upstream algorithm, agent, search or stitching is reimplemented.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

# -I deliberately excludes the script directory and all inherited PYTHONPATHs.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from generation_service import (
    KEY_VARIABLES, KEY_VARIABLE_NAMES, MAX_ARTIFACT_BYTES, MODEL_NAMES, _DIGEST, _Failure, _ID,
    _digest, _error, _contains_secret, _inventory, _json_bytes, _llm_selection, _read_json,
    _readiness, _readiness_failure, _redact_text, _safe_relative,
)

STAGES = ("strategies", "routes", "reviewed_routes", "analyses", "stitched_routes")
SYNTHEX_ARTIFACT_IO = {
    "name": "original-store-utf8-lf-1",
    "scope": "SynthEx ArtifactStore root paths only",
    "encoding": "utf-8",
    "newline_translation": False,
    "digest_validation": "unchanged upstream byte digests",
}


class _ArtifactPath(type(Path())):
    """Write the exact UTF-8/LF bytes hashed by the original ArtifactStore.

    Windows text-mode LF translation would otherwise change a checkpoint after
    upstream hashes it. Only store-owned paths use this adapter; pathlib and
    the original hash/checkpoint/read/validation algorithms remain untouched.
    """
    def write_text(self, data, encoding=None, errors=None, newline=None):
        return super().write_text(data, encoding=encoding or "utf-8", errors=errors,
                                  newline="" if newline is None else newline)

    def read_text(self, encoding=None, errors=None):
        return super().read_text(encoding=encoding or "utf-8", errors=errors)


def _utf8_lf_store(store_type, root):
    store = store_type(root)
    store.root = _ArtifactPath(store.root)
    return store


@contextmanager
def _synthex_artifact_io():
    """Scope the platform boundary to this isolated original CLI invocation.

    CLI functions import ArtifactStore from artifacts when called; pipeline
    captures its own alias at import. Adapt both seams and restore both even
    after an exception. No global Path patch or upstream source edit is used.
    """
    import synthex.artifacts as artifacts
    import synthex.pipeline as pipeline

    original_store = artifacts.ArtifactStore
    original_pipeline_store = pipeline.ArtifactStore

    def store_factory(root):
        return _utf8_lf_store(original_store, root)

    artifacts.ArtifactStore = store_factory
    pipeline.ArtifactStore = store_factory
    try:
        yield store_factory
    finally:
        artifacts.ArtifactStore = original_store
        pipeline.ArtifactStore = original_pipeline_store


def _engine():
    return importlib.import_module("engine")


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key_variable(request):
    """Name the one credential variable the selected provider may read."""
    return KEY_VARIABLES[(request.get("llm") or {}).get("provider", "gemini")]


def _validate_request(value):
    # "llm" is the optional nonsecret provider selection. Omitting it keeps the
    # original Gemini-from-config request shape unchanged.
    if (not isinstance(value, dict)
            or set(value) - {"llm"} != {"schema_version", "operation", "engine", "request_id", "context_id", "target", "settings", "configuration_id"}
            or value["schema_version"] != "1.0" or value["operation"] not in {"probe", "generate"}
            or value["engine"] not in {"aizynthfinder", "synthex"}
            or any(not isinstance(value[key], str) or not _ID.fullmatch(value[key]) for key in ("request_id", "context_id"))):
        raise _Failure("invalid_request")
    if "llm" in value:
        if value["engine"] != "synthex" or _llm_selection(value["llm"]) != value["llm"]:
            raise _Failure("invalid_request")
    settings = value["settings"]
    required = {"fingerprint"} if value["engine"] == "aizynthfinder" else {"fingerprint", "config", "config_sha256"}
    if (not isinstance(settings, dict) or set(settings) != required
            or not isinstance(settings["fingerprint"], str) or not _DIGEST.fullmatch(settings["fingerprint"])):
        raise _Failure("invalid_request")
    if value["engine"] == "synthex":
        if (not isinstance(settings["config"], str) or not Path(settings["config"]).is_absolute()
                or not isinstance(settings["config_sha256"], str) or not _DIGEST.fullmatch(settings["config_sha256"])):
            raise _Failure("invalid_request")
    present = KEY_VARIABLE_NAMES & set(os.environ)
    if value["operation"] == "probe":
        if value["target"] is not None or value["configuration_id"] is not None or present:
            raise _Failure("unexpected_key")
    else:
        if not isinstance(value["configuration_id"], str) or not _DIGEST.fullmatch(value["configuration_id"]):
            raise _Failure("invalid_request")
        target = value["target"]
        if not isinstance(target, str) or not target.strip() or len(target) > 4096:
            raise _Failure("target_invalid")
    if value["engine"] == "aizynthfinder" and present:
        raise _Failure("unexpected_key")
    # The selected provider's variable is the only one allowed to be present.
    if present - {_key_variable(value)}:
        raise _Failure("unexpected_key")
    return value


def _existing_model(path):
    path = Path(path)
    if (not path.is_absolute() or not path.is_relative_to(ROOT / "test" / "models")
            or not path.resolve().is_relative_to(ROOT / "test" / "models")):
        raise _Failure("config_invalid")
    if not path.is_file():
        raise _Failure("assets_missing")
    return path


def _planner_assets(path):
    """Require explicit, existing standard file-based upstream policies/stocks.

    The approved assets are the existing test/models files. Dynamic provider
    classes, database stocks, URL downloads and credential-bearing configs are
    not a second hidden setup mechanism.
    """
    import yaml

    path = Path(path)
    if str(path).startswith(("\\\\", "//")):
        raise _Failure("config_invalid")
    if not path.is_absolute() or not path.is_file():
        raise _Failure("assets_missing")
    if path.stat().st_size > 1024 * 1024:
        raise _Failure("config_invalid")
    text = path.read_text(encoding="utf-8")
    if "${" in text or _redact_text(text) != text:
        raise _Failure("config_invalid")
    value = yaml.safe_load(text)
    if (not isinstance(value, dict) or not {"expansion", "filter", "stock"} <= set(value)
            or set(value) - {"expansion", "filter", "stock", "search", "post_processing"}):
        raise _Failure("config_invalid")
    files = [path]
    for section in ("expansion", "filter", "stock"):
        mapping = value[section]
        if not isinstance(mapping, dict) or not mapping:
            raise _Failure("assets_missing")
        for entry in mapping.values():
            if section == "expansion":
                if not isinstance(entry, list) or len(entry) != 2 or any(not isinstance(item, str) for item in entry):
                    raise _Failure("config_invalid")
                files.extend(_existing_model(item) for item in entry)
            else:
                if not isinstance(entry, str):
                    raise _Failure("config_invalid")
                files.append(_existing_model(entry))
    expansions = value["expansion"]
    if "ringbreaker" not in expansions or len(expansions) < 2:
        raise _Failure("assets_missing")
    ringbreaker = expansions["ringbreaker"]
    for name, entry in expansions.items():
        if name != "ringbreaker" and any(_file_digest(ringbreaker[index]) == _file_digest(entry[index]) for index in (0, 1)):
            # Relabeling the ordinary USPTO assets is not full ring-breaker support.
            raise _Failure("assets_missing")
    return files


def _full_config(settings, selection=None):
    from synthex.completion import CompletionBudget
    from synthex.config import AgentConfig, LLMConfig, ManuscriptConfig, load_config, validate_external_assets

    path = Path(settings["config"])
    try:
        data = path.read_bytes()
    except OSError:
        raise _Failure("assets_missing") from None
    if len(data) > 1024 * 1024 or _digest(data) != settings["config_sha256"]:
        raise _Failure("config_changed")
    if b"${" in data or _redact_text(data.decode("utf-8")) != data.decode("utf-8"):
        raise _Failure("config_invalid")
    try:
        config = load_config(path)
    except (ValueError, TypeError):
        raise _Failure("config_invalid") from None
    if selection is not None:
        # Apply only the nonsecret provider/endpoint/model choice. Every other
        # upstream setting still has to equal the original default below.
        data = config.model_dump(mode="python")
        data["llm"].update(provider=selection["provider"], base_url=selection["base_url"],
                           model=selection["model"])
        try:
            config = ManuscriptConfig.model_validate(data)
        except (ValueError, TypeError):
            raise _Failure("config_invalid") from None
    if config.llm.use_vertex_ai:
        raise _Failure("incompatible_backend")
    # Whether it came from disk or from the UI, the effective endpoint must pass
    # the same nonsecret check the parent applies.
    _llm_selection({"provider": config.llm.provider, "base_url": config.llm.base_url,
                    "model": config.llm.model})
    unchecked = {"model", "provider", "base_url"}
    if (config.llm.model_dump(exclude=unchecked) != LLMConfig().model_dump(exclude=unchecked)
            or config.agents.model_dump() != AgentConfig().model_dump()
            or not config.completion.enabled
            or any(getattr(config.completion, name) is not None for name in ("stocks", "expansion_policies", "filter_policies"))
            or config.completion.budget.model_dump() != CompletionBudget().model_dump()):
        raise _Failure("incompatible_backend")
    if config.completion.aizynthfinder_config is None or config.external_asset_manifest is None:
        raise _Failure("assets_missing")
    files = _planner_assets(config.completion.aizynthfinder_config)
    if config.external_asset_manifest is not None:
        from synthex.config import ExternalAssetManifest
        manifest_path = config.external_asset_manifest
        if not manifest_path.is_absolute() or str(manifest_path).startswith(("\\\\", "//")):
            raise _Failure("config_invalid")
        manifest = ExternalAssetManifest.model_validate(_read_json(manifest_path, 1024 * 1024))
        declared = set()
        for asset in manifest.assets:
            asset_path = asset.path if asset.path.is_absolute() else manifest_path.parent / asset.path
            _existing_model(asset_path)
            if _file_digest(asset_path) != asset.sha256:
                raise _Failure("digest_mismatch")
            declared.add(asset_path.resolve())
        if not {file.resolve() for file in files[1:]} <= declared:
            raise _Failure("assets_missing")
    try:
        files.extend(validate_external_assets(config))
    except ValueError:
        raise _Failure("assets_missing") from None
    return config, sorted(set(files))


def _probe(request):
    engine = _engine()
    if request["operation"] == "generate":
        try:
            engine.validate_target_smiles(request["target"])
        except ValueError:
            raise _Failure("target_invalid") from None
    from aizynthfinder.aizynthfinder import AiZynthFinder
    if not callable(AiZynthFinder):
        raise _Failure("package_missing")
    if request["engine"] == "aizynthfinder":
        files = [_existing_model(ROOT / "test" / "models" / name) for name in MODEL_NAMES]
        return {"files": files, "config": None}
    source = ROOT / "synthex" / "src"
    sys.path.insert(0, str(source))
    import synthex.cli
    import synthex.application
    from synthex.integrations.aizynthfinder import AIZYNTHFINDER_VERSION
    if _version("aizynthfinder") != AIZYNTHFINDER_VERSION:
        raise _Failure("incompatible_backend")
    # Imports check dependencies without constructing a client or loading weights.
    config, files = _full_config(request["settings"], request.get("llm"))
    if config.llm.provider == "openai_compatible":
        import httpx  # The adapter's only transport dependency.

        if not hasattr(httpx, "AsyncClient"):
            raise _Failure("package_missing")
    return {"files": files, "config": config}


def _asset_digests(paths):
    return {str(path): _file_digest(path) for path in paths}


def _source_digests(engine):
    paths = {"engine.py": HERE / "engine.py", "generation_worker.py": HERE / "generation_worker.py"}
    if engine == "synthex":
        paths.update({f"synthex/{name}": ROOT / "synthex" / "src" / "synthex" / name
                      for name in ("cli.py", "pipeline.py", "completion.py")})
    return {name: _file_digest(path) for name, path in paths.items()}


def _configuration(request, probe):
    return {"interpreter_config": request["settings"]["fingerprint"],
            "platform_io_adapter": deepcopy(SYNTHEX_ARTIFACT_IO) if request["engine"] == "synthex" else None,
            "effective_config": probe["config"].model_dump(mode="json") if probe["config"] is not None else
                {"time_limit": 30, "iteration_limit": 30, "model_files": list(MODEL_NAMES)},
            "asset_sha256": probe["asset_sha256"], "source_sha256": _source_digests(request["engine"]),
            "versions": {name: _version(name) for name in ("aizynthfinder", "rdkit", "synthex", "google-genai", "pydantic")}}


def _provenance(request, assets):
    return {
        "backend": request["engine"], "request_id": request["request_id"],
        "context_id": request["context_id"], "target": request["target"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "versions": {name: _version(name) for name in ("aizynthfinder", "rdkit", "synthex", "google-genai", "pydantic")},
        "source_sha256": _source_digests(request["engine"]),
        "platform_io_adapter": deepcopy(SYNTHEX_ARTIFACT_IO) if request["engine"] == "synthex" else None,
        "asset_sha256": assets, "live_verified": False,
        "limitations": ["Research-only generation is not experimental or safety validation.",
                        "Dependency presence is not a compatibility or chemical-performance guarantee.",
                        "No new assets, saved-route substitutions, reduced workflow or alternative engine are used."],
    }


def _write_json(path, value):
    data = _json_bytes(value)
    if len(data) > MAX_ARTIFACT_BYTES:
        raise _Failure("artifact_limit")
    path.write_bytes(data)


def _tag_route(route, request, ordinal):
    upstream_id = route["id"]
    route["id"] = f'{request["engine"]}-{request["request_id"]}-{ordinal:04d}'
    upstream = route.setdefault("upstream", {})
    upstream.update(ordinal=ordinal, original_id=upstream_id, request_id=request["request_id"],
                    context_id=request["context_id"], engine=request["engine"])
    return route


def _aizynth(request, root, probe):
    engine = _engine()
    assets = probe["asset_sha256"]
    # Keep this exact original API and the approved bounded search settings.
    try:
        original = engine.generate_routes(request["target"], time_limit=30, iteration_limit=30)
    except engine.NoRoutesError:
        raise _Failure("no_routes") from None
    engine.validate_routes(original)
    native_trees = original.get("native_trees")
    if not isinstance(native_trees, list) or len(native_trees) < len(original["routes"]):
        raise _Failure("invalid_output")
    _write_json(root / "native_routes.json", original)
    if _asset_digests(probe["files"]) != assets:
        raise _Failure("config_changed")
    payload = deepcopy(original)
    failures = []
    seen = set()
    for route in payload["routes"]:
        ordinal = route.get("upstream", {}).get("ordinal")
        if (type(ordinal) is not int or not 1 <= ordinal <= len(native_trees) or ordinal in seen
                or route["upstream"].get("tree") != native_trees[ordinal - 1]):
            raise _Failure("invalid_output")
        seen.add(ordinal)
        _tag_route(route, request, ordinal)
        if route.get("upstream", {}).get("is_solved") is False:
            failures.append({"stage": "search", "ordinal": ordinal, "status": "unsolved"})
    for ordinal, tree in enumerate(native_trees, 1):
        if ordinal not in seen:
            failures.append({"stage": "search", "ordinal": ordinal, "status": "no_reaction_steps", "tree": tree})
    engine.validate_routes(payload)
    provenance = _provenance(request, assets)
    provenance.update(original_call="engine.generate_routes", time_limit=30, iteration_limit=30,
                      original_route_count=len(native_trees),
                      converted_route_count=len(payload["routes"]), live_verified=not failures,
                      generation_config=original.get("generation_config"))
    return {"payload": payload, "provenance": provenance, "item_failures": failures}


def _checkpoint_payloads(root, request, config):
    from synthex.artifacts import ArtifactStore, RunManifest, digest_json
    from synthex.config import ExternalAssetManifest

    run_dir = root / "run"
    _inventory(root)
    manifest_path = _safe_relative(root, "run/manifest.json")
    try:
        parsed = _read_json(manifest_path)
        manifest = RunManifest.model_validate(parsed)
        input_value = {"target": request["target"], "target_id": request["request_id"], "constraint": None}
        external_assets = {"aizynthfinder_config": _file_digest(config.completion.aizynthfinder_config)}
        if config.external_asset_manifest is not None:
            external_assets["asset_manifest"] = _file_digest(config.external_asset_manifest)
            assets = ExternalAssetManifest.model_validate(_read_json(config.external_asset_manifest, 1024 * 1024))
            external_assets.update({f"asset:{asset.identifier}": asset.sha256 for asset in assets.assets})
        if (manifest.target != request["target"] or manifest.target_id != request["request_id"]
                or manifest.constraint is not None
                or manifest.input_sha256 != digest_json(input_value)
                or manifest.run_id != digest_json({"input": input_value, "config": manifest.config})[:16]
                or manifest.config != config.model_dump(mode="json")
                or manifest.external_assets != external_assets
                or manifest.provider != config.llm.provider or manifest.model != config.llm.model
                or tuple(manifest.completed_stages) != STAGES[:len(manifest.completed_stages)]):
            raise _Failure("invalid_output")
        payloads = {}
        for stage, reference in manifest.artifacts.items():
            if (stage not in STAGES or reference.path != f"{stage}.json"
                    or not _DIGEST.fullmatch(reference.sha256)
                    or (reference.parent_sha256 is not None and not _DIGEST.fullmatch(reference.parent_sha256))):
                raise _Failure("path_escape")
            path = _safe_relative(root, f"run/{reference.path}")
            payloads[stage] = _read_json(path)
        for stage_config in manifest.stage_configs.values():
            if stage_config != manifest.config:
                raise _Failure("config_changed")
        try:
            validated = _utf8_lf_store(ArtifactStore, run_dir).validate()
        except ValueError:
            raise _Failure("digest_mismatch") from None
        if validated.model_dump(mode="json") != manifest.model_dump(mode="json"):
            raise _Failure("config_changed")
        if manifest.status == "complete" and tuple(manifest.completed_stages) != STAGES:
            raise _Failure("invalid_output")
        if manifest.status == "complete" and manifest.failure is not None:
            raise _Failure("invalid_output")
        if manifest.status == "complete" and manifest.current_stage != "complete":
            raise _Failure("invalid_output")
        if manifest.status in {"running", "partial"} and manifest.current_stage == "complete":
            raise _Failure("invalid_output")
        if manifest.status == "failed" and manifest.failure is None:
            raise _Failure("invalid_output")
        for digest in [manifest.input_sha256, manifest.config_sha256, *manifest.external_assets.values(),
                       *manifest.prompt_sha256.values(), *manifest.stage_config_sha256.values()]:
            if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
                raise _Failure("digest_mismatch")
        return manifest, payloads
    except _Failure:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OSError):
        raise _Failure("invalid_output") from None


def _synthex_routes(request, root, config):
    """Validate native checkpoints and adapt only final stitched chemistry."""
    from synthex.agents.models import AnalysisResult, Critique, StrategyResult
    from synthex.completion import StitchedRoute
    from synthex.routejson import Route

    engine = _engine()
    manifest, checkpoints = _checkpoint_payloads(root, request, config)
    failures = []
    if manifest.status != "complete":
        failures.append({"stage": manifest.current_stage, "status": manifest.status,
                         "failure": manifest.failure.model_dump(mode="json") if manifest.failure else None})
    expected = None
    if "strategies" in checkpoints:
        strategies = checkpoints["strategies"]
        if (not isinstance(strategies, dict) or set(strategies) != {"strategies", "origin"}
                or strategies["origin"] != "generated"):
            raise _Failure("invalid_output")
        parsed = StrategyResult.model_validate({"strategies": strategies["strategies"]})
        expected = len(parsed.strategies)
        if expected != config.agents.strategy_count:
            failures.append({"stage": "strategies", "status": "unexpected_count", "count": expected})
    for stage in STAGES[1:]:
        if stage not in checkpoints:
            continue
        items = checkpoints[stage]
        if not isinstance(items, list) or expected is None or len(items) != expected:
            raise _Failure("invalid_output")
        for index, item in enumerate(items):
            if not isinstance(item, dict) or item.get("status") not in {"ok", "failed", "review_failed"}:
                raise _Failure("invalid_output")
            status = item["status"]
            if status != "ok":
                if not isinstance(item.get("failure"), dict):
                    raise _Failure("invalid_output")
                failures.append({"stage": stage, "ordinal": index + 1, "status": status, "failure": item["failure"]})
            if stage in {"routes", "reviewed_routes"} and "route" in item:
                route = Route.model_validate(item["route"])
                if engine.validate_target_smiles(route.target) != engine.validate_target_smiles(request["target"]):
                    raise _Failure("invalid_output")
                if item.get("strategy") != checkpoints["strategies"]["strategies"][index]:
                    raise _Failure("invalid_output")
                if stage == "routes":
                    if item.get("termination") not in {"model_finished", "step_limit"}:
                        raise _Failure("invalid_output")
                    if item["termination"] == "step_limit":
                        failures.append({"stage": stage, "ordinal": index + 1, "status": "step_limit"})
                elif status == "ok":
                    critique = Critique.model_validate(item.get("final_critique"))
                    if not critique.is_resolved:
                        failures.append({"stage": stage, "ordinal": index + 1, "status": "critic_unresolved"})
            elif stage in {"routes", "reviewed_routes"} and status == "ok":
                raise _Failure("invalid_output")
            if stage == "analyses" and status == "ok":
                AnalysisResult.model_validate({key: value for key, value in item.items() if key != "status"})
    routes = []
    stitched_items = checkpoints.get("stitched_routes", [])
    for index, item in enumerate(stitched_items):
        if item["status"] != "ok":
            continue
        stitched = StitchedRoute.model_validate({key: value for key, value in item.items() if key != "status"})
        reviewed = checkpoints.get("reviewed_routes", [])[index]
        if "route" not in reviewed or stitched.strategic_route != Route.model_validate(reviewed["route"]):
            raise _Failure("invalid_output")
        if engine.validate_target_smiles(stitched.strategic_route.target) != engine.validate_target_smiles(request["target"]):
            raise _Failure("invalid_output")
        route = engine.from_native_tree(stitched.stitched_tree, f"stitched-{index + 1}", request["target"],
                                        "SynthEx original full workflow / stitched route (실험 타당성 검증 아님)")["routes"][0]
        route["upstream"].update(stitched=item, reviewed=reviewed, manifest_run_id=manifest.run_id,
                                  checkpoint_path=manifest.artifacts["stitched_routes"].path,
                                  checkpoint_sha256=manifest.artifacts["stitched_routes"].sha256)
        _tag_route(route, request, index + 1)
        routes.append(route)
        if not stitched.is_solved:
            failures.append({"stage": "stitched_routes", "ordinal": index + 1, "status": "unsolved"})
        for record in stitched.completions:
            if record.result.status != "solved":
                failures.append({"stage": "completion", "ordinal": index + 1,
                                 "status": record.result.status, "leaf_smiles": record.leaf_smiles,
                                 "result": record.result.model_dump(mode="json")})
    payload = {"schema_version": "1.0", "routes": routes} if routes else None
    if payload is not None:
        engine.validate_routes(payload)
    if not routes:
        failures.append({"stage": "stitched_routes", "status": "no_completed_routes"})
    return payload, manifest, failures


def _with_toxicity_critic():
    """Swap SynthEx's Critic for the one that also runs our rule screen.

    `PipelineComponents` is a frozen dataclass, so the swap builds a new one
    and puts it back on the pipeline. `cli._load_live_pipeline` imports
    `create_pipeline` when it runs, so replacing the module attribute is
    enough and nothing in the SynthEx checkout is edited.

    Without this the engine would run its own Critic, repair the chemistry it
    objects to, and hand back a route whose precursors had never been looked
    at until our screens ran afterwards, which is what the other engine
    already does.
    """
    import dataclasses

    import synthex.application
    from synthex.agents import Critic
    from synthex.agents.models import StepFinding
    from synthex.routejson import render_plan

    import toxicity_critic

    critic_class = toxicity_critic.build(Critic, StepFinding, render_plan)
    original = synthex.application.create_pipeline

    def create_pipeline(config, *, include_completion=True):
        pipeline = original(config, include_completion=include_completion)
        pipeline.components = dataclasses.replace(
            pipeline.components, critic=critic_class())
        return pipeline

    synthex.application.create_pipeline = create_pipeline
    return original


def _synthex(request, root, probe, secret):
    import synthex.cli

    config = probe["config"]
    if not secret or os.environ.get(_key_variable(request)) != secret:
        raise _Failure("missing_key")
    # The original personal-key branch is mandatory; no Vertex/ADC credentials exist.
    if config.llm.use_vertex_ai:
        raise _Failure("incompatible_backend")
    assets = probe["asset_sha256"]
    runtime_path = root / "runtime_config.json"
    _write_json(runtime_path, config.model_dump(mode="json"))
    arguments = ["plan", request["target"], "--config", str(runtime_path),
                 "--output", str(root / "run"), "--target-id", request["request_id"]]
    # No --resume, --env-file, --edit-rounds, partial stage or alternate backend.
    original_pipeline = _with_toxicity_critic()
    try:
        with _synthex_artifact_io():
            exit_code = synthex.cli.main(arguments)
    finally:
        import synthex.application

        synthex.application.create_pipeline = original_pipeline
    if _asset_digests(probe["files"]) != assets:
        raise _Failure("config_changed")
    if _digest(Path(request["settings"]["config"]).read_bytes()) != request["settings"]["config_sha256"]:
        raise _Failure("config_changed")
    payload, manifest, failures = _synthex_routes(request, root, config)
    if exit_code != 0:
        failures.append({"stage": "cli", "status": "failed", "exit_code": exit_code})
    provenance = _provenance(request, assets)
    provenance.update(original_call="synthex.cli.main", cli_command="plan", full_workflow_stages=list(STAGES),
                      edit_iteration_limit=config.agents.edit_iteration_limit,
                      configured_source_sha256=request["settings"]["config_sha256"],
                      manifest=manifest.model_dump(mode="json"), cli_exit_code=exit_code,
                      converted_route_count=len(payload["routes"]) if payload else 0,
                      live_verified=not failures)
    return {"payload": payload, "provenance": provenance, "item_failures": failures}


def _export_artifacts(root, secret):
    artifacts = []
    for path in _inventory(root):
        if path.parent == root and path.name in {"request.json", "output.json"}:
            continue
        original = path.read_bytes()
        if secret and original.strip() in {secret.encode("utf-8"), _json_bytes(secret)}:
            # Do not turn a secret-only artifact into an exported hash of the key.
            raise _Failure("secret_output")
        try:
            content = _redact_text(original.decode("utf-8"), secret)
        except UnicodeError:
            # Unknown binary files are not a covert export channel.
            raise _Failure("secret_output") from None
        exported = content.encode("utf-8")
        artifacts.append({"path": path.relative_to(root).as_posix(),
                          "original_sha256": _digest(original), "export_sha256": _digest(exported),
                          "redacted": original != exported, "content": content})
    return artifacts


def _sanitize(value, secret):
    try:
        return json.loads(_redact_text(_json_bytes(value).decode("utf-8"), secret))
    except (ValueError, TypeError, RecursionError):
        raise _Failure("secret_output") from None


def run_request(request, root):
    """Return a secret-free protocol response. Exceptions never cross this API."""
    root = Path(root)
    response = {"schema_version": "1.0", "request_sha256": None, "request_id": None,
                "context_id": None, "engine": None, "operation": None, "target": None,
                "status": "failed", "readiness": _readiness_failure("invalid_request"),
                "result": None, "error": _error("invalid_request")}
    secret = None
    try:
        request = _validate_request(request)
        for key in ("request_id", "context_id", "engine", "operation", "target"):
            response[key] = request[key]
        response["request_sha256"] = _digest(_json_bytes(request))
        secret = os.environ.get(_key_variable(request)) if request["engine"] == "synthex" else None
        probe = _probe(request)
        probe["asset_sha256"] = _asset_digests(probe["files"])
        configuration = _configuration(request, probe)
        if secret and _contains_secret(_json_bytes(configuration).decode("utf-8"), secret):
            raise _Failure("config_invalid")
        response["readiness"] = _readiness(ready=True, configuration=configuration)
        if request["operation"] == "probe":
            response.update(status="succeeded", error=None)
            return response
        if response["readiness"]["configuration_id"] != request["configuration_id"]:
            raise _Failure("config_changed")
        result = _aizynth(request, root, probe) if request["engine"] == "aizynthfinder" else _synthex(request, root, probe, secret)
        if _configuration(request, probe) != configuration:
            raise _Failure("config_changed")
        result["artifacts"] = _export_artifacts(root, secret)
        result = _sanitize(result, secret)
        partial = bool(result["item_failures"])
        response.update(status="partial" if partial else "succeeded", result=result,
                        error=_error("partial_output") if partial else None)
        response["readiness"] = _readiness("live_unverified" if partial else "ready", ready=True, configuration=configuration)
        response["readiness"].update(live_verified=not partial, dependency_compatibility="executed")
        return response
    except _Failure as exc:
        code = exc.code
    except (ImportError, importlib.metadata.PackageNotFoundError):
        code = "package_missing"
    except FileNotFoundError:
        code = "assets_missing"
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        code = "invalid_output"
    except Exception:
        code = "backend_failed"
    finally:
        secret = None
    ready = response["readiness"] if code == "no_routes" else _readiness_failure(code)
    response.update(status="failed", result=None, readiness=ready, error=_error(code))
    return response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # Parent assigns the process to its owned Job/session before releasing this gate.
    if sys.stdin.buffer.read(1) != b"\n":
        return 2
    try:
        root = args.input.resolve().parent
        if args.input.name != "request.json" or args.output.resolve() != root / "output.json":
            return 2
        _inventory(root)
        request = _read_json(args.input, 64 * 1024)
        response = run_request(request, root)
        data = _json_bytes(response)
        if len(data) > MAX_ARTIFACT_BYTES:
            response.update(status="failed", result=None, readiness=_readiness_failure("artifact_limit"),
                            error=_error("artifact_limit"))
            data = _json_bytes(response)
        temporary = root / "output.tmp"
        temporary.write_bytes(data)
        temporary.replace(args.output)
        return 0
    except Exception:
        # No traceback, raw SDK stderr, request or exception text is written here.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
