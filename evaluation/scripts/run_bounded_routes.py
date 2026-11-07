"""Two preregistered local searches; no toxicity model or network calls.

Run once per target with the main aizynth-env interpreter:
  python evaluation/scripts/run_bounded_routes.py acetaminophen
  python evaluation/scripts/run_bounded_routes.py ibuprofen
Existing experiment directories are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
TARGETS = {"acetaminophen": "pubchem-1983", "ibuprofen": "pubchem-3672"}


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def worker(target, destination):
    import engine
    from route_selection import select_routes
    started = time.perf_counter()
    try:
        routes = engine.generate_routes(target["smiles"], time_limit=30, iteration_limit=30)
        save(destination / "routes.json", routes)
        report = engine.analyze_routes(routes)
        save(destination / "report.json", report)
        save(destination / "policies.json", {
            "default": select_routes(report), "strict": select_routes(report, strict_alerts=True)})
        result = {"status": "success", "route_count": len(routes["routes"])}
    except Exception as exc:
        result = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    result["elapsed_worker_seconds"] = time.perf_counter() - started
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save(destination / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=TARGETS)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    from molecule_catalog import MOLECULES
    from toxicology import canonicalize_smiles
    target = next(row for row in MOLECULES if row["id"] == TARGETS[args.target])
    destination = ROOT / "evaluation" / "runs" / ("bounded_route_" + args.target)
    if args.worker:
        worker(target, destination)
        return
    destination.mkdir(exist_ok=False)
    models = ROOT / "test" / "models"
    resources = ("uspto_model.onnx", "uspto_templates.csv.gz", "uspto_filter_model.onnx", "zinc_stock.hdf5")
    save(destination / "preregistration.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "target": target,
        "canonical_smiles": canonicalize_smiles(target["smiles"]),
        "time_limit_seconds": 30, "iteration_limit": 30, "worker_timeout_seconds": 300,
        "aizynthfinder_version": version("aizynthfinder"),
        "runner_sha256": digest(Path(__file__)),
        "resource_sha256": {name: digest(models / name) for name in resources},
        "limitations": ["Search time excludes model loading; worker wall time includes it.",
                        "No random seed is imposed; identical routes are not guaranteed.",
                        "Zero routes or errors are retained, not replaced by examples.",
                        "Not SynthEx generation or experimental chemical validation.",
                        "Ibuprofen stereochemistry remains unspecified in the catalog input.",
                        "Conditions, byproduct formation, yields and stoichiometry remain unknown."]})
    command = [sys.executable, str(Path(__file__).resolve()), args.target, "--worker"]
    with (destination / "worker.stdout.log").open("x", encoding="utf-8") as stdout, \
            (destination / "worker.stderr.log").open("x", encoding="utf-8") as stderr:
        try:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=300, check=False)
            outcome = {"returncode": result.returncode, "timeout": False}
        except subprocess.TimeoutExpired:
            outcome = {"returncode": None, "timeout": True, "error": "Worker exceeded 300 seconds"}
    outcome["result_saved"] = (destination / "result.json").is_file()
    save(destination / "process_outcome.json", outcome)
    print(json.dumps({"directory": str(destination), **outcome}, ensure_ascii=False))
    if outcome["timeout"] or outcome["returncode"] != 0 or not outcome["result_saved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
