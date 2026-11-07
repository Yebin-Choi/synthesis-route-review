"""Software-only tests; these fixtures are not actual route-search evidence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "evaluation" / "scripts")]
import engine
import run_bounded_routes as runner


class BoundedRouteExperimentTests(unittest.TestCase):
    def test_existing_artifact_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            runner.save(path, {"original": True})
            with self.assertRaises(FileExistsError):
                runner.save(path, {"replacement": True})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_worker_uses_fixed_budget_and_saves_all_routes_before_analysis(self):
        routes = engine.load_example()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            engine, "generate_routes", return_value=routes
        ) as generate:
            destination = Path(directory)
            runner.worker({"smiles": "CCO"}, destination)
            generate.assert_called_once_with("CCO", time_limit=30, iteration_limit=30)
            self.assertEqual(json.loads((destination / "routes.json").read_text(encoding="utf-8")), routes)
            outcome = json.loads((destination / "result.json").read_text())
            self.assertEqual(outcome["status"], "success")
            self.assertEqual(outcome["route_count"], len(routes["routes"]))
            policies = json.loads((destination / "policies.json").read_text(encoding="utf-8"))
            for policy in policies.values():
                self.assertEqual(len(policy["routes"]), len(routes["routes"]))

    def test_search_failure_is_saved_without_fabricated_routes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            engine, "generate_routes", side_effect=ValueError("No routes in synthetic fixture")
        ):
            destination = Path(directory)
            runner.worker({"smiles": "CCO"}, destination)
            result = json.loads((destination / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_type"], "ValueError")
            self.assertFalse((destination / "routes.json").exists())

    def test_analysis_failure_preserves_generated_input(self):
        routes = engine.load_example()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            engine, "generate_routes", return_value=routes
        ), patch.object(engine, "analyze_routes", side_effect=ValueError("Synthetic analysis failure")):
            destination = Path(directory)
            runner.worker({"smiles": "CCO"}, destination)
            self.assertTrue((destination / "routes.json").exists())
            self.assertFalse((destination / "report.json").exists())
            self.assertEqual(json.loads((destination / "result.json").read_text())["status"], "failed")

    def test_parent_timeout_is_recorded_and_cannot_be_retried_over_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation" / "runs").mkdir(parents=True)
            with patch.object(runner, "ROOT", root), patch.object(runner, "digest", return_value="fixture-hash"), \
                    patch.object(runner, "version", return_value="fixture-version"), \
                    patch.object(sys, "argv", ["run_bounded_routes.py", "acetaminophen"]), \
                    patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("fixture", 300)) as run:
                with self.assertRaises(SystemExit):
                    runner.main()
                self.assertEqual(run.call_args.kwargs["timeout"], 300)
                outcome = json.loads((root / "evaluation" / "runs" / "bounded_route_acetaminophen" / "process_outcome.json").read_text())
                self.assertTrue(outcome["timeout"])
                self.assertFalse(outcome["result_saved"])
                with self.assertRaises(FileExistsError):
                    runner.main()
                run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
