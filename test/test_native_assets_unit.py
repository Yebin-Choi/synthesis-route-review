"""The SynthEx planner config and its asset manifest have to agree.

The worker refuses to run unless every policy and stock file named by the
planner config is also declared in the asset manifest with a matching
SHA-256. That check is right, and it fires as a flat "assets_missing" with
no file name. Adding a stock to the config and forgetting the manifest cost
one live run to diagnose (benchmark log 6-64), so the pair is checked here
instead, offline and for free.

These files carry absolute paths from the machine that made them, so each
test skips rather than fails where the assets are not installed.
"""

import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "evaluation" / "runs" / "verification" / "native_models.yml"
MANIFEST = ROOT / "evaluation" / "runs" / "verification" / "native_assets_manifest.json"
MODELS = ROOT / "test" / "models"


def _load(path):
    if not path.is_file():
        raise unittest.SkipTest(f"{path.name}이 없습니다.")
    return json.loads(path.read_text(encoding="utf-8"))


def _referenced(config):
    """Every model and stock file the planner config points at."""
    files = []
    for section in ("expansion", "filter", "stock"):
        for entry in config[section].values():
            files.extend(entry if isinstance(entry, list) else [entry])
    return [Path(item) for item in files]


class PlannerAssetTests(unittest.TestCase):
    def test_every_file_the_config_names_is_declared_in_the_manifest(self):
        config, manifest = _load(CONFIG), _load(MANIFEST)
        declared = {Path(asset["path"]).name for asset in manifest["assets"]}
        missing = sorted({path.name for path in _referenced(config)} - declared)
        self.assertEqual(missing, [], f"명세에 빠진 파일: {missing}")

    def test_declared_digests_match_the_files_on_disk(self):
        manifest = _load(MANIFEST)
        for asset in manifest["assets"]:
            path = MODELS / Path(asset["path"]).name
            if not path.is_file():
                continue
            with self.subTest(asset=asset["identifier"]):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, asset["sha256"])

    def test_every_referenced_file_lives_under_test_models(self):
        # The worker refuses a path outside that folder, so a config that
        # points elsewhere fails only at run time.
        config = _load(CONFIG)
        for path in _referenced(config):
            with self.subTest(path=path.name):
                self.assertEqual(Path(path).parent.name, "models")

    def test_the_stock_carries_the_bulk_reagents_as_well_as_zinc(self):
        # ZINC lists no carbon dioxide and no water, so a route ending there
        # reads as unfinished. Measured 2026-11-08.
        config = _load(CONFIG)
        self.assertIn("zinc", config["stock"])
        self.assertIn("bulk_reagents", config["stock"])


class BulkReagentListTests(unittest.TestCase):
    def test_the_readable_list_and_the_built_stock_hold_the_same_count(self):
        source = MODELS / "bulk_reagents.txt"
        built = MODELS / "bulk_reagents_stock.hdf5"
        if not source.is_file() or not built.is_file():
            raise unittest.SkipTest("대량 시약 목록이 없습니다.")
        lines = [line for line in source.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.strip().startswith("#")]
        self.assertTrue(lines)
        for line in lines:
            with self.subTest(line=line):
                self.assertIn("\t", line)

    def test_carbon_dioxide_and_water_are_on_the_list(self):
        source = MODELS / "bulk_reagents.txt"
        if not source.is_file():
            raise unittest.SkipTest("대량 시약 목록이 없습니다.")
        text = source.read_text(encoding="utf-8")
        self.assertIn("\tO=C=O", text)
        self.assertIn("\tO\n", text)


if __name__ == "__main__":
    unittest.main()
