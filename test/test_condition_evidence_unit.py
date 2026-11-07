"""Software-only fixtures, not chemical experiments or empirical model validation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "app", ROOT / "evaluation" / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_recollected_kb import SOURCE_SPECS, build_evidence, group_records, wilson_lower_bound
from condition_evidence import get_combinations, list_sources, load_evidence
from toxicology import canonicalize_smiles


DOYLE_FILE = "Dreher_and_Doyle_input_data.xlsx"
SUZUKI_FILE = "aap9112_Data_File_S1.xlsx"


def doyle_row(**overrides):
    # Arbitrary parser inputs; these structures are not a proposed condition set.
    row = {"Ligand": "OC", "Additive": "O", "Base": "CCN",
           "Aryl halide": "Brc1ccccc1", "Output": 20}
    row.update(overrides)
    return row


def suzuki_row(**overrides):
    # All fixture names are intentionally unresolved, including SMILES-like CCO.
    row = {
        "Reaction_No": 1, "Reactant_1_Name": "fixture substrate A",
        "Reactant_1_Short_Hand": "fixture A", "Reactant_1_eq": 1,
        "Reactant_1_mmol": 0.001, "Reactant_2_Name": "fixture partner X",
        "Reactant_2_eq": 1.5, "Catalyst_1_Short_Hand": "fixture catalyst",
        "Catalyst_1_eq": 0.05, "Ligand_Short_Hand": "CCO", "Ligand_eq": 0.1,
        "Reagent_1_Short_Hand": "fixture reagent", "Reagent_1_eq": 2,
        "Solvent_1_Short_Hand": "fixture solvent", "Product_Yield_PCT_Area_UV": 25,
        "Product_Yield_Mass_Ion_Count": 12345,
    }
    row.update(overrides)
    return row


class WorkbookFixtureCase(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "derived"

    def tearDown(self):
        self.temporary.cleanup()

    def write_sources(self, *, doyle=None, suzuki=None, extra_doyle=None, headers_override=None):
        raw_dir = self.root / "raw"
        raw_dir.mkdir(exist_ok=True)
        entries = []
        for filename, data in ((DOYLE_FILE, doyle), (SUZUKI_FILE, suzuki)):
            if data is None:
                continue
            spec = SOURCE_SPECS[filename]
            headers = list(spec["required_headers"])
            if headers_override is not None and filename in headers_override:
                headers = headers_override[filename]
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = spec["selected_sheet"]
            sheet.append(headers)
            for row in data:
                sheet.append([row.get(field) for field in headers])
            if filename == DOYLE_FILE:
                for title, other_rows in (extra_doyle or {}).items():
                    other = workbook.create_sheet(title)
                    other.append(headers)
                    for row in other_rows:
                        other.append([row.get(field) for field in headers])
            path = raw_dir / filename
            workbook.save(path)
            workbook.close()
            entries.append({
                "file": f"raw/{filename}",
                "url": f"https://example.invalid/software-only-fixture/{filename}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
            })
        manifest = self.root / "sources.json"
        manifest.write_text(json.dumps({"sources": entries}), encoding="utf-8")
        return manifest

    def collect(self, **kwargs):
        manifest = self.write_sources(**kwargs)
        kb = build_evidence(manifest, self.output, source_root=self.root)
        audit = json.loads((self.output / "audit.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (self.output / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        return kb, audit, records


class RecollectionBuilderTests(WorkbookFixtureCase):
    def test_only_one_cv_sheet_is_collected_and_original_rows_survive(self):
        kb, audit, records = self.collect(
            doyle=[doyle_row(), doyle_row(**{"Ligand": "CO", "Aryl halide": "Clc1ccccc1", "Output": 40})],
            extra_doyle={
                "FullCV_02": [doyle_row(), doyle_row(**{"Output": 999})],
                "Test1": [doyle_row()],
            },
        )
        self.assertEqual(len(records), 2)
        self.assertEqual([record["worksheet_row"] for record in records], [2, 3])
        self.assertEqual(records[0]["source_record_identifier"], {"field": "worksheet_row", "value": 2})
        self.assertEqual({record["sheet"] for record in records}, {"FullCV_01"})
        self.assertEqual(audit["exclusions"], [])
        summary = audit["summary"]["sources"][0]
        self.assertEqual(summary["selected_data_rows_seen"], 2)
        self.assertEqual({item["sheet"] for item in summary["ignored_sheets"]}, {"FullCV_02", "Test1"})
        group, = kb["combinations"]
        self.assertEqual(group["n"], 2)
        ligand = next(component for component in group["components"] if component["role"] == "ligand")
        self.assertEqual(ligand["canonical_smiles"], canonicalize_smiles("OC"))
        self.assertEqual(ligand["input_smiles_variants"], ["CO", "OC"])
        self.assertEqual(len(group["substrate_contexts"]), 2)
        self.assertEqual(group["measurement"]["median"], 30)
        self.assertEqual(group["measurement_basis"], "reported_Output_percent_basis_unverified")
        self.assertFalse(group["measurement"]["calibrated_yield_established"])
        for role in ("catalyst", "solvent"):
            component = next(component for component in group["components"] if component["role"] == role)
            self.assertEqual(component["status"], "missing")
            self.assertIsNone(component["canonical_smiles"])
            self.assertEqual(group["missing_fields"][role], 2)
        self.assertEqual(group["temperature"]["status"], "unknown")

    def test_invalid_structures_and_numeric_rows_are_quarantined_without_clamping(self):
        rows = [
            doyle_row(**{"Ligand": "CC("}),
            doyle_row(**{"Output": -0.1}), doyle_row(**{"Output": 100.1}),
            doyle_row(**{"Output": "NaN"}), doyle_row(**{"Output": "Infinity"}),
            doyle_row(**{"Output": True}), doyle_row(**{"Output": None}),
            doyle_row(**{"Output": 10}),
        ]
        kb, audit, records = self.collect(doyle=rows)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(audit["exclusions"]), 6)
        counts = audit["exclusion_reason_counts"]
        self.assertEqual(counts["invalid_smiles"], 1)
        self.assertEqual(counts["percentage_out_of_range"], 2)
        self.assertEqual(counts["nonfinite_number"], 2)
        self.assertEqual(counts["invalid_number"], 1)
        self.assertEqual(audit["exclusions"][2]["raw"]["Output"], 100.1)
        self.assertEqual(audit["exclusions"][0]["raw"]["Ligand"], "CC(")
        self.assertEqual(len(audit["metric_exclusions"]), 1)
        group, = kb["combinations"]
        self.assertEqual(group["n"], 2)
        self.assertEqual(group["measurement"]["n_valid"], 1)
        self.assertEqual(group["measurement"]["n_missing"], 1)
        self.assertEqual(group["measurement"]["median"], 10)
        self.assertEqual(group["research_threshold"]["count"], 1)
        self.assertEqual(group["missing_fields"]["Output"], 1)

    def test_suzuki_preserves_whole_condition_tuple_amounts_and_paired_context(self):
        rows = [
            suzuki_row(),
            suzuki_row(Reaction_No=2, Solvent_1_Short_Hand="fixture solvent B"),
            suzuki_row(Reaction_No=3, Ligand_Short_Hand="fixture ligand B"),
            suzuki_row(Reaction_No=4, Catalyst_1_eq=0.2),
            suzuki_row(Reaction_No=5, Reactant_1_Name="fixture substrate B",
                       Reactant_1_Short_Hand="fixture B", Reactant_2_Name="fixture partner Y"),
        ]
        kb, audit, records = self.collect(suzuki=rows)
        self.assertEqual(audit["exclusions"], [])
        self.assertEqual(len(kb["combinations"]), 4)
        group, = [group for group in kb["combinations"] if group["n"] == 2]
        self.assertEqual(group["measurement_basis"], "UV_area_percentage")
        self.assertEqual(group["measurement"]["unit"], "UV_area_percent")
        self.assertEqual(group["measurement"]["n_valid"], 2)
        self.assertEqual(group["measurement"]["median"], 25)
        components = {component["role"]: component for component in group["components"]}
        self.assertEqual(set(components), {"catalyst", "ligand", "reagent", "solvent"})
        self.assertEqual(components["catalyst"]["equivalents"], 0.05)
        self.assertEqual(components["reagent"]["equivalents"], 2)
        self.assertEqual(components["ligand"]["name"], "CCO")
        for component in components.values():
            self.assertEqual(component["status"], "unresolved")
            self.assertIsNone(component["canonical_smiles"])
        context_pairs = {
            tuple(component["name"] for component in context["components"])
            for context in group["substrate_contexts"]
        }
        self.assertEqual(context_pairs, {("fixture substrate A", "fixture partner X"),
                                        ("fixture substrate B", "fixture partner Y")})
        first_context = records[0]["substrate_context"][0]
        self.assertEqual(first_context["mmol"], 0.001)
        self.assertEqual(first_context["shorthand"], "fixture A")
        self.assertEqual(records[0]["source_record_identifier"], {"field": "Reaction_No", "value": 1})
        self.assertEqual(records[0]["raw"]["Product_Yield_Mass_Ion_Count"], 12345)
        self.assertNotIn("Product_Yield_Mass_Ion_Count", group["measurement"])
        self.assertEqual(group["unresolved_components"]["count"], 12)
        self.assertEqual(group["unresolved_components"]["by_role"]["reactant_1"], 2)

    def test_explicit_absent_ligand_is_not_a_missing_or_invalid_structure(self):
        kb, audit, records = self.collect(suzuki=[
            suzuki_row(Ligand_Short_Hand="None", Ligand_eq=None, Catalyst_1_Short_Hand=None),
            suzuki_row(Reaction_No=2, Ligand_Short_Hand=None, Ligand_eq=None),
            suzuki_row(Reaction_No=3, Ligand_Short_Hand=123),
        ])
        self.assertEqual(len(records), 2)
        self.assertEqual(audit["exclusion_reason_counts"], {"invalid_name": 1})
        absent = next(component for component in records[0]["components"] if component["role"] == "ligand")
        missing = next(component for component in records[1]["components"] if component["role"] == "ligand")
        self.assertEqual(absent["status"], "absent")
        self.assertEqual(absent["name"], "None")
        self.assertIsNone(absent["canonical_smiles"])
        self.assertNotIn("Ligand_eq", records[0]["missing_fields"])
        self.assertNotIn("Ligand_Short_Hand", records[0]["missing_fields"])
        self.assertIn("Catalyst_1_Short_Hand", records[0]["missing_fields"])
        self.assertEqual(missing["status"], "missing")
        self.assertIn("Ligand_eq", records[1]["missing_fields"])
        self.assertIn("Ligand_Short_Hand", records[1]["missing_fields"])
        self.assertEqual(len(kb["combinations"]), 2)

    def test_nonfinite_amount_and_mass_count_are_not_accepted_as_raw_numeric_evidence(self):
        _, audit, records = self.collect(suzuki=[
            suzuki_row(Catalyst_1_eq="NaN"),
            suzuki_row(Reaction_No=2, Product_Yield_Mass_Ion_Count="Infinity"),
            suzuki_row(Reaction_No=3, Product_Yield_Mass_Ion_Count=1000000),
        ])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["raw"]["Product_Yield_Mass_Ion_Count"], 1000000)
        self.assertEqual(audit["exclusion_reason_counts"], {"nonfinite_number": 2})

    def test_duplicate_original_reaction_identifier_is_quarantined_not_double_counted(self):
        kb, audit, records = self.collect(suzuki=[suzuki_row(), suzuki_row(Product_Yield_PCT_Area_UV=90)])
        self.assertEqual(len(records), 1)
        self.assertEqual(audit["exclusion_reason_counts"], {"duplicate_source_record_id": 1})
        self.assertEqual(kb["combinations"][0]["measurement"]["median"], 25)

    def test_groups_never_mix_sources_or_measurement_bases(self):
        _, _, records = self.collect(doyle=[doyle_row()])
        original = records[0]
        other_source = deepcopy(original)
        other_source.update(id="software-other-source", source_id="software-other-source")
        other_basis = deepcopy(original)
        other_basis.update(id="software-other-basis", measurement_basis="software-other-basis")
        other_basis["measurement"]["basis"] = "software-other-basis"
        other_basis["measurement"]["value"] = 80
        groups = group_records([original, other_source, other_basis])
        self.assertEqual(len(groups), 3)
        self.assertEqual({(group["source_id"], group["measurement_basis"]) for group in groups}, {
            ("dreher_doyle", "reported_Output_percent_basis_unverified"),
            ("software-other-source", "reported_Output_percent_basis_unverified"),
            ("dreher_doyle", "software-other-basis"),
        })
        self.assertEqual([group["n"] for group in groups], [1, 1, 1])

    def test_missing_measurement_has_no_fabricated_median_or_wilson_endpoint(self):
        kb, audit, records = self.collect(doyle=[doyle_row(**{"Output": None})])
        group, = kb["combinations"]
        self.assertEqual(group["n"], 1)
        self.assertEqual(group["measurement"]["n_valid"], 0)
        for field in ("median", "minimum", "maximum"):
            self.assertIsNone(group["measurement"][field])
        self.assertEqual(group["research_threshold"]["count"], 0)
        self.assertIsNone(group["research_threshold"]["wilson_lower_bound"])
        self.assertEqual(records[0]["measurement"]["status"], "missing")
        self.assertEqual(len(audit["metric_exclusions"]), 1)
        self.assertEqual(audit["summary"]["missing_fields"]["Output"], 1)

    def test_threshold_boundary_counts_only_observed_values(self):
        kb, _, _ = self.collect(doyle=[doyle_row(**{"Output": value}) for value in (9.99, 10, 100, None)])
        group, = kb["combinations"]
        threshold = group["research_threshold"]
        self.assertEqual(threshold["count"], 2)
        self.assertEqual(threshold["n_valid"], 3)
        self.assertEqual(group["measurement"]["median"], 10)
        self.assertAlmostEqual(threshold["wilson_lower_bound"], wilson_lower_bound(2, 3))
        self.assertIn("not_prediction", threshold["label"])
        self.assertIn("p=count/n", threshold["formula"])

    def test_hashes_are_mandatory_and_mismatches_fail_before_writing(self):
        manifest = self.write_sources(doyle=[doyle_row()])
        original = json.loads(manifest.read_text(encoding="utf-8"))
        raw_path = self.root / original["sources"][0]["file"]
        raw_bytes = raw_path.read_bytes()
        for invalid_sha in (None, "not-a-hash", "0" * 64):
            with self.subTest(sha=invalid_sha):
                altered = deepcopy(original)
                if invalid_sha is None:
                    del altered["sources"][0]["sha256"]
                else:
                    altered["sources"][0]["sha256"] = invalid_sha
                manifest.write_text(json.dumps(altered), encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_evidence(manifest, self.output, source_root=self.root)
                self.assertFalse(self.output.exists())
                self.assertEqual(raw_path.read_bytes(), raw_bytes)

    def test_missing_required_header_fails_instead_of_guessing_a_column(self):
        manifest = self.write_sources(doyle=[doyle_row()], headers_override={
            DOYLE_FILE: ["Ligand", "Additive", "Base", "Aryl halide", "Unverified other metric"],
        })
        with self.assertRaisesRegex(ValueError, "Missing source headers"):
            build_evidence(manifest, self.output, source_root=self.root)
        self.assertFalse(self.output.exists())

    def test_rebuild_is_byte_reproducible_and_keeps_raw_file_hashes(self):
        manifest = self.write_sources(doyle=[doyle_row()], suzuki=[suzuki_row()])
        entries = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
        kb = build_evidence(manifest, self.output, source_root=self.root)
        before = {name: (self.output / name).read_bytes()
                  for name in ("condition_evidence.json", "audit.json", "records.jsonl")}
        build_evidence(manifest, self.output, source_root=self.root)
        after = {name: (self.output / name).read_bytes() for name in before}
        self.assertEqual(before, after)
        sources = {source["file"]: source for source in kb["sources"]}
        for entry in entries:
            self.assertEqual(hashlib.sha256((self.root / entry["file"]).read_bytes()).hexdigest(), entry["sha256"])
            self.assertEqual(sources[entry["file"]]["sha256"], entry["sha256"])
            self.assertEqual(sources[entry["file"]]["url"], entry["url"])


class WilsonArithmeticTests(unittest.TestCase):
    def test_known_mathematical_examples_and_empty_denominator(self):
        self.assertAlmostEqual(wilson_lower_bound(5, 10), 0.2365895936154873)
        self.assertAlmostEqual(wilson_lower_bound(10, 10), 0.7224598312333834)
        self.assertAlmostEqual(wilson_lower_bound(0, 10), 0)
        self.assertIsNone(wilson_lower_bound(0, 0))
        for count, n in ((1, 0), (-1, 10), (11, 10), (1.0, 10), (1, True), (False, 2)):
            with self.subTest(count=count, n=n), self.assertRaises(ValueError):
                wilson_lower_bound(count, n)


class EvidenceInspectionTests(WorkbookFixtureCase):
    def test_queries_are_exact_source_ordered_copies_not_score_rankings(self):
        kb, _, _ = self.collect(
            doyle=[doyle_row(), doyle_row(**{"Base": "CN", "Output": 90})],
            suzuki=[suzuki_row()],
        )
        loaded = load_evidence(self.output / "condition_evidence.json")
        loaded["sources"].reverse()
        loaded["combinations"].reverse()
        sources = list_sources(loaded)
        self.assertEqual([source["id"] for source in sources], ["dreher_doyle", "suzuki_aap9112"])
        groups = get_combinations(loaded, "Buchwald-Hartwig", limit=2)
        self.assertEqual([group["id"] for group in groups], sorted(group["id"] for group in groups))
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(get_combinations(loaded, "Buchwald-Hartwig", limit=1)), 1)
        for group in groups:
            self.assertEqual(group["source"]["id"], "dreher_doyle")
            self.assertEqual(group["source"]["selected_sheet"], "FullCV_01")
            self.assertEqual(len(group["source"]["sha256"]), 64)
        groups[0]["components"][0]["name"] = "mutated inspection copy"
        sources[0]["url"] = "mutated source copy"
        self.assertEqual(get_combinations(loaded, "Buchwald-Hartwig", limit=2)[0]["components"][0]["name"], None)
        self.assertNotEqual(list_sources(loaded)[0]["url"], "mutated source copy")
        self.assertEqual(get_combinations(kb, "CCO>>CC=O"), [])
        self.assertEqual(get_combinations(kb, "buchwald-hartwig"), [])
        self.assertEqual(get_combinations(kb, "Suzuki-Miyaura", limit=0), [])
        for limit in (-1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                get_combinations(kb, "Suzuki-Miyaura", limit=limit)
        with self.assertRaises(ValueError):
            get_combinations(kb, None)

    def test_loading_never_substitutes_generated_or_unprovenanced_evidence(self):
        missing = self.output / "missing.json"
        with self.assertRaises(FileNotFoundError):
            load_evidence(missing)
        kb, _, _ = self.collect(doyle=[doyle_row()])
        malformed = self.output / "malformed.json"
        malformed.write_text("not json", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            load_evidence(malformed)
        invalid_variants = []
        wrong_schema = deepcopy(kb)
        wrong_schema["schema"] = "claimed-team-release"
        invalid_variants.append(wrong_schema)
        no_hash = deepcopy(kb)
        del no_hash["sources"][0]["sha256"]
        invalid_variants.append(no_hash)
        unknown_source = deepcopy(kb)
        unknown_source["combinations"][0]["source_id"] = "unattributed"
        invalid_variants.append(unknown_source)
        mixed_basis = deepcopy(kb)
        mixed_basis["combinations"][0]["measurement_basis"] = "UV_area_percentage"
        invalid_variants.append(mixed_basis)
        for index, variant in enumerate(invalid_variants):
            with self.subTest(variant=index):
                malformed.write_text(json.dumps(variant), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_evidence(malformed)


if __name__ == "__main__":
    unittest.main()
