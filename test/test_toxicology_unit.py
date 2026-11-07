"""Focused unittest coverage for conservative research screening contracts."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "app", ROOT / "test"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import env_support
from toxicology import assess_material, canonicalize_smiles, concentration_limit


class MaterialAssessmentTests(unittest.TestCase):
    def test_public_canonicalization_is_shared_by_assessment_and_lookup(self):
        mapped = "[Cl:1][CH2:2][CH:3]1[CH2:4][O:5]1"
        self.assertEqual(canonicalize_smiles(mapped), canonicalize_smiles("ClCC1CO1"))
        self.assertEqual(canonicalize_smiles("CO"), canonicalize_smiles("OC"))
        self.assertEqual(assess_material(mapped)["canonical_smiles"], canonicalize_smiles(mapped))
        self.assertEqual(assess_material(mapped)["name"], "Epichlorohydrin")
        self.assertEqual(canonicalize_smiles(" CCO "), canonicalize_smiles("CCO"))
        self.assertIsNone(canonicalize_smiles(""))
        self.assertIsNone(canonicalize_smiles("CC("))

    def test_exact_public_contract_for_all_statuses(self):
        keys = {
            "input_smiles", "canonical_smiles", "status", "name", "alerts",
            "special_concern", "intake_ug_day", "intake_route_note", "intake_basis",
            "source_url", "limitations",
        }
        for smiles in ("ClCC1CO1", "C1CO1", "CCO", "bad-smiles"):
            with self.subTest(smiles=smiles):
                result = assess_material(smiles)
                self.assertEqual(set(result), keys)
                self.assertEqual(result["input_smiles"], smiles)
                self.assertIsInstance(result["alerts"], list)
                self.assertIsInstance(result["limitations"], list)
                self.assertIsInstance(result["special_concern"], bool)

    def test_atom_maps_do_not_change_known_matching_or_alerts(self):
        pairs = (
            ("ClCC1CO1", "[Cl:1][CH2:2][CH:3]1[CH2:4][O:5]1"),
            ("CN(C)N=O", "[CH3:1][N:2]([CH3:3])[N:4]=[O:5]"),
        )
        for plain, mapped in pairs:
            with self.subTest(smiles=plain):
                expected = assess_material(plain)
                actual = assess_material(mapped)
                for key in ("canonical_smiles", "status", "name", "alerts",
                            "special_concern", "intake_ug_day"):
                    self.assertEqual(actual[key], expected[key])

    def test_canonicalization_preserves_stereochemistry(self):
        left = assess_material("ClC[C@H]1CO1")
        right = assess_material("ClC[C@@H]1CO1")
        mapped = assess_material("[Cl:1][CH2:2][C@H:3]1[CH2:4][O:5]1")
        self.assertNotEqual(left["canonical_smiles"], right["canonical_smiles"])
        self.assertIn("@", left["canonical_smiles"])
        self.assertEqual(left["canonical_smiles"], mapped["canonical_smiles"])
        self.assertIsNone(left["intake_ug_day"])

    def test_canonicalization_does_not_merge_charge_tautomer_or_isotope_forms(self):
        for left, right in (("OCC1CO1", "[O-]CC1CO1"),
                            ("CC=O", "C=CO"), ("CCO", "[13CH3]CO")):
            with self.subTest(left=left, right=right):
                self.assertNotEqual(canonicalize_smiles(left), canonicalize_smiles(right))
        self.assertNotEqual(canonicalize_smiles("CCO.[Na+]"), canonicalize_smiles("CCO"))

    def test_invalid_and_empty_inputs_do_not_become_unflagged_materials(self):
        for smiles in ("", " ", ".", "CC(", "not-smiles", "CCO trailing-name",
                       "CCO\x00junk", "*", "C*", "[*:1]C", None):
            with self.subTest(smiles=smiles):
                result = assess_material(smiles)
                self.assertEqual(result["status"], "invalid")
                self.assertIsNone(result["canonical_smiles"])
                self.assertIsNone(result["name"])
                self.assertIsNone(result["intake_ug_day"])

    def test_no_alert_is_explicitly_not_a_safety_or_class_5_claim(self):
        for smiles in ("CCO", "c1ccccc1"):
            with self.subTest(smiles=smiles):
                result = assess_material(smiles)
                self.assertEqual(result["status"], "no_alert_detected")
                self.assertIsNone(result["intake_ug_day"])
                wording = " ".join(result["limitations"]).lower()
                self.assertIn("does not mean safe or class 5", wording)
                self.assertIn("not evidence of safety or class 5", result["intake_basis"].lower())

    def test_ndma_and_other_detected_special_concern_never_receive_ttc(self):
        for smiles in ("CN(C)N=O", "C[N+](C)=N[O-]", "CN=[N+]([O-])C"):
            with self.subTest(smiles=smiles):
                result = assess_material(smiles)
                self.assertEqual(result["status"], "structural_alert")
                self.assertTrue(result["special_concern"])
                self.assertIsNone(result["intake_ug_day"])
                self.assertIn("No TTC fallback", result["intake_basis"])
                self.assertIsNone(concentration_limit(result["intake_ug_day"], 320))

    def test_epichlorohydrin_reference_and_ppm_units(self):
        result = assess_material("ClCC1CO1")
        self.assertEqual(result["name"], "Epichlorohydrin")
        self.assertEqual(result["intake_ug_day"], 3.0)
        self.assertEqual(result["source_url"], "https://www.fda.gov/media/170459/download#page=18")
        self.assertIn("ICH M7(R2) Addendum", result["intake_basis"])
        self.assertIn("lifetime oral impurity reference", result["intake_basis"])
        self.assertIn("not an API dose", " ".join(result["limitations"]))
        self.assertAlmostEqual(concentration_limit(result["intake_ug_day"], 320), 9.375)

    # Expected intakes are the ICH M7(R2) Addendum summary table figures for the
    # oral or "all other routes" column (FDA copy, printed pages 12-14), not
    # values read back out of the screen. Entries the table splits by route must
    # carry a note so the stored figure is never read as covering every route.
    ADDENDUM_TABLE = (
        ("C=CC#N", "Acrylonitrile", 6.0, False),
        ("ClCc1ccccc1", "Benzyl chloride", 41.0, False),
        ("ClCOCCl", "Bis(chloromethyl)ether", 0.004, False),
        ("O=[N+]([O-])c1ccc(Cl)cc1", "1-Chloro-4-nitrobenzene", 117.0, False),
        ("BrCCBr", "1,2-Dibromoethane", 2.0, False),
        ("CN(C)C(=O)Cl", "Dimethylcarbamyl chloride", 5.0, True),
        ("ClCC1CO1", "Epichlorohydrin", 3.0, False),
        ("CCBr", "Ethyl bromide", 32.0, False),
        ("CCCl", "Ethyl chloride", 1810.0, False),
        ("OCC1CO1", "Glycidol", 4.0, False),
        ("NN", "Hydrazine", 39.0, True),
        ("CCl", "Methyl chloride", 1361.0, False),
        ("C=Cc1ccccc1", "Styrene", 154.0, False),
        ("Nc1ccccc1", "Aniline", 720.0, True),
        ("CC=O", "Acetaldehyde", 2000.0, True),
        ("C=O", "Formaldehyde", 10000.0, True),
        ("C=COC(C)=O", "Vinyl acetate", 2000.0, True),
        ("Nc1ccc(Cl)cc1", "p-Chloroaniline", 34.0, True),
    )

    def test_every_listed_material_reports_its_addendum_intake(self):
        for smiles, name, intake, has_note in self.ADDENDUM_TABLE:
            with self.subTest(smiles=smiles):
                result = assess_material(smiles)
                self.assertEqual(result["status"], "known_concern")
                self.assertEqual(result["name"], name)
                self.assertEqual(result["intake_ug_day"], intake)
                self.assertIn("ICH M7(R2) Addendum", result["intake_basis"])
                self.assertEqual(
                    result["source_url"],
                    "https://www.fda.gov/media/170459/download#page=18",
                )
                if has_note:
                    self.assertIsInstance(result["intake_route_note"], str)
                    self.assertIn(result["intake_route_note"], result["intake_basis"])
                else:
                    self.assertIsNone(result["intake_route_note"])

    def test_route_split_entries_name_the_route_the_stored_figure_covers(self):
        hydrazine = assess_material("NN")
        self.assertEqual(hydrazine["intake_ug_day"], 39.0)
        self.assertIn("inhalation", hydrazine["intake_route_note"])
        acetaldehyde = assess_material("CC=O")
        self.assertEqual(acetaldehyde["intake_ug_day"], 2000.0)
        self.assertIn("oral", acetaldehyde["intake_route_note"])
        self.assertIn("all other routes", acetaldehyde["intake_route_note"])

    def test_no_numeric_intake_carries_no_route_note(self):
        for smiles in ("C1CO1", "CCO", "bad-smiles", "ClCC1CO1.CN(C)N=O"):
            with self.subTest(smiles=smiles):
                result = assess_material(smiles)
                self.assertIsNone(result["intake_ug_day"])
                self.assertIsNone(result["intake_route_note"])

    def test_known_entries_and_custom_alerts_are_distinct_statuses(self):
        known = assess_material("C=Cc1ccccc1")
        alert = assess_material("C1CO1")
        self.assertEqual(known["status"], "known_concern")
        self.assertEqual(known["name"], "Styrene")
        self.assertEqual(known["alerts"], [])
        self.assertEqual(alert["status"], "structural_alert")
        self.assertIsNone(alert["name"])
        self.assertIn("Epoxide", alert["alerts"])
        self.assertIsNone(alert["intake_ug_day"])

    def test_disconnected_components_match_without_transferring_an_intake(self):
        first = assess_material("[Na+].ClCC1CO1")
        reordered = assess_material("[Cl:1][CH2:2][CH:3]1[CH2:4][O:5]1.[Na+]")
        self.assertEqual(first["status"], "known_concern")
        self.assertEqual(first["name"], "Epichlorohydrin")
        self.assertIsNone(first["intake_ug_day"])
        self.assertEqual(first["canonical_smiles"], reordered["canonical_smiles"])
        both = assess_material("ClCC1CO1.OCC1CO1")
        self.assertEqual(set(both["name"].split("; ")), {"Epichlorohydrin", "Glycidol"})
        self.assertIsNone(both["intake_ug_day"])
        self.assertIn("whole-material intake", " ".join(both["limitations"]))
        separate = assess_material("CCl.C1CO1")
        self.assertNotIn("Epichlorohydrin", separate["name"])
        self.assertIsNone(separate["intake_ug_day"])

    def test_salt_fragments_are_not_arbitrarily_equated_to_neutral_parents(self):
        result = assess_material("[NH3+]c1ccccc1.[Cl-]")
        self.assertIsNone(result["name"])
        self.assertIsNone(result["intake_ug_day"])
        self.assertIn("not merged or neutralized", " ".join(result["limitations"]))
        glycidolate = assess_material("[O-]CC1CO1.[Na+]")
        self.assertIsNone(glycidolate["name"])
        self.assertEqual(glycidolate["status"], "structural_alert")
        self.assertIsNone(glycidolate["intake_ug_day"])

    def test_mixture_containing_ndma_cannot_inherit_a_known_reference(self):
        result = assess_material("ClCC1CO1.CN(C)N=O")
        self.assertEqual(result["status"], "known_concern")
        self.assertTrue(result["special_concern"])
        self.assertIsNone(result["intake_ug_day"])
        self.assertIn("No TTC fallback", result["intake_basis"])


class ConcentrationConversionTests(unittest.TestCase):
    def test_unknown_intake_has_no_conversion(self):
        self.assertIsNone(concentration_limit(None, 320))
        self.assertIsNone(concentration_limit(None, 0))

    def test_zero_intake_and_milligram_to_gram_conversion(self):
        self.assertEqual(concentration_limit(0, 320), 0.0)
        self.assertAlmostEqual(concentration_limit(3, 320), 9.375)
        self.assertEqual(concentration_limit(4, 1000), 4.0)

    def test_invalid_doses_raise_value_error(self):
        for dose in (0, -1, float("nan"), float("inf"), -float("inf"), None, "bad", True):
            with self.subTest(dose=dose), self.assertRaises(ValueError):
                concentration_limit(3, dose)

    def test_invalid_intakes_raise_value_error(self):
        for intake in (-1, float("nan"), float("inf"), -float("inf"), "bad", True):
            with self.subTest(intake=intake), self.assertRaises(ValueError):
                concentration_limit(intake, 320)


@env_support.requires("aizynthfinder")
class PluginAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from aizynthfinder.utils.exceptions import RejectionException
        import genotox_plugin

        cls.plugin = genotox_plugin
        cls.rejection = RejectionException

    def test_plugin_uses_shared_known_entry_and_alerts(self):
        mol = Chem.MolFromSmiles("[Cl:1][CH2:2][CH:3]1[CH2:4][O:5]1")
        self.assertEqual(self.plugin.find_ich_m7(mol), ("Epichlorohydrin", 3.0))
        self.assertEqual(
            [name for name, _ in self.plugin.find_alerts(mol)],
            assess_material("ClCC1CO1")["alerts"],
        )
        self.assertEqual(self.plugin.risk_of(None), 1.0)

    def test_filter_rejects_invalid_missing_or_unavailable_reactants(self):
        reactions = (
            SimpleNamespace(reactants=((SimpleNamespace(rd_mol=None),),), index=0),
            SimpleNamespace(reactants=((SimpleNamespace(rd_mol=Chem.MolFromSmiles("")),),), index=0),
            SimpleNamespace(reactants=(), index=0),
            SimpleNamespace(reactants=((),), index=0),
            SimpleNamespace(reactants=((SimpleNamespace(rd_mol=Chem.MolFromSmiles("CCO")),),), index=-1),
        )
        for reaction in reactions:
            with self.subTest(reaction=reaction):
                strategy = self.plugin.GenotoxFilter("test", None)
                with self.assertRaises(self.rejection):
                    strategy.apply(reaction)
                self.assertTrue(strategy.rejected)

    def test_filter_rejects_special_concern_at_default_cutoff(self):
        reaction = SimpleNamespace(
            reactants=((SimpleNamespace(rd_mol=Chem.MolFromSmiles("CN(C)N=O")),),), index=0
        )
        with self.assertRaises(self.rejection):
            self.plugin.GenotoxFilter("test", None).apply(reaction)

    def test_scorer_excludes_only_root_object_not_equal_smiles_impurity(self):
        root = SimpleNamespace(rd_mol=Chem.MolFromSmiles("ClCC1CO1"))
        precursor = SimpleNamespace(rd_mol=Chem.MolFromSmiles("ClCC1CO1"))
        tree = SimpleNamespace(root=root, molecules=lambda: iter((root,)), depth=lambda mol: 2)
        scorer = self.plugin.GenotoxScorer()
        self.assertEqual(scorer._score_reaction_tree(tree), 1.0)
        node = SimpleNamespace(to_reaction_tree=lambda: tree)
        self.assertEqual(scorer._score_node(node), 1.0)
        tree.molecules = lambda: iter((root, precursor))
        self.assertLess(scorer._score_reaction_tree(tree), 1.0)


if __name__ == "__main__":
    unittest.main()
