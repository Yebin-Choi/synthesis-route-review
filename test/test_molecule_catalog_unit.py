"""Catalog integrity; not a chemical synthesis or toxicity validation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from molecule_catalog import MOLECULES
from toxicology import canonicalize_smiles


class MoleculeCatalogTests(unittest.TestCase):
    def test_unique_valid_structures_and_verified_identity_links(self):
        self.assertEqual(len(MOLECULES), 25)
        self.assertEqual(len({m['id'] for m in MOLECULES}), 25)
        structures = []
        for item in MOLECULES:
            with self.subTest(name=item['name_en']):
                canonical = canonicalize_smiles(item['smiles'])
                self.assertIsNotNone(canonical)
                structures.append(canonical)
                self.assertTrue(item['name_ko'])
                cid = item['id'].removeprefix('pubchem-')
                self.assertTrue(cid.isdigit())
                self.assertEqual(item['source_url'], 'https://pubchem.ncbi.nlm.nih.gov/compound/' + cid)
                self.assertIn('안전성 목록이 아닙니다', item['note'])
        self.assertEqual(len(set(structures)), 25)

    def test_defaults_and_stereochemistry_preserved(self):
        by_name = {m['name_en'].lower(): m for m in MOLECULES}
        self.assertEqual(canonicalize_smiles(by_name['ethanol']['smiles']), 'CCO')
        self.assertEqual(by_name['propranolol']['id'], 'pubchem-4946')
        self.assertIn('@', canonicalize_smiles(by_name['l-alanine']['smiles']))
        self.assertNotIn('@', canonicalize_smiles(by_name['ibuprofen']['smiles']))


if __name__ == '__main__':
    unittest.main()
