"""Own input-contract tests, not reconstructed team KB records."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import engine


class ConditionContractTests(unittest.TestCase):
    def setUp(self):
        self.payload = engine.load_example()
        self.payload['routes'] = [self.payload['routes'][2]]
        self.step = self.payload['routes'][0]['steps'][0]

    def evaluate(self):
        return engine.analyze_routes(self.payload)['routes'][0]

    def test_each_additional_role_is_screened(self):
        for role in engine.ADDITIONAL_ROLES:
            with self.subTest(role=role):
                self.step[role] = ['ClCC1CO1']
                result = self.evaluate()
                self.assertEqual(result['known_concern_count'], 1)
                material = next(m for m in result['steps'][0]['materials'] if m['role'] == role)
                self.assertEqual(material['canonical_smiles'], 'ClCC1CO1')
                self.assertEqual(material['formation_kind'], 'coproduct' if role == 'coproducts' else 'unknown')
                del self.step[role]

    def test_supplied_condition_source_not_route_source(self):
        self.step['catalysts'] = ['ClCC1CO1']
        self.step['material_provenance'] = {'catalysts': {'status': 'proposed', 'source': 'synthetic test source; not actual KB'}}
        material = next(m for m in self.evaluate()['steps'][0]['materials'] if m['role'] == 'catalysts')
        self.assertEqual(material['evidence_status'], 'proposed')
        self.assertEqual(material['origin'], 'synthetic test source; not actual KB')
        self.assertNotEqual(material['origin'], material['route_source'])
        self.assertEqual(self.evaluate()['condition_coverage'], 0)

    def test_missing_material_evidence_remains_unknown(self):
        for material in self.evaluate()['steps'][0]['materials']:
            self.assertEqual(material['evidence_status'], 'unknown')
            self.assertEqual(material['origin'], '물질별 출처 미지정')

    def test_nested_or_unknown_material_fields_not_silently_dropped(self):
        self.step['conditions']['catalysts'] = ['ClCC1CO1']
        with self.assertRaises(ValueError):
            self.evaluate()
        del self.step['conditions']['catalysts']
        self.step['unhandled_materials'] = ['ClCC1CO1']
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_invalid_provenance_rejected(self):
        for value in ([], {'catalysts': {'status': 'safe', 'source': 'x'}},
                      {'catalysts': {'status': 'observed'}}, {'unknown_role': {'status': 'unknown', 'source': 'x'}}):
            with self.subTest(value=value):
                self.step['material_provenance'] = value
                with self.assertRaises(ValueError):
                    self.evaluate()

    def test_conditional_warning_checks_new_roles(self):
        self.step['ligands'] = ['CNC']
        self.step['bases'] = ['[Na+].[O-]N=O']
        self.assertEqual(self.evaluate()['conditional_warning_count'], 1)

    def test_display_preserves_products_but_target_not_impurity(self):
        original = copy.deepcopy(self.payload)
        result = self.evaluate()['steps'][0]
        self.assertEqual(result['reaction_structures']['products'], self.step['products'])
        self.assertFalse(any(m['role'] == 'products' for m in result['materials']))
        self.assertEqual(self.payload, original)


if __name__ == '__main__':
    unittest.main()
