import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation' / 'scripts'))
import run_java_tox as runner


class JavaToxParserTests(unittest.TestCase):
    def test_category_mapping(self):
        def row(p, n, e):
            return {runner.POSITIVE: p, runner.NEGATIVE: n, runner.ERROR: e}
        self.assertEqual(runner.map_prediction(row('YES', 'NO', 'NO'))[0], '1')
        self.assertEqual(runner.map_prediction(row('NO', 'YES', 'NO'))[0], '0')
        for value in (None, row('YES', 'YES', 'NO'), row('YES', 'NO', 'YES'), row('?', 'NO', 'NO')):
            self.assertEqual(runner.map_prediction(value)[0], 'abstain')

    def test_unknown_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.csv'
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['evaluation_id', runner.POSITIVE, runner.NEGATIVE, runner.ERROR])
                writer.writeheader()
                writer.writerow({'evaluation_id': 'unexpected', runner.POSITIVE: 'YES', runner.NEGATIVE: 'NO', runner.ERROR: 'NO'})
            with self.assertRaises(ValueError):
                runner.load_results(path, {'expected'})

    def test_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.csv'
            path.write_text('evaluation_id,arbitrary\nx,YES\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.load_results(path, {'x'})


if __name__ == '__main__':
    unittest.main()
