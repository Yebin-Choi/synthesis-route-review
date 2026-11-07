"""Score the project's own rule screen on an Ames evaluation file.

The rules are a transparent alert lookup, not a validated (Q)SAR system. A
no-alert result is a negative screen, never evidence of safety.
"""
import argparse
import csv
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'
sys.path.insert(0, str(PROJECT / 'app'))

from toxicology import assess_material  # noqa: E402

FIELDS = ['id', 'smiles', 'tool', 'model', 'version', 'endpoint', 'prediction',
          'applicability', 'source', 'reason']
POSITIVE_STATUSES = ('known_concern', 'structural_alert')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'ames_independent.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'rules_independent_predictions.csv')
    args = parser.parse_args()

    with args.input.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    counts = {'0': 0, '1': 0, 'abstain': 0}
    with args.output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            assessment = assess_material(row['smiles'])
            status = assessment['status']
            if status in POSITIVE_STATUSES:
                prediction = '1'
            elif status == 'no_alert_detected':
                prediction = '0'
            else:
                prediction = 'abstain'
            counts[prediction] += 1
            writer.writerow({'id': row['id'], 'smiles': row['smiles'], 'tool': 'custom_rules',
                             'model': 'assess_material', 'version': 'local', 'endpoint': 'ames',
                             'prediction': prediction, 'applicability': 'unknown',
                             'source': 'app/toxicology.py', 'reason': status})
    print(f'Completed {len(rows)} rows; {counts}')


if __name__ == '__main__':
    main()
