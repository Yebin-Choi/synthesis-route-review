"""Run installed ADMET-AI on AMES structures; fixed 0.5 threshold, no label tuning."""
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'ames_evaluation.csv')
    parser.add_argument('--prefix', default='admet',
                        help='Output basename prefix, so a second evaluation set does not '
                             'overwrite the first run')
    args = parser.parse_args()
    raw_path = ROOT / f'{args.prefix}_raw_predictions.csv'
    predictions_path = ROOT / f'{args.prefix}_predictions.csv'
    manifest_path = ROOT / f'{args.prefix}_manifest.json'

    from admet_ai import ADMETModel
    with args.input.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    model = ADMETModel(include_physchem=False, drugbank_path=None, num_workers=0)
    predictions = model.predict([row['smiles'] for row in rows])
    predictions.to_csv(raw_path, encoding='utf-8')
    if 'AMES' not in predictions.columns:
        raise ValueError('Installed model did not return AMES endpoint')
    version = importlib.metadata.version('admet-ai')
    fields = ['id', 'smiles', 'tool', 'model', 'version', 'endpoint', 'prediction', 'applicability', 'source', 'score']
    with predictions_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            score = float(predictions.loc[row['smiles'], 'AMES']) if row['smiles'] in predictions.index else float('nan')
            writer.writerow({'id': row['id'], 'smiles': row['smiles'], 'tool': 'ADMET-AI',
                             'model': 'bundled-AMES-ensemble', 'version': version, 'endpoint': 'ames',
                             'prediction': str(int(score >= 0.5)) if math.isfinite(score) else 'abstain',
                             'applicability': 'unknown', 'source': 'https://github.com/swansonk14/admet_ai',
                             'score': score if math.isfinite(score) else ''})
    model_dir = Path(importlib.metadata.distribution('admet-ai').locate_file('admet_ai/resources/models'))
    manifest = {'tool': 'ADMET-AI', 'version': version, 'threshold': 0.5,
                'input_file': str(args.input), 'dataset_rows': len(rows), 'returned_rows': len(predictions),
                'model_files_sha256': {str(p.relative_to(model_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in model_dir.rglob('*') if p.is_file()},
                'limitation': 'Public TDC AMES may overlap model training. These are descriptive predictions on known data, NOT independent generalization accuracy. Scores are not clinically calibrated probabilities. Applicability domain is unknown.'}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Completed {len(rows)} rows; ADMET-AI {version}; threshold fixed at 0.5; training overlap unknown.')


if __name__ == '__main__':
    main()
