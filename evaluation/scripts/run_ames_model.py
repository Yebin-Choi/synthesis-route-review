"""Score the in-house Ames model on the independent evaluation set.

The other four tools were already scored on this file, so adding this model
makes the comparison in the application's performance table a comparison on
one set instead of two. Until now the model's figures came from the shared
held-out test set and the rest from here, which is not a ranking.

The model was fitted on the shared Ames collection, and nothing guarantees
that collection is disjoint from this evaluation file. Every structure is
therefore checked against the model's own training and validation structures
and flagged, so the overlapping rows can be dropped before the comparison is
read. A tool scored partly on its own training data is not being evaluated.

Only the model's prediction leaves this script. Labels are read to write the
overlap flag and are never shown to the model.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'
DEFAULT_MODEL_DIR = PROJECT / 'models' / 'ames_xgboost'
TRAINING_DATA = PROJECT / 'data' / 'ames' / 'Ames_모델용_전체_22157개.csv'
FIELDS = ['id', 'smiles', 'tool', 'model', 'version', 'endpoint', 'prediction',
          'applicability', 'source', 'score', 'reason', 'training_overlap']
SOURCE_TEMPLATE = 'models/{name} (in-house XGBoost, shared Ames collection)'


def canonical(smiles):
    molecule = Chem.MolFromSmiles((smiles or '').strip())
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def training_structures(splits=('train', 'validation')):
    """Canonical structures the model was fitted or selected on.

    The test split is included as well: those rows were used to report the
    model's headline figures, so a structure from there is not unseen either.
    """
    seen = set()
    with TRAINING_DATA.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            item = canonical(row.get('canonical_smiles'))
            if item is None:
                continue
            if row['split'] in splits or row['split'] == 'test':
                seen.add(item)
    return seen


def read_inputs(path, limit=None):
    rows = []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if not row.get('id') or not row.get('smiles'):
                continue
            rows.append({'id': row['id'], 'smiles': row['smiles']})
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'ames_independent.csv')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'ames_model_independent_predictions.csv')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'ames_model_independent_manifest.json')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR,
                        help='채점할 모델 폴더. 가중치를 준 재학습본을 같은 '
                             '방식으로 채점할 때 바꿉니다.')
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    source = SOURCE_TEMPLATE.format(name=model_dir.name)

    sys.path.insert(0, str(model_dir))
    import pandas as pd
    from predict_expanded import predict_frame

    rows = read_inputs(args.input, args.limit)
    if not rows:
        raise SystemExit(f'입력이 비어 있습니다: {args.input}')
    print(f'구조 {len(rows)}개')

    trained_on = training_structures()
    print(f'모델이 본 구조 {len(trained_on)}개와 대조합니다')

    frame = pd.DataFrame({'smiles': [row['smiles'] for row in rows]})
    predicted = predict_frame(frame, model_dir, 'smiles', threshold=args.threshold)
    spec = json.loads((model_dir / 'model_spec.json').read_text(encoding='utf-8'))

    counts = {'1': 0, '0': 0, 'abstain': 0}
    overlapping = 0
    with args.output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for position, row in enumerate(rows):
            record = predicted.iloc[position]
            status = str(record['structure_status'])
            score = record['ames_score']
            item = canonical(row['smiles'])
            overlap = item is not None and item in trained_on
            overlapping += int(overlap)
            if status != 'valid' or score != score:
                prediction, reason, value = 'abstain', f'unreadable:{status}', ''
            else:
                value = float(score)
                prediction = '1' if value >= args.threshold else '0'
                reason = 'model_positive' if prediction == '1' else 'model_negative'
                value = f'{value:.6f}'
            counts[prediction] += 1
            writer.writerow({
                'id': row['id'], 'smiles': row['smiles'], 'tool': 'in-house Ames XGBoost',
                'model': spec.get('experiment_id'), 'version': '1',
                'endpoint': 'ames', 'prediction': prediction,
                'applicability': 'unknown', 'source': source, 'score': value,
                'reason': reason, 'training_overlap': 'yes' if overlap else 'no'})

    manifest = {
        'input': str(args.input), 'model_dir': str(model_dir),
        'structures': len(rows), 'threshold': args.threshold,
        'predictions': counts,
        'structures_the_model_had_already_seen': overlapping,
        'share_already_seen': overlapping / len(rows),
        'model_spec': {'experiment_id': spec.get('experiment_id'),
                       'validation_roc_auc': spec.get('roc_auc'),
                       'train_n': spec.get('train_n'),
                       'balanced': spec.get('balanced'),
                       'scale_pos_weight': spec.get('scale_pos_weight')},
        'notes': [
            'Overlap is an exact canonical-structure match against the shared '
            'collection the model was fitted, selected and reported on.',
            'Rows flagged as overlapping must be dropped before this model is '
            'compared with tools that never saw them.',
            'A structure the model could not read is an abstention, not a '
            'negative prediction.',
        ],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding='utf-8')
    print(f"판정: 양성 {counts['1']} · 음성 {counts['0']} · 기권 {counts['abstain']}")
    print(f'모델이 이미 본 구조: {overlapping}개 ({overlapping / len(rows):.1%})')
    print(f'기록: {args.output}')


if __name__ == '__main__':
    main()
