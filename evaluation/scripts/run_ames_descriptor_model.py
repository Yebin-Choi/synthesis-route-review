"""Score the descriptor-only Ames model on the independent evaluation set.

Written so the explainable model enters the comparison table on the same file
as the other five tools. Mirrors `run_ames_model.py`, including the overlap
flag: this model was fitted on the train and validation rows of the shared
collection, which is not guaranteed disjoint from this file, so every
structure is checked and flagged and the overlapping rows can be dropped
before the numbers are read.

Only the model's prediction leaves this script. Labels are read to write the
overlap flag and are never shown to the model.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'
sys.path.insert(0, str(PROJECT / 'evaluation' / 'scripts'))

from run_ames_model import canonical, read_inputs, training_structures  # noqa: E402

DEFAULT_MODEL_DIR = PROJECT / 'models' / 'ames_descriptors'
# Set from --model-dir in main, so the weighted refit is scored the same way.
model_dir = DEFAULT_MODEL_DIR
WORKER = PROJECT / 'app' / 'ames_descriptor_worker.py'
FIELDS = ['id', 'smiles', 'tool', 'model', 'version', 'endpoint', 'prediction',
          'applicability', 'source', 'score', 'reason', 'training_overlap']
SOURCE_TEMPLATE = ('models/{name} (in-house XGBoost on 215 named 2D '
          'descriptors, shared Ames collection)')
BATCH = 200


def score_all(smiles_list):
    """Scores keyed by canonical structure, via the worker in its own process."""
    import os
    import subprocess
    import tempfile

    python = PROJECT / 'external_tools' / 'ames-env' / 'Scripts' / 'python.exe'
    if not python.is_file():
        raise SystemExit(f'격리 환경이 없습니다: {python}')
    distinct = list(dict.fromkeys(item for item in smiles_list if item))
    scored = {}
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    with tempfile.TemporaryDirectory(prefix='ames_desc_eval_') as directory:
        run_dir = Path(directory)
        for start in range(0, len(distinct), BATCH):
            chunk = distinct[start:start + BATCH]
            input_path = run_dir / 'in.json'
            output_path = run_dir / 'out.json'
            input_path.write_text(
                json.dumps({'canonical_smiles': chunk}, ensure_ascii=False),
                encoding='utf-8')
            if output_path.exists():
                output_path.unlink()
            subprocess.run(
                [str(python), '-X', 'utf8', '-B', str(WORKER),
                 '--input', str(input_path), '--output', str(output_path),
                 '--model-dir', str(model_dir)],
                capture_output=True, env=env, check=False, timeout=1800)
            if not output_path.is_file():
                raise SystemExit('작업자가 결과를 남기지 않았습니다')
            payload = json.loads(output_path.read_text(encoding='utf-8'))
            if payload.get('error'):
                raise SystemExit(f"작업자 오류: {payload['error']}")
            for row in payload['assessments']:
                scored[row['canonical_smiles']] = row
            print(f'  {min(start + BATCH, len(distinct))}/{len(distinct)}')
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'ames_independent.csv')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'ames_descriptor_independent_predictions.csv')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'ames_descriptor_independent_manifest.json')
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR,
                        help='채점할 모델 폴더.')
    args = parser.parse_args()
    global model_dir
    model_dir = args.model_dir.resolve()
    source = SOURCE_TEMPLATE.format(name=model_dir.name)

    rows = read_inputs(args.input, args.limit)
    if not rows:
        raise SystemExit(f'입력이 비어 있습니다: {args.input}')
    print(f'구조 {len(rows)}개')

    trained_on = training_structures()
    print(f'모델이 본 구조 {len(trained_on)}개와 대조합니다')

    canonical_inputs = [canonical(row['smiles']) for row in rows]
    scored = score_all(canonical_inputs)
    spec = json.loads((model_dir / 'model_spec.json').read_text(encoding='utf-8'))

    counts = {'1': 0, '0': 0, 'abstain': 0}
    overlapping = 0
    with args.output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row, item in zip(rows, canonical_inputs):
            overlap = item is not None and item in trained_on
            overlapping += int(overlap)
            record = scored.get(item) if item else None
            if record is None or record.get('status') != 'success':
                prediction, value = 'abstain', ''
                reason = ('unreadable:unparsed' if item is None
                          else f"unreadable:{(record or {}).get('error')}")
            else:
                score = float(record['score'])
                prediction = '1' if score >= args.threshold else '0'
                reason = 'model_positive' if prediction == '1' else 'model_negative'
                value = f'{score:.6f}'
            counts[prediction] += 1
            writer.writerow({
                'id': row['id'], 'smiles': row['smiles'],
                'tool': 'in-house Ames descriptors', 'model': 'descriptors_all',
                'version': '1', 'endpoint': 'ames', 'prediction': prediction,
                'applicability': 'unknown', 'source': source, 'score': value,
                'reason': reason, 'training_overlap': 'yes' if overlap else 'no'})

    manifest = {
        'input': str(args.input), 'structures': len(rows),
        'threshold': args.threshold, 'predictions': counts,
        'structures_the_model_had_already_seen': overlapping,
        'share_already_seen': overlapping / len(rows),
        'model_spec': {'features': spec.get('features'),
                       'columns': spec.get('columns'),
                       'grouped_cv_auc': spec.get('grouped_cv_auc'),
                       'fitted_on': spec.get('fitted_on')},
        'notes': [
            'Overlap is an exact canonical-structure match against the shared '
            'collection this model was fitted and selected on.',
            'Rows flagged as overlapping must be dropped before this model is '
            'compared with tools that never saw them.',
            'A structure the model could not read is an abstention, not a '
            'negative prediction.',
            'Every column of this model has a name, so a prediction can be '
            'described. That is why it exists beside the fingerprint model.',
        ],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding='utf-8')
    print(f"양성 {counts['1']} · 음성 {counts['0']} · 판정 못함 {counts['abstain']}")
    print(f"이미 본 구조 {overlapping}개 ({overlapping / len(rows):.1%})")
    print(f'기록: {args.output}')


if __name__ == '__main__':
    main()
