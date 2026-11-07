"""Fit the explainable Ames model: named 2D descriptors only, no fingerprint.

Why this model exists. Measured in benchmark log 6-6, the model in use puts
almost its whole decision into the 2,048 unnamed fingerprint bits: for
ethylene oxide the fingerprint block contributed +3.074 log-odds against
+0.006 from all 215 named descriptors together. That model cannot answer "why
did you say that". A descriptor-only fit scores 0.8999 against its 0.9018 on
the same grouped folds, a difference smaller than the fold spread, so the
explainable model is available at no measured cost.

It is a second tool, not a replacement. The two agree on 89.8% of calls, so
one in ten structures gets a different answer, and the point of the screen is
to show that rather than hide it.

Fitted on the train and validation rows, which is what the model in use was
fitted and selected on, so the two can be scored on one evaluation file. The
test split is not touched here.

The kept descriptor names are written out with the weights. Dropping
zero-variance columns depends on the rows in hand, so a model that recomputed
the column list at prediction time would be featurising differently from how
it was fitted.

The collection is 30.2% active. `--balanced` carries that in the training
weights instead of leaving it to the decision line, which is what lets the
line stay at the 0.50 published work uses (benchmark log 6-19).
"""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from rdkit.Chem.Descriptors import CalcMolDescriptors

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ames_common import SEED, load, models  # noqa: E402

MODELS = ROOT.parents[1] / 'models'


def descriptor_names(molecules):
    """Column names that vary over these molecules, in a fixed order."""
    described = [CalcMolDescriptors(molecule) for molecule in molecules]
    keys = sorted(described[0])
    matrix = np.array([[float(row.get(key) or 0.0) for key in keys]
                       for row in described], dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    keep = matrix.std(axis=0) > 1e-9
    return [key for key, kept in zip(keys, keep) if kept], matrix[:, keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--balanced', action='store_true',
                        help='불균형을 학습 가중치로 반영하고 별도 폴더에 저장합니다.')
    args = parser.parse_args()
    out_dir = MODELS / ('ames_descriptors_balanced' if args.balanced
                        else 'ames_descriptors')

    data = load(splits=('train', 'validation'))
    molecules, labels = data['molecules'], data['labels']
    print(f'구조 {len(molecules)}개 · 활성 비율 {labels.mean():.3f}')

    names, matrix = descriptor_names(molecules)
    print(f'기술자 열 {len(names)}개 (변하지 않는 열은 제외)')

    model = models(SEED)['xgboost_balanced' if args.balanced else 'xgboost']()
    model.fit(matrix, labels)
    print(f"scale_pos_weight = {model.get_params()['scale_pos_weight']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / 'model.joblib')
    (out_dir / 'model_spec.json').write_text(json.dumps({
        'tool': 'in-house Ames descriptors',
        'balanced': bool(args.balanced),
        'features': 'descriptors_all',
        'descriptor_names': names,
        'columns': len(names),
        'model': 'xgboost',
        'parameters': {key: (float(value) if isinstance(value, float) else value)
                       for key, value in model.get_params().items()
                       if key in ('n_estimators', 'max_depth', 'learning_rate',
                                  'subsample', 'colsample_bytree', 'reg_lambda',
                                  'min_child_weight', 'random_state',
                                  'scale_pos_weight')},
        'fitted_on': {'splits': ['train', 'validation'],
                      'structures': int(len(molecules)),
                      'active_fraction': float(labels.mean())},
        'grouped_cv_auc': 0.8999,
        'grouped_cv_note': 'From the feature sweep in benchmark log 6-3 and '
                           '6-6, scaffold-grouped folds. Not measured here.',
        'why': 'Every column has a name, so a prediction can be described. '
               'The model in use decides almost entirely on unnamed '
               'fingerprint bits (benchmark log 6-6).',
        'limits': 'Research-only AMES screening. Not a safety determination '
                  'and not an ICH M7 classification.',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'저장: {out_dir}')


if __name__ == '__main__':
    main()
