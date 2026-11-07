"""Why the earlier Ames numbers came out the way they did.

The measurements in sections 6-1 to 6-5 of the benchmark log state what
happens. These stages ask why, because a reason that holds up is worth more in
a report than a number without one.

  grouping     Why a scaffold split costs 0.015 against a leakage-group split.
               The proposed mechanism is that scaffold folds hold out
               structures that are less like their training part, so the
               similarity distribution under each grouping is measured
               directly rather than assumed.
  redundancy   Why 2,048 fingerprint bits add 0.002 on top of the descriptors.
               Either the two views say the same thing, or they differ and the
               model cannot use the difference. Those are distinguishable: the
               agreement between a descriptor-only and a fingerprint-only
               model is measured, and an average of the two is scored.
  difficulty   What makes a scaffold family hard. Family score against family
               size, how skewed its labels are, how alike its members are and
               how close it sits to the rest of the training set.
  calibration  Whether the number the screen shows behaves like a probability,
               and where the cut-off should sit. The application treats 0.5 as
               the line and nothing has checked that here.
  tuning       Nested hyperparameter search. The settings are chosen inside
               each outer fold, so the reported figure is not the flattering
               one that comes from choosing while looking at the test side.

The test split is never touched.
"""
import argparse
import collections
import json
from pathlib import Path
import sys
import time

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ames_common import (DEFAULT_MODEL, FEATURES, FOLDS, RESULTS, SEED,  # noqa: E402
                         grouped_score, load, models, splitter)

BANDS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.01)
CUTOFFS = tuple(round(value, 2) for value in np.arange(0.20, 0.85, 0.05))
MIN_FAMILY = 100
OUTER_FOLDS = 5
INNER_FOLDS = 3


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding='utf-8')
    print(f'  기록: {path}')


def _prints(molecules):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [generator.GetFingerprint(molecule) for molecule in molecules]


def stage_grouping(data, args):
    """How alike are held-out structures to their training part, per grouping."""
    prints = _prints(data['molecules'])
    labels = data['labels']
    report = {'question': 'Why does a scaffold split score lower than a '
                          'leakage-group split on the same data?',
              'groupings': {}}
    for name in ('leakage', 'scaffold'):
        groups = data[f'{name}_groups']
        nearest = np.zeros(len(prints))
        for train, test in splitter(args.seed, 0).split(prints, labels, groups):
            reference = [prints[index] for index in train]
            for index in test:
                nearest[index] = max(DataStructs.BulkTanimotoSimilarity(
                    prints[index], reference))
        report['groupings'][name] = {
            'median_similarity_to_training_part': float(np.median(nearest)),
            'mean_similarity_to_training_part': float(np.mean(nearest)),
            'share_above_0_7': float((nearest >= 0.7).mean()),
            'share_below_0_4': float((nearest < 0.4).mean()),
        }
        info = report['groupings'][name]
        print(f"  {name:9} 유사도 중앙값 {info['median_similarity_to_training_part']:.3f} · "
              f"0.7 이상 {info['share_above_0_7']:.1%} · "
              f"0.4 미만 {info['share_below_0_4']:.1%}")
    leak = report['groupings']['leakage']
    scaf = report['groupings']['scaffold']
    report['difference'] = {
        'median_similarity': scaf['median_similarity_to_training_part']
        - leak['median_similarity_to_training_part'],
        'share_above_0_7': scaf['share_above_0_7'] - leak['share_above_0_7'],
    }
    report['reading'] = (
        'A scaffold split holding out less similar structures is the mechanism '
        'that would explain the lower score. If the two distributions are the '
        'same, the explanation is wrong and something else causes the gap.')
    write(RESULTS / 'ames_why_grouping.json', report)


def stage_redundancy(data, args):
    """Do the descriptor and fingerprint views say the same thing."""
    labels = data['labels']
    groups = data[f'{args.grouping}_groups']
    builder = models(args.seed)[args.model]
    views = {'descriptors_all': FEATURES['descriptors_all'](data['molecules']),
             'rdkit_fp_2048': FEATURES['rdkit_fp_2048'](data['molecules'])}

    out_of_fold = {name: np.zeros(len(labels)) for name in views}
    counted = np.zeros(len(labels), dtype=bool)
    for train, test in splitter(args.seed, 0).split(views['descriptors_all'],
                                                    labels, groups):
        counted[test] = True
        for name, matrix in views.items():
            model = builder()
            model.fit(matrix[train], labels[train])
            out_of_fold[name][test] = model.predict_proba(matrix[test])[:, 1]

    descriptor_scores = out_of_fold['descriptors_all'][counted]
    fingerprint_scores = out_of_fold['rdkit_fp_2048'][counted]
    truth = labels[counted]
    averaged = (descriptor_scores + fingerprint_scores) / 2
    agree = ((descriptor_scores >= 0.5) == (fingerprint_scores >= 0.5)).mean()

    report = {
        'question': 'Do the descriptor and fingerprint views carry the same '
                    'information, or different information the model cannot use?',
        'grouping': args.grouping, 'model': args.model,
        'descriptors_only_auc': float(roc_auc_score(truth, descriptor_scores)),
        'fingerprint_only_auc': float(roc_auc_score(truth, fingerprint_scores)),
        'averaged_auc': float(roc_auc_score(truth, averaged)),
        'score_correlation': float(np.corrcoef(descriptor_scores,
                                               fingerprint_scores)[0, 1]),
        'calls_agree': float(agree),
        'reading': 'A high correlation means the two views are redundant. A low '
                   'correlation with no gain from averaging means they differ '
                   'but the difference does not help.',
    }
    print(f"  기술자만 AUC {report['descriptors_only_auc']:.4f} · "
          f"지문만 {report['fingerprint_only_auc']:.4f} · "
          f"둘을 평균 {report['averaged_auc']:.4f}")
    print(f"  점수 상관 {report['score_correlation']:.3f} · "
          f"판정 일치 {report['calls_agree']:.1%}")
    write(RESULTS / 'ames_why_redundancy.json', report)


def stage_difficulty(data, args):
    """What separates a scaffold family the model reads from one it does not."""
    matrix = FEATURES[args.features](data['molecules'])
    labels = data['labels']
    groups = data[f'{args.grouping}_groups']
    builder = models(args.seed)[args.model]
    prints = _prints(data['molecules'])

    scores = np.zeros(len(labels))
    nearest_outside = np.zeros(len(labels))
    counted = np.zeros(len(labels), dtype=bool)
    for train, test in splitter(args.seed, 0).split(matrix, labels, groups):
        model = builder()
        model.fit(matrix[train], labels[train])
        scores[test] = model.predict_proba(matrix[test])[:, 1]
        counted[test] = True
        reference = [prints[index] for index in train]
        for index in test:
            nearest_outside[index] = max(DataStructs.BulkTanimotoSimilarity(
                prints[index], reference))

    rows = []
    for scaffold, size in collections.Counter(data['scaffold_groups']).most_common():
        if size < MIN_FAMILY or scaffold.startswith('acyclic::'):
            continue
        inside = (data['scaffold_groups'] == scaffold) & counted
        if inside.sum() < MIN_FAMILY or len(set(labels[inside])) < 2:
            continue
        members = np.where(inside)[0]
        # How alike the family's own members are. A family that is internally
        # varied is a harder thing to call one family.
        sample = members[:120]
        similarities = []
        for position, index in enumerate(sample):
            rest = [prints[other] for other in sample[position + 1:]]
            if rest:
                similarities.extend(
                    DataStructs.BulkTanimotoSimilarity(prints[index], rest))
        active = float(labels[inside].mean())
        rows.append({
            'scaffold': scaffold, 'structures': int(inside.sum()),
            'active_fraction': active,
            'label_skew': abs(active - 0.5) * 2,
            'internal_similarity': float(np.mean(similarities)) if similarities else None,
            'similarity_to_training': float(np.mean(nearest_outside[inside])),
            'auc': float(roc_auc_score(labels[inside], scores[inside])),
        })
    report = {'question': 'What makes a scaffold family hard to predict?',
              'families': rows}
    if len(rows) >= 4:
        for field in ('structures', 'label_skew', 'internal_similarity',
                      'similarity_to_training'):
            values = [row[field] for row in rows if row[field] is not None]
            aucs = [row['auc'] for row in rows if row[field] is not None]
            if len(values) >= 4:
                report.setdefault('correlation_with_auc', {})[field] = float(
                    np.corrcoef(values, aucs)[0, 1])
        for field, value in report.get('correlation_with_auc', {}).items():
            print(f'  {field:24} AUC와의 상관 {value:+.3f}')
    report['reading'] = (
        'Correlations over a handful of families are weak evidence and are '
        'reported as a direction to look at, not as a finding.')
    write(RESULTS / 'ames_why_difficulty.json', report)


def stage_calibration(data, args):
    """Does the screen's number behave like a probability, and where is the line."""
    matrix = FEATURES[args.features](data['molecules'])
    labels = data['labels']
    groups = data[f'{args.grouping}_groups']
    builder = models(args.seed)[args.model]

    scores = np.zeros(len(labels))
    counted = np.zeros(len(labels), dtype=bool)
    for repeat in range(2):
        for train, test in splitter(args.seed, repeat).split(matrix, labels, groups):
            model = builder()
            model.fit(matrix[train], labels[train])
            scores[test] = model.predict_proba(matrix[test])[:, 1]
            counted[test] = True
    truth = labels[counted]
    shown = scores[counted]

    report = {'question': 'Is the displayed number a probability, and is 0.5 '
                          'the right line?',
              'grouping': args.grouping, 'features': args.features,
              'overall_active_fraction': float(truth.mean()),
              'auc': float(roc_auc_score(truth, shown)),
              'brier': float(brier_score_loss(truth, shown)),
              'reliability': [], 'cutoffs': []}
    placed = np.digitize(shown, BANDS) - 1
    for band in range(len(BANDS) - 1):
        inside = placed == band
        entry = {'shown_from': BANDS[band], 'shown_to': round(BANDS[band + 1], 2),
                 'structures': int(inside.sum())}
        if inside.sum() >= 50:
            entry['mean_number_shown'] = float(shown[inside].mean())
            entry['actually_active'] = float(truth[inside].mean())
            entry['gap'] = entry['actually_active'] - entry['mean_number_shown']
            print(f"  화면 {entry['shown_from']:.1f}~{entry['shown_to']:.2f} "
                  f"{entry['structures']:6}개: 평균 표시 "
                  f"{entry['mean_number_shown']:.3f} → 실제 활성 "
                  f"{entry['actually_active']:.3f} (차이 {entry['gap']:+.3f})")
        report['reliability'].append(entry)

    for cutoff in CUTOFFS:
        predicted = (shown >= cutoff).astype(int)
        if len(set(predicted)) < 2:
            continue
        true_positive = int(((predicted == 1) & (truth == 1)).sum())
        false_positive = int(((predicted == 1) & (truth == 0)).sum())
        false_negative = int(((predicted == 0) & (truth == 1)).sum())
        true_negative = int(((predicted == 0) & (truth == 0)).sum())
        sensitivity = true_positive / max(true_positive + false_negative, 1)
        specificity = true_negative / max(true_negative + false_positive, 1)
        report['cutoffs'].append({
            'cutoff': cutoff, 'sensitivity': sensitivity, 'specificity': specificity,
            'balanced_accuracy': (sensitivity + specificity) / 2,
            'precision': true_positive / max(true_positive + false_positive, 1),
            'kept_fraction': float(predicted.mean())})
    best = max(report['cutoffs'], key=lambda row: row['balanced_accuracy'])
    report['best_cutoff_by_balanced_accuracy'] = best
    report['model_and_features'] = f'{args.model} · {args.features}'
    print(f"  균형이 가장 좋은 기준선 {best['cutoff']:.2f} "
          f"(균형 {best['balanced_accuracy']:.3f})")
    # Each model gets its own line, so each measurement gets its own file.
    # Writing both to one name would leave the second run quietly describing
    # a model the file is not named after.
    write(args.output or RESULTS / 'ames_why_calibration.json', report)


def stage_tuning(data, args):
    """Nested search: settings chosen inside each outer fold, never on the test side."""
    matrix = FEATURES[args.features](data['molecules'])
    labels = data['labels']
    groups = data[f'{args.grouping}_groups']
    grids = {
        'xgboost': {'max_depth': [4, 6, 8], 'learning_rate': [0.03, 0.07, 0.15],
                    'n_estimators': [400, 800]},
        'random_forest': {'n_estimators': [500, 1000],
                          'min_samples_leaf': [1, 2, 4],
                          'max_features': ['sqrt', 0.1]},
    }
    builders = models(args.seed)
    report = {'question': 'Does choosing hyperparameters honestly beat the '
                          'settings already in use?',
              'grouping': args.grouping, 'features': args.features,
              'outer_folds': OUTER_FOLDS, 'inner_folds': INNER_FOLDS, 'rows': []}
    for name, grid in grids.items():
        if name not in builders:
            continue
        started = time.time()
        baseline = grouped_score(matrix, labels, groups, builders[name],
                                 args.seed, repeats=1)
        aucs, picked = [], []
        outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                     random_state=args.seed)
        for train, test in outer.split(matrix, labels, groups):
            inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                         random_state=args.seed)
            search = GridSearchCV(builders[name](), grid, scoring='roc_auc',
                                  cv=inner.split(matrix[train], labels[train],
                                                 groups[train]),
                                  n_jobs=1, refit=True)
            search.fit(matrix[train], labels[train])
            picked.append({key: str(value)
                           for key, value in search.best_params_.items()})
            aucs.append(roc_auc_score(
                labels[test], search.predict_proba(matrix[test])[:, 1]))
        row = {'model': name, 'baseline_auc': baseline['auc_mean'],
               'nested_auc_mean': float(np.mean(aucs)),
               'nested_auc_std': float(np.std(aucs)),
               'difference': float(np.mean(aucs)) - baseline['auc_mean'],
               'settings_chosen_per_fold': picked,
               'elapsed_s': round(time.time() - started, 1)}
        report['rows'].append(row)
        print(f"  {name:16} 기본 {row['baseline_auc']:.4f} → 중첩 "
              f"{row['nested_auc_mean']:.4f} ± {row['nested_auc_std']:.4f} "
              f"(차이 {row['difference']:+.4f}) · {row['elapsed_s']:.0f}초")
        write(RESULTS / 'ames_why_tuning.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=('grouping', 'redundancy', 'difficulty',
                                 'calibration', 'tuning'))
    parser.add_argument('--grouping', default='scaffold',
                        choices=('scaffold', 'leakage'))
    parser.add_argument('--features', default='rdkit_fp_2048+descriptors_all')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--output', type=Path, default=None,
                        help='Where the stage writes. Give a separate path when '
                             'measuring a second model, so one run does not '
                             'overwrite the other.')
    args = parser.parse_args()

    started = time.time()
    data = load()
    print(f"구조 {len(data['smiles'])}개 · 묶음 기준 {args.grouping}")
    {'grouping': stage_grouping, 'redundancy': stage_redundancy,
     'difficulty': stage_difficulty, 'calibration': stage_calibration,
     'tuning': stage_tuning}[args.stage](data, args)
    print(f'\n{args.stage} 단계 완료 · {time.time() - started:.0f}초')


if __name__ == '__main__':
    main()
