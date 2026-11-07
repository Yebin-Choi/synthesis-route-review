"""Follow-up Ames analysis: how few features, which chemistry, which labels.

The first run established that the descriptor block carries almost all of the
signal and that a stricter grouping costs 0.015. These stages ask what that
means in practice.

  descriptors  How far the 215-descriptor block can be cut before the score
               moves. Selection happens inside each fold's training part, so
               the reported number is not a figure chosen by looking at the
               held-out side.
  scaffolds    Out-of-fold performance per scaffold family. An average over
               the whole set hides that some chemistry is predicted well and
               some is not, and the families are what a reader recognises.
  agreement    The label agreement threshold compared under grouped folds.
               The shared work compared these on a single split.
  figures      The plots, drawn from the stored reports rather than recomputed.

The test split is never touched.
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ames_common import (DATA, DEFAULT_FEATURES, DEFAULT_MODEL, FEATURES, RESULTS,  # noqa: E402
                         SEED, _descriptors, grouped_score, load, models, splitter)

DESCRIPTOR_COUNTS = (5, 10, 20, 40, 80, 160, 215)
MIN_FAMILY = 100
AGREEMENT_LEVELS = (0.6, 0.7, 0.8, 0.9, 1.0)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding='utf-8')
    print(f'  기록: {path}')


def _importance_order(model):
    """Column indices from most to least useful, whatever the model exposes."""
    if hasattr(model, 'feature_importances_'):
        return np.argsort(model.feature_importances_)[::-1]
    raise RuntimeError('이 모델은 중요도를 내놓지 않습니다')


def stage_descriptors(data, args):
    """How many descriptors are actually needed.

    The columns are ranked on the training part of each fold and the shorter
    model is refitted there, so no fold's score was helped by looking at the
    structures it is scored on.
    """
    matrix = _descriptors(data['molecules'])
    labels = data['labels']
    groups = data[f'{args.grouping}_groups']
    builders = models(args.seed)
    report = {
        'question': 'How few descriptors keep the score the full block gives?',
        'grouping': args.grouping, 'model': args.model,
        'available_descriptors': int(matrix.shape[1]),
        'selection': 'ranked by model importance inside each fold training part',
        'rows': [],
    }
    print(f'  기술자 {matrix.shape[1]}개에서 시작')

    for count in DESCRIPTOR_COUNTS:
        if count > matrix.shape[1]:
            continue
        started = time.time()
        aucs, chosen = [], collections.Counter()
        for repeat in range(2):
            for train, test in splitter(args.seed, repeat).split(matrix, labels, groups):
                if len(set(labels[test])) < 2:
                    continue
                ranker = builders[args.model]()
                ranker.fit(matrix[train], labels[train])
                keep = _importance_order(ranker)[:count]
                chosen.update(int(index) for index in keep)
                model = builders[args.model]()
                model.fit(matrix[train][:, keep], labels[train])
                aucs.append(roc_auc_score(
                    labels[test], model.predict_proba(matrix[test][:, keep])[:, 1]))
        row = {'descriptors': count, 'auc_mean': float(np.mean(aucs)),
               'auc_std': float(np.std(aucs)), 'folds_scored': len(aucs),
               'elapsed_s': round(time.time() - started, 1),
               'columns_chosen_in_every_fold': int(
                   sum(1 for _, n in chosen.items() if n == len(aucs)))}
        report['rows'].append(row)
        print(f"  기술자 {count:4}개: AUC {row['auc_mean']:.4f} ± {row['auc_std']:.4f} "
              f"· 모든 겹에서 뽑힌 열 {row['columns_chosen_in_every_fold']}개 "
              f"· {row['elapsed_s']:.0f}초")
        write(RESULTS / f'ames_descriptors_{args.grouping}.json', report)
    return report


def stage_scaffolds(data, args):
    """Where the model is right and where it is not, by scaffold family."""
    matrix = FEATURES[args.features](data['molecules'])
    labels = data['labels']
    groups = data[f'{args.grouping}_groups']
    builder = models(args.seed)[args.model]

    scores = np.zeros(len(labels))
    counted = np.zeros(len(labels), dtype=bool)
    for train, test in splitter(args.seed, 0).split(matrix, labels, groups):
        model = builder()
        model.fit(matrix[train], labels[train])
        scores[test] = model.predict_proba(matrix[test])[:, 1]
        counted[test] = True

    report = {
        'question': 'Which chemistry does the model get right, and which not?',
        'features': args.features, 'model': args.model, 'grouping': args.grouping,
        'structures_scored': int(counted.sum()),
        'overall_auc': float(roc_auc_score(labels[counted], scores[counted])),
        'families': [],
    }
    print(f"  전체 out-of-fold AUC {report['overall_auc']:.4f}")

    families = collections.Counter(data['scaffold_groups'])
    for scaffold, size in families.most_common():
        if size < MIN_FAMILY or scaffold.startswith('acyclic::'):
            continue
        inside = (data['scaffold_groups'] == scaffold) & counted
        if inside.sum() < MIN_FAMILY or len(set(labels[inside])) < 2:
            continue
        entry = {'scaffold': scaffold, 'structures': int(inside.sum()),
                 'active_fraction': float(labels[inside].mean()),
                 'auc': float(roc_auc_score(labels[inside], scores[inside]))}
        report['families'].append(entry)
        print(f"  {scaffold[:44]:46} {entry['structures']:5}개 "
              f"활성 {entry['active_fraction']:.2f} AUC {entry['auc']:.3f}")

    # Acyclic structures have no family of their own but are worth reporting as
    # one block, because a model trained mostly on ring systems may not read
    # them well and they are a large share of the set.
    acyclic = np.array([s.startswith('acyclic::') for s in data['scaffold_groups']])
    inside = acyclic & counted
    if inside.sum() >= MIN_FAMILY and len(set(labels[inside])) > 1:
        report['acyclic_block'] = {
            'structures': int(inside.sum()),
            'active_fraction': float(labels[inside].mean()),
            'auc': float(roc_auc_score(labels[inside], scores[inside]))}
        print(f"  (고리 없는 구조 전체) {inside.sum()}개 "
              f"AUC {report['acyclic_block']['auc']:.3f}")
    report['notes'] = [
        'Scores are out of fold: the model that scored a structure never saw '
        'it in training.',
        'A family with a very skewed active fraction makes any AUC easy or '
        'impossible, so the fraction is reported beside it.',
    ]
    write(RESULTS / f'ames_scaffolds_{args.grouping}.json', report)
    return report


def stage_agreement(data, args):
    """Does a stricter label agreement threshold give a better set to learn from.

    Raising the threshold removes structures whose sources disagreed. That
    leaves cleaner labels and less data at the same time, so which effect wins
    is a measurement rather than a principle.
    """
    agreement = {}
    with DATA.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            try:
                agreement[row['canonical_smiles']] = float(row['agreement'])
            except (TypeError, ValueError):
                continue
    values = np.array([agreement.get(item, 0.0) for item in data['smiles']])
    matrix = FEATURES[args.features](data['molecules'])
    labels = data['labels']
    groups = data[f'{args.grouping}_groups']
    builder = models(args.seed)[args.model]

    report = {'question': 'Does a stricter label agreement threshold help?',
              'features': args.features, 'model': args.model,
              'grouping': args.grouping, 'rows': []}
    for level in AGREEMENT_LEVELS:
        keep = np.where(values >= level - 1e-9)[0]
        if len(keep) < 500 or len(set(labels[keep])) < 2:
            continue
        score = grouped_score(matrix[keep], labels[keep], groups[keep], builder,
                              args.seed)
        if score is None:
            continue
        row = {'agreement_at_least': level, 'structures': int(len(keep)),
               'active_fraction': float(labels[keep].mean()), **score}
        report['rows'].append(row)
        print(f"  합의율 {level:.0%} 이상: 구조 {row['structures']:6}개 · "
              f"AUC {row['auc_mean']:.4f} ± {row['auc_std']:.4f}")
        write(RESULTS / f'ames_agreement_{args.grouping}.json', report)
    report['notes'] = [
        'Each threshold is scored on its own structures, so a higher score may '
        'mean an easier remaining set rather than a better model.',
    ]
    write(RESULTS / f'ames_agreement_{args.grouping}.json', report)
    return report


def stage_figures(data, args):
    """Draw the plots from the stored reports. Nothing is recomputed here."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = RESULTS / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 150, 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    drawn = []

    def stored(name):
        path = RESULTS / name
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None

    groups = stored('ames_groups.json')
    if groups:
        sizes = {int(k): v for k, v in groups['scaffold_size_distribution'].items()}
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(list(sizes), list(sizes.values()), color='#4C72B0')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('structures in the scaffold family')
        ax.set_ylabel('number of families')
        ax.set_title('Scaffold family sizes')
        fig.tight_layout()
        fig.savefig(out / 'scaffold_sizes.png')
        plt.close(fig)
        drawn.append('scaffold_sizes.png')

    checks = stored(f'ames_checks_{args.grouping}.json')
    if checks:
        bands = [row for row in checks['applicability'] if 'auc' in row]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        labels_ = [f"{row['similarity_from']:.1f}–{row['similarity_to']:.1f}"
                   for row in bands]
        ax.bar(labels_, [row['auc'] for row in bands], color='#55A868')
        ax.axhline(checks['noise_floor']['auc_mean'], color='#C44E52',
                   linestyle='--', linewidth=1, label='noise floor')
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel('Tanimoto similarity to the nearest training structure')
        ax.set_ylabel('ROC-AUC')
        ax.set_title('Where the model holds')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / 'applicability.png')
        plt.close(fig)
        drawn.append('applicability.png')

        fig, ax = plt.subplots(figsize=(5, 3.2))
        curve = checks['learning_curve']
        ax.plot([row['training_fraction'] for row in curve],
                [row['auc_mean'] for row in curve], marker='o', color='#4C72B0')
        ax.set_xlabel('share of scaffold families used')
        ax.set_ylabel('ROC-AUC')
        ax.set_title('Learning curve')
        fig.tight_layout()
        fig.savefig(out / 'learning_curve.png')
        plt.close(fig)
        drawn.append('learning_curve.png')

    sweep = stored(f'ames_sweep_features_{args.grouping}.json')
    if sweep:
        rows = sorted(sweep['rows'], key=lambda r: r['auc_mean'])
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.barh([row['features'] for row in rows],
                [row['auc_mean'] for row in rows],
                xerr=[row['auc_std'] for row in rows], color='#8172B2')
        ax.set_xlim(0.75, 0.93)
        ax.set_xlabel('ROC-AUC (grouped folds)')
        ax.set_title('Feature compositions')
        fig.tight_layout()
        fig.savefig(out / 'feature_comparison.png')
        plt.close(fig)
        drawn.append('feature_comparison.png')

    reduced = stored(f'ames_descriptors_{args.grouping}.json')
    if reduced:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot([row['descriptors'] for row in reduced['rows']],
                [row['auc_mean'] for row in reduced['rows']],
                marker='o', color='#CCB974')
        ax.set_xscale('log')
        ax.set_xlabel('descriptors kept')
        ax.set_ylabel('ROC-AUC')
        ax.set_title('How few descriptors are enough')
        fig.tight_layout()
        fig.savefig(out / 'descriptor_reduction.png')
        plt.close(fig)
        drawn.append('descriptor_reduction.png')

    print(f"  그림 {len(drawn)}장: {', '.join(drawn)}")
    write(RESULTS / 'ames_figures.json', {'figures': drawn, 'directory': str(out)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=('descriptors', 'scaffolds', 'agreement', 'figures'))
    parser.add_argument('--grouping', default='scaffold',
                        choices=('scaffold', 'leakage'))
    parser.add_argument('--features', default=DEFAULT_FEATURES)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()
    data = load()
    print(f"구조 {len(data['smiles'])}개 · 묶음 기준 {args.grouping}")
    {'descriptors': stage_descriptors, 'scaffolds': stage_scaffolds,
     'agreement': stage_agreement, 'figures': stage_figures}[args.stage](data, args)
    print(f'\n{args.stage} 단계 완료 · {time.time() - started:.0f}초')


if __name__ == '__main__':
    main()
