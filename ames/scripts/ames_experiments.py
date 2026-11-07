"""Overnight Ames experiments: what the model can do and where it stops.

The shared work ran 198 configurations and found that the differences between
them sat inside their confidence intervals. That result was obtained on a single
split. This run asks the same questions with grouped cross-validation and a
measured noise floor, which is the method that changed the enzyme numbers in
this repository, and adds the scaffold-side analysis the shared work did not do.

Stages, each writing its own report so stopping loses only the stage running:

  groups      How the two groupings compare and how the structures are
              distributed over scaffolds. Decides what every later split means.
  checks      Noise floor, label scrambling, learning curve, applicability
              domain. These say whether a score can be read at all.
  features    Feature compositions compared under the same grouped folds.
  algorithms  Learning algorithms compared at a fixed feature set.

The test split is never touched. It was already used for the ADMET-AI
comparison, so it is reserved for one final evaluation.
"""
import argparse
import collections
import json
from pathlib import Path
import sys
import time

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ames_common import (BINARY_ONLY, DEFAULT_FEATURES, DEFAULT_MODEL, FEATURES,  # noqa: E402
                         REPEATS, RESULTS, SEED, grouped_score, group_summary,
                         load, models, noise_floor, splitter)

SCRAMBLE_ROUNDS = 5
TRAINING_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
SIMILARITY_EDGES = (0.0, 0.3, 0.4, 0.5, 0.7, 1.01)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding='utf-8')
    print(f'  기록: {path}')


def stage_groups(data, args):
    """Which grouping is stricter, and how the structures sit over scaffolds."""
    report = {
        'question': 'How are the structures grouped, and does the grouping '
                    'choice change what a held-out fold means?',
        'structures': len(data['smiles']),
        'actives': int(data['labels'].sum()),
        'active_fraction': float(data['labels'].mean()),
        'splits_used': list(data['splits']),
        'leakage_group': group_summary(data['leakage_groups']),
        'scaffold_group': group_summary(data['scaffold_groups']),
    }
    for name in ('leakage_group', 'scaffold_group'):
        info = report[name]
        print(f"  {name}: 묶음 {info['groups']}개 · 가장 큰 묶음 {info['largest']}개 · "
              f"1개짜리 묶음 {info['singletons']}개 "
              f"({info['share_in_singletons']:.1%})")

    # Whether the two groupings disagree matters: if scaffolds cut across
    # leakage groups, a leakage-group split can still place close analogues on
    # both sides and the reported score is optimistic.
    pairs = collections.defaultdict(set)
    for leak, scaffold in zip(data['leakage_groups'], data['scaffold_groups']):
        pairs[scaffold].add(leak)
    split_scaffolds = {s: g for s, g in pairs.items() if len(g) > 1}
    affected = sum(1 for s in data['scaffold_groups'] if s in split_scaffolds)
    report['scaffolds_spanning_several_leakage_groups'] = len(split_scaffolds)
    report['structures_in_those_scaffolds'] = affected
    report['share_of_structures_affected'] = float(affected / len(data['smiles']))
    print(f"  여러 leakage_group에 걸친 골격: {len(split_scaffolds)}개 "
          f"(구조 {affected}개, {affected / len(data['smiles']):.1%})")

    sizes = collections.Counter(collections.Counter(data['scaffold_groups']).values())
    report['scaffold_size_distribution'] = {str(k): v for k, v in sorted(sizes.items())}
    biggest = collections.Counter(data['scaffold_groups']).most_common(15)
    report['largest_scaffolds'] = [
        {'scaffold': scaffold, 'structures': count,
         'active_fraction': float(data['labels'][
             data['scaffold_groups'] == scaffold].mean())}
        for scaffold, count in biggest if not scaffold.startswith('acyclic::')]
    report['notes'] = [
        'A scaffold spanning several leakage groups means the two groupings '
        'are not nested, so the choice changes what a held-out fold contains.',
        'Acyclic structures have no Murcko scaffold and each keeps its own '
        'group rather than being pooled into one.',
    ]
    write(RESULTS / 'ames_groups.json', report)
    return report


def stage_checks(data, args):
    """Can a score from this set be read, and over what structures does it hold."""
    matrix = FEATURES[args.features](data['molecules'])
    labels = data['labels']
    groups = data[f'{args.grouping}_groups']
    builder = models(args.seed)[args.model]
    report = {'question': 'Is the score real, and where does it hold?',
              'features': args.features, 'model': args.model,
              'grouping': args.grouping, 'structures': len(labels),
              'seed': args.seed}

    print('\n1. 잡음 바닥과 진짜 성적')
    real = grouped_score(matrix, labels, groups, builder, args.seed)
    floor = noise_floor(matrix, labels, groups, args.seed)
    report['real'] = real
    report['noise_floor'] = floor
    print(f"  실제: AUC {real['auc_mean']:.3f} ± {real['auc_std']:.3f}")
    print(f"  무작위 기준선: AUC {floor['auc_mean']:.3f} ± {floor['auc_std']:.3f}")
    print(f"  → {floor['auc_std']:.3f}보다 작은 차이는 차이로 읽지 않는다")

    print('\n2. 라벨을 뒤섞으면')
    rng = np.random.default_rng(args.seed)
    scrambled = []
    for _ in range(SCRAMBLE_ROUNDS):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        score = grouped_score(matrix, shuffled, groups, builder, args.seed,
                              repeats=1)
        if score:
            scrambled.append(score['auc_mean'])
    report['label_scrambling'] = {
        'rounds': len(scrambled),
        'scrambled_auc_mean': float(np.mean(scrambled)) if scrambled else None,
        'scrambled_auc_max': float(np.max(scrambled)) if scrambled else None,
        'real_auc_mean': real['auc_mean'],
        'reading': 'The real figure has to stand clear of the scrambled one.',
    }
    if scrambled:
        print(f"  뒤섞은 라벨: AUC {np.mean(scrambled):.3f} "
              f"(최대 {np.max(scrambled):.3f})")

    print('\n3. 자료를 줄여가며 (묶음을 통째로 뺀다)')
    report['learning_curve'] = []
    unique = np.array(sorted(set(groups)))
    rng = np.random.default_rng(args.seed)
    for fraction in TRAINING_FRACTIONS:
        if fraction >= 1.0:
            report['learning_curve'].append(
                {'training_fraction': 1.0, **real})
            print(f"  100%: AUC {real['auc_mean']:.3f}")
            continue
        chosen = set(rng.choice(unique, size=max(2, int(len(unique) * fraction)),
                                replace=False))
        keep = np.array([index for index, group in enumerate(groups)
                         if group in chosen])
        if len(set(labels[keep])) < 2:
            continue
        score = grouped_score(matrix[keep], labels[keep], groups[keep], builder,
                              args.seed, repeats=1)
        if score:
            report['learning_curve'].append(
                {'training_fraction': fraction, 'structures': int(len(keep)),
                 **score})
            print(f"  {fraction:.0%}: AUC {score['auc_mean']:.3f} "
                  f"(구조 {len(keep)}개)")

    print('\n4. 학습 구조와 얼마나 닮아야 맞히는가')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    prints = [generator.GetFingerprint(m) for m in data['molecules']]
    nearest = np.zeros(len(prints))
    scores = np.zeros(len(prints))
    counted = np.zeros(len(prints), dtype=bool)
    for train, test in splitter(args.seed, 0).split(matrix, labels, groups):
        model = builder()
        model.fit(matrix[train], labels[train])
        scores[test] = model.predict_proba(matrix[test])[:, 1]
        counted[test] = True
        reference = [prints[i] for i in train]
        for index in test:
            nearest[index] = max(DataStructs.BulkTanimotoSimilarity(
                prints[index], reference))
    report['applicability'] = []
    placed = np.digitize(nearest, SIMILARITY_EDGES) - 1
    for band in range(len(SIMILARITY_EDGES) - 1):
        inside = (placed == band) & counted
        entry = {'similarity_from': SIMILARITY_EDGES[band],
                 'similarity_to': round(SIMILARITY_EDGES[band + 1], 2),
                 'structures': int(inside.sum())}
        if inside.sum() < 50 or len(set(labels[inside])) < 2:
            entry['result'] = 'too few structures in this band to read'
        else:
            entry['auc'] = float(roc_auc_score(labels[inside], scores[inside]))
            entry['active_fraction'] = float(labels[inside].mean())
            print(f"  유사도 {entry['similarity_from']:.2f}~"
                  f"{entry['similarity_to']:.2f}: {entry['structures']:6}개 · "
                  f"AUC {entry['auc']:.3f}")
        report['applicability'].append(entry)
    report['median_similarity_to_nearest_training_structure'] = float(
        np.median(nearest[counted]))

    report['notes'] = [
        'The noise floor is the same folds scored by a model that learns '
        'nothing. Differences smaller than its spread are not differences.',
        'The learning curve drops whole groups, not random structures, so the '
        'smaller sets are genuinely smaller rather than thinner.',
        'Applicability is measured out of fold: the similarity is to the '
        'training part of the fold that scored the structure.',
    ]
    write(RESULTS / f'ames_checks_{args.grouping}.json', report)
    return report


def _sweep(data, args, axis):
    """Feature or algorithm comparison over the same folds, written as it goes."""
    path = RESULTS / f'ames_sweep_{axis}_{args.grouping}.json'
    stored = {}
    if path.is_file():
        previous = json.loads(path.read_text(encoding='utf-8'))
        stored = {(row['features'], row['model']): row
                  for row in previous.get('rows', [])}
    labels = data['labels']
    groups = data[f'{args.grouping}_groups']

    if axis == 'features':
        plan = [(name, args.model) for name in sorted(FEATURES)]
    else:
        plan = [(args.features, name) for name in sorted(models(args.seed))]

    cache = {}
    for features, model_name in plan:
        if (features, model_name) in stored:
            print(f'  건너뜀 (이미 있음): {features} / {model_name}')
            continue
        if model_name in BINARY_ONLY and 'descriptors' in features:
            print(f'  건너뜀 (이진 입력만 받음): {features} / {model_name}')
            continue
        if features not in cache:
            started = time.time()
            cache[features] = FEATURES[features](data['molecules'])
            print(f'  {features} 수치화 {time.time() - started:.0f}초 '
                  f'({cache[features].shape[1]}열)')
        builder = models(args.seed).get(model_name)
        if builder is None:
            print(f'  건너뜀 (설치 안 됨): {model_name}')
            continue
        started = time.time()
        score = grouped_score(cache[features], labels, groups, builder, args.seed)
        if score is None:
            continue
        row = {'features': features, 'model': model_name,
               'columns': int(cache[features].shape[1]),
               'elapsed_s': round(time.time() - started, 1), **score}
        stored[(features, model_name)] = row
        print(f"  {features:34} / {model_name:22} AUC {row['auc_mean']:.3f} "
              f"± {row['auc_std']:.3f} · {row['elapsed_s']:.0f}초")
        write(path, {
            'question': f'Which {axis} actually help, measured on grouped folds?',
            'grouping': args.grouping, 'seed': args.seed,
            'folds': f'StratifiedGroupKFold, 5 folds, {REPEATS} repeats',
            'fixed': {'model': args.model} if axis == 'features'
                     else {'features': args.features},
            'structures': len(labels),
            'rows': sorted(stored.values(), key=lambda r: -r['auc_mean']),
        })
    return stored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=('groups', 'checks', 'features', 'algorithms'))
    parser.add_argument('--grouping', default='scaffold',
                        choices=('scaffold', 'leakage'))
    parser.add_argument('--features', default=DEFAULT_FEATURES)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()
    data = load()
    print(f"구조 {len(data['smiles'])}개 · 양성 {int(data['labels'].sum())}개 "
          f"({data['labels'].mean():.1%}) · 분할 {', '.join(data['splits'])}")
    print(f"묶음 기준: {args.grouping}")

    if args.stage == 'groups':
        stage_groups(data, args)
    elif args.stage == 'checks':
        stage_checks(data, args)
    else:
        _sweep(data, args, args.stage)
    print(f'\n{args.stage} 단계 완료 · {time.time() - started:.0f}초')


if __name__ == '__main__':
    main()
