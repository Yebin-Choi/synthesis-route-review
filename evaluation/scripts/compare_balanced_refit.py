"""Did carrying the class imbalance in the weights help, at 0.50?

The shipped fit left the collection's 30% positive rate to the decision line,
which is why the balance point sat at 0.25 and why 0.50 missed a third of the
active structures. The refit carries it in `scale_pos_weight` instead. The
question here is the only one that matters for the software: at the 0.50 that
published work uses, is the refit better on structures neither fit has seen?

Both are scored on the same rows: not overlapping the shared collection, and
answered by both, so one fit's abstentions cannot flatter the other. AUC is
reported too, because it says whether the ranking changed or only the scale.

    python evaluation/scripts/compare_balanced_refit.py --threshold 0.50
"""
import argparse
import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'


def read_scores(path):
    rows = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            rows[row['id']] = {
                'score': float(row['score']) if row['score'] else None,
                'abstained': row['prediction'] == 'abstain',
                'overlap': row['training_overlap'] == 'yes'}
    return rows


def read_truth(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return {row['id']: row['label'] for row in csv.DictReader(handle)
                if row['label'] in ('0', '1')}


def roc_auc(pairs):
    """Rank-based AUC, ties sharing a rank."""
    ordered = sorted(pairs, key=lambda item: item[1])
    ranks, position = {}, 0
    while position < len(ordered):
        end = position
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[position][1]:
            end += 1
        shared = (position + end) / 2 + 1
        for index in range(position, end + 1):
            ranks[index] = shared
        position = end + 1
    positives = sum(1 for label, _ in ordered if label == 1)
    negatives = len(ordered) - positives
    if not positives or not negatives:
        return None
    rank_sum = sum(ranks[index] for index, (label, _) in enumerate(ordered)
                   if label == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def figures(name, scores, truth, rows, threshold):
    pairs = []
    tp = fp = tn = fn = 0
    for identity in rows:
        label = int(truth[identity] == '1')
        value = scores[identity]['score']
        pairs.append((label, value))
        called = value >= threshold
        if label and called:
            tp += 1
        elif label:
            fn += 1
        elif called:
            fp += 1
        else:
            tn += 1
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = ((sensitivity + specificity) / 2
                if sensitivity is not None and specificity is not None else None)
    return {'fit': name, 'threshold': threshold, 'rows': len(rows),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'sensitivity': sensitivity, 'specificity': specificity,
            'balanced_accuracy': balanced,
            'accuracy': (tp + tn) / len(rows) if rows else None,
            'auc': roc_auc(pairs)}


def sweep(scores, truth, rows):
    """Where this fit's own balance point falls, for reference only.

    It is not the line the software runs at. It is here because the whole
    argument for the refit is that it moves this point towards 0.50, so the
    claim should be checkable rather than asserted.
    """
    best = None
    for step in range(5, 96):
        row = figures('sweep', scores, truth, rows, step / 100)
        if row['balanced_accuracy'] is None:
            continue
        if best is None or row['balanced_accuracy'] > best['balanced_accuracy']:
            best = row
    return None if best is None else {
        'cutoff': best['threshold'],
        'balanced_accuracy': best['balanced_accuracy'],
        'sensitivity': best['sensitivity'], 'specificity': best['specificity']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', type=Path, default=ROOT / 'ames_independent.csv')
    parser.add_argument('--shipped', type=Path,
                        default=ROOT / 'ames_model_independent_predictions.csv')
    parser.add_argument('--balanced', type=Path,
                        default=ROOT / 'ames_balanced_independent_predictions.csv')
    parser.add_argument('--threshold', type=float, default=0.50)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'balanced_refit_comparison.json')
    args = parser.parse_args()

    truth = read_truth(args.truth)
    shipped = read_scores(args.shipped)
    refit = read_scores(args.balanced)

    rows = sorted(
        identity for identity in truth
        if (left := shipped.get(identity)) is not None
        and (right := refit.get(identity)) is not None
        and not left['overlap'] and not right['overlap']
        and not left['abstained'] and not right['abstained']
        and left['score'] is not None and right['score'] is not None)
    if not rows:
        raise SystemExit('두 채점본이 함께 답한 행이 없습니다')
    actives = sum(1 for identity in rows if truth[identity] == '1')

    report = {
        'truth_file': str(args.truth), 'threshold': args.threshold,
        'scored_rows': len(rows), 'active_rows': actives,
        'active_fraction': actives / len(rows),
        'at_threshold': [figures('shipped (weighting off)', shipped, truth, rows,
                                 args.threshold),
                         figures('refit (weighting on)', refit, truth, rows,
                                 args.threshold)],
        'own_balance_point': {
            'shipped (weighting off)': sweep(shipped, truth, rows),
            'refit (weighting on)': sweep(refit, truth, rows)},
        'notes': [
            "Scored only on rows that overlap neither fit's shared collection "
            "and that both fits answered, so abstentions cannot flatter one.",
            'AUC does not depend on the threshold. If it is unchanged while '
            'the figures at 0.50 move, the ranking is the same and only the '
            'scale moved, which is what the weighting was for.',
            'own_balance_point is reference, not the line the software runs '
            'at. The software runs at the stated threshold.',
            'This file is an evaluation set, not the locked test split.',
            'A negative prediction is a screening result, not evidence of safety.',
        ],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding='utf-8')

    print(f"채점 {len(rows)}개 · 양성 {actives}개 ({actives / len(rows):.1%})")
    for row in report['at_threshold']:
        print(f"{row['fit']:28s} 기준선 {row['threshold']:.2f} · "
              f"민감도 {row['sensitivity']:.3f} · 특이도 {row['specificity']:.3f} · "
              f"균형정확도 {row['balanced_accuracy']:.3f} · AUC {row['auc']:.4f}")
    for name, point in report['own_balance_point'].items():
        print(f"  {name:26s} 자체 균형점 {point['cutoff']:.2f} "
              f"(균형정확도 {point['balanced_accuracy']:.3f})")
    print(f'기록: {args.output}')


if __name__ == '__main__':
    main()
