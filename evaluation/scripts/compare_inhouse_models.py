"""Compare the two in-house Ames models on rows neither of them has seen.

The fingerprint model decides almost entirely on unnamed bits; the descriptor
model uses 215 named columns and can say why. The question this script answers
is whether choosing the explainable one costs anything measurable on an
evaluation file, and how often the two disagree, because a disagreement is
what the comparison screen is for.

Only rows flagged as not overlapping the shared collection are scored. A model
scored on its own training data is not being evaluated. Abstentions are
excluded from the ranking metric and counted separately; they are never
treated as negative predictions.
"""
import argparse
import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'


def read_scores(path):
    """Score, prediction and overlap flag per row id."""
    rows = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            rows[row['id']] = {
                'score': float(row['score']) if row['score'] else None,
                'prediction': row['prediction'],
                'overlap': row['training_overlap'] == 'yes'}
    return rows


def read_truth(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return {row['id']: row['label'] for row in csv.DictReader(handle)}


def roc_auc(pairs):
    """Rank-based AUC over (label, score) pairs, ties shared."""
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


def summarise(name, scores, truth, threshold, *, only=None):
    """One model's figures. `only` restricts scoring to a fixed row set.

    The restriction is the point, not a detail. The fingerprint model abstains
    on structures the descriptor model reads, so scoring each on whatever it
    happened to answer would compare two different sets again, which is the
    error this file exists to avoid.
    """
    pairs, abstained, tp = [], 0, 0
    positives = decided = 0
    for identity, label in truth.items():
        row = scores.get(identity)
        if row is None or row['overlap']:
            continue
        if label not in ('0', '1'):
            continue
        if only is not None and identity not in only:
            continue
        positives += int(label == '1')
        if row['score'] is None or row['prediction'] == 'abstain':
            abstained += 1
            continue
        decided += 1
        pairs.append((int(label == '1'), row['score']))
        if label == '1' and row['score'] >= threshold:
            tp += 1
    return {'tool': name, 'unseen_rows_scored': decided, 'abstained': abstained,
            'labelled_positives': positives, 'auc': roc_auc(pairs),
            'positives_caught_at_threshold': tp,
            'sensitivity_at_threshold': tp / positives if positives else None,
            'threshold': threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', type=Path, default=ROOT / 'ames_independent.csv')
    parser.add_argument('--fingerprint', type=Path,
                        default=ROOT / 'ames_model_independent_predictions.csv')
    parser.add_argument('--descriptors', type=Path,
                        default=ROOT / 'ames_descriptor_independent_predictions.csv')
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'inhouse_model_comparison.json')
    args = parser.parse_args()

    truth = read_truth(args.truth)
    fingerprint = read_scores(args.fingerprint)
    descriptors = read_scores(args.descriptors)

    # The row set both models actually answered on, which is what the two
    # figures must be computed over if they are to be read side by side.
    shared_rows = {
        identity for identity, label in truth.items()
        if label in ('0', '1')
        and (row := fingerprint.get(identity)) is not None
        and (other := descriptors.get(identity)) is not None
        and not row['overlap'] and not other['overlap']
        and row['prediction'] != 'abstain' and other['prediction'] != 'abstain'}

    report = {
        'truth_file': str(args.truth), 'rows_in_file': len(truth),
        'threshold': args.threshold,
        'scored_on_rows_both_models_answered': len(shared_rows),
        'models': [summarise('fingerprint+descriptors (in use)', fingerprint,
                             truth, args.threshold, only=shared_rows),
                   summarise('descriptors only (explainable)', descriptors,
                             truth, args.threshold, only=shared_rows)],
        'each_model_on_everything_it_answered': [
            summarise('fingerprint+descriptors (in use)', fingerprint,
                      truth, args.threshold),
            summarise('descriptors only (explainable)', descriptors,
                      truth, args.threshold)],
        'notes': [
            'Scored only on rows flagged as not overlapping the shared '
            'collection both models were fitted on.',
            'The headline figures use only rows BOTH models answered. The '
            'fingerprint model abstains on structures the descriptor model '
            'reads, so scoring each on its own row set would compare two '
            'different sets.',
            'each_model_on_everything_it_answered is kept for reference and '
            'must not be read as a ranking: the two row sets differ.',
            'AUC is a ranking measure and does not depend on the threshold; '
            'sensitivity does.',
            'An abstention is counted separately and never read as negative.',
            'A negative prediction is a screening result, not evidence of safety.',
        ],
    }

    both, agree, disagreements = 0, 0, []
    for identity, label in truth.items():
        left, right = fingerprint.get(identity), descriptors.get(identity)
        if (left is None or right is None or left['overlap'] or right['overlap']
                or left['prediction'] == 'abstain' or right['prediction'] == 'abstain'):
            continue
        both += 1
        if left['prediction'] == right['prediction']:
            agree += 1
        else:
            disagreements.append({
                'id': identity, 'label': label,
                'fingerprint': left['prediction'], 'descriptors': right['prediction'],
                'fingerprint_score': left['score'], 'descriptor_score': right['score'],
                'which_was_right': ('fingerprint' if left['prediction'] == label
                                    else 'descriptors')})
    report['agreement'] = {
        'rows_both_decided': both, 'agreed': agree,
        'agreement_rate': agree / both if both else None,
        'disagreed': both - agree,
        'fingerprint_right_when_they_differ': sum(
            1 for row in disagreements if row['which_was_right'] == 'fingerprint'),
        'descriptors_right_when_they_differ': sum(
            1 for row in disagreements if row['which_was_right'] == 'descriptors'),
    }
    report['disagreements'] = disagreements[:200]

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding='utf-8')
    print(f"두 모델이 모두 답한 {len(shared_rows)}개로 채점합니다")
    for row in report['models']:
        auc = 'n/a' if row['auc'] is None else f"{row['auc']:.4f}"
        print(f"{row['tool']:40s} AUC {auc} · 기준선 {row['threshold']}에서 "
              f"민감도 {row['sensitivity_at_threshold']:.3f}")
    print('참고: 각자 답한 것 전부로 재면 (행 집합이 달라 순위로 읽으면 안 됨)')
    for row in report['each_model_on_everything_it_answered']:
        auc = 'n/a' if row['auc'] is None else f"{row['auc']:.4f}"
        print(f"  {row['tool']:38s} AUC {auc} · 채점 "
              f"{row['unseen_rows_scored']}개 · 기권 {row['abstained']}개")
    deal = report['agreement']
    print(f"판정 일치 {deal['agreement_rate']:.1%} ({deal['agreed']}/"
          f"{deal['rows_both_decided']}) · 엇갈린 {deal['disagreed']}개 중 "
          f"지문 모델이 맞은 것 {deal['fingerprint_right_when_they_differ']}개, "
          f"기술자 모델이 맞은 것 {deal['descriptors_right_when_they_differ']}개")
    print(f'기록: {args.output}')


if __name__ == '__main__':
    main()
