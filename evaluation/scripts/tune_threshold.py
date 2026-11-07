"""Choose an ADMET-AI cut-off on one half of the data and report it on the other.

The 0.5 cut-off shipped with the model is a default, not a tuned optimum. A
cut-off chosen and reported on the same rows would repeat the exact mistake
this evaluation set exists to avoid, so the split is made once, by a fixed
hash of the structure, and the held-out half is scored only after the choice.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'


def split_half(identity):
    """Deterministic, order-independent split that does not look at the label."""
    digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()
    return 'tune' if int(digest, 16) % 2 == 0 else 'holdout'


def load(truth_path, scores_path):
    with truth_path.open(encoding='utf-8-sig', newline='') as handle:
        truth = {row['id']: row['label'] for row in csv.DictReader(handle)}
    rows = []
    with scores_path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            score = row.get('score', '')
            if row['id'] not in truth or score in ('', None):
                continue
            rows.append({'id': row['id'], 'score': float(score),
                         'label': truth[row['id']], 'half': split_half(row['id'])})
    return rows


def measure(rows, cutoff):
    tp = sum(1 for row in rows if row['score'] >= cutoff and row['label'] == '1')
    fp = sum(1 for row in rows if row['score'] >= cutoff and row['label'] == '0')
    fn = sum(1 for row in rows if row['score'] < cutoff and row['label'] == '1')
    tn = sum(1 for row in rows if row['score'] < cutoff and row['label'] == '0')
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = ((sensitivity + specificity) / 2
                if sensitivity is not None and specificity is not None else None)
    return {'cutoff': round(cutoff, 3), 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'sensitivity': sensitivity, 'specificity': specificity,
            'balanced_accuracy': balanced}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', type=Path, default=ROOT / 'ames_independent.csv')
    parser.add_argument('--scores', type=Path,
                        default=ROOT / 'admet_independent_predictions.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'threshold_tuning.json')
    parser.add_argument('--min-specificity', type=float, default=0.85,
                        help='Route screening needs a high specificity, because one flagged '
                             'material excludes the whole route')
    args = parser.parse_args()

    rows = load(args.truth, args.scores)
    tune = [row for row in rows if row['half'] == 'tune']
    holdout = [row for row in rows if row['half'] == 'holdout']
    grid = [index / 100 for index in range(5, 100, 5)]

    tuned = [measure(tune, cutoff) for cutoff in grid]
    best_balanced = max(tuned, key=lambda item: item['balanced_accuracy'] or 0)
    eligible = [item for item in tuned
                if (item['specificity'] or 0) >= args.min_specificity]
    best_constrained = max(eligible, key=lambda item: item['sensitivity'] or 0) if eligible else None

    report = {
        'scores_file': str(args.scores),
        'split': 'sha256(id) parity; label never used to split',
        'tune_rows': len(tune), 'holdout_rows': len(holdout),
        'grid': tuned,
        'default_cutoff_on_holdout': measure(holdout, 0.5),
        'best_balanced_on_tune': best_balanced,
        'best_balanced_on_holdout': measure(holdout, best_balanced['cutoff']),
        'min_specificity': args.min_specificity,
        'best_constrained_on_tune': best_constrained,
        'best_constrained_on_holdout': (measure(holdout, best_constrained['cutoff'])
                                        if best_constrained else None),
        'notes': [
            'The cut-off is selected on the tune half only; holdout numbers are the ones to report.',
            'A cut-off raises specificity by calling fewer materials positive; it does not make '
            'the model better at telling them apart.',
            'Scores are model outputs, not calibrated probabilities of mutagenicity.',
        ],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    def line(label, item):
        if item is None:
            print(f'{label:34s} 해당 없음')
            return
        print(f'{label:34s} 기준값 {item["cutoff"]:.2f} · 민감도 {item["sensitivity"]:.3f} '
              f'· 특이도 {item["specificity"]:.3f} · 균형 {item["balanced_accuracy"]:.3f}')

    print(f'조정용 {len(tune)}개 / 시험용 {len(holdout)}개')
    line('기본값 0.5 (시험용)', report['default_cutoff_on_holdout'])
    line('균형 최대 (조정용에서 선택)', best_balanced)
    line('  그 기준값을 시험용에', report['best_balanced_on_holdout'])
    line(f'특이도 {args.min_specificity} 이상 중 민감도 최대', best_constrained)
    line('  그 기준값을 시험용에', report['best_constrained_on_holdout'])


if __name__ == '__main__':
    main()
