"""Re-score the score-based tools at a chosen decision line, on one row set.

The screen's performance table and the screen's decision line have to be the
same measurement. When the line moved from 0.50 to the measured balance point
0.25 (benchmark log 6-8), the table's sensitivity and specificity stopped
describing what the screen does, so they are recomputed here.

Only tools that return a score can be re-scored. Toxtree returns a verdict and
the rule screen returns an alert, so their rows are unchanged by the line and
are reported as measured. EPA TEST's command-line output is a verdict, not a
score, so it is in the same position.

All tools are scored on the same rows. Rows the model had already seen are
dropped. An abstention is reported, never counted as negative.
"""
import argparse
import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'

# Tool id, prediction file, and whether moving the line can change its calls.
SCORED = {
    'ames_xgb': ROOT / 'ames_model_independent_predictions.csv',
    'ames_desc': ROOT / 'ames_descriptor_independent_predictions.csv',
    'admet': ROOT / 'admet_independent_predictions.csv',
}
OVERLAP_SOURCE = ROOT / 'ames_model_independent_predictions.csv'


def read_truth(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return {row['id']: row['label'] for row in csv.DictReader(handle)}


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return {row['id']: row for row in csv.DictReader(handle)}


def unseen_ids(truth):
    """Row ids the in-house models had not been fitted on.

    Taken from the overlap flag the in-house scoring script wrote, so every
    tool is scored on the same rows as in benchmark log 6-2.
    """
    rows = read_rows(OVERLAP_SOURCE)
    return {identity for identity, label in truth.items()
            if label in ('0', '1')
            and rows.get(identity, {}).get('training_overlap') == 'no'}


def score_at(rows, truth, keep, threshold):
    tp = tn = fp = fn = abstain = 0
    for identity in keep:
        row = rows.get(identity)
        label = truth[identity]
        raw = (row or {}).get('score')
        if row is None or not raw or (row.get('prediction') == 'abstain'):
            abstain += 1
            continue
        call = '1' if float(raw) >= threshold else '0'
        if label == '1':
            tp += call == '1'
            fn += call == '0'
        else:
            fp += call == '1'
            tn += call == '0'
    decided = tp + tn + fp + fn
    positives, negatives = tp + fn, tn + fp
    sensitivity = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    balanced = (None if sensitivity is None or specificity is None
                else (sensitivity + specificity) / 2)
    return {'threshold': threshold, 'rows': len(keep), 'decided': decided,
            'abstained': abstain, 'answered': decided / len(keep) if keep else None,
            'true_positive': tp, 'true_negative': tn, 'false_positive': fp,
            'false_negative': fn, 'sensitivity': sensitivity,
            'specificity': specificity, 'balanced': balanced}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', type=Path, default=ROOT / 'ames_independent.csv')
    parser.add_argument('--threshold', type=float, action='append', default=[],
                        help='Repeatable; defaults to 0.25 and 0.50')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'tool_scores_by_threshold.json')
    args = parser.parse_args()
    thresholds = args.threshold or [0.25, 0.50]

    truth = read_truth(args.truth)
    keep = unseen_ids(truth)
    print(f'같은 행 {len(keep)}개에서 채점합니다 '
          f"(양성 {sum(1 for i in keep if truth[i] == '1')}개)")

    report = {'truth_file': str(args.truth), 'rows_scored': len(keep),
              'thresholds': thresholds, 'tools': {}, 'notes': [
                  'Every tool is scored on the same rows, with rows the '
                  'in-house models had already seen dropped.',
                  'Only score-returning tools appear here. Toxtree, EPA TEST '
                  'and the rule screen return verdicts, so a decision line '
                  'does not apply to them.',
                  'An abstention is counted separately and never read as a '
                  'negative prediction.',
                  'A negative prediction is a screening result, not evidence '
                  'of safety.']}
    for tool, path in SCORED.items():
        if not path.is_file():
            print(f'  {tool}: 예측 파일이 없습니다 ({path.name})')
            continue
        rows = read_rows(path)
        report['tools'][tool] = {str(line): score_at(rows, truth, keep, line)
                                 for line in thresholds}
        for line in thresholds:
            result = report['tools'][tool][str(line)]
            print(f"  {tool:10s} 기준선 {line:.2f} · 민감도 "
                  f"{result['sensitivity']:.3f} · 특이도 "
                  f"{result['specificity']:.3f} · 균형 {result['balanced']:.3f} "
                  f"· 판정 비율 {result['answered']:.3f}")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding='utf-8')
    print(f'기록: {args.output}')


if __name__ == '__main__':
    main()
