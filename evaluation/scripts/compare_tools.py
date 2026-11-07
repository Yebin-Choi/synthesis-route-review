"""Score every tool and every combination rule on one Ames evaluation file.

All tools are scored on the same rows so the numbers are comparable. Rows a
tool abstains on are reported, never counted as negative. Sensitivity and
specificity condition on rows the combination actually decided; coverage says
how much of the set that was, so a rule cannot look good by deciding little.
"""
import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'


def read_predictions(path, *, prediction_field='prediction'):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return {row['id']: row[prediction_field] for row in csv.DictReader(handle)}


def read_truth(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return {row['id']: row['label'] for row in csv.DictReader(handle)}


def score(decisions, truth):
    tp = tn = fp = fn = abstain = 0
    for identity, label in truth.items():
        call = decisions.get(identity, 'abstain')
        if call == 'abstain':
            abstain += 1
        elif call == '1':
            if label == '1':
                tp += 1
            else:
                fp += 1
        else:
            if label == '0':
                tn += 1
            else:
                fn += 1
    decided = tp + tn + fp + fn
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = ((sensitivity + specificity) / 2
                if sensitivity is not None and specificity is not None else None)
    # Positives found out of every labelled positive, abstentions included, so
    # a rule cannot hide missed positives behind "no decision".
    positives = sum(1 for label in truth.values() if label == '1')
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'abstain': abstain,
        'decided': decided, 'coverage': decided / len(truth) if truth else None,
        'sensitivity': sensitivity, 'specificity': specificity,
        'balanced_accuracy': balanced,
        'positives_caught_of_all_positives': tp / positives if positives else None,
    }


def combine(calls, rule):
    """calls is the list of each tool's call for one structure."""
    decided = [call for call in calls if call in ('0', '1')]
    if not decided:
        return 'abstain'
    if rule == 'any_positive':
        # A tool that abstained cannot vouch for a negative, so a negative
        # verdict requires every tool to have decided.
        if '1' in decided:
            return '1'
        return '0' if len(decided) == len(calls) else 'abstain'
    if rule == 'unanimous':
        if len(decided) != len(calls) or len(set(decided)) != 1:
            return 'abstain'
        return decided[0]
    if rule == 'majority':
        if len(decided) != len(calls):
            return 'abstain'
        positive = decided.count('1')
        if positive * 2 == len(decided):
            return 'abstain'
        return '1' if positive * 2 > len(decided) else '0'
    raise ValueError(f'Unknown rule: {rule}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', type=Path, default=ROOT / 'ames_independent.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'tool_comparison.json')
    parser.add_argument('--tool', action='append', default=[], metavar='NAME=PATH',
                        help='Repeatable; prediction CSV with id and prediction columns')
    args = parser.parse_args()

    truth = read_truth(args.truth)
    tools = {}
    for entry in args.tool:
        name, _, path = entry.partition('=')
        tools[name] = read_predictions(Path(path))

    report = {
        'truth_file': str(args.truth),
        'rows': len(truth),
        'positives': sum(1 for label in truth.values() if label == '1'),
        'negatives': sum(1 for label in truth.values() if label == '0'),
        'tools': {name: str(entry.partition('=')[2]) for name, entry in
                  zip(tools, args.tool)},
        'single': {}, 'combinations': {},
        'notes': [
            'Sensitivity and specificity condition on decided rows; read them with coverage.',
            'positives_caught_of_all_positives divides true positives by every labelled '
            'positive, including rows the rule abstained on.',
            'A no-alert or negative prediction is a screening result, not evidence of safety.',
            'Label provenance is taken from the source collection and is not re-verified here.',
        ],
    }
    for name, calls in tools.items():
        report['single'][name] = score({identity: calls.get(identity, 'abstain')
                                        for identity in truth}, truth)
    names = list(tools)
    for size in range(2, len(names) + 1):
        for chosen in combinations(names, size):
            for rule in ('any_positive', 'unanimous', 'majority'):
                decisions = {}
                for identity in truth:
                    decisions[identity] = combine(
                        [tools[name].get(identity, 'abstain') for name in chosen], rule)
                report['combinations'][f"{'+'.join(chosen)} | {rule}"] = score(decisions, truth)

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    def line(label, result):
        def show(value):
            return f'{value:.3f}' if isinstance(value, float) else '-'
        print(f'{label:52s} 민감도 {show(result["sensitivity"])} '
              f'특이도 {show(result["specificity"])} '
              f'균형 {show(result["balanced_accuracy"])} '
              f'판정비율 {show(result["coverage"])} '
              f'놓친양성 {result["fn"]}')

    print(f"=== 단일 도구 ({report['rows']}개, 양성 {report['positives']})")
    for name, result in report['single'].items():
        line(name, result)
    print('=== 조합')
    for label, result in sorted(report['combinations'].items(),
                                key=lambda item: -(item[1]['balanced_accuracy'] or 0)):
        line(label, result)


if __name__ == '__main__':
    main()
