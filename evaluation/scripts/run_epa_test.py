"""Run the EPA TEST mutagenicity consensus over an Ames evaluation file.

Only the prediction columns are read. The output also carries Exp_Value and
Exp_Result, which are experimental values EPA TEST already holds for some
compounds; scoring against those would be reading the answer key, so they are
dropped. A row EPA TEST could not predict stays an abstention rather than
becoming a negative.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'
# EPA TEST is installed per machine by its own installer, so the location
# differs. EPA_TEST_HOME points at it; the default is where version 5.1.2.0 put
# itself on the machine these measurements were run on.
INSTALL = Path(os.environ.get('EPA_TEST_HOME')
               or r'C:\Users\hahab\AppData\Local\Programs\TEST 5.1.2.0')
MAIN_CLASS = 'ToxPredictor.Application.Calculations.RunFromCommandLine'
SOURCE = 'https://www.epa.gov/comptox-tools/toxicity-estimation-software-tool-test'
FIELDS = ['id', 'smiles', 'tool', 'model', 'version', 'endpoint', 'prediction',
          'applicability', 'source', 'score', 'reason']
POSITIVE = 'mutagenicity positive'
NEGATIVE = 'mutagenicity negative'


def run_test(install, smi_path, out_path, timeout):
    # Without headless mode the tool opens a results window on the user's
    # desktop part way through a batch run.
    command = [str(install / 'jre' / 'bin' / 'java.exe'),
               '-Djava.awt.headless=true', '-cp', 'WebTEST.jar',
               MAIN_CLASS, '-i', str(smi_path), '-o', str(out_path),
               '-e', 'Mutagenicity', '-m', 'consensus']
    completed = subprocess.run(command, cwd=install, capture_output=True,
                               text=True, timeout=timeout)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'ames_independent.csv')
    parser.add_argument('--install', type=Path, default=INSTALL)
    parser.add_argument('--workdir', type=Path, default=ROOT / 'epa_test_run')
    parser.add_argument('--output', type=Path, default=ROOT / 'epa_predictions.csv')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'epa_manifest.json')
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))

    smi_path = args.workdir / 'input.smi'
    raw_path = args.workdir / 'raw.csv'
    if raw_path.exists():
        raw_path.unlink()
    # The CLI reads structures only; identity is restored by row order, so the
    # output is rejected below unless its Query column still matches.
    smi_path.write_text('\n'.join(row['smiles'] for row in rows) + '\n', encoding='utf-8')

    completed = run_test(args.install, smi_path, raw_path, args.timeout)
    (args.workdir / 'stdout.log').write_text(completed.stdout or '', encoding='utf-8')
    (args.workdir / 'stderr.log').write_text(completed.stderr or '', encoding='utf-8')
    if not raw_path.exists():
        raise SystemExit(f'EPA TEST wrote no output (exit {completed.returncode})')

    with raw_path.open(encoding='utf-8-sig', newline='') as handle:
        results = list(csv.DictReader(handle))
    by_query = {}
    for result in results:
        by_query.setdefault((result.get('Query') or '').strip(), result)

    counts = {'0': 0, '1': 0, 'abstain': 0}
    with args.output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            result = by_query.get(row['smiles'].strip())
            verdict = (result or {}).get('Pred_Result', '').strip().lower()
            raw_score = (result or {}).get('Pred_Value', '').strip()
            if result is None:
                prediction, reason = 'abstain', 'no_returned_row'
            elif verdict == POSITIVE:
                prediction, reason = '1', 'consensus_positive'
            elif verdict == NEGATIVE:
                prediction, reason = '0', 'consensus_negative_not_proof_of_safety'
            else:
                prediction, reason = 'abstain', f'unusable_result:{verdict or "empty"}'
            counts[prediction] += 1
            writer.writerow({'id': row['id'], 'smiles': row['smiles'], 'tool': 'EPA TEST',
                             'model': 'Mutagenicity consensus', 'version': '5.1.2',
                             'endpoint': 'ames', 'prediction': prediction,
                             'applicability': 'unknown', 'source': SOURCE,
                             'score': raw_score if raw_score not in ('', 'N/A') else '',
                             'reason': reason})

    args.manifest.write_text(json.dumps({
        'tool': 'EPA TEST', 'version': '5.1.2', 'endpoint': 'Mutagenicity',
        'method': 'consensus', 'install': str(args.install),
        'input_rows': len(rows), 'returned_rows': len(results), 'counts': counts,
        'returncode': completed.returncode,
        'dropped_columns': ['Exp_Value', 'Exp_Result'],
        'limitations': [
            'Experimental columns in the tool output were not used for scoring.',
            'EPA TEST training data may overlap public Ames collections; the '
            'independent evaluation set does not remove that overlap.',
            'A consensus negative is a screening result, not evidence of safety.',
        ],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'EPA TEST: {len(rows)} inputs, {len(results)} returned, {counts}')


if __name__ == '__main__':
    main()
