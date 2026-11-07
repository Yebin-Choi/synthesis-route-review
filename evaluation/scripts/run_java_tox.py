"""Run the official Toxtree Ames CLI and map its real structural-alert CSV.

Only Toxtree's documented -n/-i/-o/-m interface is used. VEGA is not executed.
The no-alert category is a negative SCREEN, not evidence of experimental safety.
Source: https://toxtree.sourceforge.net/faq.html#k3
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'
ARCHIVE = PROJECT / 'external_tools' / 'Toxtree-v3.1.0.1851.zip'
ARCHIVE_SHA256 = '222479c996f6d91616c135b7f433c8176139d63672bd28c1ccdad6372b3d82e8'
DISTRIBUTION_PREFIX = 'Toxtree-v3.1.0.1851/Toxtree/'
JAR_NAME = 'Toxtree-3.1.0.1851.jar'
MODULE = 'toxtree.plugins.ames.AmesMutagenicityRules'
SOURCE = 'https://toxtree.sourceforge.net/ames.html'
POSITIVE = 'Structural Alert for S. typhimurium mutagenicity'
NEGATIVE = 'No alerts for S. typhimurium mutagenicity'
ERROR = 'Error when applying the decision tree'
AMES_RULE_FRAME = 'at toxtree.plugins.ames.rules.'
FIELDS = ['id', 'smiles', 'tool', 'model', 'version', 'endpoint', 'prediction',
          'applicability', 'source', 'reason']


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_inputs(path, limit):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'id', 'smiles'}.issubset(reader.fieldnames or []):
            raise ValueError('Input CSV requires id and smiles columns')
        rows = []
        seen = set()
        for row in reader:
            if not row['id'] or not row['smiles'] or row['id'] in seen:
                raise ValueError('Input IDs must be nonempty and unique; SMILES must be nonempty')
            seen.add(row['id'])
            rows.append({'id': row['id'], 'smiles': row['smiles']})
            if limit is not None and len(rows) == limit:
                break
    if not rows:
        raise ValueError('Input CSV has no molecules')
    return rows


def extract_distribution(archive, destination):
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Toxtree archive does not match the inspected official distribution SHA256')
    home = destination / 'Toxtree'
    home.mkdir()
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            if not info.filename.startswith(DISTRIBUTION_PREFIX) or info.is_dir():
                continue
            relative = Path(info.filename[len(DISTRIBUTION_PREFIX):])
            target = home / relative
            if relative.is_absolute() or '..' in relative.parts or not target.resolve().is_relative_to(home.resolve()):
                raise ValueError('Unsafe archive member')
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as reader, target.open('xb') as writer:
                shutil.copyfileobj(reader, writer)
    for required in [home / JAR_NAME, home / 'ext' / 'toxtree-ames-3.1.0.jar']:
        if not required.is_file():
            raise ValueError(f'Official distribution is missing {required.name}')
    return home


def normalize_header(value):
    return ' '.join(value.split())


def load_results(path, expected_ids):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        original = reader.fieldnames or []
        headers = [normalize_header(name) for name in original]
        if len(headers) != len(set(headers)):
            raise ValueError('Duplicate Toxtree output columns')
        required = {'evaluation_id', POSITIVE, NEGATIVE, ERROR}
        if not required.issubset(headers):
            raise ValueError(f'Unrecognized Toxtree CSV schema; missing {sorted(required - set(headers))}; got {headers}')
        results = {}
        for values in reader:
            if None in values or any(value is None for value in values.values()):
                raise ValueError('Malformed Toxtree CSV record')
            row = {normalize_header(key): value for key, value in values.items()}
            molecule_id = row['evaluation_id']
            if molecule_id not in expected_ids or molecule_id in results:
                raise ValueError(f'Unknown or repeated Toxtree evaluation_id: {molecule_id}')
            results[molecule_id] = row
    if not results:
        raise ValueError('Toxtree returned no molecular records')
    return results


def map_prediction(row):
    if row is None:
        return 'abstain', 'missing_tool_record'
    values = [row[column].strip().upper() for column in (POSITIVE, NEGATIVE, ERROR)]
    if any(value not in {'YES', 'NO'} for value in values):
        return 'abstain', 'unrecognized_category_value'
    positive, negative, error = values
    if error == 'YES':
        return 'abstain', 'tool_error_category'
    if positive == 'YES' and negative == 'NO':
        return '1', 'structural_alert_present'
    if negative == 'YES' and positive == 'NO':
        return '0', 'no_structural_alert_not_proof_of_safety'
    return 'abstain', 'ambiguous_structural_alert_categories'


def run_batch(rows, *, home, java, timeout, work_dir, label):
    """Run the documented CLI once and report whether Java logged an exception.

    Toxtree catches rule exceptions internally: the process still exits 0 and
    the per-record error column still reads NO, so the log is the only signal
    that a rule did not actually evaluate. The log never names the record, so
    an exception here only means "somewhere in this batch".
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / f'toxtree_input_{label}.csv'
    raw_path = work_dir / f'toxtree_raw_{label}.csv'
    with input_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['evaluation_id', 'SMILES'])
        writer.writeheader()
        writer.writerows({'evaluation_id': row['id'], 'SMILES': row['smiles']} for row in rows)
    command = [java, '-Xmx2048m', '-Djava.awt.headless=true', '-Dfile.encoding=UTF-8',
               '-jar', str(home / JAR_NAME), '-n', '-i', str(input_path), '-o', str(raw_path),
               '-m', MODULE]
    stdout_path = work_dir / f'toxtree_stdout_{label}.log'
    stderr_path = work_dir / f'toxtree_stderr_{label}.log'
    with stdout_path.open('w', encoding='utf-8') as stdout, stderr_path.open('w', encoding='utf-8') as stderr:
        subprocess.run(command, cwd=home, stdout=stdout, stderr=stderr, check=True, timeout=timeout)
    stderr_text = stderr_path.read_text(encoding='utf-8', errors='replace')
    # Startup noise (for example an unrelated plugin's ClassNotFoundException)
    # is not an Ames failure. Only a trace inside the Ames rule package means a
    # rule did not evaluate for some record in this batch.
    raised = AMES_RULE_FRAME in stderr_text
    try:
        results = load_results(raw_path, {row['id'] for row in rows})
    except ValueError:
        # Toxtree sometimes writes nothing or an unusable file for a batch and
        # still exits 0. Treat that as a batch failure so the records are
        # re-run individually instead of silently disappearing.
        return {}, True
    return results, raised


def resolve_batch(rows, *, home, java, timeout, work_dir, label):
    """Return {id: (row_or_None, reason)} with exceptions attributed per record.

    A clean batch is accepted as it stands. A batch whose log carries an
    exception is re-run one record at a time, so only the records that
    actually raise are withheld instead of the whole batch.
    """
    results, raised = run_batch(rows, home=home, java=java, timeout=timeout,
                                work_dir=work_dir, label=label)
    if not raised:
        return {row['id']: (results.get(row['id']), None) for row in rows}
    if len(rows) == 1:
        identity = rows[0]['id']
        record = results.get(identity)
        # A crashed rule can only hide an alert, never invent one. So a found
        # alert still stands, while a no-alert verdict from this record cannot
        # be trusted and is withheld rather than counted as a negative.
        if record is not None and map_prediction(record)[0] == '1':
            return {identity: (record, None)}
        return {identity: (None, 'record_model_exception')}
    # Measured on this dataset: about one record in five raises, so halving
    # keeps splitting all the way down and costs more Java starts than simply
    # re-running each record once. Attribute linearly instead.
    resolved = {}
    for index, row in enumerate(rows):
        resolved.update(resolve_batch([row], home=home, java=java, timeout=timeout,
                                      work_dir=work_dir, label=f'{label}_{index}'))
    return resolved


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'ames_evaluation.csv')
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'toxtree_run',
                        help='Must not already exist; contains raw input/output, logs, predictions and extracted binaries')
    parser.add_argument('--java', default='java')
    parser.add_argument('--limit', type=positive_integer, help='First N input records for a parent-run smoke check')
    parser.add_argument('--timeout', type=positive_integer, default=1800, help='Java subprocess timeout in seconds')
    parser.add_argument('--chunk-size', type=positive_integer, default=50,
                        help='Records per Java batch; a batch that logs an exception is re-run record by record')
    args = parser.parse_args()
    rows = load_inputs(args.input.resolve(), args.limit)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    home = extract_distribution(args.archive.resolve(), output_dir)
    input_path = output_dir / 'toxtree_input.csv'
    raw_path = output_dir / 'toxtree_raw.csv'
    predictions_path = output_dir / 'toxtree_predictions.csv'
    with input_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['evaluation_id', 'SMILES'])
        writer.writeheader()
        writer.writerows({'evaluation_id': row['id'], 'SMILES': row['smiles']} for row in rows)
    command = [args.java, '-Xmx2048m', '-Djava.awt.headless=true', '-Dfile.encoding=UTF-8',
               '-jar', str(home / JAR_NAME), '-n', '-i', str(input_path), '-o', str(raw_path), '-m', MODULE]
    manifest = {
        'tool': 'Toxtree', 'version': '3.1.0.1851', 'plugin_version': '3.1.0',
        'module': MODULE, 'archive_sha256': ARCHIVE_SHA256,
        'application_jar_sha256': sha256(home / JAR_NAME),
        'ames_plugin_sha256': sha256(home / 'ext' / 'toxtree-ames-3.1.0.jar'),
        'input_sha256': sha256(args.input.resolve()), 'input_rows': len(rows),
        'command': command, 'working_directory': str(home), 'status': 'started',
        'mapping': 'Structural Alert YES / No alerts NO => 1; converse => 0; errors, ambiguous categories, missing records => abstain. QSAR-specific categories are retained in raw output but are not used by this structural-alert model.',
        'limitations': 'No general applicability-domain claim; applicability is unknown. A negative alert screen is not experimental proof of safety. Possible public AMES/training overlap has not been assessed.'
    }
    manifest_path = output_dir / 'toxtree_manifest.json'
    manifest['chunk_size'] = args.chunk_size
    manifest['exception_attribution'] = (
        'Batches whose Java log carries an exception are re-run one record at a '
        'time. Only records that raise on their own are withheld; a rule that '
        'crashed is never recorded as a no-alert result.'
    )
    try:
        batches = [rows[start:start + args.chunk_size]
                   for start in range(0, len(rows), args.chunk_size)]
        resolved = {}
        for index, batch in enumerate(batches):
            resolved.update(resolve_batch(batch, home=home, java=args.java,
                                          timeout=args.timeout, work_dir=output_dir / 'batches',
                                          label=f'b{index:04d}'))
        withheld = sorted(identity for identity, (_, reason) in resolved.items() if reason)
        manifest['records_withheld_for_model_exception'] = withheld
        results = {identity: row for identity, (row, _) in resolved.items() if row is not None}
        counts = {'0': 0, '1': 0, 'abstain': 0}
        with predictions_path.open('x', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                record, failure = resolved.get(row['id'], (None, 'missing_record'))
                if failure:
                    prediction, reason = 'abstain', failure
                else:
                    prediction, reason = map_prediction(record)
                counts[prediction] += 1
                writer.writerow({**row, 'tool': 'Toxtree', 'model': 'ISS-Ames-structural-alerts',
                                 'version': '3.1.0', 'endpoint': 'ames', 'prediction': prediction,
                                 'applicability': 'unknown', 'source': SOURCE, 'reason': reason})
        manifest.update(status='completed', returned_rows=len(results), counts=counts,
                        predictions_sha256=sha256(predictions_path))
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Toxtree completed: {len(rows)} inputs, {len(results)} tool records, {counts}; {predictions_path}')


if __name__ == '__main__':
    main()
