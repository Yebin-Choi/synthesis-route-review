"""Build an Ames evaluation set that ADMET-AI's training data does not cover.

ADMET-AI's AMES model is trained on the TDC AMES collection, which is the same
set as evaluation/results/ames_evaluation.csv. Scoring it there measures recall of its
own training data, so it cannot be compared with rule systems that never saw
those labels. This script removes every structure that appears in that file
from the independently published MutagenLou2023 collection and keeps the rest.

Structure identity uses the project's own canonicalization, so it does not
detect overlap across salts, tautomers or related stereoisomers. Remaining
overlap is therefore possible and is reported as a limitation, not denied.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'
sys.path.insert(0, str(PROJECT / 'app'))

from toxicology import canonicalize_smiles  # noqa: E402

SOURCE = 'https://huggingface.co/datasets/maomlab/MutagenLou2023'
LICENSE = 'CC BY 4.0'
FIELDS = ['id', 'smiles', 'label', 'source', 'split']


def read_training_structures(path):
    seen = set()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            identity = canonicalize_smiles(row['smiles'])
            if identity:
                seen.add(identity)
    return seen


def read_candidates(paths):
    rows = []
    for path in paths:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            # The published validation split carries no ID column, so rows from
            # it are identified by file and position instead of an invented id.
            for index, row in enumerate(csv.DictReader(handle)):
                label = row['Y'].strip()
                if label not in ('0', '1'):
                    continue
                raw_id = row.get('ID') or f'{path.stem}-{index}'
                rows.append({'origin': path.name, 'raw_id': raw_id,
                             'raw_smiles': row['SMILES'], 'label': label})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--excluded', type=Path, default=ROOT / 'ames_evaluation.csv',
                        help='Benchmark whose structures must not appear in the output')
    parser.add_argument('--candidates', type=Path, nargs='+',
                        default=sorted((ROOT / 'independent_set').glob('*.csv')))
    parser.add_argument('--output', type=Path, default=ROOT / 'ames_independent.csv')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'ames_independent_manifest.json')
    args = parser.parse_args()

    excluded = read_training_structures(args.excluded)
    candidates = read_candidates(args.candidates)

    kept, seen, overlap, unreadable, duplicate = [], set(), 0, 0, 0
    for row in candidates:
        identity = canonicalize_smiles(row['raw_smiles'])
        if identity is None:
            unreadable += 1
            continue
        if identity in excluded:
            overlap += 1
            continue
        if identity in seen:
            duplicate += 1
            continue
        seen.add(identity)
        kept.append({'id': f"lou-{row['raw_id']}", 'smiles': identity, 'label': row['label'],
                     'source': SOURCE, 'split': 'test'})

    with args.output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept)

    manifest = {
        'purpose': 'Ames evaluation set disjoint from the TDC-derived benchmark, so that a '
                   'model trained on that benchmark and a rule system are scored on the same rows.',
        'candidate_files': [str(path) for path in args.candidates],
        'candidate_rows': len(candidates),
        'excluded_file': str(args.excluded),
        'excluded_structures': len(excluded),
        'removed_as_overlap': overlap,
        'removed_as_duplicate': duplicate,
        'removed_as_unreadable': unreadable,
        'retained_rows': len(kept),
        'positives': sum(row['label'] == '1' for row in kept),
        'negatives': sum(row['label'] == '0' for row in kept),
        'source': SOURCE,
        'license': LICENSE,
        'normalization': 'toxicology.canonicalize_smiles: atom-map-free isomeric canonical SMILES. '
                         'No salt stripping, neutralization or tautomer merging, so overlap across '
                         'those forms is not detected and may remain.',
        'limitations': [
            'Independence from ADMET-AI training data is established only for exact canonical '
            'structures present in the excluded benchmark, not for the model vendor\'s full '
            'training corpus, which is not published here.',
            'Toxtree rule development may still overlap these labels; rule systems have no '
            'published training split to subtract.',
            'Labels are taken from the source collection and are not independently verified.',
        ],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"retained {len(kept)} rows "
          f"(positives {manifest['positives']}, negatives {manifest['negatives']}); "
          f"removed overlap {overlap}, duplicate {duplicate}, unreadable {unreadable}")


if __name__ == '__main__':
    main()
