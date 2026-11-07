"""Prepare downloaded TDC AMES without label tuning or claiming independent holdout."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'evaluation' / 'results'
sys.path.insert(0, str(PROJECT / 'app'))
from toxicology import canonicalize_smiles


def main():
    raw = ROOT / 'ames_tdc.tab'
    groups = defaultdict(list)
    excluded = []
    count = 0
    with raw.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            count += 1
            canonical = canonicalize_smiles(row['Drug'])
            if canonical is None or row['Y'] not in ('0', '1'):
                excluded.append({**row, 'reason': 'invalid_structure_or_label'})
            else:
                groups[canonical].append(row)
    records = []
    for canonical, rows in groups.items():
        if len({row['Y'] for row in rows}) != 1:
            excluded.extend({**row, 'reason': 'conflicting_labels_same_canonical_structure'} for row in rows)
            continue
        row = rows[0]
        records.append({'id': row['Drug_ID'], 'smiles': canonical, 'label': row['Y'],
                        'source': 'https://dataverse.harvard.edu/api/access/datafile/4259564',
                        'split': 'test'})
        excluded.extend({**other, 'reason': 'duplicate_canonical_structure'} for other in rows[1:])
    with (ROOT / 'ames_evaluation.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'smiles', 'label', 'source', 'split'])
        writer.writeheader()
        writer.writerows(records)
    manifest = {'source': 'https://dataverse.harvard.edu/api/access/datafile/4259564',
                'metadata': 'https://github.com/mims-harvard/TDC/blob/main/tdc/metadata.py',
                'sha256': hashlib.sha256(raw.read_bytes()).hexdigest(), 'raw_rows': count,
                'retained': len(records), 'class_counts': dict(Counter(r['label'] for r in records)),
                'exclusion_counts': dict(Counter(r['reason'] for r in excluded)),
                'exclusions': excluded,
                'limitation': 'All retained records use split=test solely for fixed-rule descriptive evaluation. This is NOT an independent holdout for pretrained external models. Training overlap is unknown; no thresholds or rules were tuned on these labels. Conflicting structures excluded as a group, not resolved by majority.'}
    (ROOT / 'ames_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'exclusions'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
