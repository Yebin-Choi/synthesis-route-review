"""25 PubChem-verified identities/structures, retrieved 2026-10-29.

Examples are not safety labels or guarantees of a nontrivial synthesizable route.
Parent/free-base entries are not equated with marketed salts or formulations.
"""
_ROWS = [
    (4946, '프로프라놀롤', 'Propranolol', '의약품', 'CC(C)NCC(COC1=CC=CC2=CC=CC=C21)O'),
    (2244, '아스피린', 'Aspirin', '의약품', 'CC(=O)OC1=CC=CC=C1C(=O)O'),
    (1983, '아세트아미노펜', 'Acetaminophen', '의약품', 'CC(=O)NC1=CC=C(C=C1)O'),
    (3672, '이부프로펜', 'Ibuprofen', '의약품', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'),
    (3676, '리도카인', 'Lidocaine', '의약품', 'CCN(CC)CC(=O)NC1=C(C=CC=C1C)C'),
    (4091, '메트포르민', 'Metformin', '의약품', 'CN(C)C(=N)N=C(N)N'),
    (60823, '아토르바스타틴', 'Atorvastatin', '의약품', 'CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4'),
    (2719, '클로로퀸', 'Chloroquine', '의약품', 'CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl'),
    (338, '살리실산', 'Salicylic acid', '생활 속 물질', 'C1=CC=C(C(=C1)C(=O)O)O'),
    (2519, '카페인', 'Caffeine', '식품·천연물', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'),
    (54670067, '비타민 C', 'L-Ascorbic acid', '식품·천연물', 'C([C@@H]([C@@H]1C(=C(C(=O)O1)O)O)O)O'),
    (938, '나이아신 (비타민 B3)', 'Nicotinic acid', '식품·천연물', 'C1=CC(=CN=C1)C(=O)O'),
    (5793, '포도당', 'D-Glucose', '식품·천연물', 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O'),
    (5988, '설탕 (자당)', 'Sucrose', '식품·천연물', 'C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@]2([C@H]([C@@H]([C@H](O2)CO)O)O)CO)O)O)O)O'),
    (311, '구연산', 'Citric acid', '식품·천연물', 'C(C(=O)O)C(CC(=O)O)(C(=O)O)O'),
    (445154, '레스베라트롤', 'Resveratrol', '식품·천연물', 'C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O'),
    (5280343, '퀘르세틴', 'Quercetin', '식품·천연물', 'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O'),
    (969516, '커큐민', 'Curcumin', '식품·천연물', 'COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O'),
    (750, '글리신', 'Glycine', '아미노산', 'C(C(=O)O)N'),
    (5950, 'L-알라닌', 'L-Alanine', '아미노산', 'C[C@@H](C(=O)O)N'),
    (6322, 'L-아르기닌', 'L-Arginine', '아미노산', 'C(C[C@@H](C(=O)O)N)CN=C(N)N'),
    (702, '에탄올', 'Ethanol', '생활 속 물질', 'CCO'),
    (887, '메탄올', 'Methanol', '생활 속 물질', 'CO'),
    (180, '아세톤', 'Acetone', '생활 속 물질', 'CC(=O)C'),
    (176, '아세트산', 'Acetic acid', '생활 속 물질', 'CC(=O)O'),
]
MOLECULES = [
    {'id': f'pubchem-{cid}', 'name_ko': ko, 'name_en': en, 'smiles': smiles,
     'category': category, 'source_url': f'https://pubchem.ncbi.nlm.nih.gov/compound/{cid}',
     'note': 'PubChem 구조 예시입니다. 제품·염 형태와 다를 수 있으며 선택 목록은 안전성 목록이 아닙니다. 재고에 있는 물질은 탐색 경로가 생략될 수 있습니다.'}
    for cid, ko, en, category, smiles in _ROWS
]
