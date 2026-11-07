"""
테스트 2: 받아온 부산물이 유전독성인지 판정할 수 있는가? (1단계 "파악")

확인하려는 것:
  (1) ICH M7이 지정한 위험 물질 목록과 대조해서 걸러낼 수 있는가
  (2) 경고구조(위험할 수 있다고 알려진 부분 구조)로 걸러낼 수 있는가
  (3) 둘이 각각 다른 것을 잡아내는가 (= 둘 다 필요한가)
"""
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------
# (1) ICH M7 Addendum 지정 위험 물질 (Class 1/2). CAS + 1일 허용량(ug/day)
# ---------------------------------------------------------------
ICH_M7_LIST = [
    ("Acrylonitrile",            "107-13-1", "C=CC#N",            6),
    ("Benzyl chloride",          "100-44-7", "ClCc1ccccc1",       41),
    ("Bis(chloromethyl)ether",   "542-88-1", "ClCOCCl",           0.004),
    ("1-Chloro-4-nitrobenzene",  "100-00-5", "O=[N+]([O-])c1ccc(Cl)cc1", 117),
    ("1,2-Dibromoethane",        "106-93-4", "BrCCBr",            2),
    ("Dimethylcarbamyl chloride","79-44-7",  "CN(C)C(=O)Cl",      5),
    ("Epichlorohydrin",          "106-89-8", "ClCC1CO1",          3),
    ("Ethyl bromide",            "74-96-4",  "CCBr",              32),
    ("Ethyl chloride",           "75-00-3",  "CCCl",              1810),
    ("Glycidol",                 "556-52-5", "OCC1CO1",           4),
    ("Hydrazine",                "302-01-2", "NN",                39),
    ("Methyl chloride",          "74-87-3",  "CCl",               1361),
    ("Styrene",                  "100-42-5", "C=Cc1ccccc1",       154),
    ("Aniline",                  "62-53-3",  "Nc1ccccc1",         720),
    ("Acetaldehyde",             "75-07-0",  "CC=O",              2000),
    ("Formaldehyde",             "50-00-0",  "C=O",               None),
    ("Vinyl acetate",            "108-05-4", "C=COC(C)=O",        2000),
    ("p-Chloroaniline",          "106-47-8", "Nc1ccc(Cl)cc1",     None),
]

# ---------------------------------------------------------------
# (2) 경고구조: DNA와 반응할 수 있다고 알려진 부분 구조
#     (Benigni-Bossa / ICH M7 계열에서 흔히 쓰이는 대표 패턴)
# ---------------------------------------------------------------
STRUCTURAL_ALERTS = [
    ("에폭사이드",          "C1OC1"),
    ("아지리딘",            "C1NC1"),
    ("N-니트로소",          "[NX3][NX2]=[OX1]"),
    ("방향족 니트로",       "c[N+](=O)[O-]"),
    ("방향족 아민",         "[NX3;H2,H1;!$(NC=O)]c"),
    ("알킬 할라이드",       "[CX4;H2,H3][F,Cl,Br,I]"),
    ("설폰산 에스터",       "[#6]OS(=O)(=O)[#6]"),
    ("알데히드",            "[CX3H1](=O)[#6,#1]"),
    ("마이클 수용체",       "C=CC(=O)"),
    ("히드라진",            "[NX3][NX3]"),
    ("아조",                "[#6]N=N[#6]"),
    ("아자이드",            "[N-]=[N+]=[N-]"),
]


def canon(smiles):
    """구조식을 표준형으로 바꿔서 같은 물질인지 비교 가능하게 만든다"""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


# 미리 표준형으로 변환해 둔 ICH M7 조회 표
ICH_LOOKUP = {}
for name, cas, smi, limit in ICH_M7_LIST:
    c = canon(smi)
    if c:
        ICH_LOOKUP[c] = (name, cas, limit)

ALERT_PATTERNS = [(n, Chem.MolFromSmarts(s)) for n, s in STRUCTURAL_ALERTS]


def check_ich_m7(smiles):
    """ICH M7 지정 목록에 있는 물질인가?"""
    return ICH_LOOKUP.get(canon(smiles))


def check_alerts(smiles):
    """경고구조가 몇 개나 걸리는가?"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [n for n, p in ALERT_PATTERNS if p is not None and mol.HasSubstructMatch(p)]


# ---------------------------------------------------------------
# 테스트 1에서 ASKCOS가 돌려준 부산물들
# ---------------------------------------------------------------
BYPRODUCTS = [
    ("CCOC(C)=O",         "Normal reaction"),
    ("CC(=O)O",           "Over-reaction"),
    ("CCOC(=O)CC(C)=O",   "Over-reaction"),
    ("CC=O",              "Over-reaction"),
    ("O=C(Cl)CCC(=O)Cl",  "Dimerization"),
]


def main():
    print("=" * 74)
    print("ASKCOS가 돌려준 부산물 5개를 유전독성 판정")
    print("=" * 74)
    print(f"{'구조식':<20} {'분류':<17} {'ICH M7 지정':<22} {'경고구조'}")
    print(f"{'-'*20} {'-'*17} {'-'*22} {'-'*20}")

    flagged = []
    for smi, mode in BYPRODUCTS:
        ich = check_ich_m7(smi)
        alerts = check_alerts(smi)

        if ich:
            name, cas, limit = ich
            lim = f"{limit} ug/day" if limit else "한도 별도"
            ich_s = f"{name} ({lim})"
        else:
            ich_s = "-"

        alert_s = ", ".join(alerts) if alerts else "-"
        print(f"{smi:<20} {mode:<17} {ich_s:<22} {alert_s}")

        if ich or alerts:
            flagged.append((smi, ich, alerts))

    print()
    print(f"경고가 붙은 물질: {len(flagged)} / {len(BYPRODUCTS)}개")
    print()

    # 두 방법이 각각 다른 것을 잡는지 확인
    only_ich = [s for s, i, a in flagged if i and not a]
    only_alert = [s for s, i, a in flagged if a and not i]
    both = [s for s, i, a in flagged if i and a]

    print("두 방법의 역할 비교")
    print(f"  ICH M7 목록만 잡음 : {only_ich or '없음'}")
    print(f"  경고구조만 잡음   : {only_alert or '없음'}")
    print(f"  둘 다 잡음         : {both or '없음'}")


if __name__ == "__main__":
    main()
