"""
테스트 3: 발사르탄 사고를 우리 도구가 잡아낼 수 있는가? (3단계 "검증")

배경 (EMA/217823/2019 조사보고서):
  - NDMA는 발사르탄 제조의 "테트라졸 고리 형성 단계"에서 생겼다.
  - 그 단계에서 남은 아자이드를 "아질산(NaNO2)"으로 제거하는 공정을 썼다.
  - 용매로 쓴 DMF가 고온에서 분해되어 디메틸아민(2급 아민)이 생겼다.
  - 2급 아민 + 아질산 + 산성 조건 = 니트로사민 생성.

확인하려는 것:
  우리 도구가 "이 조합은 위험하다"를 경고할 수 있는가?
"""
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------
# 위험 조합 규칙: "이 물질들이 같이 있으면 저 물질이 생긴다"
# ---------------------------------------------------------------
FORMATION_RULES = [
    {
        "이름": "니트로사민 생성",
        "조건": [
            ("2급 아민", "[NX3;H1]([#6])[#6]"),
            ("니트로화제", "[N+](=O)[O-],[NX2]=O"),
        ],
        "생성물": "N-니트로소 화합물 (발암 가능)",
        "근거": "ICH M7 cohort of concern. 발사르탄 사고 원인",
        "심각도": "최고",
    },
    {
        "이름": "알킬 설포네이트 생성",
        "조건": [
            ("알코올", "[OX2H][CX4]"),
            ("설폰산/할라이드", "S(=O)(=O)[OH,Cl]"),
        ],
        "생성물": "알킬 설포네이트 (DNA 반응성)",
        "근거": "ICH M7 대표 사례",
        "심각도": "높음",
    },
]

# 아자이드를 아질산으로 제거하는 공정에서 쓰이는 물질들
NITROSATING_AGENTS = {
    "아질산나트륨 (NaNO2)": "[Na+].[O-][N+]=O",
    "아질산": "ON=O",
}

SECONDARY_AMINE_SOURCES = {
    "디메틸아민 (DMF 분해산물)": "CNC",
    "DMF (용매, 분해되면 위 물질 생성)": "CN(C)C=O",
}


def has_pattern(smiles, smarts):
    mol = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts(smarts)
    if mol is None or patt is None:
        return False
    return mol.HasSubstructMatch(patt)


def check_nitrosamine_risk(materials):
    """공정에 쓰인 물질 목록을 받아 니트로사민 위험을 판정"""
    sec_amine = []
    nitrosating = []

    for name, smi in materials.items():
        if has_pattern(smi, "[NX3;H1]([CX4])[CX4]"):
            sec_amine.append(name)
        # 아질산염/아질산: N에 O가 이중결합 + 단일결합 산소
        if has_pattern(smi, "[NX2](=O)[OX1,OX2H,OX1-]"):
            nitrosating.append(name)

    return sec_amine, nitrosating


def main():
    print("=" * 74)
    print("발사르탄 사고 재현: 당시 공정 물질을 우리 도구에 넣으면?")
    print("=" * 74)
    print()
    print("[EMA 조사보고서에 기술된 당시 공정 물질]")

    # 당시 공정에 실제로 등장한 물질들
    process_materials = {}
    process_materials.update(SECONDARY_AMINE_SOURCES)
    process_materials.update(NITROSATING_AGENTS)

    for name, smi in process_materials.items():
        print(f"  - {name:<38} {smi}")
    print()

    sec_amine, nitrosating = check_nitrosamine_risk(process_materials)

    print("[판정]")
    print(f"  2급 아민 검출    : {sec_amine or '없음'}")
    print(f"  니트로화제 검출  : {nitrosating or '없음'}")
    print()

    if sec_amine and nitrosating:
        print("  >> 경고 발생: 니트로사민 생성 위험")
        print(f"     근거: 2급 아민({sec_amine[0]})과 니트로화제({nitrosating[0]})가 같은 공정에 있음")
        print("     ICH M7 cohort of concern: TTC 1.5 ug/day로도 불충분, 개별 한도 필요")
        print()
        print("  판정: 우리 도구는 이 경로에 경고를 냈을 것이다.")
    else:
        print("  >> 경고 없음")
        print("  판정: 우리 도구는 이 경로를 놓쳤을 것이다.")

    print()
    print("-" * 74)
    print("주의: DMF 자체는 2급 아민이 아니다. 고온에서 분해되어야 디메틸아민이 된다.")
    print("      즉 '분해 경로'까지 보려면 규칙을 추가해야 한다. (한계)")


if __name__ == "__main__":
    main()
