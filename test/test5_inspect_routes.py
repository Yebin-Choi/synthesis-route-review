"""
테스트 5: 필터를 켠 뒤 남은 경로는 정말 안전한가?

테스트 4에서 필터를 켜니 경로가 8개에서 5개로 줄었다.
그런데 남은 5개가 정말 안전한지는 확인하지 않았다.
우리 목록에 없는 다른 위험 물질을 쓰고 있을 수도 있다.

확인하려는 것:
  (1) 남은 경로들이 실제로 어떤 물질을 쓰는가
  (2) 그중 구조 경보가 걸리는 것은 없는가
  (3) 필터가 놓친 위험은 무엇인가 (정직하게)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aizynthfinder.aizynthfinder import AiZynthFinder
from genotox_plugin import GenotoxFilter, find_ich_m7, find_alerts, risk_of

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "config.yml")
TARGET = "CC(C)NCC(O)COc1cccc2ccccc12"   # 프로프라놀롤


def run(use_filter):
    finder = AiZynthFinder(configfile=CONFIG)
    finder.stock.select_first()
    finder.expansion_policy.select_first()
    if use_filter:
        finder.filter_policy.load(GenotoxFilter("genotox", finder.config))
        finder.filter_policy.select("genotox")
    finder.target_smiles = TARGET
    finder.tree_search()
    finder.build_routes()
    return finder


def report(finder, label):
    trees = finder.routes.reaction_trees
    print("=" * 74)
    print(f"{label}: 경로 {len(trees)}개")
    print("=" * 74)

    all_flagged = {}
    for i, tree in enumerate(trees, 1):
        mats = {mol.smiles for mol in tree.molecules()}
        flagged = []
        for smi in mats:
            r = risk_of_smiles(smi)
            if r:
                flagged.append(r)
        status = "위험 물질 없음" if not flagged else f"위험 {len(flagged)}건"
        print(f"  경로 {i}: 물질 {len(mats)}개, {status}")
        for name, smi, basis in flagged:
            print(f"      - {name} ({smi}) : {basis}")
            all_flagged[smi] = (name, basis)
    print()
    return all_flagged


def risk_of_smiles(smi):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    ich = find_ich_m7(mol)
    if ich:
        return (ich[0], smi, f"ICH M7 지정 (허용량 {ich[1]} ug/day)")
    alerts = find_alerts(mol)
    if alerts:
        names = ", ".join(n for n, _ in alerts)
        return ("(목록에 없음)", smi, f"구조 경보: {names}")
    return None


def main():
    print()
    off = run(False)
    flagged_off = report(off, "필터 끔")

    on = run(True)
    flagged_on = report(on, "필터 켬")

    print("=" * 74)
    print("정리")
    print("=" * 74)
    print(f"  필터 끔: 경로 {len(off.routes.reaction_trees)}개, 위험 물질 {len(flagged_off)}종")
    print(f"  필터 켬: 경로 {len(on.routes.reaction_trees)}개, 위험 물질 {len(flagged_on)}종")
    print()

    removed = set(flagged_off) - set(flagged_on)
    remaining = set(flagged_on)

    print("  필터가 제거한 위험 물질:")
    for smi in removed:
        name, basis = flagged_off[smi]
        print(f"    - {name} ({smi}) : {basis}")
    if not removed:
        print("    없음")
    print()

    print("  필터를 켰는데도 남은 위험 물질:")
    for smi in remaining:
        name, basis = flagged_on[smi]
        print(f"    - {name} ({smi}) : {basis}")
    if not remaining:
        print("    없음")
    print()
    print("  참고: 현재 필터는 ICH M7 지정 물질만 차단한다(cutoff 1.0).")
    print("        구조 경보만 걸리는 물질은 통과시키고 점수기로 감점하는 설계다.")


if __name__ == "__main__":
    main()
