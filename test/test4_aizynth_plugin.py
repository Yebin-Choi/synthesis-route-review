"""
테스트 4: AiZynthFinder에 우리 필터와 점수기를 끼워넣을 수 있는가? (2단계 "제시")

확인하려는 것:
  (1) 원본 코드를 고치지 않고 우리 규칙을 등록할 수 있는가
  (2) 필터가 실제로 위험한 반응 단계를 차단하는가
  (3) 점수기가 경로 전체를 보고 점수를 매기는가
  (4) 필터를 켜고 끌 때 결과가 달라지는가

주의: AI 모델 파일(약 790MB)이 있어야 실제 경로 탐색이 된다.
      모델이 없으면 (1)(2)(3)까지만 확인하고 (4)는 건너뛴다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem


def step1_import():
    """(1) 원본을 안 고치고 우리 클래스를 붙일 수 있는가"""
    print("=" * 70)
    print("(1) 원본 수정 없이 우리 규칙을 등록할 수 있는가")
    print("=" * 70)

    from genotox_plugin import GenotoxFilter, GenotoxScorer
    from aizynthfinder.context.policy.filter_strategies import FilterStrategy
    from aizynthfinder.context.scoring import Scorer

    ok_filter = issubclass(GenotoxFilter, FilterStrategy)
    ok_scorer = issubclass(GenotoxScorer, Scorer)

    print(f"  GenotoxFilter 가 AiZynthFinder 의 FilterStrategy 를 상속: {ok_filter}")
    print(f"  GenotoxScorer 가 AiZynthFinder 의 Scorer 를 상속       : {ok_scorer}")
    print()
    print("  -> AiZynthFinder 원본 파일은 하나도 고치지 않았다.")
    print("     설정 파일에 이름만 적으면 불러온다.")
    print()
    return ok_filter and ok_scorer


def step2_risk_judgement():
    """(2) 위험 판정 규칙이 제대로 동작하는가"""
    print("=" * 70)
    print("(2) 위험 판정이 제대로 되는가")
    print("=" * 70)

    from genotox_plugin import risk_of, find_ich_m7, find_alerts

    cases = [
        ("CCO",            "에탄올 (안전해야 함)"),
        ("CC=O",           "아세트알데히드 (ICH M7 지정)"),
        ("ClCC1CO1",       "에피클로로히드린 (ICH M7 지정)"),
        ("NN",             "히드라진 (ICH M7 지정)"),
        ("CN(C)N=O",       "NDMA (발사르탄 사고 물질)"),
        ("c1ccccc1",       "벤젠 (목록에 없음)"),
        ("Nc1ccc(C)cc1",   "p-톨루이딘 (목록엔 없지만 방향족 아민)"),
    ]

    print(f"  {'구조식':<16} {'설명':<32} {'위험도':<7} {'근거'}")
    print(f"  {'-'*16} {'-'*32} {'-'*7} {'-'*30}")
    for smi, desc in cases:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  {smi:<16} {desc:<32} 파싱실패")
            continue
        risk = risk_of(mol)
        ich = find_ich_m7(mol)
        alerts = find_alerts(mol)
        if ich:
            basis = f"ICH M7 지정: {ich[0]}"
        elif alerts:
            basis = "구조 경보: " + ", ".join(n for n, _ in alerts)
        else:
            basis = "-"
        print(f"  {smi:<16} {desc:<32} {risk:<7.2f} {basis}")
    print()


def step3_scorer_depth():
    """(3) 점수기가 '마지막 단계에 가까울수록 더 감점'을 하는가"""
    print("=" * 70)
    print("(3) 점수기가 경로 내 위치를 반영하는가")
    print("=" * 70)

    from genotox_plugin import GenotoxScorer

    scorer = GenotoxScorer()
    mol = Chem.MolFromSmiles("ClCC1CO1")   # 에피클로로히드린, 위험도 1.0

    print("  같은 위험 물질이 경로의 어디에 있느냐에 따른 점수")
    print()
    print(f"  {'깊이':<6} {'의미':<28} {'점수'}")
    print(f"  {'-'*6} {'-'*28} {'-'*6}")
    labels = {
        0: "최종 단계 (씻어낼 기회 없음)",
        1: "한 단계 앞",
        2: "두 단계 앞",
        3: "세 단계 앞",
        5: "다섯 단계 앞 (충분히 씻김)",
    }
    for depth in [0, 1, 2, 3, 5]:
        p = scorer._penalty(mol, depth)
        print(f"  {depth:<6} {labels[depth]:<28} {p:.3f}")
    print()
    print("  -> 깊이가 얕을수록(마지막 단계에 가까울수록) 점수가 낮다.")
    print("     이것은 반응 한 단계만 보는 '필터'로는 낼 수 없는 판단이다.")
    print()


def step4_real_search():
    """(4) 실제 경로 탐색에서 필터가 작동하는가 (모델 파일 필요)"""
    print("=" * 70)
    print("(4) 실제 경로 탐색에서 필터를 켜고 끄면 결과가 달라지는가")
    print("=" * 70)

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "models", "config.yml")
    if not os.path.exists(config_path):
        print(f"  건너뜀: 모델 설정 파일이 없다 ({config_path})")
        print("  먼저 'download_public_data' 로 모델을 받아야 한다.")
        return False

    from aizynthfinder.aizynthfinder import AiZynthFinder

    # 프로프라놀롤 (고혈압약).
    # 공업적으로 1-나프톨 + 에피클로로히드린으로 만든다.
    # 에피클로로히드린은 ICH M7 지정 위험 물질 (허용량 3 ug/day).
    # 즉 "위험 시약을 쓰는 경로"가 실제로 제안될 가능성이 높은 표적이다.
    target = "CC(C)NCC(O)COc1cccc2ccccc12"

    results = {}
    for use_filter in [False, True]:
        finder = AiZynthFinder(configfile=config_path)
        finder.stock.select_first()
        finder.expansion_policy.select_first()

        rejected = []
        if use_filter:
            from genotox_plugin import GenotoxFilter
            f = GenotoxFilter("genotox", finder.config)
            finder.filter_policy.load(f)
            finder.filter_policy.select("genotox")
            rejected = f.rejected

        finder.target_smiles = target
        finder.tree_search()
        finder.build_routes()

        n = len(finder.routes)
        stats = finder.search_stats or {}
        solved = stats.get("is_solved", "?")
        iters = stats.get("iterations", "?")
        results["필터 켬" if use_filter else "필터 끔"] = (n, solved, iters, list(rejected))

    print(f"  표적 물질: {target}")
    print()
    print(f"  {'설정':<10} {'찾은 경로 수':<14} {'해결':<8} {'반복 횟수'}")
    print(f"  {'-'*10} {'-'*14} {'-'*8} {'-'*10}")
    for k, (n, s, it, _) in results.items():
        print(f"  {k:<10} {n:<14} {str(s):<8} {it}")

    rej = results.get("필터 켬", (0, 0, 0, []))[3]
    print()
    if rej:
        print(f"  필터가 차단한 반응: {len(rej)}건")
        seen = set()
        for smi, name in rej:
            if name not in seen:
                seen.add(name)
                print(f"    - {name} ({smi})")
    else:
        print("  필터가 차단한 반응: 없음")
        print("  (이 표적은 위험 시약을 쓰는 경로가 애초에 제안되지 않았다는 뜻)")
    print()
    return True


def main():
    ok = step1_import()
    if not ok:
        print("상속 확인 실패. 이후 단계 중단.")
        return
    step2_risk_judgement()
    step3_scorer_depth()
    step4_real_search()


if __name__ == "__main__":
    main()
