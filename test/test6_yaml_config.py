"""
테스트 6: 설정 파일(YAML)에 이름만 적어도 우리 모듈이 불러와지는가?

앞선 테스트 4는 파이썬 코드로 직접 등록했다.
실제 사용은 설정 파일에 적는 방식이므로, 그것도 되는지 확인한다.

확인하려는 것:
  설정 파일에 두 줄 적는 것만으로 filter policy와 scorer가 등록되는가
  (= AiZynthFinder 원본을 정말 안 고쳐도 되는가)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # genotox_plugin.py 를 찾을 수 있게

from aizynthfinder.aizynthfinder import AiZynthFinder

CONFIG = os.path.join(HERE, "models", "config_genotox.yml")


def main():
    print("=" * 70)
    print("설정 파일만으로 우리 모듈이 등록되는가")
    print("=" * 70)
    print(f"설정 파일: {CONFIG}\n")

    finder = AiZynthFinder(configfile=CONFIG)

    print("불러와진 filter policy:")
    for name in finder.filter_policy.items:
        obj = finder.filter_policy[name]
        mark = "  <- 우리가 만든 것" if "Genotox" in type(obj).__name__ else ""
        print(f"  - {name:<12} ({type(obj).__name__}){mark}")

    print("\n불러와진 scorer:")
    for name in finder.scorers.names():
        obj = finder.scorers[name]
        mark = "  <- 우리가 만든 것" if "Genotox" in type(obj).__name__ else ""
        print(f"  - {name:<24} ({type(obj).__name__}){mark}")

    ok_filter = any("Genotox" in type(finder.filter_policy[n]).__name__
                    for n in finder.filter_policy.items)
    ok_scorer = any("Genotox" in type(finder.scorers[n]).__name__
                    for n in finder.scorers.names())

    print()
    print("-" * 70)
    print(f"  filter policy 등록: {'성공' if ok_filter else '실패'}")
    print(f"  scorer 등록       : {'성공' if ok_scorer else '실패'}")
    print()
    if ok_filter and ok_scorer:
        print("  AiZynthFinder 원본 파일을 고치지 않고")
        print("  설정 파일에 이름만 적어서 우리 모듈을 붙였다.")


if __name__ == "__main__":
    main()
