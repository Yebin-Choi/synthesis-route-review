"""외부 서비스 없이 시험용 경로와 선택한 SMILES를 평가하여 JSON으로 출력한다."""
import argparse
import json

import engine


def main():
    parser = argparse.ArgumentParser(
        description="소프트웨어 시험용 예제 경로를 오프라인으로 평가합니다. 모델이나 외부 API를 예열하지 않습니다."
    )
    parser.add_argument(
        "smiles", nargs="*", help="추가로 검사할 SMILES. 물질 이름 검색은 수행하지 않습니다."
    )
    args = parser.parse_args()
    try:
        result = {
            "route_analysis": engine.analyze_routes(engine.load_example()),
            "individual_assessments": [engine.assess_material(value) for value in args.smiles],
        }
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
