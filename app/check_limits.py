"""출처와 적용 범위를 확인한 입력값의 농도 환산만 수행한다."""
import argparse
import json

import engine


def main():
    parser = argparse.ArgumentParser(
        description="확인된 섭취량과 제제의 1일량을 ppm으로 환산합니다. 허용량을 자동 설정하지 않습니다."
    )
    parser.add_argument(
        "--intake-ug-day", type=float,
        help="출처와 적용 범위를 확인한 섭취량 (µg/day). 생략하면 미산정입니다.",
    )
    parser.add_argument(
        "--dose-mg-day", type=float, required=True,
        help="해당 제제의 확인된 1일량 (mg/day). 임의 기본값을 사용하지 않습니다.",
    )
    args = parser.parse_args()
    try:
        concentration = engine.concentration_limit(args.intake_ug_day, args.dose_mg_day)
    except ValueError as exc:
        parser.error(str(exc))
    result = {
        "intake_ug_day": args.intake_ug_day,
        "daily_dose_mg": args.dose_mg_day,
        "concentration_ppm": concentration,
        "display": "미산정" if concentration is None else f"{concentration:g} ppm",
        "note": "입력값의 단위 환산일 뿐 허용 기준의 적합성이나 규제 분류를 결정하지 않습니다.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
