"""
테스트 1: ASKCOS 공개 서버에서 부산물을 받아올 수 있는가?

확인하려는 것:
  - 설치 없이 인터넷 요청만으로 부산물 예측을 받을 수 있는가
  - 부산물이 종류별로(주생성물/과반응 등) 구분되어 오는가
"""
import json
import requests

API = "https://askcos.mit.edu/api/impurity-predictor/call-sync"

# 에틸아세테이트 합성: 에탄올 + 염화아세틸 -> 에틸아세테이트
CASES = [
    {
        "이름": "에스터화 (에탄올 + 염화아세틸)",
        "rct_smi": "CCO.CC(=O)Cl",
        "prd_smi": "CCOC(C)=O",
    },
]


def predict_byproducts(rct_smi, prd_smi, timeout=180):
    resp = requests.post(
        API,
        json={"rct_smi": rct_smi, "prd_smi": prd_smi},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    for case in CASES:
        print("=" * 60)
        print(f"반응: {case['이름']}")
        print(f"  반응물: {case['rct_smi']}")
        print(f"  주생성물: {case['prd_smi']}")
        print("-" * 60)

        data = predict_byproducts(case["rct_smi"], case["prd_smi"])
        results = data.get("result", {}).get("predict_expand", [])

        if not results:
            print("  결과 없음. 원본 응답:")
            print(json.dumps(data, ensure_ascii=False)[:1000])
            continue

        print(f"  받은 물질 수: {len(results)}개\n")
        print(f"  {'구조식(SMILES)':<32} {'분류':<20} {'점수'}")
        print(f"  {'-'*32} {'-'*20} {'-'*6}")
        for r in results:
            smi = r.get("prd_smiles", "?")
            mode = r.get("modes_name", "?")
            score = r.get("avg_insp_score", "")
            score_s = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
            print(f"  {smi:<32} {mode:<20} {score_s}")


if __name__ == "__main__":
    main()
