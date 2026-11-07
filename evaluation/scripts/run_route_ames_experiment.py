"""Fresh local ADMET inference over a fixed saved route set; not independent validation."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
from engine import analyze_routes
from admet_service import assess_admet_materials
from ames_route_review import review_ames_routes
from route_selection import select_routes


def main():
    input_path = ROOT / "evaluation" / "results" / "live_routes_input.json"
    raw = input_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    report = analyze_routes(payload)
    structures = list(dict.fromkeys(
        material["input_smiles"]
        for route in report["routes"] for step in route["steps"]
        for material in step["materials"]
    ))
    start = time.perf_counter()
    admet = assess_admet_materials(structures, threshold=0.5, timeout=300)
    elapsed = time.perf_counter() - start
    review = review_ames_routes(report, admet)
    result = {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "source_input": str(input_path.relative_to(ROOT)),
        "scope": "Saved AiZynthFinder routes; target excluded from impurity screening. No new route generation or byproduct prediction.",
        "limitations": [
            "Fresh inference is not experimental mutagenicity validation.",
            "AND exclusion is a research policy, not ICH M7 classification.",
            "Missing conditions/byproducts remain missing; no safe route guarantee.",
        ],
        "elapsed_admet_seconds": elapsed,
        "requested_distinct_input_structures": len(structures),
        "structure_only_review": select_routes(report),
        "structure_only_strict": select_routes(report, strict_alerts=True),
        "analysis_report": report,
        "admet": admet,
        "combined_review": review,
    }
    destination = ROOT / "evaluation" / "results" / "live_route_ames_experiment.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(destination), "structures": len(structures),
                      "status_counts": {s: sum(row["status"] == s for row in admet["assessments"])
                                        for s in ("success", "invalid", "failed")},
                      "review_counts": review["counts"], "elapsed_seconds": elapsed},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
