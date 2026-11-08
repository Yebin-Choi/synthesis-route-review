"""Research-only route screening, with explicit missing-data and provenance states."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import requests
from rdkit import Chem

from toxicology import assess_material, canonicalize_smiles

HERE = Path(__file__).resolve().parent
ASKCOS_API = "https://askcos.mit.edu/api/impurity-predictor/call-sync"
MAX_TARGET_LENGTH = 4096
DISCLAIMER = (
    "연구용 초기 선별입니다. 경보 미검출은 안전성 입증이 아니며, 부산물 후보는 실제 생성·잔류량이 아닙니다. "
    "표시된 섭취량은 출처가 있는 참고치이며 규제 적합성 판정이 아닙니다. "
    "현재 규칙은 제한적이며 단계 간 잔류·정제·분해 및 모든 불순물 생성 반응을 모델링하지 않습니다."
)
ROLES = ("reactants", "products", "reagents", "solvents")
ADDITIONAL_ROLES = ("catalysts", "ligands", "bases", "coproducts")
EVIDENCE_STATES = {"observed", "proposed", "unknown"}
SECONDARY_AMINE = Chem.MolFromSmarts("[NX3;H1;!$(NC=O);!$(NS(=O)=O)]([#6])[#6]")
NITRITE = Chem.MolFromSmarts("[O;H1,-1][N]=O")


class NoRoutesError(ValueError):
    """The bounded native search returned no route containing reaction steps."""


def load_example() -> dict:
    return json.loads((HERE / "example_routes.json").read_text(encoding="utf-8"))


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: 비어 있지 않은 문자열이 필요합니다.")
    return value.strip()


def _strings(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) > 200:
        raise ValueError(f"{label}: 최대 200개 문자열의 배열이 필요합니다.")
    for item in value:
        _text(item, label)


def validate_target_smiles(smiles: str) -> str:
    """Validate a target before parsing; preserve its full structural identity.

    The 4096-character limit includes surrounding whitespace. The returned
    identity uses the existing map-free canonicalizer, without changing salts,
    charge, isotopes or stereochemistry. Raw input is not modified.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("목표물 SMILES는 비어 있지 않은 문자열이어야 합니다.")
    if len(smiles) > MAX_TARGET_LENGTH:
        raise ValueError("목표물 SMILES는 최대 4096자까지 입력할 수 있습니다. 잘라서 처리하지 않습니다.")
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        raise ValueError("유효한 목표물 SMILES를 입력하세요. 이름 자동 변환은 사용하지 않습니다.")
    return canonical


def validate_routes(payload: Any) -> dict:
    """Reject schema errors; retain invalid material SMILES for visible assessment."""
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("schema_version이 '1.0'인 경로 JSON이 필요합니다. 입력 예제를 참고하세요.")
    routes = payload.get("routes")
    if not isinstance(routes, list) or not 1 <= len(routes) <= 50:
        raise ValueError("routes는 1~50개 경로 배열이어야 합니다.")
    route_ids = set()
    for route in routes:
        if not isinstance(route, dict):
            raise ValueError("경로는 JSON 객체여야 합니다.")
        rid = _text(route.get("id"), "경로 id")
        if rid in route_ids:
            raise ValueError(f"중복 경로 id: {rid}")
        route_ids.add(rid)
        target = validate_target_smiles(route.get("target"))
        _text(route.get("source"), f"{rid} source")
        steps = route.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= 100:
            raise ValueError(f"{rid}: steps는 1~100개 단계 배열이어야 합니다.")
        step_ids = set()
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("반응 단계는 JSON 객체여야 합니다.")
            sid = _text(step.get("id"), "단계 id")
            allowed = {"id", "conditions", "byproducts", "material_provenance", "reaction_type",
                       *ROLES, *ADDITIONAL_ROLES}
            unsupported = set(step) - allowed
            if unsupported:
                raise ValueError(f"{rid}/{sid}: 지원하지 않는 단계 필드 {sorted(unsupported)}. 물질 정보를 조용히 생략하지 않습니다.")
            if sid in step_ids:
                raise ValueError(f"{rid}: 중복 단계 id {sid}")
            step_ids.add(sid)
            for role in ROLES:
                _strings(step.get(role), f"{rid}/{sid}/{role}")
            for role in ADDITIONAL_ROLES:
                _strings(step.get(role, []), f"{rid}/{sid}/{role}")
            provenance = step.get("material_provenance", {})
            if not isinstance(provenance, dict) or set(provenance) - {*ROLES, *ADDITIONAL_ROLES, "byproducts"}:
                raise ValueError(f"{rid}/{sid}: material_provenance는 지원되는 물질 역할별 근거 객체여야 합니다.")
            for role, evidence in provenance.items():
                if (not isinstance(evidence, dict) or not isinstance(evidence.get("status"), str)
                        or evidence["status"] not in EVIDENCE_STATES):
                    raise ValueError(f"{rid}/{sid}/{role}: 근거 상태 observed/proposed/unknown이 필요합니다.")
                _text(evidence.get("source"), f"{rid}/{sid}/{role} 물질 근거 출처")
            if not step["reactants"] or not step["products"]:
                raise ValueError(f"{rid}/{sid}: 반응물·생성물이 필요합니다.")
            condition = step.get("conditions")
            bp = step.get("byproducts")
            for field, metadata in (("conditions", condition), ("byproducts", bp)):
                try:
                    if not json_depth_within(metadata):
                        raise ValueError(f"{MAX_JSON_DEPTH}단계보다 깊게 중첩되었습니다.")
                    json.dumps(metadata, allow_nan=False, sort_keys=True)
                except (TypeError, ValueError, RecursionError) as exc:
                    raise ValueError(f"{rid}/{sid}/{field}: 유한한 수와 JSON으로 직렬화할 수 있는 값만 허용합니다.") from exc
            if not isinstance(condition, dict) or condition.get("status") not in ("observed", "proposed", "unknown"):
                raise ValueError(f"{rid}/{sid}: conditions.status는 observed/proposed/unknown이어야 합니다.")
            if condition["status"] != "unknown":
                _text(condition.get("source"), f"{rid}/{sid} 조건 출처")
                _text(condition.get("description"), f"{rid}/{sid} 조건 설명")
            unsupported_conditions = set(condition) - {"status", "source", "description"}
            if unsupported_conditions:
                raise ValueError(f"{rid}/{sid}: 조건 내부 필드 {sorted(unsupported_conditions)}는 아직 지원하지 않습니다. 촉매·리간드·염기·용매 구조는 해당 단계 역할 배열에 넣어야 검사됩니다.")
            if not isinstance(bp, dict) or bp.get("status") not in ("success", "not_run", "failed"):
                raise ValueError(f"{rid}/{sid}: byproducts.status는 success/not_run/failed이어야 합니다.")
            _strings(bp.get("candidates"), f"{rid}/{sid} 부산물 후보")
            if bp["status"] == "success":
                _text(bp.get("source"), f"{rid}/{sid} 부산물 출처")
            elif bp["candidates"]:
                raise ValueError(f"{rid}/{sid}: 미실행·실패 상태에 성공 후보를 혼합할 수 없습니다.")
        if not any(canonicalize_smiles(s) == target for st in steps for s in st["products"]):
            raise ValueError(f"{rid}: 목표물 target이 어떤 단계의 생성물에도 없습니다.")
    return payload


def _conditional_warnings(step: dict) -> list[dict]:
    structures = [canonicalize_smiles(s) for role in
                  ("reactants", "reagents", "solvents", *ADDITIONAL_ROLES)
                  for s in step.get(role, [])]
    mols = [Chem.MolFromSmiles(s) for s in structures if s is not None]
    if any(m.HasSubstructMatch(SECONDARY_AMINE) for m in mols) and any(m.HasSubstructMatch(NITRITE) for m in mols):
        return [{
            "code": "secondary_amine_nitrite",
            "message": "같은 단계에 2급 아민과 아질산/아질산염이 함께 기재되어 있습니다. 니트로사민 생성 가능성 검토가 필요합니다. 생성 확정이나 생성량 예측은 아닙니다.",
            "basis": "동일 단계 구조 공존 규칙. pH·온도·농도·처리 순서 확인 필요. 단계 간 잔류 및 DMF 분해는 미모델링.",
            "source_url": "https://www.ema.europa.eu/en/documents/referral/sartan-article-31-referral-assessment-report_en.pdf",
        }]
    return []


# Validation used to lean on json.dumps refusing pathological input. It no
# longer does. Measured on Python 3.14: a 2,000-deep list serialises without
# error, while copy.deepcopy of that same value raises RecursionError. So a
# value could pass validation here and blow the stack later, somewhere with
# no handling for it, which is what two of this project's tests were quietly
# catching.
#
# The bound is therefore checked directly rather than inferred from what the
# interpreter's serialiser happens to tolerate. A real report nests 9 deep
# and a generated route payload 18, so 64 leaves plenty of room; anything
# deeper is malformed input rather than data.
MAX_JSON_DEPTH = 64


def json_depth_within(value, limit=MAX_JSON_DEPTH) -> bool:
    """Is every branch of this value shallower than the limit?

    Iterative on purpose: a recursive check would consume the very stack the
    bound exists to protect.
    """
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if depth > limit:
            return False
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend((child, depth + 1) for child in item)
    return True


def analyze_routes(payload: dict) -> dict:
    validate_routes(payload)
    output = []
    for route in payload["routes"]:
        target = canonicalize_smiles(route["target"])
        unique = {}
        steps_out = []
        conditions_known = byproducts_run = warning_count = 0
        for step in route["steps"]:
            materials = []
            missing = []
            if step["conditions"]["status"] != "observed":
                missing.append("실제 반응 조건 미확인: 추천 조건은 관측 조건이 아닙니다.")
            else:
                conditions_known += 1
            if step["byproducts"]["status"] != "success":
                missing.append("부산물 평가 " + ("실패" if step["byproducts"]["status"] == "failed" else "미실행"))
            else:
                byproducts_run += 1
            for role in (*ROLES, *ADDITIONAL_ROLES, "byproducts"):
                values = step.get(role, []) if role != "byproducts" else step[role]["candidates"]
                evidence = step.get("material_provenance", {}).get(role, {})
                origin = (evidence.get("source", step["byproducts"].get("source", "")) if role == "byproducts"
                          else evidence.get("source", "물질별 출처 미지정"))
                evidence_status = evidence.get("status", "unknown")
                for smiles in values:
                    # Exact canonical target identity only; do not strip salts or stereo here.
                    if canonicalize_smiles(smiles) == target:
                        continue
                    result = assess_material(smiles)
                    result = {**result, "role": role, "origin": origin,
                              "evidence_status": evidence_status, "route_source": route["source"],
                              "formation_kind": "coproduct" if role == "coproducts" else "unknown"}
                    materials.append(result)
                    key = result["canonical_smiles"] or ("invalid:" + smiles)
                    unique[key] = result
                    if result["status"] == "invalid":
                        missing.append(f"구조 해석 실패: {smiles}")
            warnings = _conditional_warnings(step)
            warning_count += len(warnings)
            steps_out.append({"id": step["id"], "materials": materials,
                              "conditional_warnings": warnings, "missing_information": missing,
                              "conditions": step["conditions"], "byproducts": step["byproducts"],
                              "reaction_structures": {"reactants": step["reactants"],
                                                      "products": step["products"]}})
        counts = {s: sum(m["status"] == s for m in unique.values()) for s in
                  ("known_concern", "structural_alert", "invalid", "no_alert_detected")}
        missing_steps = sum(bool(s["missing_information"]) for s in steps_out)
        review = "우려 근거 검토 필요" if counts["known_concern"] or counts["structural_alert"] or warning_count else "평가 범위 내 경보 미검출"
        if missing_steps:
            review += " · 미평가 정보 있음"
        output.append({"id": route["id"], "target": route["target"], "source": route["source"],
                       "step_count": len(steps_out), "known_concern_count": counts["known_concern"],
                       "structural_alert_count": counts["structural_alert"],
                       "invalid_material_count": counts["invalid"],
                       "evaluated_material_count": len(unique) - counts["invalid"],
                       "material_count": len(unique), "condition_coverage": conditions_known / len(steps_out),
                       "byproduct_coverage": byproducts_run / len(steps_out),
                       "missing_step_count": missing_steps,
                       "missing_information_count": sum(len(s["missing_information"]) for s in steps_out),
                       "conditional_warning_count": warning_count, "review_status": review, "steps": steps_out})
    return {"schema_version": "1.0", "disclaimer": DISCLAIMER, "routes": output,
            "comparison_note": "입력 순서를 유지합니다. 우려 물질 수만으로 안전 순위를 정하지 않습니다. 조건 커버리지는 observed 표기 단계 비율, 부산물 커버리지는 호출/평가 완료 단계 비율이며 정확도·검출률이 아닙니다. 목표물 자체는 정확히 일치하는 구조만 불순물 집계에서 제외합니다."}


def predict_byproducts(reactants_smiles: str, product_smiles: str, timeout: int = 20) -> dict:
    """Explicitly call ASKCOS; retain decoded raw evidence even on response failure.

    Candidate provider fields are preserved without inventing absent scores.
    Candidates are proposals, and failure or an empty list is not absence.
    """
    for value in (reactants_smiles, product_smiles):
        if not isinstance(value, str) or not value.strip() or canonicalize_smiles(value) is None:
            raise ValueError("반응물과 주생성물에 유효한 SMILES가 필요합니다.")
    limitations = ["입력은 공개 ASKCOS 서버로 전송됩니다.", "이 호출은 구체적 용매·온도·정제 조건을 전달하지 않습니다.",
                   "예측 후보는 실험 검출 결과가 아니며 미반환은 불순물 부재를 뜻하지 않습니다."]
    payload = None
    try:
        response = requests.post(ASKCOS_API, json={"rct_smi": reactants_smiles, "prd_smi": product_smiles}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("result", {}).get("predict_expand")
        if not isinstance(rows, list):
            raise ValueError("예상한 ASKCOS 응답 배열이 없습니다.")
        candidates = []
        seen = {canonicalize_smiles(product_smiles)}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("prd_smiles"), str):
                raise ValueError("부산물 응답 항목이 손상되었습니다.")
            smi = row["prd_smiles"]
            assessment = assess_material(smi)
            canonical = assessment["canonical_smiles"]
            if canonical in seen:
                continue
            if canonical:
                seen.add(canonical)
            candidate = {"smiles": smi, "mode": str(row.get("modes_name", "")), "assessment": assessment}
            candidate.update({key: value for key, value in row.items() if key not in candidate})
            candidates.append(candidate)
        return {"status": "success", "candidates": candidates, "error": None, "limitations": limitations,
                "source": ASKCOS_API, "raw_response": payload}
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        return {"status": "failed", "candidates": [], "error": f"{type(exc).__name__}: {exc}", "limitations": limitations,
                "source": ASKCOS_API, "raw_response": payload}


def _tree_steps(node: dict, steps: list[dict]) -> None:
    """Convert an AiZynthFinder molecule/reaction tree without inventing conditions."""
    if not isinstance(node, dict) or node.get("type") != "mol":
        raise ValueError("AiZynthFinder 분자 트리 형식이 아닙니다.")
    for reaction in node.get("children", []):
        if reaction.get("type") != "reaction":
            raise ValueError("AiZynthFinder 반응 노드가 없습니다.")
        precursors = reaction.get("children", [])
        steps.append({"id": str(len(steps) + 1), "reactants": [p["smiles"] for p in precursors],
                      "products": [node["smiles"]], "reagents": [], "solvents": [],
                      "conditions": {"status": "unknown"},
                      "byproducts": {"status": "not_run", "candidates": []}})
        for precursor in precursors:
            _tree_steps(precursor, steps)


def from_native_tree(tree: dict, route_id: str, target: str, source: str) -> dict:
    """Convert an original molecule/reaction tree, retaining its native evidence.

    SynthEx's final stitched tree uses the same upstream representation as
    AiZynthFinder. Converting that tree, rather than its pre-stitch CLI view,
    includes reactions added by the original leaf-completion workflow.
    """
    identity = validate_target_smiles(target)
    if not isinstance(tree, dict) or canonicalize_smiles(tree.get("smiles")) != identity:
        raise ValueError("원본 경로 트리의 목표물이 요청과 일치하지 않습니다.")
    pending = [(tree, 0)]
    reactions = 0
    while pending:
        node, depth = pending.pop()
        if not isinstance(node, dict) or depth > 200:
            raise ValueError("원본 경로 트리의 구조 또는 깊이가 올바르지 않습니다.")
        children = node.get("children", [])
        if not isinstance(children, list):
            raise ValueError("원본 경로 트리의 children 배열이 필요합니다.")
        kind = node.get("type")
        if kind == "mol":
            if canonicalize_smiles(node.get("smiles")) is None:
                raise ValueError("원본 경로 트리의 분자 구조를 해석할 수 없습니다.")
            expected = "reaction"
        elif kind == "reaction":
            reactions += 1
            if not children or reactions > 100:
                raise ValueError("원본 경로의 반응 단계 수 또는 전구체가 올바르지 않습니다.")
            expected = "mol"
        else:
            raise ValueError("원본 경로 트리의 노드 유형이 올바르지 않습니다.")
        if any(not isinstance(child, dict) or child.get("type") != expected for child in children):
            raise ValueError("원본 경로 트리의 분자/반응 연결이 올바르지 않습니다.")
        pending.extend((child, depth + 1) for child in children)
    steps = []
    _tree_steps(tree, steps)
    return validate_routes({"schema_version": "1.0", "routes": [{
        "id": route_id, "target": target, "source": source, "steps": steps,
        "upstream": {"tree": tree},
    }]})


def generate_routes(smiles: str, time_limit: int = 30, iteration_limit: int = 30) -> dict:
    validate_target_smiles(smiles)
    if not 1 <= time_limit <= 120 or not 1 <= iteration_limit <= 200:
        raise ValueError("탐색 시간은 1~120초, 횟수는 1~200이어야 합니다.")
    from aizynthfinder.aizynthfinder import AiZynthFinder
    models = HERE.parent / "test" / "models"
    names = ("uspto_model.onnx", "uspto_templates.csv.gz", "uspto_filter_model.onnx", "zinc_stock.hdf5")
    if any(not (models / name).is_file() for name in names):
        raise FileNotFoundError("test/models의 AiZynthFinder 모델 또는 재고 파일이 없습니다.")
    config = {"expansion": {"uspto": [str(models / names[0]), str(models / names[1])]},
              "filter": {"uspto": str(models / names[2])}, "stock": {"zinc": str(models / names[3])}}
    # ZINC is a drug-like screening library and lists almost no small
    # inorganics: measured 2026-11-08, it has neither carbon dioxide nor
    # water. A route ending there read as unfinished, which stopped the
    # search from ever calling it solved. The bulk list is optional so a
    # checkout without it still runs on ZINC alone.
    bulk = models / "bulk_reagents_stock.hdf5"
    if bulk.is_file():
        config["stock"]["bulk_reagents"] = str(bulk)
    # Read the names before constructing: AiZynthFinder consumes the dict.
    stock_names = list(config["stock"])
    finder = AiZynthFinder(configdict=config)
    finder.stock.select(stock_names)
    finder.expansion_policy.select_first()
    finder.filter_policy.select("uspto")
    finder.config.search.time_limit = time_limit
    finder.config.search.iteration_limit = iteration_limit
    finder.target_smiles = smiles
    finder.tree_search()
    finder.build_routes()
    routes = []
    native_trees = []
    for index, tree in enumerate(finder.routes.reaction_trees):
        native_tree = tree.to_dict(include_metadata=True)
        native_trees.append(native_tree)
        if native_tree.get("children"):
            route = from_native_tree(native_tree, f"aizynth-{index+1}", smiles,
                                     "AiZynthFinder live search (시약·용매·부산물 정보 미제공)")["routes"][0]
            route["upstream"].update(ordinal=index + 1, is_solved=bool(tree.is_solved))
            routes.append(route)
    if not routes:
        raise NoRoutesError("분석할 반응 단계가 있는 경로를 찾지 못했습니다. 목표물이 이미 재고에 있거나 탐색 한도가 부족할 수 있습니다.")
    return {"schema_version": "1.0", "routes": routes,
            "native_trees": native_trees,
            "generation_config": {"config": config, "time_limit": time_limit,
                                  "iteration_limit": iteration_limit}}


def from_synthex_rendered(rendered: dict, route_id: str, target: str) -> dict:
    """Import SynthEx's PRODUCT>>PRECURSORS connected-fragment summary.

    render_plan splits executed precursors with Chem.GetMolFrags(asMols=True)
    and joins their unmapped SMILES with dots. Each right-side fragment is one
    reactant; the complete product remains one entry even when disconnected.
    No salt association, neutralization, or charge removal is inferred.

    Standalone external strings do not encode precursor grouping either.
    Salt-preserving canonicalization retains components supplied together as
    one material; it cannot reconstruct associations absent from this summary.
    """
    validate_target_smiles(target)
    entries = rendered.get("route") if isinstance(rendered, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("SynthEx render_plan 출력의 route 배열이 필요합니다.")
    steps = []
    for index, entry in enumerate(entries, 1):
        reaction = entry.get("smiles", "") if isinstance(entry, dict) else ""
        if not isinstance(reaction, str) or reaction.count(">>") != 1:
            raise ValueError("SynthEx 반응은 product>>precursors 형식이어야 합니다.")
        product, precursors = reaction.split(">>")
        steps.append({"id": str(entry.get("id", index)), "reactants": precursors.split("."),
                      "products": [product], "reagents": [], "solvents": [],
                      "conditions": {"status": "unknown"}, "byproducts": {"status": "not_run", "candidates": []}})
    payload = {"schema_version": "1.0", "routes": [{"id": route_id, "target": target,
               "source": "SynthEx rendered route (구조 렌더링은 실험 타당성 검증이 아님)", "steps": steps}]}
    return validate_routes(payload)


def import_synthex(payload: dict) -> dict:
    """Render local RouteJSON without LLM calls or API credentials.

    Precursor entries follow render_plan's connected-fragment contract, not
    inferred groupings of salts or other disconnected materials.
    """
    if not isinstance(payload, dict) or payload.get("format_version") != "routejson-1.0":
        raise ValueError("SynthEx routejson-1.0 단일 경로 객체가 필요합니다.")
    validate_target_smiles(payload.get("target"))
    source = str(HERE.parent / "synthex" / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from synthex.routejson import Route, render_plan
    route = Route.model_validate(payload)
    return from_synthex_rendered(render_plan(route.to_op_plan()), "synthex-import", route.target)
