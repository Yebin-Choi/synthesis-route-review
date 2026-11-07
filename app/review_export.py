"""Local, read-only exports of a caller-supplied review snapshot."""

import base64
import csv
import html
from io import BytesIO, StringIO
import json

from rdkit import Chem
from rdkit.Chem import Draw


_COLUMNS = (
    ("경로 ID", "route_id"),
    ("단계 ID", "step_id"),
    ("역할", "role"),
    ("입력 SMILES", "input_smiles"),
    ("정규 SMILES", "canonical_smiles"),
    ("평가 상태", "status"),
    ("물질명", "name"),
    ("평가 이유", "intake_basis"),
    ("구조 경보", "alerts"),
    ("근거 상태", "evidence_status"),
    ("물질 출처", "origin"),
    ("참고 출처 URL", "source_url"),
    ("경로 출처", "route_source"),
    ("현재 경로 판정", "route_status"),
    ("현재 경로 판정 이유", "route_reasons"),
    ("현재 정책", "policy"),
    ("조건 상태", "conditions_status"),
    ("부산물 상태", "byproducts_status"),
    ("부산물 오류", "byproducts_error"),
    ("누락 정보", "missing_information"),
    ("조건부 경고", "conditional_warnings"),
)
_HTML_COLUMNS = tuple(
    column for column in _COLUMNS
    if column[1] in {
        "role", "input_smiles", "canonical_smiles", "status", "name",
        "intake_basis", "alerts", "evidence_status", "origin", "source_url",
    }
)


def _json(value, *, indent=None):
    return json.dumps(value, ensure_ascii=False, indent=indent, allow_nan=False)


def export_json(snapshot: dict) -> str:
    """Preserve every supplied field, rejecting non-JSON and nonfinite values."""
    if not isinstance(snapshot, dict):
        raise TypeError("스냅샷은 JSON 객체여야 합니다.")
    return _json(snapshot, indent=2)


def _object(value, label):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{label}: JSON 객체가 필요합니다.")
    return value


def _array(value, label):
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{label}: JSON 배열이 필요합니다.")
    return value


def _mode(snapshot):
    mode = snapshot.get("mode")
    if mode not in ("route", "material"):
        raise ValueError("스냅샷 mode는 route 또는 material이어야 합니다.")
    return mode


def _text(value):
    if value is None:
        return ""
    return value if isinstance(value, str) else _json(value)


def _escaped(value):
    return html.escape("미제공" if value is None else _text(value), quote=True)


def _decision(snapshot, route):
    selection = _object(snapshot.get("selection"), "selection")
    for decision in _array(selection.get("routes"), "selection.routes"):
        decision = _object(decision, "selection.route")
        if decision.get("id") == route.get("id"):
            return decision
    return {}


def _material(snapshot):
    result = snapshot.get("material_result")
    if result is None:
        return {"input_smiles": snapshot.get("original_input"), "status": "미평가"}
    return _object(result, "material_result")


def _row(snapshot, route, step, material, decision):
    conditions = _object(step.get("conditions"), "conditions")
    byproducts = _object(step.get("byproducts"), "byproducts")
    return {
        "route_id": route.get("id"),
        "step_id": step.get("id"),
        "role": material.get("role"),
        "input_smiles": material.get("input_smiles"),
        "canonical_smiles": material.get("canonical_smiles"),
        "status": material.get("status"),
        "name": material.get("name"),
        "intake_basis": material.get("intake_basis"),
        "alerts": material.get("alerts"),
        "evidence_status": material.get("evidence_status"),
        "origin": material.get("origin"),
        "source_url": material.get("source_url"),
        "route_source": material.get("route_source", route.get("source")),
        "route_status": decision.get("status"),
        "route_reasons": decision.get("reasons"),
        "policy": snapshot.get("policy"),
        "conditions_status": conditions.get("status"),
        "byproducts_status": byproducts.get("status"),
        "byproducts_error": byproducts.get("error"),
        "missing_information": step.get("missing_information"),
        "conditional_warnings": step.get("conditional_warnings"),
    }


def _rows(snapshot):
    if _mode(snapshot) == "material":
        yield _row(snapshot, {}, {}, _material(snapshot), {})
        return
    report = _object(snapshot.get("report"), "report")
    for route in _array(report.get("routes"), "report.routes"):
        route = _object(route, "route")
        decision = _decision(snapshot, route)
        for step in _array(route.get("steps"), "route.steps"):
            step = _object(step, "step")
            for material in _array(step.get("materials"), "step.materials"):
                yield _row(snapshot, route, step, _object(material, "material"), decision)


def _csv_cell(value):
    text = _text(value)
    # Do not strip or rewrite the original: neutralize the whole exported cell.
    for char in text:
        if char in "=+-@\t\r":
            return "'" + text
        if not char.isspace():
            break
    return text


def export_csv(snapshot: dict) -> str:
    """Write every report material occurrence; targets are not reconstructed."""
    export_json(snapshot)
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(label for label, _ in _COLUMNS)
    for row in _rows(snapshot):
        writer.writerow(_csv_cell(row[key]) for _, key in _COLUMNS)
    return output.getvalue()


def _structure(smiles, cache):
    if not isinstance(smiles, str) or not smiles.strip():
        return "<span>구조 미제공</span>"
    if smiles in cache:
        return cache[smiles]
    if len(smiles) > 4096:
        result = "<span>구조 그림 없음: 입력이 4096자를 초과합니다.</span>"
    else:
        text = smiles.strip()
        try:
            if "\x00" in text or any(char.isspace() for char in text):
                mol = None
            else:
                params = Chem.SmilesParserParams()
                params.parseName = False
                params.allowCXSMILES = False
                mol = Chem.MolFromSmiles(text, params)
            if (mol is None or mol.GetNumAtoms() == 0
                    or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())):
                result = "<span>구조 그림 없음: 유효한 구조를 해석할 수 없습니다.</span>"
            elif mol.GetNumAtoms() > 256:
                result = "<span>구조 그림 생략: 256원자 초과. SMILES 원문과 평가 결과를 확인하세요.</span>"
            else:
                image = Draw.MolToImage(mol, size=(220, 150))
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                result = (
                    '<img width="220" height="150" alt="입력 SMILES의 로컬 구조 그림" '
                    f'src="data:image/png;base64,{encoded}">'
                )
        except (ValueError, RuntimeError, OSError) as exc:
            result = f"<span>구조 그림 생성 실패: {_escaped(str(exc))}</span>"
    cache[smiles] = result
    return result


def _scalar_summary(value):
    if not isinstance(value, dict):
        return ""
    items = [
        f"<dt>{_escaped(key)}</dt><dd>{_escaped(item)}</dd>"
        for key, item in value.items() if not isinstance(item, (dict, list))
    ]
    return "<dl>" + "".join(items) + "</dl>" if items else ""


def _pre(value):
    return "<pre>" + html.escape(_json(value, indent=2), quote=True) + "</pre>"


def _details(title, value):
    return (f"<details><summary>{_escaped(title)}</summary>"
            + _scalar_summary(value) + _pre(value) + "</details>")


def _table(rows, cache):
    if not rows:
        return "<p>보고서에 기재된 비목표 물질 행이 없습니다.</p>"
    parts = ['<div class="table-wrap"><table><thead><tr><th scope="col">구조</th>']
    parts.extend(f'<th scope="col">{_escaped(label)}</th>' for label, _ in _HTML_COLUMNS)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr><td>" + _structure(row["input_smiles"], cache) + "</td>")
        parts.extend(f"<td>{_escaped(row[key])}</td>" for _, key in _HTML_COLUMNS)
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def export_html(snapshot: dict) -> str:
    """Render a standalone Korean report, with text-only sources and local PNGs."""
    original_json = export_json(snapshot)
    mode = _mode(snapshot)
    cache = {}
    parts = [
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        'img-src data:; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">',
        "<title>화학 위험 검토 보고서</title><style>",
        "body{font-family:system-ui,sans-serif;line-height:1.6;margin:1.5rem;color:#172334;"
        "background:#fff}main{max-width:100rem;margin:auto}h1,h2,h3{line-height:1.3}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #8591a0;"
        "padding:.6rem;text-align:left;vertical-align:top;white-space:pre-wrap;"
        "overflow-wrap:anywhere}th{background:#e9eef5}.table-wrap{overflow-x:auto}"
        "pre,dd{white-space:pre-wrap;overflow-wrap:anywhere}pre{padding:1rem;"
        "background:#f0f3f7}dt{font-weight:700}dd{margin:0 0 .5rem}"
        "details{margin:1rem 0}summary{cursor:pointer;font-weight:700}"
        "img{max-width:220px;height:auto}section{margin:2rem 0}",
        "</style></head><body><main><h1>화학 위험 검토 보고서</h1>",
    ]
    if mode == "route":
        parts.append("<p>범위: 보고서의 비목표 물질을 경로·단계·역할별로 표시합니다. "
                     "목표물은 물질 행에 추가하지 않습니다. 정책 제외는 독성 입증이 아니며, "
                     "경보 미검출은 안전 보장이 아닙니다.</p>")
    else:
        parts.append("<p>범위: 단일 물질 평가이며 경로·단계 판정을 의미하지 않습니다. "
                     "경보 미검출은 안전 보장이 아닙니다.</p>")
    parts.append(_scalar_summary(snapshot))
    parts.append("<section><h2>현재 정책</h2>" + _pre(snapshot.get("policy")) + "</section>")
    if mode == "material":
        parts.append("<section><h2>단일 물질 평가</h2>")
        parts.append(_table([_row(snapshot, {}, {}, _material(snapshot), {})], cache))
        parts.append(_details("공급된 물질 평가 전체", snapshot.get("material_result")))
        parts.append("</section>")
    else:
        report = _object(snapshot.get("report"), "report")
        routes = _array(report.get("routes"), "report.routes")
        if not routes:
            parts.append("<p>경로 보고서가 제공되지 않았습니다.</p>")
        for route in routes:
            route = _object(route, "route")
            decision = _decision(snapshot, route)
            parts.append(f'<section><h2>경로 {_escaped(route.get("id"))}</h2>')
            parts.append(_scalar_summary({
                "목표물 입력 SMILES (물질 표에서 제외)": route.get("target"),
                "경로 출처": route.get("source"),
                "보고서 검토 상태": route.get("review_status"),
                "현재 경로 판정": decision.get("status"),
            }))
            parts.append(_structure(route.get("target"), cache))
            parts.append("<h3>현재 경로 판정 이유</h3>" + _pre(decision.get("reasons")))
            for step in _array(route.get("steps"), "route.steps"):
                step = _object(step, "step")
                parts.append(f'<h3>단계 {_escaped(step.get("id"))}</h3>')
                rows = [
                    _row(snapshot, route, step, _object(material, "material"), decision)
                    for material in _array(step.get("materials"), "step.materials")
                ]
                parts.append(_table(rows, cache))
                parts.append("<h4>누락 정보</h4>" + _pre(step.get("missing_information")))
                parts.append("<h4>조건부 경고</h4>" + _pre(step.get("conditional_warnings")))
                parts.append("<h4>조건·부산물 근거 (실패 포함)</h4>" + _pre({
                    "conditions": step.get("conditions"), "byproducts": step.get("byproducts"),
                }))
            parts.append("</section>")
    for title, key in (
        ("부산물 시도·실패 이력", "byproduct_attempts"),
        ("모델 근거·출처", "model_evidence"),
    ):
        parts.append(_details(title, snapshot.get(key)))
    parts.append("<details><summary>전체 스냅샷 JSON (원본·이력 포함)</summary><pre>"
                 + html.escape(original_json, quote=True) + "</pre></details>")
    parts.append("</main></body></html>")
    return "".join(parts)
