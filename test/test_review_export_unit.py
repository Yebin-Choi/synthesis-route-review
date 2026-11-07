"""Offline export contracts for immutable review snapshots, not safety evidence."""

import base64
import copy
import csv
from html.parser import HTMLParser
from io import StringIO
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import engine
import review_export
from route_selection import select_routes
from toxicology import assess_material


TARGET = "CCOC(C)=O"


def _snapshot():
    payload = {"schema_version": "1.0", "routes": [{
        "id": "경로-1", "target": TARGET, "source": "소프트웨어 시험 입력; 실험 아님",
        "steps": [{
            "id": "단계-1", "reactants": ["CCO", "CCO"], "products": [TARGET],
            "reagents": ["CCO", "C1CC"], "solvents": ["CCO"],
            "catalysts": ["CCO"], "coproducts": ["O"],
            "material_provenance": {
                "reactants": {"status": "observed", "source": "기재된 시험 출처"},
                "catalysts": {"status": "proposed", "source": "제안일 뿐 관측 아님"},
            },
            "conditions": {"status": "unknown"},
            "byproducts": {"status": "failed", "candidates": [], "error": "시험 요청 실패"},
        }, {
            "id": "단계-2", "reactants": ["CCO"], "products": [TARGET, "CC=O"],
            "reagents": [], "solvents": [],
            "conditions": {"status": "proposed", "source": "시험 제안", "description": "관측 아님"},
            "byproducts": {"status": "success", "candidates": ["CO"], "source": "시험 후보"},
        }],
    }]}
    report = engine.analyze_routes(payload)
    return {
        "mode": "route", "original_input": payload, "report": report,
        "selection": select_routes(report, excluded_smiles=["CCO"]),
        "policy": {"excluded_smiles": ["CCO"], "strict_alerts": False, "note": "현재 정책"},
        "material_result": None,
        "model_evidence": {"status": "failed", "error": "모델 파일 없음", "provenance": None},
        "byproduct_attempts": [
            {"request_id": "이전 성공", "status": "success", "candidates": ["CO"], "historical": True},
            {"request_id": "현재 실패", "status": "failed", "error": "응답 실패"},
        ],
        "history": [{"status": "failed", "reason": "이전 입력 실패", "original_input": "원문"}],
        "report_hash": "공급된 해시", "extra_unknown_key": {"값": [None, False, "보존"]},
    }


def _material_snapshot(smiles="CCO"):
    return {
        "mode": "material", "original_input": smiles, "report": None,
        "selection": None, "policy": {"strict_alerts": False},
        "material_result": assess_material(smiles), "model_evidence": None,
        "byproduct_attempts": [],
    }


def _csv_rows(snapshot):
    return list(csv.DictReader(StringIO(review_export.export_csv(snapshot), newline="")))


class _Document(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.text = []
        self.pre = []
        self._current_pre = None
        self.feed(source)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))
        if tag == "pre":
            self._current_pre = []

    def handle_data(self, data):
        self.text.append(data)
        if self._current_pre is not None:
            self._current_pre.append(data)

    def handle_endtag(self, tag):
        if tag == "pre" and self._current_pre is not None:
            self.pre.append("".join(self._current_pre))
            self._current_pre = None


class ReviewExportTests(unittest.TestCase):
    def test_json_preserves_original_keys_korean_and_history(self):
        snapshot = _snapshot()
        result = review_export.export_json(snapshot)
        self.assertEqual(json.loads(result), snapshot)
        self.assertIn('\n  "mode": "route"', result)
        self.assertIn("이전 입력 실패", result)
        self.assertIn("기재된 시험 출처", result.encode("utf-8").decode("utf-8"))
        self.assertNotIn("\\uc774", result)
        self.assertEqual(json.loads(result)["original_input"], snapshot["original_input"])

    def test_all_exports_reject_nonfinite_values_even_in_unused_history(self):
        for exporter in (review_export.export_json, review_export.export_csv, review_export.export_html):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(exporter=exporter.__name__, value=value):
                    snapshot = _material_snapshot()
                    snapshot["history"] = [{"unused_number": value}]
                    with self.assertRaises(ValueError):
                        exporter(snapshot)

    def test_all_exports_reject_circular_nonjson_and_nondict_values(self):
        for exporter in (review_export.export_json, review_export.export_csv, review_export.export_html):
            with self.subTest(exporter=exporter.__name__):
                snapshot = _material_snapshot()
                snapshot["history"] = snapshot
                with self.assertRaises(ValueError):
                    exporter(snapshot)
                for value in ({"not-json"}, object(), b"not-json"):
                    snapshot = _material_snapshot()
                    snapshot["history"] = value
                    with self.assertRaises(TypeError):
                        exporter(snapshot)
                with self.assertRaises(TypeError):
                    exporter([])

    def test_csv_keeps_every_material_occurrence_in_report_order_and_excludes_target(self):
        snapshot = _snapshot()
        expected = [
            (route["id"], step["id"], material["role"], material["input_smiles"])
            for route in snapshot["report"]["routes"] for step in route["steps"]
            for material in step["materials"]
        ]
        rows = _csv_rows(snapshot)
        self.assertEqual(
            [(row["경로 ID"], row["단계 ID"], row["역할"], row["입력 SMILES"]) for row in rows],
            expected,
        )
        self.assertEqual(sum(row["입력 SMILES"] == "CCO" for row in rows), 6)
        self.assertEqual(sum(row["역할"] == "reactants" for row in rows), 3)
        self.assertNotIn(TARGET, [row["입력 SMILES"] for row in rows])
        self.assertIn(("단계-2", "products", "CC=O"),
                      [(row["단계 ID"], row["역할"], row["입력 SMILES"]) for row in rows])
        self.assertIn("byproducts", [row["역할"] for row in rows])

    def test_csv_does_not_merge_identical_materials_across_routes(self):
        snapshot = _snapshot()
        route = copy.deepcopy(snapshot["report"]["routes"][0])
        per_route_count = sum(len(step["materials"]) for step in route["steps"])
        route["id"] = "다른 경로"
        snapshot["report"]["routes"].append(route)
        decision = copy.deepcopy(snapshot["selection"]["routes"][0])
        decision["id"] = route["id"]
        decision["status"] = "별도 판정"
        snapshot["selection"]["routes"].append(decision)
        rows = _csv_rows(snapshot)
        self.assertEqual(len(rows), 2 * per_route_count)
        self.assertEqual([row["입력 SMILES"] for row in rows[:per_route_count]],
                         [row["입력 SMILES"] for row in rows[per_route_count:]])
        self.assertEqual({row["현재 경로 판정"] for row in rows[per_route_count:]}, {"별도 판정"})

    def test_csv_preserves_invalid_rows_failure_reasons_sources_and_current_policy(self):
        snapshot = _snapshot()
        rows = _csv_rows(snapshot)
        invalid = next(row for row in rows if row["입력 SMILES"] == "C1CC")
        self.assertEqual(invalid["평가 상태"], "invalid")
        self.assertEqual(invalid["정규 SMILES"], "")
        self.assertEqual(invalid["부산물 상태"], "failed")
        self.assertEqual(invalid["부산물 오류"], "시험 요청 실패")
        self.assertIn("구조 해석 실패", invalid["누락 정보"])
        self.assertEqual(json.loads(invalid["현재 정책"]), snapshot["policy"])
        self.assertNotEqual(snapshot["policy"], snapshot["selection"]["policy"])
        self.assertEqual(json.loads(invalid["현재 경로 판정 이유"]),
                         snapshot["selection"]["routes"][0]["reasons"])
        catalyst = next(row for row in rows if row["역할"] == "catalysts")
        self.assertEqual(catalyst["근거 상태"], "proposed")
        self.assertEqual(catalyst["물질 출처"], "제안일 뿐 관측 아님")

    def test_csv_quotes_commas_quotes_and_multiline_text_without_losing_it(self):
        snapshot = _material_snapshot()
        raw = '한글, "인용"\n다음 줄'
        snapshot["material_result"]["origin"] = raw
        snapshot["material_result"]["source_url"] = "https://example.invalid/a?x=1&y=2"
        row, = _csv_rows(snapshot)
        self.assertEqual(row["물질 출처"], raw)
        self.assertEqual(row["참고 출처 URL"], snapshot["material_result"]["source_url"])

    def test_csv_neutralizes_formula_prefixes_after_whitespace_without_mutating_json(self):
        dangerous = (
            "=1+1", "+SUM(1,2)", "-cmd|' /C calc'!A0", "@SUM(1)",
            "\tplain", "\rplain", "  =1", " \tplain", " \rplain",
            "\n  @SUM(1)", " +1", " \t",
        )
        for raw in dangerous:
            with self.subTest(raw=raw):
                snapshot = _material_snapshot()
                snapshot["material_result"]["input_smiles"] = raw
                row, = _csv_rows(snapshot)
                self.assertEqual(row["입력 SMILES"], "'" + raw)
                self.assertEqual(json.loads(review_export.export_json(snapshot))["material_result"]["input_smiles"], raw)
        for raw in ("C[NH2+]C", "[Na+]", "x=1", "  ordinary", "https://example.invalid"):
            with self.subTest(safe=raw):
                snapshot = _material_snapshot()
                snapshot["material_result"]["input_smiles"] = raw
                row, = _csv_rows(snapshot)
                self.assertEqual(row["입력 SMILES"], raw)

    def test_csv_neutralizes_every_untrusted_scalar_column_not_only_smiles(self):
        snapshot = _snapshot()
        raw = "  @SUM(1)"
        route = snapshot["report"]["routes"][0]
        route["id"] = raw
        decision = snapshot["selection"]["routes"][0]
        decision.update(id=raw, status=raw)
        step = route["steps"][0]
        step["id"] = raw
        step["conditions"]["status"] = raw
        step["byproducts"].update(status=raw, error=raw)
        material = step["materials"][0]
        for key in ("role", "input_smiles", "canonical_smiles", "status", "name",
                    "intake_basis", "evidence_status", "origin", "source_url", "route_source"):
            material[key] = raw
        row = _csv_rows(snapshot)[0]
        for label in ("경로 ID", "단계 ID", "역할", "입력 SMILES", "정규 SMILES", "평가 상태",
                      "물질명", "평가 이유", "근거 상태", "물질 출처", "참고 출처 URL", "경로 출처",
                      "현재 경로 판정", "조건 상태", "부산물 상태", "부산물 오류"):
            self.assertEqual(row[label], "'" + raw, label)

    def test_material_mode_is_one_independent_row_even_if_a_report_is_also_present(self):
        snapshot = _snapshot()
        snapshot.update(mode="material", original_input="C1CC", material_result=assess_material("C1CC"))
        row, = _csv_rows(snapshot)
        self.assertEqual(row["경로 ID"], "")
        self.assertEqual(row["단계 ID"], "")
        self.assertEqual(row["현재 경로 판정"], "")
        self.assertEqual(row["입력 SMILES"], "C1CC")
        self.assertEqual(row["평가 상태"], "invalid")
        document = _Document(review_export.export_html(snapshot))
        self.assertIn("단일 물질 평가", "".join(document.text))
        self.assertEqual(sum(tag == "tbody" for tag, _ in document.elements), 1)
        self.assertEqual(json.loads(document.pre[-1]), snapshot)

    def test_null_assessment_keeps_raw_input_and_truthful_unevaluated_status(self):
        snapshot = _material_snapshot()
        snapshot.update(original_input="원문 <검토 전>", material_result=None)
        row, = _csv_rows(snapshot)
        self.assertEqual(row["입력 SMILES"], "원문 <검토 전>")
        self.assertEqual(row["정규 SMILES"], "")
        self.assertEqual(row["평가 상태"], "미평가")
        document = _Document(review_export.export_html(snapshot))
        self.assertIn("미평가", "".join(document.text))
        self.assertIsNone(json.loads(document.pre[-1])["material_result"])

    def test_unknown_status_and_null_fields_remain_unknown_not_negative(self):
        snapshot = _material_snapshot()
        snapshot["material_result"].update(status="future_unknown_status", canonical_smiles=None,
                                            name=None, alerts=None, intake_basis=None, evidence_status=None)
        row, = _csv_rows(snapshot)
        self.assertEqual(row["평가 상태"], "future_unknown_status")
        self.assertEqual(row["정규 SMILES"], "")
        self.assertEqual(row["평가 이유"], "")
        document = _Document(review_export.export_html(snapshot))
        self.assertIn("future_unknown_status", "".join(document.text))
        self.assertEqual(json.loads(document.pre[-1]), snapshot)

    def test_no_report_does_not_rebuild_routes_from_original_input(self):
        snapshot = _snapshot()
        snapshot.update(report=None, selection=None)
        self.assertEqual(_csv_rows(snapshot), [])
        document = _Document(review_export.export_html(snapshot))
        self.assertIn("경로 보고서가 제공되지 않았습니다.", "".join(document.text))
        self.assertFalse(any(tag == "table" for tag, _ in document.elements))
        self.assertEqual(json.loads(document.pre[-1]), snapshot)

    def test_empty_step_retains_failure_and_missing_evidence_without_fake_material_row(self):
        snapshot = _snapshot()
        route = snapshot["report"]["routes"][0]
        route["steps"] = route["steps"][:1]
        step = route["steps"][0]
        step["materials"] = []
        step["conditional_warnings"] = [{"code": "conditional_fixture", "message": "조건부 시험 경고"}]
        self.assertEqual(_csv_rows(snapshot), [])
        document = _Document(review_export.export_html(snapshot))
        visible = "".join(document.text)
        for expected in ("비목표 물질 행이 없습니다", "시험 요청 실패", "실제 반응 조건 미확인", "조건부 시험 경고"):
            self.assertIn(expected, visible)
        self.assertEqual(json.loads(document.pre[-1]), snapshot)

    def test_html_escapes_titles_tags_quotes_urls_and_unknown_keys(self):
        snapshot = _snapshot()
        attack = '</title></h2><script>alert("x")</script><img src="https://evil.invalid/x" onerror="x">'
        url = 'javascript:alert("x")?q=<iframe src="https://evil.invalid/frame">&x=1'
        snapshot["title"] = attack
        snapshot[attack] = attack
        route = snapshot["report"]["routes"][0]
        route["id"] = attack
        snapshot["selection"]["routes"][0]["id"] = attack
        route["source"] = url
        step = route["steps"][0]
        step["id"] = attack
        step["materials"][0].update(input_smiles=attack, name=attack, role=attack,
                                      intake_basis=attack, origin=url, source_url=url)
        step["byproducts"]["error"] = attack
        snapshot["model_evidence"]["provenance"] = {"url": url, "title": attack}
        output = review_export.export_html(snapshot)
        document = _Document(output)
        forbidden = {"script", "iframe", "svg", "a", "link", "object", "embed", "form", "base"}
        self.assertFalse(forbidden.intersection(tag for tag, _ in document.elements))
        for tag, attrs in document.elements:
            self.assertFalse(any(name.lower().startswith("on") for name in attrs), (tag, attrs))
            self.assertNotIn("href", attrs)
            if "src" in attrs:
                self.assertEqual(tag, "img")
                self.assertTrue(attrs["src"].startswith("data:image/png;base64,"))
        self.assertIn("&lt;script&gt;", output)
        self.assertIn("&quot;x&quot;", output)
        self.assertIn(attack, "".join(document.text))
        self.assertIn(url, "".join(document.text))
        self.assertEqual(json.loads(document.pre[-1]), snapshot)

    def test_html_embeds_only_small_local_pngs_with_no_remote_assets(self):
        snapshot = _material_snapshot()
        document = _Document(review_export.export_html(snapshot))
        images = [attrs for tag, attrs in document.elements if tag == "img"]
        self.assertEqual(len(images), 1)
        for attrs in images:
            data = base64.b64decode(attrs["src"].split(",", 1)[1], validate=True)
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack(">II", data[16:24])
            self.assertLessEqual(width, 220)
            self.assertLessEqual(height, 150)
        for tag, attrs in document.elements:
            for name in ("src", "href", "srcset", "poster", "action"):
                if name in attrs:
                    self.assertEqual(tag, "img")
                    self.assertTrue(attrs[name].startswith("data:image/png;base64,"))
        self.assertFalse(any(tag in {"link", "script", "iframe"} for tag, _ in document.elements))

    def test_html_depiction_is_cached_only_within_each_export_without_deduplicating_rows(self):
        snapshot = _snapshot()
        route = snapshot["report"]["routes"][0]
        route["steps"] = route["steps"][:1]
        route["steps"][0]["materials"] = [copy.deepcopy(route["steps"][0]["materials"][0]) for _ in range(4)]
        with patch.object(review_export.Draw, "MolToImage", wraps=review_export.Draw.MolToImage) as draw:
            output = review_export.export_html(snapshot)
            self.assertEqual(draw.call_count, 2)  # Target and one unique material structure.
            review_export.export_html(snapshot)
            self.assertEqual(draw.call_count, 4)
        self.assertEqual(len(_csv_rows(snapshot)), 4)
        self.assertEqual(sum(tag == "img" for tag, _ in _Document(output).elements), 5)

    def test_html_checks_4096_limit_before_parsing_and_preserves_overlength_input(self):
        for length, expected_calls in ((4096, 1), (4097, 0)):
            with self.subTest(length=length):
                raw = "C" + " " * (length - 1)
                snapshot = _material_snapshot()
                snapshot["original_input"] = raw
                snapshot["material_result"].update(input_smiles=raw, canonical_smiles=None)
                with patch.object(review_export.Chem, "MolFromSmiles", wraps=review_export.Chem.MolFromSmiles) as parse:
                    output = review_export.export_html(snapshot)
                self.assertEqual(parse.call_count, expected_calls)
                self.assertEqual(json.loads(_Document(output).pre[-1]), snapshot)
                if length == 4097:
                    self.assertIn("4096자를 초과", output)

    def test_large_valid_structure_keeps_text_without_expensive_depiction(self):
        snapshot = _material_snapshot()
        raw = "C" * 257
        snapshot["original_input"] = raw
        snapshot["material_result"].update(input_smiles=raw, canonical_smiles=raw)
        with patch.object(review_export.Draw, "MolToImage") as draw:
            output = review_export.export_html(snapshot)
        draw.assert_not_called()
        self.assertIn("256원자 초과", output)
        self.assertEqual(json.loads(_Document(output).pre[-1]), snapshot)

    def test_html_depiction_failure_is_escaped_and_does_not_replace_assessment(self):
        snapshot = _material_snapshot()
        before = copy.deepcopy(snapshot)
        with patch.object(review_export.Draw, "MolToImage", side_effect=RuntimeError('<img src="https://evil.invalid">')):
            output = review_export.export_html(snapshot)
        document = _Document(output)
        self.assertIn("구조 그림 생성 실패", "".join(document.text))
        self.assertFalse(any(tag == "img" for tag, _ in document.elements))
        self.assertEqual(json.loads(document.pre[-1]), before)
        self.assertEqual(snapshot, before)

    def test_html_full_snapshot_includes_attempt_model_and_history_details(self):
        snapshot = _snapshot()
        document = _Document(review_export.export_html(snapshot))
        visible = "".join(document.text)
        for expected in ("부산물 시도·실패 이력", "모델 근거·출처", "이전 성공", "현재 실패",
                         "모델 파일 없음", "이전 입력 실패"):
            self.assertIn(expected, visible)
        self.assertEqual(json.loads(document.pre[-1]), snapshot)

    def test_exports_do_not_mutate_reanalyze_rescreen_or_call_network(self):
        snapshot = _snapshot()
        before = copy.deepcopy(snapshot)
        with patch("engine.analyze_routes", side_effect=AssertionError("must not analyze")), \
                patch("route_selection.select_routes", side_effect=AssertionError("must not select")), \
                patch("toxicology.assess_material", side_effect=AssertionError("must not assess")), \
                patch("requests.sessions.Session.request", side_effect=AssertionError("must not call network")):
            for exporter in (review_export.export_json, review_export.export_csv, review_export.export_html):
                exporter(snapshot)
                self.assertEqual(snapshot, before, exporter.__name__)

    def test_unsupported_mode_is_not_silently_rendered_as_a_different_scope(self):
        snapshot = _material_snapshot()
        snapshot["mode"] = "unknown"
        self.assertEqual(json.loads(review_export.export_json(snapshot)), snapshot)
        for exporter in (review_export.export_csv, review_export.export_html):
            with self.assertRaises(ValueError):
                exporter(snapshot)


if __name__ == "__main__":
    unittest.main()
