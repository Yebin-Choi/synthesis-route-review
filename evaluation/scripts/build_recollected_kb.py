"""Independently recollect two public workbooks; this is not a team KB release.

Only the named source sheets are read. Statistics describe the retained source
observations, not calibrated yields, target-reaction predictions or safety.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, datetime, time
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import sys
from urllib.parse import urlparse

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
from toxicology import canonicalize_smiles


SCHEMA = "independent-condition-evidence/1"
DEFAULT_DIRECTORY = ROOT / "data" / "recollected"
WILSON_Z = 1.96
WILSON_FORMULA = (
    "(p + z^2/(2*n) - z*sqrt(p*(1-p)/n + z^2/(4*n^2))) / "
    "(1 + z^2/n), where p=count/n and z=1.96"
)
SOURCE_SPECS = {
    "Dreher_and_Doyle_input_data.xlsx": {
        "id": "dreher_doyle",
        "reaction_type": "Buchwald-Hartwig",
        "selected_sheet": "FullCV_01",
        "measurement_field": "Output",
        "measurement_basis": "reported_Output_percent_basis_unverified",
        "measurement_unit": "reported_percent",
        "required_headers": ("Ligand", "Additive", "Base", "Aryl halide", "Output"),
        "raw_only_measurement_fields": [],
    },
    "aap9112_Data_File_S1.xlsx": {
        "id": "suzuki_aap9112",
        "reaction_type": "Suzuki-Miyaura",
        "selected_sheet": "Sheet1",
        "measurement_field": "Product_Yield_PCT_Area_UV",
        "measurement_basis": "UV_area_percentage",
        "measurement_unit": "UV_area_percent",
        "required_headers": (
            "Reaction_No", "Reactant_1_Name", "Reactant_1_Short_Hand",
            "Reactant_1_eq", "Reactant_1_mmol", "Reactant_2_Name", "Reactant_2_eq",
            "Catalyst_1_Short_Hand", "Catalyst_1_eq", "Ligand_Short_Hand", "Ligand_eq",
            "Reagent_1_Short_Hand", "Reagent_1_eq", "Solvent_1_Short_Hand",
            "Product_Yield_PCT_Area_UV", "Product_Yield_Mass_Ion_Count",
        ),
        "raw_only_measurement_fields": ["Product_Yield_Mass_Ion_Count"],
    },
}
LIMITATIONS = [
    "Independent recollection from public workbooks, not exact restoration of Jaeyun v1.8/v2.3.",
    "Literature-reference inspection only; a selected reaction type does not establish a match to a target.",
    "No condition application, price, total-route yield, target prediction or safety ranking is provided.",
    "Doyle Output has unverified measurement basis; no calibrated-yield interpretation is established.",
    "UV area percentages and unverified Output are separate bases; do not compare across sources or bases.",
    "Mass ion counts are retained only in raw records, never interpreted as percentages or summarized yields.",
    "Names and shorthands are unresolved structures, not SMILES and not negative safety screens.",
    "Missing catalyst, solvent, temperature or other source information is not imputed.",
    "Threshold counts and Wilson endpoints describe collected observations only, not predictions; "
    "independence and target applicability have not been established.",
    "Only FullCV_01 is collected for Doyle; alternate CV/OOS sheets are not pooled. "
    "Within-sheet experimental rows are not deduplicated by structure or measurement.",
]


def wilson_lower_bound(count, n):
    """Descriptive 95% Wilson interval lower endpoint; None without observations."""
    if (isinstance(count, bool) or isinstance(n, bool)
            or not isinstance(count, int) or not isinstance(n, int)
            or n < 0 or not 0 <= count <= n):
        raise ValueError("Wilson count and n must be integers with 0 <= count <= n")
    if n == 0:
        return None
    if count == 0:
        return 0.0
    p = count / n
    z2 = WILSON_Z ** 2
    return (p + z2 / (2 * n) - WILSON_Z * math.sqrt(
        p * (1 - p) / n + z2 / (4 * n * n)
    )) / (1 + z2 / n)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_numeric_value": repr(value)}
    if isinstance(value, (datetime, date, time)):
        return {"excel_datetime_value": value.isoformat()}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _missing(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _issue(issues, code, field, value):
    issues.append({"code": code, "field": field, "raw_value": _json_safe(value)})


def _number(raw, field, missing_fields, issues, *, percentage=False, required=False):
    value = raw.get(field)
    if _missing(value):
        missing_fields.add(field)
        if required:
            _issue(issues, "missing_required_number", field, value)
        return None
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not a measurement")
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        _issue(issues, "invalid_number", field, value)
        return None
    if not math.isfinite(number):
        _issue(issues, "nonfinite_number", field, value)
        return None
    if percentage and not 0 <= number <= 100:
        _issue(issues, "percentage_out_of_range", field, value)
        return None
    if not percentage and number < 0:
        _issue(issues, "negative_quantity", field, value)
        return None
    return number


def _component(role, *, status, name=None, input_smiles=None, canonical_smiles=None,
               equivalents=None, mmol=None, source_fields=None, reason=None):
    return {
        "role": role, "status": status, "name": name, "input_smiles": input_smiles,
        "canonical_smiles": canonical_smiles, "equivalents": equivalents, "mmol": mmol,
        "source_fields": source_fields or {}, "reason": reason,
    }


def _smiles_component(raw, field, role, missing_fields, issues):
    value = raw.get(field)
    if _missing(value):
        missing_fields.add(field)
        return _component(role, status="missing", source_fields={"smiles": field},
                          reason="missing_source_cell")
    canonical = canonicalize_smiles(value)
    if canonical is None:
        _issue(issues, "invalid_smiles", field, value)
    return _component(role, status="resolved" if canonical else "invalid",
                      input_smiles=value, canonical_smiles=canonical,
                      source_fields={"smiles": field})


def _named_component(raw, role, name_field, missing_fields, issues, *, eq_field=None,
                     mmol_field=None, shorthand_field=None, allow_absent=False):
    name = raw.get(name_field)
    absent = allow_absent and isinstance(name, str) and name.strip() == "None"
    if _missing(name):
        status, reason = "missing", "missing_source_cell"
        missing_fields.add(name_field)
    elif not isinstance(name, str):
        status, reason = "invalid", "invalid_name_cell"
        _issue(issues, "invalid_name", name_field, name)
    else:
        status = "absent" if absent else "unresolved"
        reason = "explicit_source_absence" if absent else "name_only_not_structure"
    fields = {"name": name_field}
    quantities = {}
    for key, field in (("equivalents", eq_field), ("mmol", mmol_field)):
        if field is not None:
            fields[key] = field
            if absent and _missing(raw.get(field)):
                quantities[key] = None
            else:
                quantities[key] = _number(raw, field, missing_fields, issues)
    component = _component(role, status=status, name=name, source_fields=fields,
                           reason=reason, **quantities)
    if shorthand_field is not None:
        component["shorthand"] = raw.get(shorthand_field)
        fields["shorthand"] = shorthand_field
        if _missing(component["shorthand"]):
            missing_fields.add(shorthand_field)
        elif not isinstance(component["shorthand"], str):
            _issue(issues, "invalid_name", shorthand_field, component["shorthand"])
    return component


def _parse_row(raw, source, worksheet_row):
    missing_fields, issues = set(), []
    if source["id"] == "dreher_doyle":
        components = [
            _smiles_component(raw, field, role, missing_fields, issues)
            for field, role in (("Base", "base"), ("Ligand", "ligand"), ("Additive", "additive"))
        ]
        for role in ("catalyst", "solvent"):
            components.append(_component(role, status="missing", reason="not_provided_by_source"))
        missing_fields.update(("catalyst", "solvent", "temperature", "amine_substrate"))
        missing_fields.update(f"{role}.equivalents" for role in ("base", "ligand", "additive"))
        context = [_smiles_component(raw, "Aryl halide", "aryl_halide", missing_fields, issues)]
        identifier = {"field": "worksheet_row", "value": worksheet_row}
    else:
        definitions = (
            ("catalyst", "Catalyst_1_Short_Hand", "Catalyst_1_eq", False),
            ("ligand", "Ligand_Short_Hand", "Ligand_eq", True),
            ("reagent", "Reagent_1_Short_Hand", "Reagent_1_eq", True),
            ("solvent", "Solvent_1_Short_Hand", None, False),
        )
        components = [
            _named_component(raw, role, name_field, missing_fields, issues,
                             eq_field=eq_field, allow_absent=allow_absent)
            for role, name_field, eq_field, allow_absent in definitions
        ]
        context = [
            _named_component(raw, "reactant_1", "Reactant_1_Name", missing_fields, issues,
                             eq_field="Reactant_1_eq", mmol_field="Reactant_1_mmol",
                             shorthand_field="Reactant_1_Short_Hand"),
            _named_component(raw, "reactant_2", "Reactant_2_Name", missing_fields, issues,
                             eq_field="Reactant_2_eq"),
        ]
        missing_fields.add("temperature")
        reaction_no = _number(raw, "Reaction_No", missing_fields, issues, required=True)
        if reaction_no is not None and (reaction_no < 1 or not reaction_no.is_integer()):
            _issue(issues, "invalid_source_record_id", "Reaction_No", raw["Reaction_No"])
        identifier = {"field": "Reaction_No", "value": raw.get("Reaction_No")}
        _number(raw, "Product_Yield_Mass_Ion_Count", missing_fields, issues)
    value = _number(raw, source["measurement_field"], missing_fields, issues, percentage=True)
    for field, cell in raw.items():
        if field.startswith("__unnamed_column_") and not _missing(cell):
            _issue(issues, "unlabelled_column_value", field, cell)
    record = {
        "id": f"{source['id']}:{source['selected_sheet']}:row:{worksheet_row}",
        "source_id": source["id"], "source_sha256": source["sha256"],
        "sheet": source["selected_sheet"], "worksheet_row": worksheet_row,
        "source_record_identifier": identifier, "reaction_type": source["reaction_type"],
        "measurement_basis": source["measurement_basis"],
        "measurement": {"field": source["measurement_field"], "basis": source["measurement_basis"],
                        "unit": source["measurement_unit"], "value": value,
                        "status": "missing" if _missing(raw.get(source["measurement_field"])) else "reported"},
        "components": components, "substrate_context": context,
        "temperature": {"value": None, "unit": None, "status": "unknown"},
        "missing_fields": sorted(missing_fields), "raw": _json_safe(raw),
    }
    return record, issues


def _component_identity(component):
    return [
        component["role"], component["status"], component["canonical_smiles"],
        component["name"], component.get("shorthand"),
        component["equivalents"], component["mmol"],
    ]


def _stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _display_components(rows, field):
    components = deepcopy(rows[0][field])
    for index, component in enumerate(components):
        component["input_smiles_variants"] = sorted({
            row[field][index]["input_smiles"] for row in rows
            if row[field][index]["input_smiles"] is not None
        })
    return components


def _unresolved_counts(records):
    by_role = Counter()
    for record in records:
        for component in record["components"] + record["substrate_context"]:
            if component["status"] == "unresolved":
                by_role[component["role"]] += 1
    return {"count": sum(by_role.values()), "by_role": dict(sorted(by_role.items())),
            "count_basis": "name-only component occurrences across records, including substrate context"}


def group_records(records):
    """Group whole conditions within a source and basis; substrates remain paired context."""
    buckets = defaultdict(list)
    for record in records:
        key = _stable_json([
            record["source_id"], record["reaction_type"], record["measurement_basis"],
            record["measurement"]["field"], record["measurement"]["unit"],
            [_component_identity(component) for component in record["components"]],
        ])
        buckets[key].append(record)
    groups = []
    for key, rows in sorted(buckets.items()):
        rows = sorted(rows, key=lambda row: (row["worksheet_row"], row["id"]))
        first = rows[0]
        values = [row["measurement"]["value"] for row in rows if row["measurement"]["value"] is not None]
        contexts = defaultdict(list)
        for row in rows:
            contexts[_stable_json([_component_identity(c) for c in row["substrate_context"]])].append(row)
        threshold_count = sum(value >= 10 for value in values)
        missing = Counter(field for row in rows for field in row["missing_fields"])
        groups.append({
            "id": f"{first['source_id']}:{hashlib.sha256(key.encode('utf-8')).hexdigest()}",
            "source_id": first["source_id"], "reaction_type": first["reaction_type"],
            "measurement_basis": first["measurement_basis"],
            "components": _display_components(rows, "components"),
            "substrate_contexts": [
                {"components": _display_components(context_rows, "substrate_context"),
                 "n": len(context_rows), "record_ids": [row["id"] for row in context_rows]}
                for _, context_rows in sorted(contexts.items())
            ],
            "n": len(rows), "record_ids": [row["id"] for row in rows],
            "measurement": {
                "field": first["measurement"]["field"], "basis": first["measurement_basis"],
                "unit": first["measurement"]["unit"], "n_valid": len(values),
                "n_missing": len(rows) - len(values), "median": median(values) if values else None,
                "minimum": min(values) if values else None, "maximum": max(values) if values else None,
                "calibrated_yield_established": False,
            },
            "research_threshold": {
                "label": "research_observation_count_not_prediction", "operator": ">=", "value": 10,
                "unit": first["measurement"]["unit"], "n_valid": len(values), "count": threshold_count,
                "wilson_lower_bound": wilson_lower_bound(threshold_count, len(values)),
                "method": "descriptive_Wilson_95pct_interval_lower_endpoint", "z": WILSON_Z,
                "formula": WILSON_FORMULA, "independence_established": False,
            },
            "temperature": {"value": None, "unit": None, "status": "unknown"},
            "missing_fields": dict(sorted(missing.items())),
            "unresolved_components": _unresolved_counts(rows),
            "use": "literature_reference_inspection_only_not_target_matching",
        })
    return sorted(groups, key=lambda group: (group["source_id"], group["measurement_basis"], group["id"]))


def _source_entry(entry, source_root):
    file_value = entry.get("file")
    if not isinstance(file_value, str) or not file_value:
        raise ValueError("Each source must identify its raw file")
    path = Path(file_value.replace("\\", "/"))
    if not path.is_absolute():
        path = source_root / path
    spec = SOURCE_SPECS.get(path.name)
    if spec is None:
        raise ValueError(f"Unsupported workbook: {path.name}")
    expected = entry.get("sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
        raise ValueError(f"A raw SHA256 is mandatory for {path.name}")
    url = entry.get("url")
    parsed = urlparse(url) if isinstance(url, str) else None
    if parsed is None or parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"A source HTTP(S) URL is required for {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected.lower():
        raise ValueError(f"Raw SHA256 mismatch: {path.name}; no evidence was written")
    if "bytes" in entry and entry["bytes"] != path.stat().st_size:
        raise ValueError(f"Raw byte count mismatch: {path.name}")
    source = {key: deepcopy(value) for key, value in spec.items() if key != "required_headers"}
    source.update(file=file_value.replace("\\", "/"), url=url, sha256=actual, bytes=path.stat().st_size)
    return path, source, spec


def _collect_source(path, source, spec):
    records, exclusions, metric_exclusions = [], [], []
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        selected = source["selected_sheet"]
        if selected not in workbook.sheetnames:
            raise ValueError(f"Required source sheet is missing: {path.name}:{selected}")
        ignored_sheets = [
            {"sheet": sheet.title, "worksheet_data_row_span": max(0, (sheet.max_row or 1) - 1),
             "reason": "not_selected_alternate_split_or_auxiliary_sheet"}
            for sheet in workbook if sheet.title != selected
        ]
        rows = workbook[selected].iter_rows(values_only=True)
        headers = next(rows, ())
        named_headers = [header for header in headers if header is not None]
        if any(not isinstance(header, str) or not header.strip() for header in named_headers):
            raise ValueError(f"Source headers must be non-empty strings: {path.name}:{selected}")
        if len(named_headers) != len(set(named_headers)):
            raise ValueError(f"Duplicate source headers: {path.name}:{selected}")
        missing_headers = set(spec["required_headers"]) - set(named_headers)
        if missing_headers:
            raise ValueError(f"Missing source headers in {path.name}:{selected}: {sorted(missing_headers)}")
        seen_identifiers, rows_seen = set(), 0
        for worksheet_row, cells in enumerate(rows, start=2):
            rows_seen += 1
            raw = {
                header if header is not None else f"__unnamed_column_{index + 1}": cells[index]
                for index, header in enumerate(headers)
                if header is not None or not _missing(cells[index])
            }
            record, issues = _parse_row(raw, source, worksheet_row)
            identifier_key = _stable_json(_json_safe(record["source_record_identifier"]))
            if identifier_key in seen_identifiers:
                _issue(issues, "duplicate_source_record_id", "Reaction_No", raw.get("Reaction_No"))
            if not any(not _missing(value) for value in raw.values()):
                _issue(issues, "empty_row", "worksheet_row", worksheet_row)
            if issues:
                exclusions.append({
                    "record_id": record["id"], "source_id": source["id"], "sheet": selected,
                    "worksheet_row": worksheet_row,
                    "source_record_identifier": _json_safe(record["source_record_identifier"]),
                    "reasons": issues, "raw": record["raw"],
                })
                continue
            seen_identifiers.add(identifier_key)
            records.append(record)
            if record["measurement"]["value"] is None:
                metric_exclusions.append({
                    "record_id": record["id"], "source_id": source["id"],
                    "field": source["measurement_field"], "reason": "missing_measurement_not_in_statistics",
                })
        summary = {
            "source_id": source["id"], "sha256": source["sha256"], "selected_sheet": selected,
            "selected_data_rows_seen": rows_seen, "records_retained": len(records),
            "records_quarantined": len(exclusions), "missing_primary_measurements": len(metric_exclusions),
            "ignored_sheets": ignored_sheets,
        }
        return records, exclusions, metric_exclusions, summary
    finally:
        workbook.close()


def build_evidence(manifest_path=None, output_dir=None, *, source_root=ROOT):
    """Verify manifest hashes, collect selected sheets and write three derived artifacts.

    Manifest file paths are relative to source_root, never guessed relative to
    another directory. Invalid observations are quarantined in audit.json.
    """
    manifest_path = Path(manifest_path) if manifest_path is not None else DEFAULT_DIRECTORY / "sources.json"
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_DIRECTORY
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("sources") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("Manifest must contain a non-empty sources array")
    verified = [_source_entry(entry, Path(source_root)) for entry in entries]
    if len({source["id"] for _, source, _ in verified}) != len(verified):
        raise ValueError("A source workbook must appear only once in the manifest")
    records, sources, exclusions, metric_exclusions, summaries = [], [], [], [], []
    for path, source, spec in sorted(verified, key=lambda item: item[1]["id"]):
        collected, quarantined, missing_metrics, summary = _collect_source(path, source, spec)
        records.extend(collected)
        sources.append(source)
        exclusions.extend(quarantined)
        metric_exclusions.extend(missing_metrics)
        summaries.append(summary)
    combinations = group_records(records)
    missing = Counter(field for record in records for field in record["missing_fields"])
    kb = {
        "schema": SCHEMA, "provenance": "independent_recollection_not_Jaeyun_release_restoration",
        "sources": sources, "combinations": combinations, "limitations": LIMITATIONS,
        "record_artifact": "records.jsonl", "audit_artifact": "audit.json",
        "summary": {"records_retained": len(records), "combinations": len(combinations),
                    "records_quarantined": len(exclusions), "sources": summaries,
                    "missing_fields": dict(sorted(missing.items())),
                    "unresolved_components": _unresolved_counts(records)},
    }
    exclusion_counts = Counter(reason["code"] for item in exclusions for reason in item["reasons"])
    audit = {
        "schema": SCHEMA, "provenance": kb["provenance"], "sources": sources,
        "summary": kb["summary"], "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "exclusions": exclusions, "metric_exclusions": metric_exclusions,
        "policies": {
            "selected_sheets_only": True, "raw_files_modified": False,
            "invalid_structures_and_invalid_numeric_rows": "quarantine_entire_row_without_repair_or_clamping",
            "missing_primary_measurement": "retain_record_exclude_from_numeric_statistics",
            "explicit_None_ligand_or_reagent": "absent_not_missing; retain_source_quantities_unchanged",
            "grouping": "source + reaction_type + measurement_basis/field/unit + whole_condition_tuple_including_amounts",
            "missing_field_counts": "record occurrences, including unavailable source fields; absent-component amounts are not missing",
            "exclusion_reason_counts": "reason occurrences; one quarantined row may have multiple reasons",
            "ignored_sheet_spans": "worksheet dimensions only, not collected observations or unique experiments",
        },
        "limitations": LIMITATIONS,
    }
    # Serialize everything before touching output files; NaN/Infinity are never emitted.
    kb_text = json.dumps(kb, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    audit_text = json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    records_text = "".join(_stable_json(record) + "\n" for record in records)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "condition_evidence.json").write_text(kb_text, encoding="utf-8", newline="\n")
    (output_dir / "audit.json").write_text(audit_text, encoding="utf-8", newline="\n")
    (output_dir / "records.jsonl").write_text(records_text, encoding="utf-8", newline="\n")
    return kb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DIRECTORY / "sources.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="Base directory for raw file paths in the manifest")
    args = parser.parse_args()
    kb = build_evidence(args.manifest, args.output_dir, source_root=args.source_root)
    print(json.dumps(kb["summary"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
