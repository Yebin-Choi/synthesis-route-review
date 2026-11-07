"""Inspect independently recollected literature conditions, never apply them.

A user's selected reaction type only filters references. It does not establish
target applicability, calibrated yield or a safety assessment. Name-only
components remain unresolved; this module does not attempt structure resolution.
"""

from copy import deepcopy
import json
from pathlib import Path
import re
from urllib.parse import urlparse


SCHEMA = "independent-condition-evidence/1"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "recollected" / "condition_evidence.json"


def _validate(kb):
    if not isinstance(kb, dict) or kb.get("schema") != SCHEMA:
        raise ValueError(f"Expected evidence schema {SCHEMA}")
    sources, combinations = kb.get("sources"), kb.get("combinations")
    if not isinstance(sources, list) or not isinstance(combinations, list):
        raise ValueError("Evidence sources and combinations must be arrays")
    by_id = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Evidence sources must be objects")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id or source_id in by_id:
            raise ValueError("Evidence source IDs must be non-empty and unique")
        sha256 = source.get("sha256")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
            raise ValueError(f"Raw source SHA256 is missing or invalid: {source_id}")
        url = source.get("url")
        parsed = urlparse(url) if isinstance(url, str) else None
        if parsed is None or parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Source URL is missing or invalid: {source_id}")
        for field in ("selected_sheet", "reaction_type", "measurement_basis"):
            if not isinstance(source.get(field), str) or not source[field]:
                raise ValueError(f"Source {field} is missing: {source_id}")
        by_id[source_id] = source
    seen = set()
    for group in combinations:
        if not isinstance(group, dict):
            raise ValueError("Evidence combinations must be objects")
        group_id, source_id = group.get("id"), group.get("source_id")
        if not isinstance(group_id, str) or not group_id or group_id in seen:
            raise ValueError("Evidence combination IDs must be non-empty and unique")
        if not isinstance(source_id, str) or source_id not in by_id:
            raise ValueError(f"Unknown source for combination: {group_id}")
        source = by_id[source_id]
        if (group.get("reaction_type") != source["reaction_type"]
                or group.get("measurement_basis") != source["measurement_basis"]):
            raise ValueError(f"Combination source/type/measurement basis mismatch: {group_id}")
        measurement = group.get("measurement")
        if not isinstance(measurement, dict) or measurement.get("basis") != group["measurement_basis"]:
            raise ValueError(f"Combination measurement basis is missing or inconsistent: {group_id}")
        seen.add(group_id)
    return by_id


def load_evidence(path=None):
    """Load the explicit JSON artifact; missing/bad files raise, with no fake fallback.

    Embedded hashes identify the collected raw files. Loading does not reverify
    the original workbooks or establish experimental validity.
    """
    path = Path(path) if path is not None else DEFAULT_PATH
    kb = json.loads(path.read_text(encoding="utf-8"))
    _validate(kb)
    return kb


def list_sources(kb):
    """Return independent source metadata ordered by ID, not evidence or safety scores."""
    sources = _validate(kb)
    return [deepcopy(sources[source_id]) for source_id in sorted(sources)]


def get_combinations(kb, reaction_type, limit=20):
    """Return source-separated copies for inspection, never recommended conditions.

    reaction_type must be an explicit exact label; no target classifier runs.
    Ordering is source ID, measurement basis, then stable combination ID. The
    limit is a display bound, not a chemical selection or performance ranking.
    """
    if not isinstance(reaction_type, str) or not reaction_type.strip():
        raise ValueError("An explicit reaction-type label is required")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    sources = _validate(kb)
    matches = sorted(
        (group for group in kb["combinations"] if group["reaction_type"] == reaction_type),
        key=lambda group: (group["source_id"], group["measurement_basis"], group["id"]),
    )
    results = []
    for group in matches[:limit]:
        item = deepcopy(group)
        item["source"] = deepcopy(sources[group["source_id"]])
        results.append(item)
    return results
