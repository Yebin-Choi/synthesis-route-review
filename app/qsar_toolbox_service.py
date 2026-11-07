"""Run the installed OECD QSAR Toolbox Ames profilers over its local WebAPI.

The Toolbox is a desktop application; its WebAPI is a self-hosted Kestrel
server that ships in the same install. Nothing leaves this machine: the base
URL must resolve to loopback or this module refuses to call it.

Three things about the API shape a correct reading of its answers, and each
was measured against the running 4.9 install rather than assumed:

* A SMILES search returns every database record carrying that structure, and
  the order is reshuffled on every call. Taking the first row would make the
  same structure read positive on one run and abstain on the next. Records are
  therefore filtered and sorted before one is picked.
* Polymer and UVCB records answer "(N/A)": the profiler did not evaluate them.
  That is an abstention, not a negative.
* A mixture record answers with the alerts of one constituent AND the literal
  "No alert found" for another, in a single list. A negative is therefore only
  a list that carries nothing except "No alert found".

The profilers are alert sets applied to the parent structure. No metabolic
simulator is requested (the empty GUID), so these are parent-structure alerts
and not a simulation of metabolic activation.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

if __package__:
    from .toxicology import canonicalize_smiles
else:
    from toxicology import canonicalize_smiles


DEFAULT_BASE_URL = "http://127.0.0.1:5000"
API_VERSION = "v6"
MAX_DISTINCT_VALID = 256

# No metabolic simulator: profile the parent structure as supplied.
NO_SIMULATOR = "00000000-0000-0000-0000-000000000000"

SOURCE = "https://qsartoolbox.org/developers/webapi/"

# Captions as the running install reports them. The GUID is resolved from the
# caption at run time rather than hard-coded, so a reinstall that renumbers
# its profilers fails loudly instead of profiling with the wrong alert set.
OASIS_AMES_PROFILER = "DNA alerts for AMES, CA and MNT by OASIS"
ISS_AMES_PROFILER = "in vitro mutagenicity (Ames test) alerts by ISS"

# Literal strings the profiler returns instead of an alert name.
NO_ALERT = "No alert found"
NOT_APPLICABLE = "(N/A)"

# Record kinds the profiler does not evaluate; it answers "(N/A)" for them.
UNPROFILABLE_TYPES = ("Polymer", "Uvcb")

DECLINED = {
    "tool_rejected_structure": "QSAR Toolbox가 이 구조를 읽지 못했습니다.",
    "not_applicable_record": "QSAR Toolbox가 이 기록에는 경고구조를 적용하지 않았습니다.",
    "no_record": "QSAR Toolbox가 이 구조에 해당하는 기록을 돌려주지 않았습니다.",
}
FAILED = {
    "tool_not_running": "QSAR Toolbox WebAPI가 실행되고 있지 않습니다.",
    "run_failed": "QSAR Toolbox 호출이 실패했습니다.",
    "timed_out": "QSAR Toolbox 호출이 제한 시간을 넘겼습니다.",
    "unreadable_output": "QSAR Toolbox 응답을 읽지 못했습니다.",
    "profiler_not_found": "요청한 경고구조 목록이 설치본에 없습니다.",
    "remote_base_url": "QSAR Toolbox 주소가 이 컴퓨터가 아닙니다.",
}

_LIMITATIONS = (
    "Research-only screening output, not a safety determination or an ICH M7 classification.",
    "A no-alert result is the absence of a known structural alert, not evidence of safety.",
    "A tool that declines to answer is recorded as an abstention with its reason; "
    "an abstention is never a negative prediction.",
    "Alerts are applied to the parent structure; no metabolic simulator is requested.",
)

_LOOPBACK = ("127.0.0.1", "::1", "localhost", "[::1]")

# Resolved once per process: caption -> guid, and the reported versions.
_profiler_cache = {}
_version_cache = {}


def base_url() -> str:
    """The WebAPI root, overridable for a different local port."""
    return os.environ.get("QSAR_TOOLBOX_WEBAPI", DEFAULT_BASE_URL).rstrip("/")


def _check_local(url):
    """Refuse any base URL that is not loopback. Structures stay on this PC."""
    host = urllib.parse.urlparse(url).hostname
    if host is None or host.lower() not in _LOOPBACK:
        raise ValueError(
            f"QSAR Toolbox 주소는 이 컴퓨터(localhost)여야 합니다: {url!r}")


def _get(path, *, timeout, query=None):
    root = base_url()
    _check_local(root)
    url = f"{root}/api/{API_VERSION}/{path.lstrip('/')}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", "replace")
    return json.loads(body) if body.strip() else None


def toolbox_versions(*, timeout=60) -> dict:
    """Versions as the running install reports them, not as we assume."""
    if not _version_cache:
        try:
            _version_cache["toolbox"] = str(_get("about/toolbox/version", timeout=timeout))
            _version_cache["webapi"] = str(_get("about/webapi/version", timeout=timeout))
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            return {"toolbox": None, "webapi": None}
    return dict(_version_cache)


def profiler_guid(caption, *, timeout=300) -> str:
    """The GUID the install gives this profiler caption. Unknown is an error."""
    if caption in _profiler_cache:
        return _profiler_cache[caption]
    profilers = _get("profiling", timeout=timeout) or []
    for entry in profilers:
        name = entry.get("Caption") or entry.get("caption")
        guid = entry.get("Guid") or entry.get("guid")
        if name and guid:
            _profiler_cache[name] = guid
    if caption not in _profiler_cache:
        raise ValueError(
            f"설치된 QSAR Toolbox에 '{caption}' 경고구조 목록이 없습니다.")
    return _profiler_cache[caption]


def _pick_record(hits):
    """One record, chosen the same way every call.

    Search order is reshuffled per call, so the pick cannot depend on it.
    Polymer and UVCB records are set aside because the profiler answers
    "(N/A)" for them; if they are all that came back they are still used, so
    the caller sees an abstention with a reason rather than a silent drop.
    """
    usable = [hit for hit in hits
              if (hit.get("SubstanceType") or "") not in UNPROFILABLE_TYPES]
    pool = usable or list(hits)
    if not pool:
        return None
    return sorted(pool, key=lambda hit: str(hit.get("ChemId") or ""))[0]


def _read_alerts(strings):
    """Map the profiler's string list to a verdict.

    Returns (verdict, alerts, reason). A mixture answers with a constituent's
    alerts and "No alert found" in one list, so a negative is only a list
    holding nothing but "No alert found".
    """
    values = [str(value) for value in (strings or [])]
    if not values:
        return None, [], "no_record"
    alerts = [value for value in values
              if value not in (NO_ALERT, NOT_APPLICABLE)]
    if alerts:
        return "positive", alerts, None
    if NO_ALERT in values:
        return "negative", [], None
    return None, [], "not_applicable_record"


def _assessment(value, canonical):
    text = value if isinstance(value, str) else repr(value)
    if canonical is None:
        return {"input_smiles": text, "canonical_smiles": None, "status": "invalid",
                "verdict": None, "score": None, "alerts": [],
                "reason": "unreadable_structure",
                "error": "구조를 해석하지 못했습니다."}
    return {"input_smiles": text, "canonical_smiles": canonical, "status": "pending",
            "verdict": None, "score": None, "alerts": [], "reason": None, "error": None}


def _collect(smiles):
    """Distinct valid structures in first-seen order, aliases kept."""
    if isinstance(smiles, (str, bytes)) or smiles is None:
        values = [smiles]
    else:
        values = list(smiles)
    rows, valid = [], {}
    for value in values:
        canonical = canonicalize_smiles(value)
        if canonical is not None and canonical in valid:
            row = valid[canonical]
            aliases = row.setdefault("input_smiles_list", [row["input_smiles"]])
            text = value if isinstance(value, str) else repr(value)
            if text not in aliases:
                aliases.append(text)
            continue
        row = _assessment(value, canonical)
        rows.append(row)
        if canonical is not None:
            valid[canonical] = row
            if len(valid) > MAX_DISTINCT_VALID:
                raise ValueError(
                    f"한 번에 서로 다른 유효 구조를 {MAX_DISTINCT_VALID}개까지만 받습니다")
    return rows, valid


def _mark(row, status, reason):
    row["status"] = status
    row["verdict"] = None
    row["score"] = None
    row["reason"] = reason
    row["error"] = DECLINED.get(reason) or FAILED.get(reason) or reason


def _fail_all(rows, reason):
    for row in rows:
        _mark(row, "failed", reason)


def _declined(reason):
    return {"status": "declined", "verdict": None, "alerts": [], "reason": reason}


def _failed(reason):
    return {"status": "failed", "verdict": None, "alerts": [], "reason": reason}


def _search_record(smiles, *, timeout):
    """Find the one record to profile. Returns (record, failure_outcome)."""
    try:
        hits = _get("search/smiles/true/true", timeout=timeout,
                    query={"smiles": smiles})
    except urllib.error.HTTPError as error:
        # The Toolbox's own SMILES reader rejects structures RDKit accepts
        # (measured: "Unrealistic valent state for atom"). That is the tool
        # declining, so it stays an abstention with its reason.
        if error.code == 400:
            return None, _declined("tool_rejected_structure")
        return None, _failed("run_failed")
    except urllib.error.URLError as error:
        reason = ("tool_not_running"
                  if isinstance(getattr(error, "reason", None),
                                (ConnectionRefusedError, OSError))
                  else "run_failed")
        return None, _failed(reason)
    except (TimeoutError, OSError):
        return None, _failed("timed_out")
    except (ValueError, json.JSONDecodeError):
        return None, _failed("unreadable_output")

    record = _pick_record(hits or [])
    if record is None or not (record.get("ChemId") or record.get("chemId")):
        return None, _declined("no_record")
    return record, None


def _profile_record(record, guid, *, timeout):
    chem_id = record.get("ChemId") or record.get("chemId")
    try:
        strings = _get(f"profiling/{guid}/{chem_id}/{NO_SIMULATOR}", timeout=timeout)
    except (urllib.error.HTTPError, urllib.error.URLError):
        return _failed("run_failed")
    except (TimeoutError, OSError):
        return _failed("timed_out")
    except (ValueError, json.JSONDecodeError):
        return _failed("unreadable_output")

    verdict, alerts, reason = _read_alerts(strings)
    if verdict is None:
        return _declined(reason)
    return {"status": "success", "verdict": verdict, "alerts": alerts,
            "reason": None, "chem_id": str(chem_id),
            "substance_type": record.get("SubstanceType")}


def profile_multi(smiles, guids, *, timeout=120) -> dict:
    """Profile one structure with several profilers, searching only once.

    The search is the slow half of the call, so a batch comparing two alert
    sets over the same structures pays for it once instead of twice.
    """
    record, failure = _search_record(smiles, timeout=timeout)
    if failure is not None:
        return {guid: dict(failure) for guid in guids}
    return {guid: _profile_record(record, guid, timeout=timeout) for guid in guids}


def profile_one(smiles, guid, *, timeout=120) -> dict:
    """Profile one structure. Returns the same fields a row carries.

    Exposed so a batch runner can go structure by structure without the
    distinct-structure cap that the app's entry point applies.
    """
    return profile_multi(smiles, [guid], timeout=timeout)[guid]


def assess_qsar_toolbox_materials(smiles, *, profiler=OASIS_AMES_PROFILER,
                                  timeout=120) -> dict:
    """Screen distinct valid structures with one Toolbox Ames profiler."""
    rows, valid = _collect(smiles)
    versions = toolbox_versions()
    report = {"model": {"tool": "OECD QSAR Toolbox",
                        "version": versions.get("toolbox"),
                        "endpoint": "ames",
                        "source": SOURCE,
                        "profiler": profiler,
                        "webapi_version": versions.get("webapi"),
                        "webapi_endpoint": base_url()},
              "assessments": rows, "limitations": list(_LIMITATIONS)}
    pending = list(valid.values())
    if not pending:
        return report

    try:
        _check_local(base_url())
    except ValueError as error:
        _fail_all(pending, "remote_base_url")
        for row in pending:
            row["error"] = str(error)
        return report

    try:
        guid = profiler_guid(profiler)
    except ValueError as error:
        _fail_all(pending, "profiler_not_found")
        for row in pending:
            row["error"] = str(error)
        return report
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        _fail_all(pending, "tool_not_running")
        return report

    report["model"]["profiler_guid"] = guid
    for canonical, row in valid.items():
        outcome = profile_one(canonical, guid, timeout=timeout)
        if outcome["status"] == "success":
            row["status"] = "success"
            row["verdict"] = outcome["verdict"]
            row["alerts"] = outcome["alerts"]
            row["reason"] = None
            row["error"] = None
        else:
            _mark(row, outcome["status"], outcome["reason"])
    return report
