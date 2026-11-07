"""Run the two installed Java genotoxicity tools and keep their abstentions.

Measured on an independent set, Toxtree answered 80% of structures and EPA
TEST 77%. The rest are not wrong answers but no answer, and the two decline
differently: Toxtree reports its own "Error when applying the decision tree",
and separately can throw inside a rule while exiting 0 with the error column
reading NO; EPA TEST returns N/A when a structure falls outside its consensus
model's domain. Each is reported with a reason code. Nothing here turns a
non-answer into a negative.
"""

import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

if __package__:
    from .toxicology import canonicalize_smiles
else:
    from toxicology import canonicalize_smiles


ROOT = Path(__file__).resolve().parents[1]
MAX_DISTINCT_VALID = 256

TOXTREE_ARCHIVE = ROOT / "external_tools" / "Toxtree-v3.1.0.1851.zip"
TOXTREE_ARCHIVE_SHA256 = "222479c996f6d91616c135b7f433c8176139d63672bd28c1ccdad6372b3d82e8"
TOXTREE_PREFIX = "Toxtree-v3.1.0.1851/Toxtree/"
TOXTREE_HOME = ROOT / "external_tools" / "toxtree-home"
TOXTREE_JAR = "Toxtree-3.1.0.1851.jar"
TOXTREE_MODULE = "toxtree.plugins.ames.AmesMutagenicityRules"
TOXTREE_SOURCE = "https://toxtree.sourceforge.net/ames.html"
TOXTREE_POSITIVE = "Structural Alert for S. typhimurium mutagenicity"
TOXTREE_NEGATIVE = "No alerts for S. typhimurium mutagenicity"
TOXTREE_ERROR = "Error when applying the decision tree"
TOXTREE_RULE_FRAME = "at toxtree.plugins.ames.rules."

EPA_INSTALL = Path(os.environ.get("EPA_TEST_HOME")
                   or r"C:\Users\hahab\AppData\Local\Programs\TEST 5.1.2.0")
EPA_MAIN_CLASS = "ToxPredictor.Application.Calculations.RunFromCommandLine"
EPA_SOURCE = "https://www.epa.gov/comptox-tools/toxicity-estimation-software-tool-test"
EPA_POSITIVE = "mutagenicity positive"
EPA_NEGATIVE = "mutagenicity negative"

# Reason codes, shown instead of a blank when a tool does not answer. The
# distinction that matters is whether the tool declined or the run failed.
DECLINED = {
    "tool_error_category": "Toxtree가 이 구조에 결정 트리를 적용하지 못했다고 스스로 보고했습니다.",
    "outside_applicability_domain": "EPA TEST가 적용 범위 밖이라 예측하지 않았습니다.",
    "ambiguous_categories": "도구가 서로 모순된 분류를 돌려주어 판정하지 않았습니다.",
}
FAILED = {
    "rule_exception": "Toxtree 규칙이 실행 중 예외를 냈습니다. 결과를 신뢰할 수 없습니다.",
    "missing_record": "도구가 이 구조에 대한 기록을 돌려주지 않았습니다.",
    "tool_not_installed": "도구가 설치되어 있지 않습니다.",
    "run_failed": "도구 실행이 실패했습니다.",
    "timed_out": "도구 실행이 제한 시간을 넘겼습니다.",
    "unreadable_output": "도구 출력을 읽지 못했습니다.",
}

_LIMITATIONS = (
    "Research-only screening output, not a safety determination or an ICH M7 classification.",
    "A no-alert result is the absence of a known structural alert, not evidence of safety.",
    "A tool that declines to answer is recorded as an abstention with its reason; "
    "an abstention is never a negative prediction.",
)


def _java() -> str:
    return shutil.which("java") or "java"


def _assessment(value, canonical):
    text = value if isinstance(value, str) else repr(value)
    if canonical is None:
        return {"input_smiles": text, "canonical_smiles": None, "status": "invalid",
                "score": None, "reason": "unreadable_structure",
                "error": "구조를 해석하지 못했습니다."}
    return {"input_smiles": text, "canonical_smiles": canonical, "status": "pending",
            "verdict": None,
            "score": None, "reason": None, "error": None}


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


def _fail_all(rows, reason):
    for row in rows:
        row["status"] = "failed"
        row["score"] = None
        row["reason"] = reason
        row["error"] = FAILED.get(reason, reason)


def _mark(row, status, reason):
    row["status"] = status
    row["score"] = None
    row["reason"] = reason
    row["error"] = DECLINED.get(reason) or FAILED.get(reason) or reason


# --------------------------------------------------------------------------
# Toxtree
# --------------------------------------------------------------------------

def _sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def toxtree_home() -> Path | None:
    """The extracted distribution, unpacked once and reused afterwards.

    The archive hash is checked before anything is written, so a replaced
    archive fails loudly rather than running unknown code.
    """
    jar = TOXTREE_HOME / TOXTREE_JAR
    if jar.is_file():
        return TOXTREE_HOME
    if not TOXTREE_ARCHIVE.is_file():
        return None
    if _sha256(TOXTREE_ARCHIVE) != TOXTREE_ARCHIVE_SHA256:
        raise ValueError("Toxtree 압축 파일이 확인된 공식 배포본과 다릅니다")
    staging = TOXTREE_HOME.with_name(TOXTREE_HOME.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(TOXTREE_ARCHIVE) as source:
        for info in source.infolist():
            if not info.filename.startswith(TOXTREE_PREFIX) or info.is_dir():
                continue
            relative = Path(info.filename[len(TOXTREE_PREFIX):])
            target = staging / relative
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("압축 파일에 안전하지 않은 항목이 있습니다")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
    if not (staging / TOXTREE_JAR).is_file():
        shutil.rmtree(staging)
        raise ValueError("공식 배포본에 Toxtree jar가 없습니다")
    staging.rename(TOXTREE_HOME)
    return TOXTREE_HOME


def _normalize(value):
    return " ".join(value.split())


def _read_toxtree(path, expected):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = [_normalize(name) for name in (reader.fieldnames or [])]
        required = {"evaluation_id", TOXTREE_POSITIVE, TOXTREE_NEGATIVE, TOXTREE_ERROR}
        if not required.issubset(headers):
            raise ValueError("Toxtree 출력 형식을 알아볼 수 없습니다")
        results = {}
        for values in reader:
            if any(value is None for value in values.values()):
                continue
            row = {_normalize(key): value for key, value in values.items()}
            identity = row["evaluation_id"]
            if identity in expected:
                results[identity] = row
    return results


def _toxtree_call(record):
    values = [record[column].strip().upper()
              for column in (TOXTREE_POSITIVE, TOXTREE_NEGATIVE, TOXTREE_ERROR)]
    if any(value not in {"YES", "NO"} for value in values):
        return None, "ambiguous_categories"
    positive, negative, error = values
    if error == "YES":
        return None, "tool_error_category"
    if positive == "YES" and negative == "NO":
        return 1, None
    if negative == "YES" and positive == "NO":
        return 0, None
    return None, "ambiguous_categories"


def assess_toxtree_materials(smiles, *, timeout=900) -> dict:
    """Screen distinct valid structures with the Toxtree Ames rulebase."""
    rows, valid = _collect(smiles)
    report = {"model": {"tool": "Toxtree", "version": "3.1.0.1851", "endpoint": "ames",
                        "source": TOXTREE_SOURCE},
              "assessments": rows, "limitations": list(_LIMITATIONS)}
    pending = list(valid.values())
    if not pending:
        return report
    try:
        home = toxtree_home()
    except ValueError as exc:
        _fail_all(pending, "run_failed")
        for row in pending:
            row["error"] = str(exc)
        return report
    if home is None:
        _fail_all(pending, "tool_not_installed")
        return report

    identities = {f"m{index}": item for index, item in enumerate(valid)}
    try:
        with tempfile.TemporaryDirectory(prefix="toxtree_") as directory:
            work = Path(directory)
            input_path = work / "input.csv"
            output_path = work / "output.csv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["evaluation_id", "SMILES"])
                writer.writeheader()
                for identity, item in identities.items():
                    writer.writerow({"evaluation_id": identity, "SMILES": item})
            completed = subprocess.run(
                [_java(), "-Xmx2048m", "-Djava.awt.headless=true",
                 "-Dfile.encoding=UTF-8", "-jar", str(home / TOXTREE_JAR),
                 "-n", "-i", str(input_path), "-o", str(output_path),
                 "-m", TOXTREE_MODULE],
                cwd=home, capture_output=True, text=True, errors="replace",
                timeout=timeout, check=False)
            # Toxtree swallows rule exceptions: it exits 0 and the per-record
            # error column still reads NO, so the log is the only signal that a
            # rule did not evaluate. The log never names the record, so this
            # only says "somewhere in this batch".
            raised = TOXTREE_RULE_FRAME in (completed.stderr or "")
            if not output_path.is_file():
                _fail_all(pending, "run_failed")
                return report
            results = _read_toxtree(output_path, set(identities))
    except subprocess.TimeoutExpired:
        _fail_all(pending, "timed_out")
        return report
    except (OSError, ValueError):
        _fail_all(pending, "unreadable_output")
        return report

    report["model"]["rule_exception_in_batch"] = bool(raised)
    for identity, item in identities.items():
        row = valid[item]
        record = results.get(identity)
        if record is None:
            _mark(row, "failed", "missing_record")
            continue
        call, reason = _toxtree_call(record)
        if call is None:
            _mark(row, "declined", reason)
            continue
        # A rule that crashed can only hide an alert, never invent one, so a
        # found alert still stands while a no-alert from a batch that raised
        # cannot be trusted.
        if raised and call == 0:
            _mark(row, "failed", "rule_exception")
            continue
        row["status"] = "success"
        row["score"] = float(call)
        row["reason"] = None
        row["error"] = None
    return report


# --------------------------------------------------------------------------
# EPA TEST
# --------------------------------------------------------------------------

def epa_test_readiness() -> dict:
    jar = EPA_INSTALL / "WebTEST.jar"
    java = EPA_INSTALL / "jre" / "bin" / "java.exe"
    missing = [name for name, path in (("install", EPA_INSTALL), ("jar", jar),
                                       ("jre", java))
               if not path.exists()]
    return {"ready": not missing, "missing": missing, "install": str(EPA_INSTALL)}


def _read_epa(path):
    """Results keyed by the Query column, which echoes the submitted SMILES.

    The input file is one SMILES per line with no identifier column: the tool
    accepts nothing else, so the echoed query is the only way back to the row.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        key = (row.get("Query") or "").strip()
        if key and key not in out:
            out[key] = row
    return out


def assess_epa_test_materials(smiles, *, timeout=1800) -> dict:
    """Run the EPA TEST mutagenicity consensus over distinct valid structures.

    Only the prediction columns are read. The output also carries Exp_Value and
    Exp_Result, experimental values EPA TEST already holds for some compounds;
    reading those would be reading the answer key, so they are dropped.
    """
    rows, valid = _collect(smiles)
    report = {"model": {"tool": "EPA TEST", "version": "5.1.2.0", "endpoint": "mutagenicity",
                        "source": EPA_SOURCE, "method": "consensus"},
              "assessments": rows, "limitations": list(_LIMITATIONS)}
    pending = list(valid.values())
    if not pending:
        return report
    readiness = epa_test_readiness()
    if not readiness["ready"]:
        _fail_all(pending, "tool_not_installed")
        return report

    identities = {f"m{index}": item for index, item in enumerate(valid)}
    try:
        with tempfile.TemporaryDirectory(prefix="epa_test_") as directory:
            work = Path(directory)
            smi_path = work / "input.smi"
            out_path = work / "output.csv"
            # One SMILES per line, nothing else: the tool rejects an input file
            # that carries an identifier column and then writes no output at
            # all, which is what an empty result directory looks like.
            smi_path.write_text("\n".join(identities.values()) + "\n",
                                encoding="utf-8")
            # Headless mode cannot be used. The command-line runner builds a
            # Swing JFrame on its way to the calculation, so -Djava.awt.
            # headless=true makes it throw HeadlessException before writing
            # anything. Measured 2026-11-05 on TEST 5.1.2.0: with the flag the
            # run produces no output file at all, which reads as a failed tool
            # rather than as a misconfiguration. The cost of leaving it off is
            # that a window may flash on the desktop during a batch.
            subprocess.run(
                [str(EPA_INSTALL / "jre" / "bin" / "java.exe"),
                 "-cp", "WebTEST.jar", EPA_MAIN_CLASS,
                 "-i", str(smi_path), "-o", str(out_path),
                 "-e", "Mutagenicity", "-m", "consensus"],
                cwd=EPA_INSTALL, capture_output=True, text=True, errors="replace",
                timeout=timeout, check=False)
            if not out_path.is_file():
                _fail_all(pending, "run_failed")
                return report
            results = _read_epa(out_path)
    except subprocess.TimeoutExpired:
        _fail_all(pending, "timed_out")
        return report
    except (OSError, ValueError):
        _fail_all(pending, "unreadable_output")
        return report

    for item in identities.values():
        row = valid[item]
        record = results.get(item)
        if record is None:
            _mark(row, "failed", "missing_record")
            continue
        verdict = (record.get("Pred_Result") or "").strip().lower()
        value = (record.get("Pred_Value") or "").strip()
        if verdict not in {EPA_POSITIVE, EPA_NEGATIVE}:
            # An empty verdict is the consensus model declining to extrapolate
            # outside its applicability domain, not a run that failed.
            _mark(row, "declined", "outside_applicability_domain")
            continue
        try:
            score = float(value)
        except ValueError:
            score = 1.0 if verdict == EPA_POSITIVE else 0.0
        row["status"] = "success"
        row["score"] = score
        # The tool's own call. Its consensus value can fall outside 0 to 1
        # (measured: ethanol reads -0.01), so a line drawn through the value
        # is not the same statement the tool made.
        row["verdict"] = "positive" if verdict == EPA_POSITIVE else "negative"
        row["reason"] = None
        row["error"] = None
        row["verdict"] = verdict
    return report
