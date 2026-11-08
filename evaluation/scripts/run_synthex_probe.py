"""Run SynthEx end to end with a real key, and write down where it stops.

SynthEx has never produced a route on this machine. The last live attempt
(benchmark log 6-47, Groq) got through the strategy stage and died in route
search on a schema failure and a rate limit. The Gemini Pro key now answers
(log 6-60), so this runs the same path with it and records the result rather
than guessing.

    $env:GEMINI_API_KEY = "..."      (or use run_synthex_probe.ps1)
    .venv/Scripts/python.exe evaluation/scripts/run_synthex_probe.py [SMILES]

Default target is aspirin, the same one 6-47 used, so the two runs compare.
The key is read from the environment, never printed and never written to the
artifact. Output: evaluation/results/synthex_live_run.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))

from generation_service import GenerationService  # noqa: E402

OUT = ROOT / "evaluation" / "results" / "synthex_live_run.json"
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
SELECTION = {"provider": "gemini", "base_url": None,
             "model": "gemini-3.1-pro-preview"}
BUSY = {"queued", "running"}
PROBE_TIMEOUT = 300
RUN_TIMEOUT = 1200


def _wait(service, request_id, context_id, limit, label):
    started = time.monotonic()
    last = None
    while time.monotonic() - started < limit:
        snapshot = service.poll(request_id, context_id=context_id)
        if snapshot["status"] != last:
            last = snapshot["status"]
            print(f"  [{time.monotonic() - started:6.1f}s] {label}: {last}")
        if snapshot["status"] not in BUSY:
            return snapshot
        time.sleep(2.0)
    return {"status": "timeout", "request_id": request_id, "result": None,
            "error": {"message": f"{label}이 {limit}초를 넘겼습니다."}}


def _redacted(value):
    """Nothing resembling a key reaches the artifact."""
    if isinstance(value, dict):
        return {key: ("<redacted>"
                      if isinstance(val, str) and any(
                          token in key.lower()
                          for token in ("key", "secret", "token"))
                      else _redacted(val))
                for key, val in value.items()}
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    return value


def main():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("GEMINI_API_KEY가 비어 있습니다.")
        return 2
    # The key travels as an HTTP header, and a header carries ASCII only. A
    # paste that picked up Korean text fails deep inside the first LLM call
    # as UnicodeEncodeError, 36 seconds in, naming no cause. Caught here it
    # costs nothing. The key itself is never printed.
    if not key.isascii():
        print(f"키에 ASCII가 아닌 글자가 있습니다 (길이 {len(key)}자). "
              "한글이나 따옴표가 섞여 들어간 것 같습니다. 키만 다시 붙여넣으세요.")
        return 2
    target = sys.argv[1] if len(sys.argv) > 1 else ASPIRIN
    print(f"목표: {target}")
    print(f"모델: {SELECTION['model']}")

    service = GenerationService()
    context_id = str(uuid4())

    print("\n준비 확인")
    probe_id = str(uuid4())
    service.start_readiness("synthex", request_id=probe_id,
                            context_id=context_id, llm_selection=SELECTION)
    probe = _wait(service, probe_id, context_id, PROBE_TIMEOUT, "준비 확인")
    readiness = probe.get("readiness") or service.readiness("synthex", SELECTION)
    print(f"  ready={readiness.get('ready')} "
          f"configuration_id={readiness.get('configuration_id')}")
    if readiness.get("error"):
        print(f"  오류: {readiness['error']}")

    record = {"target": target, "model": SELECTION["model"],
              "readiness": _redacted(readiness), "probe_status": probe["status"]}

    if not readiness.get("ready"):
        print("\n준비가 안 돼 생성은 시도하지 않았습니다.")
        record["generation"] = None
    else:
        print("\n경로 생성")
        run_id = str(uuid4())
        service.start("synthex", target, request_id=run_id, context_id=context_id,
                      configuration_id=readiness.get("configuration_id"),
                      gemini_api_key=key, llm_selection=SELECTION)
        snapshot = _wait(service, run_id, context_id, RUN_TIMEOUT, "경로 생성")
        record["generation"] = _redacted(snapshot)
        result = snapshot.get("result") or {}
        routes = (result.get("routes") or []) if isinstance(result, dict) else []
        print(f"\n상태: {snapshot['status']}")
        print(f"경로 수: {len(routes)}")
        if snapshot.get("error"):
            print(f"오류: {json.dumps(snapshot['error'], ensure_ascii=False)[:400]}")
        stages = (result.get("stages") if isinstance(result, dict) else None) or {}
        for name, entry in stages.items() if isinstance(stages, dict) else []:
            state = entry.get("status") if isinstance(entry, dict) else entry
            print(f"  단계 {name}: {state}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n기록: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
