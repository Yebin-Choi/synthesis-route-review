"""Check what a Gemini API key can actually do, without printing the key.

The app asks for the key in a password field and passes it to the child
process as GEMINI_API_KEY; nothing is stored. This probe reads the same
variable, so the key never has to be typed anywhere else.

    set GEMINI_API_KEY=...        (cmd)
    $env:GEMINI_API_KEY = "..."   (PowerShell)
    python evaluation/scripts/probe_gemini.py

It answers three things:

* whether the key works at all,
* which models the key is allowed to list,
* which of the models this project names actually return text.

The last one matters because a free-tier key lists a model and then refuses
it at call time (benchmark log 6-49). Listing is not permission.

Output goes to evaluation/results/gemini_probe.json. The key is never
written to it, printed, or put in a URL that is logged.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluation" / "results" / "gemini_probe.json"
API = "https://generativelanguage.googleapis.com/v1beta"

# What this project names. The config asks for the first one; the rest are
# here to say what the key can reach if that one is refused.
CANDIDATES = [
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]
PROMPT = "Reply with the single word OK."


def _request(url, key, payload=None, *, timeout=60):
    """One call. The key goes in a header, not the query string, so it does
    not end up in any proxy or server log that records URLs."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data,
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), None
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        try:
            message = json.loads(body)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = body[:300]
        return None, f"HTTP {error.code}: {message}"
    except (urllib.error.URLError, OSError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def list_models(key):
    payload, error = _request(f"{API}/models", key)
    if error:
        return None, error
    names = []
    for entry in payload.get("models", []):
        name = entry.get("name", "").removeprefix("models/")
        methods = entry.get("supportedGenerationMethods", [])
        if name and "generateContent" in methods:
            names.append(name)
    return sorted(names), None


def try_model(key, model):
    started = time.monotonic()
    payload, error = _request(
        f"{API}/models/{urllib.parse.quote(model)}:generateContent", key,
        {"contents": [{"parts": [{"text": PROMPT}]}]})
    elapsed = round(time.monotonic() - started, 2)
    if error:
        return {"model": model, "ok": False, "error": error, "seconds": elapsed}
    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return {"model": model, "ok": False, "seconds": elapsed,
                "error": "응답에 text가 없습니다.",
                "finish_reason": (payload.get("candidates") or [{}])[0].get("finishReason")}
    usage = payload.get("usageMetadata", {})
    return {"model": model, "ok": True, "seconds": elapsed, "reply": text[:80],
            "prompt_tokens": usage.get("promptTokenCount"),
            "total_tokens": usage.get("totalTokenCount")}


def main():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("GEMINI_API_KEY가 비어 있습니다. 키를 환경변수로 넣고 다시 실행하세요.")
        return 2

    listed, listing_error = list_models(key)
    if listing_error:
        print(f"모델 목록을 받지 못했습니다: {listing_error}")
    else:
        print(f"이 키로 부를 수 있다고 적힌 모델 {len(listed)}개")

    attempts = []
    for model in CANDIDATES:
        result = try_model(key, model)
        attempts.append(result)
        mark = "OK  " if result["ok"] else "거부"
        detail = result.get("reply") if result["ok"] else result.get("error", "")
        print(f"  {mark} {model:<28} {result['seconds']:>5.2f}s  {detail}")

    working = [row["model"] for row in attempts if row["ok"]]
    summary = {
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "key_source": "GEMINI_API_KEY environment variable (not stored)",
        "listed_models": listed,
        "listing_error": listing_error,
        "attempts": attempts,
        "working_models": working,
        # Listing a model is not permission to call it. The free tier lists
        # Pro and then refuses it, so only the attempts decide this.
        "listed_but_refused": sorted(
            set(listed or []) & {row["model"] for row in attempts if not row["ok"]}),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n쓸 수 있는 모델: {', '.join(working) if working else '없음'}")
    print(f"기록: {OUT.relative_to(ROOT)}")
    return 0 if working else 1


if __name__ == "__main__":
    sys.exit(main())
