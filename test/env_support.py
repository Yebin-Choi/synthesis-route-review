"""What this interpreter can actually run, so the suite can say so.

Some tests need a dependency the running interpreter does not have, and some
need a dependency that is installed but hidden. The second kind is the
awkward one: the isolated worker is started with `python -I`, which drops the
user site-packages directory, so a package installed with `pip install
--user` is importable here and invisible there. Measured 2026-11-06 on this
machine: `requests` imports normally and vanishes under `-I`, which made
eight worker tests fail with "the configured interpreter could not start the
isolated worker" and looked for days like a broken AiZynthFinder.

It was not broken. Run against the interpreter the application is actually
configured with, the readiness probe succeeds and a propranolol search
returns ten routes in about 33 seconds.

A test that cannot run here should say so rather than fail, because a suite
with twenty unexplained failures tells a reader nothing.
"""

from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import unittest

APP = str(Path(__file__).resolve().parents[1] / "app")


@lru_cache(maxsize=None)
def importable(module):
    """Can this interpreter import the module in the ordinary way?"""
    try:
        __import__(module)
    except Exception:
        return False
    return True


@lru_cache(maxsize=None)
def importable_isolated(module):
    """Can a `python -I` child import it, which is how the worker runs?

    Returns (ok, reason). The reason names the module that was missing, so a
    skip message says what to install rather than only that something failed.
    """
    probe = (f"import sys; sys.path.insert(0, {APP!r}); "
             f"import {module}")
    result = subprocess.run([sys.executable, "-I", "-B", "-c", probe],
                            capture_output=True, text=True, errors="replace")
    if result.returncode == 0:
        return True, ""
    last = [line for line in result.stderr.strip().splitlines() if line.strip()]
    return False, last[-1] if last else f"exit {result.returncode}"


def requires(*modules):
    """Skip unless this interpreter can import every module named."""
    missing = [name for name in modules if not importable(name)]
    return unittest.skipIf(
        missing,
        "이 인터프리터에 " + ", ".join(missing) + "이(가) 없습니다. "
        "앱은 generation.local.json에 설정된 별도 환경에서 돌아갑니다.")


def requires_isolated_worker():
    """Skip unless a `python -I` child can import the application modules.

    This is the condition the worker runs under, so a test that starts a
    worker cannot pass without it.
    """
    ok, reason = importable_isolated("engine")
    return unittest.skipUnless(
        ok,
        "격리 모드(python -I)에서 앱 모듈을 불러오지 못합니다: " + reason
        + ". -I는 사용자 site-packages를 제외하므로, pip install --user로 "
          "깔린 패키지는 작업 프로세스에서 보이지 않습니다.")
