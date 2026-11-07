#!/bin/sh
# Cause analysis plus the heavy nested search, ordered cheapest first so the
# quick answers land even if the long one is still running.
# Resolved from this script, so the path is not tied to one machine.
HERE=$(cd "$(dirname "$0")" && pwd) || exit 1
PY="$HERE/../../external_tools/ames-env/Scripts/python.exe"
cd "$HERE" || exit 1
run() {
  echo "===== $* started $(date -Is) ====="
  "$PY" -X utf8 -u -B ames_why.py "$@" 2>&1
  echo "===== $* finished $(date -Is) rc=$? ====="
}
run --stage redundancy
run --stage grouping
run --stage calibration
run --stage difficulty
run --stage tuning
echo "ALL AMES WHY DONE $(date -Is)"
