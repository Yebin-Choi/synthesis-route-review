#!/bin/sh
# Overnight Ames experiments. Each stage writes its own report, so stopping
# loses only the stage that was running. The sweep stages skip rows already in
# their report, so a restart resumes rather than repeating.
# Resolved from this script, so the path is not tied to one machine.
HERE=$(cd "$(dirname "$0")" && pwd) || exit 1
PY="$HERE/../../external_tools/ames-env/Scripts/python.exe"
cd "$HERE" || exit 1

run() {
  echo "===== $* started $(date -Is) ====="
  "$PY" -X utf8 -u -B ames_experiments.py "$@" 2>&1
  echo "===== $* finished $(date -Is) rc=$? ====="
}

run --stage groups
# Both groupings for the checks, because the two are not nested and the
# difference is the size of the optimism in a leakage-group split.
run --stage checks --grouping scaffold
run --stage checks --grouping leakage
run --stage features --grouping scaffold
run --stage algorithms --grouping scaffold
echo "ALL AMES EXPERIMENTS DONE $(date -Is)"
