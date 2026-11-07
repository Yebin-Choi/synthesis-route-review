#!/bin/sh
# Follow-up analysis stages. Figures run last so they can read every report.
# Resolved from this script, so the path is not tied to one machine.
HERE=$(cd "$(dirname "$0")" && pwd) || exit 1
PY="$HERE/../../external_tools/ames-env/Scripts/python.exe"
cd "$HERE" || exit 1
run() {
  echo "===== $* started $(date -Is) ====="
  "$PY" -X utf8 -u -B ames_analysis.py "$@" 2>&1
  echo "===== $* finished $(date -Is) rc=$? ====="
}
run --stage scaffolds
run --stage agreement
run --stage descriptors
run --stage figures
echo "ALL AMES ANALYSIS DONE $(date -Is)"
