#!/usr/bin/env bash
# One-command regression check for the finished project (Day 26 extra).
# Runs: pytest, lint, report check, EVAL.md link check, quick demo, disk budget (< 12 GB).
# Usage: bash scripts/regress.sh   -> prints PASS/FAIL per step, exit 0 only if all pass.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
fail=0
step() { local name=$1; shift
  if timeout 900 "$@" >/tmp/tinyllm_regress_$$.log 2>&1; then echo "PASS  $name"
  else echo "FAIL  $name"; tail -15 /tmp/tinyllm_regress_$$.log; fail=1; fi; }
step pytest     "$PY" -m pytest -q
step lint       "$PY" scripts/check_lint.py
step report     "$PY" scripts/check_report.py
step eval_links "$PY" scripts/make_eval.py --check
step demo_quick bash scripts/demo.sh --quick
bytes=$(du -sb . | cut -f1)
if [ "$bytes" -lt 12000000000 ]; then echo "PASS  disk ($bytes bytes)"; else echo "FAIL  disk ($bytes bytes >= 12 GB)"; fail=1; fi
rm -f /tmp/tinyllm_regress_$$.log
[ $fail -eq 0 ] && echo "RESULT OK" || echo "RESULT FAIL"
exit $fail
