#!/usr/bin/env bash
# Quota sensitivity analysis (ANALYSIS_PLAN_2026-11-16_quota_sensitivity.md), stages in dependency order.
# Prerequisites: `quota_sensitivity.py setup` and `quota_sensitivity.py d1` have been run.
# Every stage writes into results/v2_2026-11_quota_sensitivity; results/v2_2026-11 (primary) is only read.
# MPS jobs run one after another. Stops at the first failure. START_AT=<stage> resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
export ECG_RESULTS_DIR=results/v2_2026-11_quota_sensitivity PYTHONDONTWRITEBYTECODE=1
PY=".venv/bin/python -B"
TABPFN_PY=${TABPFN_PY:-/Volumes/KIOXIA/AI/tabpfn_env/bin/python}
LOG=$ECG_RESULTS_DIR/logs
mkdir -p "$LOG"
{ echo "started: $(date -u +%FT%TZ)"; echo "git HEAD: $(git rev-parse HEAD)"; echo "git status --porcelain (tracked and untracked code):";
  git status --porcelain -- experiments evaluation models features data tests config.py | sed 's/^/  /'; } >> "$LOG/queue_provenance.txt"
STARTED=${START_AT:+no}; STARTED=${STARTED:-yes}
stage() {
  local name=$1; shift
  if [ "$STARTED" = no ]; then
    if [ "$name" = "$START_AT" ]; then STARTED=yes; else echo "-- skip $name"; return 0; fi
  fi
  echo "== $(date '+%F %T') $name"; "$@" > "$LOG/$name.log" 2>&1; echo "   done $(date '+%F %T')"
}
stage supervised_cv        $PY -u experiments/run_supervised_cohort_cv.py
stage consolidate          $PY -u experiments/compute_consolidated_metrics.py
stage d1_linear_probe      $PY -u experiments/d1_linear_probe.py --analysis-only
stage d1_cinc2017_rescore  $PY -u experiments/d1_cinc2017_full.py --rescore
stage hubert               $PY -u experiments/quota_sensitivity.py hubert
stage external             $PY -u experiments/quota_sensitivity.py external
stage heuristic_refit      $PY -u experiments/heuristic_refit.py
stage tabpfn               $TABPFN_PY -B -u experiments/quota_sensitivity_tabpfn.py
stage paired_comparisons   $PY -u evaluation/paired_comparisons.py
stage model_cis            $PY -u evaluation/model_cis.py
stage paired_vs_best       $PY -u evaluation/paired_vs_best.py
stage analyse              $PY -u experiments/quota_sensitivity.py analyse
echo "quota sensitivity queue: done $(date '+%F %T')"
