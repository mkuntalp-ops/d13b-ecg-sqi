#!/usr/bin/env bash
# v2 analysis (revision of 2026-11-05; REVISION_PLAN_2026-11-05_post_desk_rejection.md), stages in dependency order.
# Prerequisites: experiments/cinc2011_full_extraction.py and experiments/v2_setup.py have been run.
# Waits for the primary CV if it is still running so that MPS jobs never overlap. Stops at the first failure.
# Every stage writes into $ECG_RESULTS_DIR (results/v2_2026-11); results/ (v1) is not modified.
# Latency is not measured here (it must run alone; separate step).
set -euo pipefail
cd "$(dirname "$0")/.."
export ECG_RESULTS_DIR=results/v2_2026-11
PY=.venv/bin/python
TABPFN_PY=${TABPFN_PY:-/Volumes/KIOXIA/AI/tabpfn_env/bin/python}
LOG=$ECG_RESULTS_DIR/logs
mkdir -p "$LOG"

wait_for() { while pgrep -f "$1" >/dev/null; do sleep 60; done; }
# START_AT=<stage name> resumes an interrupted queue: earlier stages are skipped (their outputs must exist).
STARTED=${START_AT:+no}; STARTED=${STARTED:-yes}
stage() {
  local name=$1; shift
  if [ "$STARTED" = no ]; then
    if [ "$name" = "$START_AT" ]; then STARTED=yes; else echo "-- skip $name"; return 0; fi
  fi
  echo "== $(date '+%F %T') $name"; "$@" > "$LOG/$name.log" 2>&1; echo "   done $(date '+%F %T')"
}

wait_for "[r]un_supervised_cohort_cv.py"
test -f "$ECG_RESULTS_DIR/raw_supervised_cohort1000_oof_predictions.csv"

stage consolidate          $PY -u experiments/compute_consolidated_metrics.py
stage d1_linear_probe      $PY -u experiments/d1_linear_probe.py          # extracts the 433 new CinC 2011 records
stage hubert_ecg_probe     $PY -u experiments/hubert_ecg_probe.py         # authors' preprocessing (primary)
stage external_validation  $PY -u experiments/external_validation.py      # BUT-only models on CinC 2011 / 2017
stage d1_cinc2017_rescore  $PY -u experiments/d1_cinc2017_full.py --rescore
stage heuristic_refit      $PY -u experiments/heuristic_refit.py
stage tabpfn               $TABPFN_PY -u experiments/tabpfn_runner.py
stage lodo                 $PY -u experiments/lodo_multisource.py
stage paired_comparisons   $PY -u evaluation/paired_comparisons.py
stage model_cis            $PY -u evaluation/model_cis.py
stage paired_vs_best       $PY -u evaluation/paired_vs_best.py
stage decision_calibration $PY -u evaluation/decision_calibration.py
echo "v2 queue: done $(date '+%F %T')"
