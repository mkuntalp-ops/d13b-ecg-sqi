#!/usr/bin/env bash
# v2 analysis, part 2 (handoff item 1 of REVISION_PLAN_2026-11-05_post_desk_rejection.md, 2026-11-07):
# the secondary analyses that retrain models, in increasing run time. Stops at the first failure.
# Run beforehand (no training, minutes): cinc2011_constant_lead_check.py, nested_thresholds.py --transfer-only,
# butqdb_official_split.py --split-only, tabpfn_runner.py --official-split (TabPFN env), local_recalibration.py,
# prevalence_subjects_features.py.
# Every stage writes into $ECG_RESULTS_DIR (results/v2_2026-11); results/ and figures/ (v1) are not modified.
set -euo pipefail
cd "$(dirname "$0")/.."
export ECG_RESULTS_DIR=results/v2_2026-11
export ECG_FIGURES_DIR=$ECG_RESULTS_DIR/figures
PY=.venv/bin/python
LOG=$ECG_RESULTS_DIR/logs
mkdir -p "$LOG"

STARTED=${START_AT:+no}; STARTED=${STARTED:-yes}
stage() {
  local name=$1; shift
  if [ "$STARTED" = no ]; then
    if [ "$name" = "$START_AT" ]; then STARTED=yes; else echo "-- skip $name"; return 0; fi
  fi
  echo "== $(date '+%F %T') $name"; "$@" > "$LOG/$name.log" 2>&1; echo "   done $(date '+%F %T')"
}

test -f "$ECG_RESULTS_DIR/tabpfn_official_split_predictions.csv"

stage butqdb_official_split  $PY -u experiments/butqdb_official_split.py     # F1b, TabPFN included
stage exp4_revised_nstdb     $PY -u experiments/exp4_revised_nstdb.py        # includes zero-shot d1-3B on NSTDB
stage butqdb_class2          $PY -u experiments/butqdb_class2_sensitivity.py
stage nested_thresholds      $PY -u experiments/nested_thresholds.py         # nested CV, then transfer
stage seed_stability         $PY -u experiments/seed_stability.py
stage seed_repeats_transfer  $PY -u experiments/seed_repeats_transfer.py     # resumable; analysis at the end
echo "v2 queue part 2: done $(date '+%F %T')"
