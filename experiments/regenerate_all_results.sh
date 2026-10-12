#!/bin/bash
# Regenerate every result cited in the manuscript after the 2026-10-12 fixes.
# Old outputs are moved to results/_backup_pre_fix_2026-10-12/previous_outputs/ first:
# run_d1_neural_inference.py resumes from existing CSVs, so stale d1 predictions must not survive.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BK=results/_backup_pre_fix_2026-10-12/previous_outputs
mkdir -p "$BK"
for f in \
    results/raw_supervised_cohort1000_oof_predictions.csv \
    results/raw_d1_neural_cohort1000_predictions.csv \
    results/raw_d1_neural_cohort1000_responses.jsonl \
    results/raw_d1_neural_cinc2011_predictions.csv \
    results/raw_d1_neural_cinc2011_responses.jsonl \
    results/cohort1000_consolidated_oof_predictions.csv \
    results/cohort1000_consolidated_summary.csv \
    results/cohort1000_consolidated_operating_points.csv \
    results/cinc2011_expanded_predictions.csv \
    results/cinc2011_expanded_transfer_summary.csv \
    results/exp4_revised_nstdb_predictions.csv \
    results/exp4_revised_nstdb_summary.csv; do
  if [ -e "$f" ]; then mv "$f" "$BK/"; fi
done
$PY experiments/recompute_sqi_harmonized.py
$PY experiments/run_supervised_cohort_cv.py
$PY experiments/run_d1_neural_inference.py
$PY experiments/compute_consolidated_metrics.py
$PY experiments/run_cinc_transfer_benchmark.py
$PY experiments/exp4_revised_nstdb.py
$PY experiments/scaling_sensitivity_cinc.py
echo "REGENERATION DONE"
