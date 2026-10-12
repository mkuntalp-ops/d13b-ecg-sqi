#!/bin/bash
# Re-run every stage that trains CNN/MLP models, now that torch is seeded (RANDOM_SEED=42).
# SQI features and the d1 zero-shot outputs are unchanged by the seeding and are not recomputed.
# Previous outputs are moved to results/_backup_pre_fix_2026-10-12/previous_outputs_unseeded/ first.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BK=results/_backup_pre_fix_2026-10-12/previous_outputs_unseeded
mkdir -p "$BK"
for f in \
    results/raw_supervised_cohort1000_oof_predictions.csv \
    results/cohort1000_consolidated_oof_predictions.csv \
    results/cohort1000_consolidated_summary.csv \
    results/cohort1000_consolidated_operating_points.csv \
    results/cinc2011_expanded_predictions.csv \
    results/cinc2011_expanded_transfer_summary.csv \
    results/exp4_revised_nstdb_predictions.csv \
    results/exp4_revised_nstdb_summary.csv \
    results/cinc2011_scaling_sensitivity.csv; do
  if [ -e "$f" ]; then cp "$f" "$BK/"; fi
done
$PY -u experiments/run_supervised_cohort_cv.py
$PY -u experiments/compute_consolidated_metrics.py
$PY -u experiments/run_cinc_transfer_benchmark.py
$PY -u experiments/exp4_revised_nstdb.py
$PY -u experiments/scaling_sensitivity_cinc.py
$PY -u evaluation/generate_publication_figures.py
$PY -u evaluation/build_manuscript_tables.py
echo "SEEDED STAGE DONE"
