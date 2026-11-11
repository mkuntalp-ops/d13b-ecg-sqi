#!/usr/bin/env bash
# v2 analysis, part 3 (handoff items 2 and 3): waits for experiments/d1_twelve_lead_full.py, then the 12-lead models, the raster
# 2D CNN and, last and alone, the latency measurements. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
export ECG_RESULTS_DIR=results/v2_2026-11
export ECG_FIGURES_DIR=$ECG_RESULTS_DIR/figures
PY=.venv/bin/python
LOG=$ECG_RESULTS_DIR/logs
while pgrep -f "[d]1_twelve_lead_full.py" >/dev/null; do sleep 60; done
test "$(wc -l < "$ECG_RESULTS_DIR/d1_twelve_lead_full_predictions.csv")" -eq 11977
echo "== $(date '+%F %T') twelve_lead_models"; $PY -u experiments/external_validation.py --twelve-lead > "$LOG/twelve_lead_models.log" 2>&1
echo "== $(date '+%F %T') pil_raster";          $PY -u experiments/pil_raster_2dcnn.py > "$LOG/pil_raster.log" 2>&1
echo "== $(date '+%F %T') latency";              bash experiments/latency_all.sh
echo "v2 queue part 3: done $(date '+%F %T')"
