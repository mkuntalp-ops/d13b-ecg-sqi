#!/usr/bin/env bash
# v2 analysis, part 4: analyses still on v1 inputs that feed manuscript tables. Waits for part 3 so that latency runs alone.
# Stops at the first failure. Afterwards rebuilds the tables (evaluation/build_manuscript_tables.py).
set -euo pipefail
cd "$(dirname "$0")/.."
export ECG_RESULTS_DIR=results/v2_2026-11
export ECG_FIGURES_DIR=$ECG_RESULTS_DIR/figures
PY=.venv/bin/python
LOG=$ECG_RESULTS_DIR/logs
while pgrep -f "[v]2_queue_part3.sh" >/dev/null; do sleep 60; done
grep -q "^exit 0" "$LOG/v2_queue_part3.log"
echo "== $(date '+%F %T') scaling_sensitivity"; $PY -u experiments/scaling_sensitivity_cinc.py > "$LOG/scaling_sensitivity.log" 2>&1
echo "== $(date '+%F %T') hubert_finetune";     $PY -u experiments/hubert_ecg_finetune.py > "$LOG/hubert_finetune.log" 2>&1
echo "== $(date '+%F %T') tables";              $PY -W ignore evaluation/build_manuscript_tables.py > "$LOG/build_tables.log" 2>&1
echo "v2 queue part 4: done $(date '+%F %T')"
