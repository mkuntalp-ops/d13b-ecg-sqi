#!/usr/bin/env bash
# v2 analysis, part 5: repeat the latency session after the parameter-count fix in empirical_latency_benchmark.py
# (LR and MLP counts had been hard-coded for 13 inputs). Waits for part 4 so that latency runs alone. The first session's
# outputs are kept in latency_first_session/. Afterwards rebuilds the tables.
set -euo pipefail
cd "$(dirname "$0")/.."
export ECG_RESULTS_DIR=results/v2_2026-11
export ECG_FIGURES_DIR=$ECG_RESULTS_DIR/figures
PY=.venv/bin/python
R=$ECG_RESULTS_DIR
while pgrep -f "[v]2_queue_part4.sh" >/dev/null; do sleep 30; done
grep -q "^exit 0" "$R/logs/v2_queue_part4.log"
mkdir -p "$R/latency_first_session"
cp -R "$R/latency_repeats" "$R/latency_first_session/"
cp "$R"/latency_*_raw.csv "$R"/latency_*_summary.csv "$R"/empirical_latency_raw_measurements.csv "$R"/empirical_computational_profile_summary.csv "$R/latency_first_session/"
echo "== $(date '+%F %T') latency"; bash experiments/latency_all.sh
echo "== $(date '+%F %T') tables"; $PY -W ignore evaluation/build_manuscript_tables.py > "$R/logs/build_tables.log" 2>&1
echo "v2 queue part 5: done $(date '+%F %T')"
