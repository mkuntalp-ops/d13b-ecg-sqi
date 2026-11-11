#!/usr/bin/env bash
# Single entry point for every latency measurement of the manuscript (plan item 3.5, v2). Run ALONE: no other MPS or
# CPU-heavy job may run at the same time. Fixed order, one session:
#   three runs of the Table 4 benchmark -> extended profiling -> foundation-model probes -> TabPFN (own environment).
# Warm-up: the first window of each run is excluded by each script.
# Outputs go to $ECG_RESULTS_DIR (default results/v2_2026-11); each benchmark run is copied to latency_repeats/.
set -euo pipefail
cd "$(dirname "$0")/.."
export ECG_RESULTS_DIR=${ECG_RESULTS_DIR:-results/v2_2026-11}
PY=.venv/bin/python
TABPFN_PY=${TABPFN_PY:-/Volumes/KIOXIA/AI/tabpfn_env/bin/python}
R=$ECG_RESULTS_DIR; mkdir -p "$R/latency_repeats" "$R/logs"
if pgrep -f "[e]xperiments/(d1_|run_supervised|pil_raster|external_validation|lodo)" >/dev/null; then
  echo "another experiment is running; latency must run alone" >&2; exit 1
fi
for k in 1 2 3; do
  $PY -u experiments/empirical_latency_benchmark.py > "$R/latency_repeats/run${k}.log" 2>&1
  cp "$R/empirical_latency_raw_measurements.csv" "$R/latency_repeats/run${k}_raw.csv"
  cp "$R/empirical_computational_profile_summary.csv" "$R/latency_repeats/run${k}_summary.csv"
  echo "benchmark run $k done $(date '+%F %T')"
done
$PY -u experiments/latency_extended.py > "$R/logs/latency_extended.log" 2>&1
$PY -u experiments/latency_foundation_probes.py > "$R/logs/latency_foundation_probes.log" 2>&1
$TABPFN_PY -u experiments/latency_tabpfn.py > "$R/logs/latency_tabpfn.log" 2>&1
echo "latency done $(date '+%F %T')"
