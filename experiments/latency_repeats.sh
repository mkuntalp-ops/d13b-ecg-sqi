#!/usr/bin/env bash
# Three sequential runs of the latency benchmark (Table 4). Run this alone: no other MPS job may run at the same time.
# Each run's outputs are copied to results/latency_repeats/ before the next run overwrites them.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/latency_repeats
for k in 1 2 3; do
  .venv/bin/python -u experiments/empirical_latency_benchmark.py > "results/latency_repeats/run${k}.log" 2>&1
  cp results/empirical_latency_raw_measurements.csv "results/latency_repeats/run${k}_raw.csv"
  cp results/empirical_computational_profile_summary.csv "results/latency_repeats/run${k}_summary.csv"
  echo "run ${k} done"
done
echo "latency repeats done"
