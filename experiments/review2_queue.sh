#!/usr/bin/env bash
# Second-review analyses (2026-10-13), in dependency order. Waits for the CinC 2017 extraction (CPU) and the
# post-renderer-fix rerun (MPS) when they are still running, so that MPS jobs never overlap and the latency
# measurements run alone.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

wait_for() { while pgrep -f "$1" >/dev/null; do sleep 30; done; }

wait_for "[c]inc2017_extraction.py"
echo "== CinC 2011 twelve-lead extraction";   $PY -u experiments/cinc2011_twelve_lead_extraction.py
wait_for "[r]erun_after_render_fix.sh"
echo "== d1-3B probe features and external d1 predictions"; $PY -u experiments/d1_linear_probe.py
echo "== External validation (CinC 2017, CinC 2011 12-lead)"; $PY -u experiments/external_validation.py
echo "== Nested and source-selected thresholds";            $PY -u experiments/nested_thresholds.py
echo "== Extended latency (alone)";                         $PY -u experiments/latency_extended.py
echo "review2 queue: done"
