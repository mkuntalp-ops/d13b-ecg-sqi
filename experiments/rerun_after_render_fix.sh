#!/usr/bin/env bash
# Recompute every result that used images of CinC 2011 or NSTDB after the 2026-10-13 renderer fix
# (experiments/rerender_images.py must have been run first). BUT QDB images and all SQI features were
# unchanged, so the BUT QDB cross-validation results are not recomputed.
# Run alone: the steps use MPS and the d1-3B weights.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BACKUP=results/_backup_pre_render_fix_2026-10-13
mkdir -p "$BACKUP"

# d1-3B CinC predictions were computed from the mis-rendered images: move them aside so they are recomputed
for f in raw_d1_neural_cinc2011_predictions.csv raw_d1_neural_cinc2011_responses.jsonl; do
  if [ -f "results/$f" ]; then mv "results/$f" "$BACKUP/$f"; fi
done

echo "== d1 CinC (BUT QDB outputs exist and are skipped)"; $PY -u experiments/run_d1_neural_inference.py
echo "== CinC transfer (Table 2, 2b)";                     $PY -u experiments/run_cinc_transfer_benchmark.py
echo "== CinC random subsets (Table 2d)";                  $PY -u experiments/cinc_random_draws.py
echo "== CinC subsets x seeds (Table 6)";                  $PY -u experiments/cinc_draws_by_seed.py
echo "== Seed stability, CinC rows (Table 5)";             $PY -u experiments/seed_stability.py --transfer-only
echo "== NSTDB (Tables 3, 3b)";                            $PY -u experiments/exp4_revised_nstdb.py
echo "== d1 prompt variants (Table 7)";                    $PY -u experiments/d1_prompt_variants.py --reuse-but
echo "rerun after render fix: done"
