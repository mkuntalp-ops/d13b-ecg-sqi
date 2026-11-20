"""
BUT QDB paired cluster bootstrap with 20,000 resamples (round V19, 2026-11-19; review finding PAPER-005 of the V17
review). Sensitivity analysis only: the primary p-values (2,000 resamples, 1/n floor) are unchanged.

Reads saved out-of-fold predictions (the BUT QDB family of evaluation/paired_comparisons.py) and refits nothing. For
each of the 55 stored pairs, in the stored order, it repeats the resampling of evaluation/bootstrap_p_audit.py
(subject-level, same seed) with B = 20,000 and records
  - n_valid, n_delta_le0, n_delta_ge0 (resamples whose subjects contain one class only are discarded);
  - p_plus1 = min(1, 2 * (min(n_le0, n_ge0) + 1) / (n_valid + 1)), which has no 1/n floor, with Holm over the 55 pairs;
  - p_floor = max(2 * min(n_le0, n_ge0) / n_valid, 1 / n_valid), capped at 1, with Holm over the 55 pairs, for
    comparison with the primary convention.
The first 2,000 resamples are the same draws as the primary analysis (same generator and seed); the script checks
that their counts reproduce the stored primary p-values exactly.

Output: <results>/butqdb_bootstrap_b20000_sensitivity.csv
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/bootstrap_b20000_sensitivity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
import paired_comparisons as pc  # noqa: E402

FAMILY = "BUT_QDB (5-fold subject CV)"
OUT = pc.RESULTS / "butqdb_bootstrap_b20000_sensitivity.csv"
B, B0, N_PAIRS = 20000, 2000, 55


def deltas(y, a, b, cluster, n=B, seed=pc.SEED):
    """Differences AUROC(a) - AUROC(b) per draw; NaN where the drawn subjects contain one class only."""
    rng = np.random.default_rng(seed)
    ug = np.unique(cluster)
    idx = {c: np.where(cluster == c)[0] for c in ug}
    d = np.full(n, np.nan)
    for k in range(n):
        i = np.concatenate([idx[c] for c in rng.choice(ug, len(ug))])
        if y[i].min() != y[i].max():
            d[k] = roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i])
    return d


def counts(d):
    d = d[~np.isnan(d)]
    return len(d), int((d <= 0).sum()), int((d >= 0).sum())


def main():
    _, df = pc.families()[FAMILY]
    stored = pd.read_csv(pc.RESULTS / "paired_comparisons.csv")
    stored = stored[stored.family == FAMILY].reset_index(drop=True)
    if len(stored) != N_PAIRS:
        raise ValueError(f"expected {N_PAIRS} stored pairs, got {len(stored)}")
    y = df["y"].values.astype(int)
    rows = []
    for _, s in stored.iterrows():
        d = deltas(y, df[s.model_a].values, df[s.model_b].values, df["cluster"].values)
        n0, le0, ge0 = counts(d[:B0])
        p0 = min(1.0, max(2 * min(le0, ge0) / n0, 1.0 / n0))
        if p0 != s.p:
            raise ValueError(f"{s.model_a} vs {s.model_b}: first {B0} draws give p {p0}, stored {s.p}")
        n, le, ge = counts(d)
        rows.append({"model_a": s.model_a, "model_b": s.model_b, "delta_a_minus_b": s.delta, "n_resamples": B,
                     "n_valid": n, "n_delta_le0": le, "n_delta_ge0": ge,
                     "p_plus1": min(1.0, 2 * (min(le, ge) + 1) / (n + 1)),
                     "p_floor": min(1.0, max(2 * min(le, ge) / n, 1.0 / n)), "p_primary_2000": s.p,
                     "p_holm_primary_2000": s.p_holm})
        print(f"{s.model_a} vs {s.model_b}: n_valid {n}, min tail {min(le, ge)}", flush=True)
    out = pd.DataFrame(rows)
    out["p_plus1_holm"] = pc.holm(out.p_plus1.values)
    out["p_floor_holm"] = pc.holm(out.p_floor.values)
    out.to_csv(OUT, index=False)
    print(f"significant after Holm (55 pairs): primary {int((out.p_holm_primary_2000 < 0.05).sum())}, "
          f"B={B} plus-one {int((out.p_plus1_holm < 0.05).sum())}, B={B} floor {int((out.p_floor_holm < 0.05).sum())}")
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
