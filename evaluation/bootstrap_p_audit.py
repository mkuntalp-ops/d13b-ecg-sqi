"""
Audit of the BUT QDB paired cluster-bootstrap p-values (round V10, 2026-11-18; review finding PAPER-001).

Reads only saved out-of-fold predictions (the BUT QDB family of evaluation/paired_comparisons.py) and refits nothing.
For each comparison of a model with the best in-domain model it re-runs the same bootstrap (same function logic, same
seed and the same pair order as paired_comparisons.py) and records what the stored p-value is made of:
  - n_valid: resamples kept (resamples whose drawn subjects contain only one class are discarded, because AUROC is
    undefined there);
  - n_le0 / n_ge0: resamples with AUROC(model) - AUROC(best) <= 0 and >= 0;
  - p = max(2 * min(n_le0, n_ge0) / n_valid, 1 / n_valid), capped at 1. This is a percentile-bootstrap p-value: it
    inverts the percentile interval of the observed difference and does not resample under a null hypothesis.
The recomputed p must equal the stored p in paired_comparisons.csv exactly, otherwise the script stops. The Holm values
are copied unrounded from paired_vs_best.csv (Holm over all 55 pairs, and over the ten comparisons with the best).

Round V11 (2026-11-18; review findings PAPER-001 / PAPER-009) extends the audit to the complete family of 55 pairs:
every pair is re-run in the stored order, its p must equal the stored p exactly, and the Holm adjustment recomputed
from the 55 p-values must equal the stored p_holm. Alongside the 1/n floor used in the manuscript, the file records a
convention-sensitivity p-value that adds one to the smaller tail count and to the number of valid resamples,
p_plus1 = min(1, 2 * (min(n_le0, n_ge0) + 1) / (n_valid + 1)), with its own Holm adjustment over the 55 pairs. It is
arithmetic on the same resamples, not a new analysis, and shows which decisions depend on the reporting convention.

Output: <results>/butqdb_bootstrap_p_audit.csv (comparisons with the best model)
        <results>/butqdb_bootstrap_p_audit_all_pairs.csv (all 55 pairs)
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/bootstrap_p_audit.py
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
OUT = pc.RESULTS / "butqdb_bootstrap_p_audit.csv"
OUT_ALL = pc.RESULTS / "butqdb_bootstrap_p_audit_all_pairs.csv"
N_PAIRS = 55


def bootstrap_counts(y, a, b, cluster, B=2000, seed=pc.SEED):
    """Same resampling as pc.cluster_bootstrap_paired, returning the counts behind its p-value."""
    rng = np.random.default_rng(seed)
    ug = np.unique(cluster)
    idx = {c: np.where(cluster == c)[0] for c in ug}
    d = []
    for _ in range(B):
        i = np.concatenate([idx[c] for c in rng.choice(ug, len(ug))])
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    n_le, n_ge = int((d <= 0).sum()), int((d >= 0).sum())
    p = min(1.0, max(2 * min(n_le, n_ge) / len(d), 1.0 / len(d)))
    return len(d), n_le, n_ge, p


def p_plus1(n_valid, n_le, n_ge):
    """Convention-sensitivity p-value: one added to the smaller tail count and to the number of valid resamples."""
    return min(1.0, 2 * (min(n_le, n_ge) + 1) / (n_valid + 1))


def all_pairs(df, stored):
    """Re-run every stored pair of the BUT QDB family; check p and the Holm adjustment against the stored values."""
    if len(stored) != N_PAIRS:
        raise ValueError(f"expected {N_PAIRS} stored BUT QDB pairs, got {len(stored)}")
    y = df["y"].values.astype(int)
    rows = []
    for _, s in stored.iterrows():
        n, n_le, n_ge, p = bootstrap_counts(y, df[s.model_a].values, df[s.model_b].values, df["cluster"].values)
        if p != s.p:
            raise ValueError(f"{s.model_a} vs {s.model_b}: recomputed p {p} differs from stored {s.p}")
        rows.append({"model_a": s.model_a, "model_b": s.model_b, "auroc_a": s.auroc_a, "auroc_b": s.auroc_b,
                     "delta_a_minus_b": s.delta, "n_resamples": 2000, "n_valid": n, "n_delta_le0": n_le,
                     "n_delta_ge0": n_ge, "p_unadjusted": p, "p_plus1": p_plus1(n, n_le, n_ge)})
    out = pd.DataFrame(rows)
    out["p_holm"] = pc.holm(out.p_unadjusted.values)
    if not np.allclose(out.p_holm.values, stored.p_holm.values, rtol=0, atol=1e-12):
        raise ValueError("recomputed Holm adjustment differs from the stored p_holm")
    out["p_plus1_holm"] = pc.holm(out.p_plus1.values)
    return out


def main():
    method, df = pc.families()[FAMILY]
    stored_all = pd.read_csv(pc.RESULTS / "paired_comparisons.csv")
    full = all_pairs(df, stored_all[stored_all.family == FAMILY].reset_index(drop=True))
    full.to_csv(OUT_ALL, index=False)
    print(full.to_string(index=False))
    print(f"significant after Holm: 1/n floor {int((full.p_holm < 0.05).sum())}, "
          f"plus-one convention {int((full.p_plus1_holm < 0.05).sum())}")
    print(f"[Saved] {OUT_ALL}")
    stored = pd.read_csv(pc.RESULTS / "paired_comparisons.csv")
    stored = stored[stored.family == FAMILY]
    vs = pd.read_csv(pc.RESULTS / "paired_vs_best.csv")
    vs = vs[vs.family == FAMILY]
    best = vs.best.iloc[0]
    y = df["y"].values.astype(int)
    rows = []
    for _, v in vs.iterrows():
        m = v.model
        hit = stored[((stored.model_a == m) & (stored.model_b == best)) | ((stored.model_a == best) & (stored.model_b == m))]
        if len(hit) != 1:
            raise ValueError(f"expected one stored comparison of {m} with {best}, got {len(hit)}")
        a, b = hit.model_a.iloc[0], hit.model_b.iloc[0]  # the order used in paired_comparisons.py
        n, n_le, n_ge, p = bootstrap_counts(y, df[a].values, df[b].values, df["cluster"].values)
        if p != hit.p.iloc[0] or p != v.p_unadjusted:
            raise ValueError(f"{m}: recomputed p {p} differs from stored {hit.p.iloc[0]} / {v.p_unadjusted}")
        sign = 1 if a == m else -1  # report counts for AUROC(model) - AUROC(best)
        rows.append({"model": m, "best": best, "auroc": v.auroc, "auroc_best": v.auroc_best,
                     "delta_model_minus_best": v.auroc - v.auroc_best, "n_resamples": 2000, "n_valid": n,
                     "n_delta_le0": n_le if sign == 1 else n_ge, "n_delta_ge0": n_ge if sign == 1 else n_le,
                     "p_unadjusted": p, "p_holm_all_pairs": v.p_holm_all_pairs, "p_holm_vs_best": v.p_holm_vs_best})
    out = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
