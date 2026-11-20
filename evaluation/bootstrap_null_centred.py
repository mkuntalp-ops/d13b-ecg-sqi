"""
Null-centred sensitivity for the bootstrap p-values (round V23, 2026-11-20; review finding PAPER-004). Post hoc;
saved predictions only, no model is refitted, and no stored output is overwritten.

The paired AUROC comparisons on BUT QDB (subject-level cluster bootstrap, evaluation/paired_comparisons.py) and the
leave-one-dataset-out comparisons of Table 3 (experiments/lodo_multisource.py; subjects for BUT QDB, recordings for
CinC 2011 and CinC 2017) report a percentile-bootstrap p-value, p = 2 x min(share of resampled differences <= 0,
share >= 0). That value inverts the percentile interval and is not computed under a null hypothesis. This script
re-runs exactly the same resamples (same generator, seed and pair order; every stored p must be reproduced exactly)
and adds a null-centred p-value: the resampled differences are shifted by the observed difference, d* - d, which
approximates the distribution of the estimate around the true value, and
    p_centred = share of resamples with |d* - d| >= |d|      (floor 1/n, n = valid resamples),
Holm-adjusted within the same family as the primary values (all 55 BUT QDB pairs; the nine comparisons of each
held-out dataset in Table 3).

It also applies to the Table 3 p-values the 1/n floor that the BUT QDB p-values have had since the code audit of
2026-11-11 (CODE-003): experiments/lodo_multisource.py returns p = 0 when no resample crosses zero, and Table 3 printed
these as "p_Holm < 0.001", a resolution that 2,000 resamples do not have. The floored values are Holm-adjusted within
each held-out dataset as before (p_floor_holm; used by Table 3 from round V23 on).

Families: BUT QDB primary cohort (988 windows), BUT QDB quota sensitivity cohort (1,041 windows,
<results>_quota_sensitivity), and the 27 comparisons of Table 3 (primary cohort).
Output: <results>/bootstrap_null_centred.csv
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/bootstrap_null_centred.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "experiments"))
import paired_comparisons as pc  # noqa: E402

PRIMARY = pc.RESULTS
SENS = PRIMARY.parent / f"{PRIMARY.name}_quota_sensitivity"
OUT = PRIMARY / "bootstrap_null_centred.csv"
FAMILY = "BUT_QDB (5-fold subject CV)"


def resampled_differences(y, a, b, cluster, B=2000, seed=pc.SEED):
    """The resampling of pc.cluster_bootstrap_paired and lodo_multisource.paired_boot (identical loops)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(cluster)
    idx = {c: np.where(cluster == c)[0] for c in ug}
    d = []
    for _ in range(B):
        i = np.concatenate([idx[c] for c in rng.choice(ug, len(ug))])
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.array(d)


def summarise(d: np.ndarray, delta: float) -> dict:
    n = len(d)
    n_le, n_ge = int((d <= 0).sum()), int((d >= 0).sum())
    n_c = int((np.abs(d - delta) >= abs(delta)).sum())
    return {"n_valid": n, "n_le0": n_le, "n_ge0": n_ge, "p_percentile_raw": min(1.0, 2 * min(n_le, n_ge) / n),
            "p_percentile_floor": min(1.0, max(2 * min(n_le, n_ge) / n, 1.0 / n)), "n_centred": n_c,
            "p_centred": min(1.0, max(n_c / n, 1.0 / n))}


def butqdb(results_dir: Path, cohort: str) -> pd.DataFrame:
    pc.RESULTS = results_dir  # families() reads the module-level results directory
    _, df = pc.families()[FAMILY]
    stored = pd.read_csv(results_dir / "paired_comparisons.csv")
    stored = stored[stored.family == FAMILY].reset_index(drop=True)
    if len(stored) != 55:
        raise ValueError(f"{cohort}: expected 55 BUT QDB pairs, got {len(stored)}")
    y, cl = df["y"].values.astype(int), df["cluster"].values
    rows = []
    for _, s in stored.iterrows():
        a, b = df[s.model_a].values, df[s.model_b].values
        delta = roc_auc_score(y, a) - roc_auc_score(y, b)
        r = summarise(resampled_differences(y, a, b, cl), delta)
        if r["p_percentile_floor"] != s.p or abs(delta - s.delta) > 1e-12:
            raise ValueError(f"{cohort} {s.model_a} vs {s.model_b}: recomputed p {r['p_percentile_floor']} differs from stored {s.p}")
        rows.append({"analysis": f"BUT QDB, {cohort}", "family": "all 55 pairs", "model_a": s.model_a, "model_b": s.model_b,
                     "delta_a_minus_b": delta, **r, "p_percentile_holm": s.p_holm})
    out = pd.DataFrame(rows)
    if not np.allclose(pc.holm(out.p_percentile_floor.values), out.p_percentile_holm.values, rtol=0, atol=1e-12):
        raise ValueError(f"{cohort}: Holm adjustment does not reproduce the stored p_holm")
    out["p_centred_holm"] = pc.holm(out.p_centred.values)
    return out


def lodo() -> pd.DataFrame:
    pc.RESULTS = PRIMARY
    import lodo_multisource as lm
    recorded = []

    def paired_boot(y, a, b, cluster, B=2000, seed=lm.SEED):
        delta = roc_auc_score(y, a) - roc_auc_score(y, b)
        r = summarise(resampled_differences(y, a, b, cluster, B, seed), delta)
        recorded.append({"delta_a_minus_b": delta, **r})
        return delta, np.nan, np.nan, r["p_percentile_raw"]  # the value lodo_multisource.paired_boot returns

    lm.paired_boot = paired_boot
    lm.boot_ci = lambda y, p, cluster, B=2000, seed=lm.SEED: (np.nan, np.nan)  # intervals are not needed here
    with tempfile.TemporaryDirectory() as tmp:
        lm.SUM_OUT = Path(tmp) / "lodo_summary_rerun.csv"  # the stored lodo_summary.csv is not touched
        lm.analyse()
    stored = pd.read_csv(PRIMARY / "lodo_summary.csv")
    stored = stored[stored.p_boot.notna()].reset_index(drop=True)
    rec = pd.DataFrame(recorded)
    if len(rec) != len(stored) or len(stored) != 27:
        raise ValueError(f"expected 27 LODO comparisons, got {len(rec)} re-run and {len(stored)} stored")
    if not (np.allclose(rec.p_percentile_raw, stored.p_boot, rtol=0, atol=1e-12)
            and np.allclose(rec.delta_a_minus_b, stored.delta, rtol=0, atol=1e-12)):
        raise ValueError("re-run LODO p-values or differences do not reproduce lodo_summary.csv")
    out = pd.concat([stored[["test", "model"]].rename(columns={"test": "family", "model": "model_a"}), rec], axis=1)
    out.insert(0, "analysis", "Table 3 (LODO)")
    out["model_b"] = out.model_a + " (BUT QDB only)"
    for col in ["p_percentile_holm", "p_floor_holm", "p_centred_holm"]:
        out[col] = np.nan
    for _, g in out.groupby("family"):
        out.loc[g.index, "p_percentile_holm"] = pc.holm(g.p_percentile_raw.values)  # as Table 3 up to round V22
        out.loc[g.index, "p_floor_holm"] = pc.holm(g.p_percentile_floor.values)
        out.loc[g.index, "p_centred_holm"] = pc.holm(g.p_centred.values)
    return out


def main():
    out = pd.concat([butqdb(PRIMARY, "primary cohort (988 windows)"), butqdb(SENS, "sensitivity cohort (1,041 windows)"),
                     lodo()], ignore_index=True)
    pc.RESULTS = PRIMARY
    out.to_csv(OUT, index=False)
    for name, g in out.groupby("analysis", sort=False):
        prim = g.p_floor_holm if name.startswith("Table 3") else g.p_percentile_holm
        print(f"{name}: significant after Holm, percentile {int((prim < 0.05).sum())}, null-centred "
              f"{int((g.p_centred_holm < 0.05).sum())} of {len(g)}")
    best = out[out.analysis.str.startswith("BUT QDB") & ((out.model_a == "2D_CNN_Image") | (out.model_b == "2D_CNN_Image"))]
    print(best[["analysis", "model_a", "model_b", "delta_a_minus_b", "p_percentile_floor", "p_percentile_holm", "p_centred",
                "p_centred_holm"]].round(4).to_string(index=False))
    print(out[out.analysis.str.startswith("Table 3")][["family", "model_a", "delta_a_minus_b", "p_percentile_raw",
                                                        "p_floor_holm", "p_centred", "p_centred_holm"]].round(4).to_string(index=False))
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
