"""
Paired AUROC comparisons between models (Q1 revision, 2026-10-25).

Every pair of models is compared on the same evaluation units; each dataset is one family and the p-values are
Holm-adjusted within it (all pairwise comparisons, specified before the comparisons were run).

- BUT QDB (pooled out-of-fold, 1,047 windows in 15 subjects): windows of a subject are not independent, so the
  AUROC difference is tested with a subject-level cluster bootstrap (2,000 resamples; two-sided percentile p).
- CinC 2011 (lead II, the 565 records of the transfer analyses) and CinC 2017 (8,512 recordings, recording
  maximum; and the balanced 556-recording subset on which d1-3B was run): records are independent, so the
  DeLong test for correlated ROC curves is used (fast midrank algorithm).

Models: the eight models of the main analysis, TabPFN and the HuBERT-ECG probe (when their outputs exist) and the d1-3B
image-token probe (BUT QDB out-of-fold; CinC 2011 and CinC 2017 subset transfer predictions). When
results/d1_cinc2017_full_predictions.csv exists (analysis plan A2), zero-shot d1-3B and the d1-3B probe are also in the
CinC 2017 (8,512 recordings) family.

Output: results/paired_comparisons.csv
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "paired_comparisons.csv"
MODELS = ["Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG", "2D_CNN_Image",
          "Multimodal_Fusion", "Rule_Based_SQI_Heuristic", "LiquidAI_d1_3B_Neural"]
SEED = 42


# ---------- DeLong (Sun & Xu 2014 midrank formulation) ----------
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_paired(y: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Two-sided DeLong test of AUROC(a) - AUROC(b) on the same labels. Returns (delta, ci_lo, ci_hi, p)."""
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    aucs, v10, v01 = [], [], []
    for s in (np.asarray(a, float), np.asarray(b, float)):
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(np.concatenate([s[pos], s[neg]]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n)
        v01.append(1.0 - (tz[m:] - ty) / m)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    d = aucs[0] - aucs[1]
    if var <= 0:
        return d, d, d, 1.0
    se = np.sqrt(var)
    return d, d - 1.96 * se, d + 1.96 * se, float(2 * norm.sf(abs(d) / se))


def cluster_bootstrap_paired(y, a, b, cluster, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    ug = np.unique(cluster)
    idx = {c: np.where(cluster == c)[0] for c in ug}
    d = []
    for _ in range(B):
        i = np.concatenate([idx[c] for c in rng.choice(ug, len(ug))])
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return roc_auc_score(y, a) - roc_auc_score(y, b), lo, hi, float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------- score tables per family ----------
def families() -> dict:
    fam = {}
    oof = pd.read_csv(RESULTS / "cohort1000_consolidated_oof_predictions.csv")
    but = pd.DataFrame({"unit": oof.segment_id, "y": oof.label, "cluster": oof.subject_id})
    for m in MODELS:
        but[m] = oof[f"prob_{m}"].values
    probe = pd.read_csv(RESULTS / "d1_probe_oof_predictions.csv")
    probe = probe[probe.features == "image"].set_index("segment_id")["prob_unusable"]
    but["d1_image_probe"] = probe.loc[but.unit].values
    tab = pd.read_csv(RESULTS / "tabpfn_predictions.csv") if (RESULTS / "tabpfn_predictions.csv").exists() else None
    if tab is not None and (tab.setting == "butqdb_cv").any():
        t = tab[tab.setting == "butqdb_cv"].set_index("segment_id")["prob_unusable"]
        but["TabPFN_SQI"] = t.loc[but.unit].values
    hub_path = RESULTS / "hubert_probe_predictions.csv"
    hub = pd.read_csv(hub_path, low_memory=False) if hub_path.exists() else None
    if hub is not None:
        h = hub[hub.setting == "butqdb_cv"].set_index("segment_id")["prob_unusable"]
        but["HuBERT_ECG_probe"] = h.loc[but.unit].values
    fam["BUT_QDB (5-fold subject CV)"] = ("cluster_bootstrap", but)

    c11 = pd.read_csv(RESULTS / "cinc_random_draws_predictions.csv").drop_duplicates("record_name")
    lab = pd.read_csv(ROOT / "cache/datasets/cinc2011/cinc2011_random_draws_manifest.csv").drop_duplicates("record_name").set_index("record_name")["label"]
    cc = pd.DataFrame({"unit": c11.record_name.astype(str), "y": lab.loc[c11.record_name].values})
    for m in MODELS:
        cc[m] = c11[m].values
    tp = pd.read_csv(RESULTS / "d1_probe_transfer_predictions.csv")
    tp11 = tp[(tp.features == "image") & (tp.dataset == "CinC2011")].set_index("record_name")["prob_unusable"]
    tp11.index = tp11.index.astype(str)
    cc["d1_image_probe"] = tp11.loc[cc.unit].values
    if tab is not None:
        t = tab[(tab.setting == "but_only_external") & (tab.dataset == "CinC2011")].set_index("segment_id")["prob_unusable"]
        cc["TabPFN_SQI"] = t.loc[c11.segment_id].values
    if hub is not None:
        h = hub[(hub.setting == "but_only_external") & (hub.dataset == "CinC2011")].set_index("segment_id")["prob_unusable"]
        cc["HuBERT_ECG_probe"] = h.loc[c11.segment_id].values
    fam["CinC2011 lead II (565 records)"] = ("delong", cc)

    w = pd.read_csv(RESULTS / "external_window_predictions.csv", low_memory=False)
    w = w[w.dataset == "CinC2017"].copy()
    if tab is not None:
        t = tab[(tab.setting == "but_only_external") & (tab.dataset == "CinC2017")].set_index("segment_id")["prob_unusable"]
        w["TabPFN_SQI"] = t.loc[w.segment_id].values
    if hub is not None:
        h = hub[(hub.setting == "but_only_external") & (hub.dataset == "CinC2017")].set_index("segment_id")["prob_unusable"]
        w["HuBERT_ECG_probe"] = h.loc[w.segment_id].values
    full17 = RESULTS / "d1_cinc2017_full_predictions.csv"  # analysis plan A2: d1-3B on all CinC 2017 windows
    if full17.exists():
        f = pd.read_csv(full17).set_index("segment_id")
        w["LiquidAI_d1_3B_Neural"] = f.loc[w.segment_id, "prob_unusable"].values
        w["d1_image_probe"] = f.loc[w.segment_id, "prob_image_probe"].values
    cols = [m for m in MODELS if m in w.columns] + [c for c in ["TabPFN_SQI", "HuBERT_ECG_probe", "d1_image_probe"]
                                                    if c in w.columns]
    r = w.groupby("record_name").agg({**{m: "max" for m in cols}, "label": "first"}).reset_index()
    r = r.rename(columns={"label": "y", "record_name": "unit"})
    fam["CinC2017 (8,512 recordings)"] = ("delong", r)

    d1 = pd.read_csv(RESULTS / "d1_external_predictions.csv")
    d1 = d1[d1.dataset == "CinC2017_subset"].groupby("record_name")["prob_unusable"].max()
    sub = r[r.unit.isin(d1.index)].copy()
    if "LiquidAI_d1_3B_Neural" in sub.columns:  # with A2 the full-set scores equal the subset run on these recordings
        assert np.allclose(sub["LiquidAI_d1_3B_Neural"].values, d1.loc[sub.unit].values)
    sub["LiquidAI_d1_3B_Neural"] = d1.loc[sub.unit].values
    tp17 = tp[(tp.features == "image") & (tp.dataset == "CinC2017")].set_index("record_name")["prob_unusable"]
    sub["d1_image_probe"] = tp17.loc[sub.unit].values
    fam["CinC2017 balanced subset (556 recordings)"] = ("delong", sub)
    return fam


def main():
    rows = []
    for fam_name, (method, df) in families().items():
        models = [c for c in df.columns if c not in ("unit", "y", "cluster")]
        y = df["y"].values.astype(int)
        res = []
        for a, b in combinations(models, 2):
            if method == "delong":
                d, lo, hi, p = delong_paired(y, df[a].values, df[b].values)
            else:
                d, lo, hi, p = cluster_bootstrap_paired(y, df[a].values, df[b].values, df["cluster"].values)
            res.append({"family": fam_name, "method": method, "n_units": len(df), "model_a": a, "model_b": b,
                        "auroc_a": roc_auc_score(y, df[a]), "auroc_b": roc_auc_score(y, df[b]), "delta": d,
                        "ci_lo": lo, "ci_hi": hi, "p": p})
        r = pd.DataFrame(res)
        r["p_holm"] = holm(r["p"].values)
        rows.append(r)
        print(f"[{fam_name}] {len(r)} comparisons, {int((r.p_holm < 0.05).sum())} significant after Holm", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    if "--check" in sys.argv:  # DeLong vs record bootstrap on one pair (sanity check)
        f = families()["CinC2017 (8,512 recordings)"][1]
        y = f.y.values.astype(int)
        print("DeLong:", delong_paired(y, f["Rule_Based_SQI_Heuristic"].values, f["2D_CNN_Image"].values))
        print("Bootstrap:", cluster_bootstrap_paired(y, f["Rule_Based_SQI_Heuristic"].values, f["2D_CNN_Image"].values,
                                                    f.unit.values, B=1000))
    else:
        main()
