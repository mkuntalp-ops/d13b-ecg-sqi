"""
External validation added after the second review (2026-10-13).

1. CinC 2017 (single lead, 8,528 recordings, natural prevalence of '~' noisy recordings). Every 10-s window is
   scored by the models trained on the full BUT QDB cohort (seed 42, same settings as the CinC 2011 transfer);
   recording-level score = maximum (primary) or mean of the window scores. Metrics at the natural prevalence:
   AUROC and AP with recording-level bootstrap CIs, and sensitivity, false reject, PPV and NPV at tau = 0.5.
   d1-3B is evaluated on a balanced subset (all noisy recordings plus as many random non-noisy recordings),
   where the other models are reported on the same subset.
2. CinC 2011, all 12 leads (565 records of the transfer analyses). Recording score = lead II, maximum over the
   12 leads, or mean over the 12 leads; AUROC on the reported first-N subset and the five random subsets.
   d1-3B on the 12 leads of the first-N subset.

d1-3B probabilities come from experiments/d1_linear_probe.py (results/d1_external_predictions.csv).

Usage: python experiments/external_validation.py                 (score windows, then analyse)
       python experiments/external_validation.py --analysis-only
Outputs: results/external_window_predictions.csv, results/external_cinc2017_summary.csv,
         results/cinc2011_twelve_lead_summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from cinc_random_draws import fit_models
from features.sqi_extractor import SQI_FEATURE_NAMES
from models.rule_heuristic_baseline import RuleBasedSQIHeuristic

RESULTS = PROJECT_ROOT / "results"
CINC11 = PROJECT_ROOT / "cache" / "datasets" / "cinc2011"
CINC17 = PROJECT_ROOT / "cache" / "datasets" / "cinc2017" / "cinc2017_windows.csv"
TWELVE = CINC11 / "twelve_lead" / "cinc2011_twelve_lead_windows.csv"
WIN_OUT = RESULTS / "external_window_predictions.csv"
C17_OUT = RESULTS / "external_cinc2017_summary.csv"
C12_OUT = RESULTS / "cinc2011_twelve_lead_summary.csv"
D1_PRED = RESULTS / "d1_external_predictions.csv"
MODELS = ["Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG", "2D_CNN_Image",
          "Multimodal_Fusion", "Rule_Based_SQI_Heuristic"]
D1 = "LiquidAI_d1_3B_Neural"
N_BOOT = 1000


def score_windows():
    lr, lgb, mlp, c1, c2, mm = fit_models()
    rule = RuleBasedSQIHeuristic()
    parts = []
    for name, path in [("CinC2017", CINC17), ("CinC2011_12lead", TWELVE)]:
        df = pd.read_csv(path)
        X = np.nan_to_num(df[SQI_FEATURE_NAMES].values.astype(float))
        out = df[["segment_id", "record_name", "label"]].copy()
        out["dataset"] = name
        out["unit_idx"] = df["window_idx"] if "window_idx" in df else df["lead_idx"]
        out["Logistic_Regression_SQI"] = lr.predict_proba(X)
        out["LightGBM_SQI"] = lgb.predict_proba(X)
        out["Independent_SQI_MLP"] = mlp.predict_proba(X)
        out["1D_CNN_RawECG"] = c1.predict_proba([np.load(p) for p in df["signal_path"]])
        out["2D_CNN_Image"] = c2.predict_proba(df["image_path"].tolist())
        out["Multimodal_Fusion"] = mm.predict_proba(df["image_path"].tolist(), X)
        rp = []
        for (_, r), x in zip(df.iterrows(), X):
            with Image.open(r["image_path"]) as img:
                rp.append(rule.decide_single(img, dict(zip(SQI_FEATURE_NAMES, x)), modality="multimodal")["prob_unusable"])
        out["Rule_Based_SQI_Heuristic"] = rp
        parts.append(out)
        print(f"[{name}] scored {len(out)} windows", flush=True)
        pd.concat(parts).to_csv(WIN_OUT, index=False)


def boot_ci(y, p, metric, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(metric(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def rec_metrics(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(float)
    pred = p >= 0.5
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    a_lo, a_hi = boot_ci(y, p, roc_auc_score)
    ap_lo, ap_hi = boot_ci(y, p, average_precision_score)
    return {"n": len(y), "n_pos": int(y.sum()), "prevalence": y.mean(), "auroc": roc_auc_score(y, p), "auroc_ci_low": a_lo,
            "auroc_ci_high": a_hi, "ap": average_precision_score(y, p), "ap_ci_low": ap_lo, "ap_ci_high": ap_hi,
            "sens_at_0.5": tp / max(1, tp + fn), "fr_at_0.5": fp / max(1, fp + tn),
            "ppv_at_0.5": tp / (tp + fp) if tp + fp else np.nan, "npv_at_0.5": tn / (tn + fn) if tn + fn else np.nan}


def analyse():
    w = pd.read_csv(WIN_OUT, low_memory=False)
    d1 = pd.read_csv(D1_PRED) if D1_PRED.exists() else pd.DataFrame(columns=["dataset", "segment_id", "prob_unusable"])
    # CinC 2017
    c17 = w[w.dataset == "CinC2017"].copy()
    d17 = d1[d1.dataset == "CinC2017_subset"].set_index("segment_id")["prob_unusable"]
    c17[D1] = c17["segment_id"].map(d17)
    rows = []
    for agg in ["max", "mean", "first_window"]:
        if agg == "first_window":
            rec = c17[c17.unit_idx == 0].set_index("record_name")
        else:
            rec = c17.groupby("record_name").agg({**{m: agg for m in MODELS + [D1]}, "label": "first"})
        subset_ids = rec.index[rec[D1].notna()]
        for m in MODELS:
            rows.append({"cohort": "all recordings", "aggregation": agg, "model": m, **rec_metrics(rec.label, rec[m])})
        for m in MODELS + [D1]:
            s = rec.loc[subset_ids]
            if len(s) and s[m].notna().all():
                rows.append({"cohort": "balanced subset (d1)", "aggregation": agg, "model": m, **rec_metrics(s.label, s[m])})
    pd.DataFrame(rows).to_csv(C17_OUT, index=False)
    print(pd.DataFrame(rows).query("aggregation == 'max'")[["cohort", "model", "n", "prevalence", "auroc", "auroc_ci_low",
                                                             "auroc_ci_high", "ap", "sens_at_0.5", "fr_at_0.5", "ppv_at_0.5"]]
          .round(3).to_string(index=False))
    # CinC 2011, 12 leads
    c12 = w[w.dataset == "CinC2011_12lead"].copy()
    c12["record_name"] = c12["record_name"].astype(int)  # shared CSV with CinC 2017 string names
    d12 = d1[d1.dataset == "CinC2011_12lead_firstN"].set_index("segment_id")["prob_unusable"]
    c12[D1] = c12["segment_id"].map(d12)
    mem = pd.read_csv(CINC11 / "cinc2011_random_draws_manifest.csv")
    rows = []
    for agg in ["lead_II", "max_12", "mean_12"]:
        if agg == "lead_II":
            rec = c12[c12.unit_idx == 1].set_index("record_name")
        else:
            rec = c12.groupby("record_name").agg({**{m: agg.split("_")[0] for m in MODELS + [D1]}, "label": "first"})
        for draw, mm in mem.groupby("draw"):
            r = rec.loc[mm.record_name.values]
            for m in MODELS + [D1]:
                if r[m].notna().all():
                    rows.append({"draw": draw, "aggregation": agg, "model": m, "n": len(r),
                                 "auroc": roc_auc_score(r.label, r[m]), "ap": average_precision_score(r.label, r[m]),
                                 "sens_at_0.5": float((r[m][r.label == 1] >= 0.5).mean()),
                                 "fr_at_0.5": float((r[m][r.label == 0] >= 0.5).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(C12_OUT, index=False)
    print(out[out.draw == "first_N_original"].pivot_table(index="model", columns="aggregation", values="auroc").round(3))
    print(f"[Saved] {C17_OUT}\n[Saved] {C12_OUT}")


if __name__ == "__main__":
    if "--analysis-only" not in sys.argv:
        score_windows()
    analyse()
