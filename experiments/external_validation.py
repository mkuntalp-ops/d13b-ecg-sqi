"""
External validation added after the second review (2026-10-13).

1. CinC 2017 (single lead, 8,528 recordings, natural prevalence of '~' noisy recordings). Every 10-s window is
   scored by the models trained on the full BUT QDB cohort (seed 42, same settings as the CinC 2011 transfer);
   recording-level score = maximum (primary) or mean of the window scores. Metrics at the natural prevalence:
   AUROC and AP with recording-level bootstrap CIs, and sensitivity, false reject, PPV and NPV at tau = 0.5.
   d1-3B is evaluated on a balanced subset (all noisy recordings plus as many random non-noisy recordings),
   where the other models are reported on the same subset.
0. CinC 2011 Set A lead II, all 998 labelled records (v2): AUROC and AP with recording-level bootstrap CIs and
   decision metrics at tau = 0.5 (results/external_cinc2011_summary.csv).
2. CinC 2011, all 12 leads of the 998 labelled records (v2; v1: 565 records in draws). Recording score = lead II,
   maximum over the 12 leads, or mean over the 12 leads, with recording-level bootstrap CIs. Zero-shot d1-3B comes from
   experiments/d1_twelve_lead_full.py. Run with --twelve-lead (fit, score, analyse) or --twelve-lead-analysis. Scores go
   to their own file, so the lead II and CinC 2017 predictions used by every other analysis are not rewritten.

d1-3B probabilities come from experiments/d1_linear_probe.py (results/d1_external_predictions.csv).

Usage: python experiments/external_validation.py                 (score windows, then analyse)
       python experiments/external_validation.py --analysis-only
       python experiments/external_validation.py --twelve-lead | --twelve-lead-analysis
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
from features.sqi_extractor import MODEL_INPUT_FEATURE_NAMES, SQI_FEATURE_NAMES
from models.rule_heuristic_baseline import RuleBasedSQIHeuristic

from config import CINC2011_LEADII_FULL, CINC2011_TWELVE_LEAD_FULL, RESULTS_DIR as _CFG_RESULTS  # noqa: E402  (ECG_RESULTS_DIR overrides)
RESULTS = _CFG_RESULTS
CINC11 = PROJECT_ROOT / "cache" / "datasets" / "cinc2011"
CINC17 = PROJECT_ROOT / "cache" / "datasets" / "cinc2017" / "cinc2017_windows.csv"
TWELVE = CINC11 / "twelve_lead" / "cinc2011_twelve_lead_windows.csv"
WIN_OUT = RESULTS / "external_window_predictions.csv"
C17_OUT = RESULTS / "external_cinc2017_summary.csv"
C12_OUT = RESULTS / "cinc2011_twelve_lead_summary.csv"
C11_OUT = RESULTS / "external_cinc2011_summary.csv"
D1_PRED = RESULTS / "d1_external_predictions.csv"
W12_OUT = RESULTS / "external_twelve_lead_window_predictions.csv"
D12_PRED = RESULTS / "d1_twelve_lead_full_predictions.csv"
MODELS = ["Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG", "2D_CNN_Image",
          "Multimodal_Fusion", "Rule_Based_SQI_Heuristic"]
D1 = "LiquidAI_d1_3B_Neural"
N_BOOT = 1000


def score_frame(models, df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Scores of the BUT-only models on one window table (one row per window)."""
    lr, lgb, mlp, c1, c2, mm = models
    rule = RuleBasedSQIHeuristic()
    X = np.nan_to_num(df[MODEL_INPUT_FEATURE_NAMES].values.astype(float))
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
            rp.append(rule.decide_single(img, dict(zip(MODEL_INPUT_FEATURE_NAMES, x)), modality="multimodal")["prob_unusable"])
    out["Rule_Based_SQI_Heuristic"] = rp
    return out


def score_windows():
    models = fit_models()
    parts = []
    # The v1 12-lead table (565 records) is no longer scored into this file: its windows share segment ids with the
    # lead II set, and the v2 12-lead analysis has its own file (W12_OUT). Code audit 2026-11-11, CODE-009. Files
    # written before this change still contain those rows under dataset == "CinC2011_12lead"; always filter by dataset.
    for name, path in [("CinC2011", CINC2011_LEADII_FULL), ("CinC2017", CINC17)]:
        df = pd.read_csv(path)
        if name == "CinC2011":
            df["lead_idx"] = 1  # one lead II window per record
        out = score_frame(models, df, name)
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
    # CinC 2011 Set A, lead II, all 998 labelled records (one window per record; v2 primary external set)
    c11 = w[w.dataset == "CinC2011"].copy()
    d11 = d1[d1.dataset == "CinC2011_leadII"].set_index("segment_id")["prob_unusable"]
    c11[D1] = c11["segment_id"].map(d11)
    if c11.record_name.duplicated().any():
        raise ValueError("CinC 2011 lead II: more than one window per record")
    rows = [{"model": m, **rec_metrics(c11.label, c11[m])} for m in MODELS + [D1] if len(c11) and c11[m].notna().all()]
    pd.DataFrame(rows).to_csv(C11_OUT, index=False)
    print(pd.DataFrame(rows)[["model", "n", "n_pos", "auroc", "auroc_ci_low", "auroc_ci_high", "ap", "sens_at_0.5",
                              "fr_at_0.5"]].round(3).to_string(index=False))
    print(f"[Saved] {C17_OUT}\n[Saved] {C11_OUT}")


def score_twelve_lead():
    """BUT-only models on all 12 leads of the 998 records (own output file; own fit of the seed-42 models)."""
    df = pd.read_csv(CINC2011_TWELVE_LEAD_FULL)
    if df.segment_id.nunique() != 11976 or df.record_name.nunique() != 998:
        raise ValueError("expected 11,976 lead windows from 998 records")
    out = score_frame(fit_models(), df, "CinC2011_12lead_full")
    out.to_csv(W12_OUT, index=False)
    # is this fit the same as the one behind the primary lead II table?
    prim = pd.read_csv(WIN_OUT, low_memory=False)
    prim = prim[prim.dataset == "CinC2011"].set_index("segment_id")
    l2 = out[out.unit_idx == 1].set_index("segment_id")
    for m in MODELS:
        print(f"[check] {m}: max |refit - primary| on lead II = {np.abs(l2[m] - prim.loc[l2.index, m]).max():.6f}, AUROC "
              f"{roc_auc_score(l2.label, l2[m]):.4f} vs {roc_auc_score(prim.loc[l2.index, 'label'], prim.loc[l2.index, m]):.4f}", flush=True)


def analyse_twelve_lead():
    w = pd.read_csv(W12_OUT, low_memory=False)
    d = pd.read_csv(D12_PRED).set_index("segment_id")["prob_unusable"]
    if not set(w.segment_id) <= set(d.index):
        raise ValueError("d1-3B scores are missing for some 12-lead windows")
    w[D1] = w["segment_id"].map(d)
    rows = []
    for agg in ["lead_II", "max_12", "mean_12"]:
        if agg == "lead_II":
            rec = w[w.unit_idx == 1].set_index("record_name")
        else:
            rec = w.groupby("record_name").agg({**{m: agg.split("_")[0] for m in MODELS + [D1]}, "label": "first"})
        if len(rec) != 998 or rec.index.duplicated().any():
            raise ValueError(f"{agg}: expected 998 records, got {len(rec)}")
        rows += [{"draw": "all_998", "aggregation": agg, "model": m, **rec_metrics(rec.label, rec[m])} for m in MODELS + [D1]]
    out = pd.DataFrame(rows)
    out.to_csv(C12_OUT, index=False)
    print(out.pivot_table(index="model", columns="aggregation", values="auroc").round(3).to_string())
    print(f"[Saved] {C12_OUT}")


if __name__ == "__main__":
    if "--twelve-lead" in sys.argv:
        score_twelve_lead()
        analyse_twelve_lead()
    elif "--twelve-lead-analysis" in sys.argv:
        analyse_twelve_lead()
    else:
        if "--analysis-only" not in sys.argv:
            score_windows()
        analyse()
