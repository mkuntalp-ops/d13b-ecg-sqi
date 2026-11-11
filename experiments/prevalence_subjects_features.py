"""
Analyses from stored predictions and annotations (second review, 2026-10-13). No model is trained.

1. Natural prevalence in BUT QDB. All non-overlapping 10-s windows inside the consensus annotations of the 18
   records (the candidate pool of the cohort) give the natural share of Class 3. The quota cohort (28.4%) is
   re-weighted to that prevalence: AP with sample weights, and PPV / NPV at tau = 0.5 from the pooled
   out-of-fold sensitivity and specificity. False rejections per hour of usable signal = FPR x 360 windows.
2. Subject-level heterogeneity: sensitivity (subjects with Class 3) and false reject rate per held-out
   subject at tau = 0.5, from the pooled out-of-fold predictions (seed 42).
3. Feature shift: univariate AUROC of each SQI (raw value as the score for "unusable") in BUT QDB, CinC 2011
   (lead II, the 565 records of the transfer analyses and the reported 150) and CinC 2017 (first window of each
   recording). Values below 0.5 mean the feature is higher in usable recordings.

v2 (revision of 2026-11-05): the quota cohort row is computed from the manifest (988 windows inside the official
intervals); the natural prevalence is also given for the candidate windows inside the official intervals
(experiments/butqdb_official_split.official_mask), the pool from which the v2 cohort is drawn; CinC 2011 in the
feature-shift table is all 998 labelled records (config.CINC2011_LEADII_FULL).

Outputs: results/natural_prevalence_summary.csv, results/butqdb_natural_prevalence_metrics.csv,
         results/butqdb_subject_heterogeneity.csv, results/feature_shift_univariate_auroc.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.extract_butqdb_1000_cohort import build_candidate_windows
from features.sqi_extractor import MODEL_INPUT_FEATURE_NAMES, SQI_FEATURE_NAMES

from config import CINC2011_LEADII_FULL  # noqa: E402
from config import RESULTS_DIR as _CFG_RESULTS  # noqa: E402  (ECG_RESULTS_DIR overrides)
RESULTS = _CFG_RESULTS
CINC11 = PROJECT_ROOT / "cache" / "datasets" / "cinc2011"
CINC17 = PROJECT_ROOT / "cache" / "datasets" / "cinc2017" / "cinc2017_windows.csv"
MODELS = ["Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG", "2D_CNN_Image",
          "Multimodal_Fusion", "Rule_Based_SQI_Heuristic", "LiquidAI_d1_3B_Neural"]


def natural_prevalence():
    w = build_candidate_windows()
    w["unusable"] = (w["original_quality_class"] == 3).astype(int)
    per_rec = w.groupby("record_name").agg(windows=("unusable", "size"), unusable=("unusable", "sum"))
    per_rec["prevalence"] = per_rec.unusable / per_rec.windows
    rows = [{"scope": "all 18 records", "windows": len(w), "unusable": int(w.unusable.sum()), "prevalence": w.unusable.mean()},
            {"scope": "excluding record 105001", "windows": int((w.record_name != "105001_ECG").sum()),
             "unusable": int(w[w.record_name != "105001_ECG"].unusable.sum()),
             "prevalence": w[w.record_name != "105001_ECG"].unusable.mean()},
            {"scope": "median over records", "windows": np.nan, "unusable": np.nan, "prevalence": per_rec.prevalence.median()}]
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    from butqdb_official_split import official_mask  # noqa: E402
    inside = official_mask(w)
    rows.append({"scope": "inside the official intervals", "windows": int(inside.sum()),
                 "unusable": int(w.unusable[inside].sum()), "prevalence": w.unusable[inside].mean()})
    man = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv")
    rows.append({"scope": "quota cohort (reported)", "windows": len(man), "unusable": int(man.label.sum()),
                 "prevalence": man.label.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "natural_prevalence_summary.csv", index=False)
    print(out.round(4).to_string(index=False))
    print(per_rec.round(3).to_string())
    return (float(w.unusable.mean()), float(w[w.record_name != "105001_ECG"].unusable.mean()),
            float(w.unusable[inside].mean()))


def reweighted_metrics(pi_all, pi_ex, pi_off):
    oof = pd.read_csv(RESULTS / "cohort1000_consolidated_oof_predictions.csv")
    y = oof["label"].values
    rows = []
    for m in MODELS:
        p = oof[f"prob_{m}"].values
        pred = p >= 0.5
        sens, spec = pred[y == 1].mean(), 1 - pred[y == 0].mean()
        for scope, pi in [("quota cohort", y.mean()), ("natural, all records", pi_all), ("natural, excl. 105001", pi_ex),
                          ("natural, official intervals", pi_off)]:
            w = np.where(y == 1, 1.0, (y.sum() * (1 - pi) / pi) / (y == 0).sum())
            ppv = sens * pi / (sens * pi + (1 - spec) * (1 - pi))
            npv = spec * (1 - pi) / (spec * (1 - pi) + (1 - sens) * pi)
            rows.append({"model": m, "prevalence_scope": scope, "prevalence": pi, "auroc": roc_auc_score(y, p),
                         "ap_reweighted": average_precision_score(y, p, sample_weight=w), "sens_at_0.5": sens,
                         "spec_at_0.5": spec, "ppv_at_0.5": ppv, "npv_at_0.5": npv,
                         "false_rejections_per_hour_usable": (1 - spec) * 360})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "butqdb_natural_prevalence_metrics.csv", index=False)
    print(out[out.prevalence_scope != "quota cohort"][["model", "prevalence_scope", "prevalence", "ap_reweighted", "ppv_at_0.5",
                                                       "npv_at_0.5", "false_rejections_per_hour_usable"]].round(3).to_string(index=False))


def subject_heterogeneity():
    oof = pd.read_csv(RESULTS / "cohort1000_consolidated_oof_predictions.csv")
    rows = []
    for s, g in oof.groupby("subject_id"):
        y = g["label"].values
        r = {"subject_id": s, "fold_id": int(g["fold_id"].iloc[0]), "n_usable": int((y == 0).sum()), "n_unusable": int((y == 1).sum())}
        for m in MODELS:
            pred = g[f"prob_{m}"].values >= 0.5
            r[f"{m}_sens"] = pred[y == 1].mean() if (y == 1).any() else np.nan
            r[f"{m}_fr"] = pred[y == 0].mean() if (y == 0).any() else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "butqdb_subject_heterogeneity.csv", index=False)
    cols = ["subject_id", "n_usable", "n_unusable"] + [f"{m}_{k}" for m in ["1D_CNN_RawECG", "2D_CNN_Image", "LiquidAI_d1_3B_Neural"] for k in ["sens", "fr"]]
    print(out[cols].round(2).to_string(index=False))


def feature_shift():
    sets = {}
    but = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv")
    sets[f"BUT QDB ({len(but):,} windows)"] = but
    c11 = pd.read_csv(CINC2011_LEADII_FULL)
    if len(c11) != 998 or c11.record_name.duplicated().any():
        raise ValueError(f"CinC 2011: expected 998 records, one window each; got {len(c11)}")
    sets["CinC 2011 lead II (998 records)"] = c11
    if CINC17.exists():
        c17 = pd.read_csv(CINC17)
        sets["CinC 2017 (first window, all recordings)"] = c17[c17.window_idx == 0]
    rows = []
    for f in MODEL_INPUT_FEATURE_NAMES:
        r = {"feature": f}
        for name, d in sets.items():
            x = np.nan_to_num(d[f].values.astype(float))
            r[name] = roc_auc_score(d["label"].values, x) if np.ptp(x) > 0 else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "feature_shift_univariate_auroc.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    pi_all, pi_ex, pi_off = natural_prevalence()
    reweighted_metrics(pi_all, pi_ex, pi_off)
    subject_heterogeneity()
    feature_shift()
