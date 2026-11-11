"""
EXPLORATORY analysis 3.3b (revision of 2026-11-06): quantization-insensitive flatline fraction.

The definition and the analysis were written into REVISION_PLAN_2026-11-05_post_desk_rejection.md before any CinC
window was scored with it. The problem it addresses (flatline_fraction counting repeats of the 5 uV ADC step of CinC
2011) was found after the CinC 2011 results were known, so this analysis is exploratory and does not replace the
primary results.

Change: flatline_fraction -> flatline_fraction_robust (features.sqi_extractor.robust_flatline_and_saturation); the other
10 model inputs, the models (LR, LightGBM, SQI MLP), their settings and seed are those of the primary analysis.
Evaluations: BUT QDB 988 windows, the five subject folds; BUT-only training applied to CinC 2011 Set A lead II (998
records) and CinC 2017 (all windows, recording maximum). Reference: the same models with the original feature, from
the v2 primary outputs (raw_supervised_cohort1000_oof_predictions.csv, external_window_predictions.csv).

Outputs ($ECG_RESULTS_DIR): flatline_robust_features.csv (per window), flatline_robust_predictions.csv,
flatline_robust_summary.csv (AUROC by model, set and feature version; univariate AUROC and class medians of both
flatline measures)
Usage: ECG_RESULTS_DIR=results/v2_2026-11 python experiments/flatline_robust_exploratory.py [--analysis-only]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CINC2011_LEADII_FULL, RESULTS_DIR  # noqa: E402
from data.units import to_millivolts  # noqa: E402
from features.sqi_extractor import MODEL_INPUT_FEATURE_NAMES, robust_flatline_and_saturation  # noqa: E402
from models.multimodal_fusion import IndependentSQIClassifier  # noqa: E402
from models.tabular_baselines import SQIGradientBoosting, SQILogisticRegression  # noqa: E402

CINC17 = ROOT / "cache" / "datasets" / "cinc2017" / "cinc2017_windows.csv"
FS = 500.0
ROBUST = [("flatline_fraction_robust" if f == "flatline_fraction" else f) for f in MODEL_INPUT_FEATURE_NAMES]
MODELS = {"Logistic_Regression_SQI": lambda: SQILogisticRegression(),
          "LightGBM_SQI": lambda: SQIGradientBoosting(),
          "Independent_SQI_MLP": lambda: IndependentSQIClassifier(epochs=35, lr=1e-3)}


def add_robust(df: pd.DataFrame, unit_col: str = "signal_unit") -> pd.DataFrame:
    vals = []
    for i, (path, unit) in enumerate(zip(df.signal_path, df[unit_col])):
        x = to_millivolts(np.load(path).astype(float), unit)
        vals.append(robust_flatline_and_saturation(x, FS)[0])
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(df)}", flush=True)
    return df.assign(flatline_fraction_robust=vals)


def fit(name, X, y):
    m = MODELS[name]()
    m.fit(X, y)
    return m


def main():
    if "--analysis-only" in sys.argv:
        return analyse()
    but = add_robust(pd.read_csv(RESULTS_DIR / "manifest_butqdb_1000_cohort.csv"))
    c11 = add_robust(pd.read_csv(CINC2011_LEADII_FULL))
    c17 = add_robust(pd.read_csv(CINC17))
    print("[features] done", flush=True)
    feats = pd.concat([d[["segment_id", "label", "flatline_fraction", "flatline_fraction_robust"]].assign(dataset=n)
                       for n, d in [("BUT_QDB", but), ("CinC2011", c11), ("CinC2017", c17)]])
    feats.to_csv(RESULTS_DIR / "flatline_robust_features.csv", index=False)

    y = but.label.values.astype(int)
    Xb = but[ROBUST].values
    preds = []
    for name in MODELS:
        oof = np.zeros(len(but))
        for k in range(5):
            tr, te = but.fold_id.values != k, but.fold_id.values == k
            oof[te] = fit(name, Xb[tr], y[tr]).predict_proba(Xb[te])
        preds.append(pd.DataFrame({"dataset": "BUT_QDB", "segment_id": but.segment_id, "record": but.subject_id.astype(str),
                                   "label": y, "model": name, "prob_unusable": oof}))
        full = fit(name, Xb, y)
        for ds, d in [("CinC2011", c11), ("CinC2017", c17)]:
            preds.append(pd.DataFrame({"dataset": ds, "segment_id": d.segment_id, "record": d.record_name.astype(str),
                                       "label": d.label.values, "model": name,
                                       "prob_unusable": full.predict_proba(d[ROBUST].values)}))
        print(f"[{name}] done", flush=True)
    pd.concat(preds, ignore_index=True).to_csv(RESULTS_DIR / "flatline_robust_predictions.csv", index=False)
    analyse()


def analyse():
    P = pd.read_csv(RESULTS_DIR / "flatline_robust_predictions.csv", low_memory=False)
    feats = pd.read_csv(RESULTS_DIR / "flatline_robust_features.csv")

    # reference: same models with the original flatline feature (v2 primary outputs)
    oof = pd.read_csv(RESULTS_DIR / "raw_supervised_cohort1000_oof_predictions.csv").set_index("segment_id")
    ext = pd.read_csv(RESULTS_DIR / "external_window_predictions.csv", low_memory=False)
    # the 12-lead set reuses the lead II segment ids of CinC 2011, so look references up within each dataset
    ext = {ds: g.set_index("segment_id") for ds, g in ext.groupby("dataset")}
    rows = []
    for (ds, name), g in P.groupby(["dataset", "model"]):
        if ds == "BUT_QDB":
            ref = oof.loc[g.segment_id, f"prob_{name}"].values
            yy, pr, po = g.label.values, g.prob_unusable.values, ref
        else:
            g = g.assign(ref=ext[ds].loc[g.segment_id, name].values)
            r = g.groupby("record").agg(y=("label", "first"), p=("prob_unusable", "max"), ref=("ref", "max"))
            yy, pr, po = r.y.values, r.p.values, r.ref.values
        rows.append({"dataset": ds, "model": name, "n_units": len(yy), "auroc_original_flatline": roc_auc_score(yy, po),
                     "auroc_robust_flatline": roc_auc_score(yy, pr),
                     "n_unique_scores_original": len(np.unique(po)), "n_unique_scores_robust": len(np.unique(pr))})
    for ds, g in feats.groupby("dataset"):
        if ds == "CinC2017":
            g = g.groupby(g.segment_id.str.rsplit("_w", n=1).str[0]).agg(
                label=("label", "first"), flatline_fraction=("flatline_fraction", "max"),
                flatline_fraction_robust=("flatline_fraction_robust", "max"))
        for f in ["flatline_fraction", "flatline_fraction_robust"]:
            rows.append({"dataset": ds, "model": f"univariate {f}", "n_units": len(g),
                         "auroc_univariate": roc_auc_score(g.label, g[f]),
                         "median_usable": g.loc[g.label == 0, f].median(), "median_unusable": g.loc[g.label == 1, f].median()})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "flatline_robust_summary.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
