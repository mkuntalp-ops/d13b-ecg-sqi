"""
Threshold selection without using the test data (second review, 2026-10-13).

Tables 1b and 2b choose operating points on the partition they report. Here the thresholds are chosen without
the evaluated data:

1. Nested selection on BUT QDB. For each outer fold k (3 test subjects), the 12 training subjects are split by
   GroupKFold into 4 inner folds; each trainable model is retrained on 9 subjects and scored on the other 3,
   giving inner out-of-fold scores for the 12 training subjects. Thresholds are chosen on those scores and
   applied to the stored outer-fold scores of fold k (seed 42, Table 1). The rule heuristic and d1-3B have no
   training, so their thresholds are chosen on their scores for the 12 training subjects.
2. Source-selected thresholds for transfer. Thresholds chosen on the pooled BUT QDB out-of-fold scores are
   applied frozen to CinC 2011 (lead II; reported subset and random subsets) and to CinC 2017 (recording max).

Operating rules (positive = unusable, p >= t): Target 1 = maximum sensitivity with FPR <= 5%
(ties: minimum FPR); Target 2 = minimum FPR with sensitivity >= 80% (ties: maximum sensitivity);
Youden = maximum sensitivity + specificity - 1.

Usage: python experiments/nested_thresholds.py                  (nested CV, then transfer)
       python experiments/nested_thresholds.py --transfer-only
Outputs: results/nested_thresholds_folds.csv, results/nested_thresholds_summary.csv,
         results/threshold_transfer_summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.sqi_extractor import SQI_FEATURE_NAMES
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import IndependentSQIClassifier, SinglePassMultimodalClassifier
from models.tabular_baselines import SQIGradientBoosting, SQILogisticRegression

RESULTS = PROJECT_ROOT / "results"
OOF = RESULTS / "cohort1000_consolidated_oof_predictions.csv"
MANIFEST = RESULTS / "manifest_butqdb_1000_cohort.csv"
FOLDS_OUT = RESULTS / "nested_thresholds_folds.csv"
SUM_OUT = RESULTS / "nested_thresholds_summary.csv"
TRANSFER_OUT = RESULTS / "threshold_transfer_summary.csv"
TRAINED = ["Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG", "2D_CNN_Image",
           "Multimodal_Fusion"]
UNTRAINED = ["Rule_Based_SQI_Heuristic", "LiquidAI_d1_3B_Neural"]
RULES = ["target1_fpr5", "target2_sens80", "youden", "fixed_0.5"]


def pick(y, p, rule, max_fpr=0.05, min_sens=0.80):
    if rule == "fixed_0.5":
        return 0.5
    fpr, tpr, thr = roc_curve(y, p, drop_intermediate=False)
    thr = np.minimum(thr, 1.0 + 1e-9)  # roc_curve's first threshold is inf
    if rule == "target1_fpr5":
        ok = np.where(fpr <= max_fpr)[0]
        best = ok[np.lexsort((fpr[ok], -tpr[ok]))[0]]
    elif rule == "target2_sens80":
        ok = np.where(tpr >= min_sens)[0]
        best = ok[np.lexsort((-tpr[ok], fpr[ok]))[0]]
    else:  # youden
        j = tpr - fpr
        best = np.lexsort((fpr, -j))[0]
    return float(thr[best])


def realised(y, p, t):
    pred = p >= t
    return {"sens": float(pred[y == 1].mean()), "fpr": float(pred[y == 0].mean()),
            "tp": int((pred & (y == 1)).sum()), "fp": int((pred & (y == 0)).sum()),
            "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}


def train_score(model, tr, te, df, X, sigs, imgs):
    y = df["label"].values
    if model == "Logistic_Regression_SQI":
        return SQILogisticRegression().fit(X[tr], y[tr]).predict_proba(X[te])
    if model == "LightGBM_SQI":
        return SQIGradientBoosting().fit(X[tr], y[tr]).predict_proba(X[te])
    if model == "Independent_SQI_MLP":
        return IndependentSQIClassifier(epochs=35, lr=1e-3).fit(X[tr], y[tr]).predict_proba(X[te])
    if model == "1D_CNN_RawECG":
        return ECG1DCNNClassifier(epochs=25, lr=1e-3).fit([sigs[i] for i in tr], y[tr]).predict_proba([sigs[i] for i in te])
    if model == "2D_CNN_Image":
        return ECG2DCNNClassifier(epochs=25, lr=1e-3).fit([imgs[i] for i in tr], y[tr]).predict_proba([imgs[i] for i in te])
    if model == "Multimodal_Fusion":
        return SinglePassMultimodalClassifier(epochs=25, lr=1e-3).fit([imgs[i] for i in tr], X[tr], y[tr]).predict_proba(
            [imgs[i] for i in te], X[te])
    raise ValueError(model)


def nested():
    df = pd.read_csv(MANIFEST)
    oof = pd.read_csv(OOF).set_index("segment_id").loc[df["segment_id"]].reset_index()
    X = np.nan_to_num(df[SQI_FEATURE_NAMES].values.astype(float))
    sigs = [np.load(p) for p in df["signal_path"]]
    imgs = df["image_path"].tolist()
    y, subj, fold = df["label"].values, df["subject_id"].values, df["fold_id"].values
    fold_rows = []
    for k in range(5):
        outer_tr = np.where(fold != k)[0]
        outer_te = np.where(fold == k)[0]
        inner = {m: np.zeros(len(outer_tr)) for m in TRAINED}
        for itr, ite in GroupKFold(n_splits=4).split(outer_tr, groups=subj[outer_tr]):
            for m in TRAINED:
                inner[m][ite] = train_score(m, outer_tr[itr], outer_tr[ite], df, X, sigs, imgs)
        for m in UNTRAINED:
            inner[m] = oof[f"prob_{m}"].values[outer_tr]
        for m in TRAINED + UNTRAINED:
            for rule in RULES:
                t = pick(y[outer_tr], inner[m], rule)
                fold_rows.append({"fold": k, "model": m, "rule": rule, "threshold": t,
                                  "inner_sens": realised(y[outer_tr], inner[m], t)["sens"],
                                  "inner_fpr": realised(y[outer_tr], inner[m], t)["fpr"],
                                  **realised(y[outer_te], oof[f"prob_{m}"].values[outer_te], t)})
        pd.DataFrame(fold_rows).to_csv(FOLDS_OUT, index=False)
        print(f"[nested] outer fold {k} done", flush=True)
    f = pd.DataFrame(fold_rows)
    summ = (f.groupby(["model", "rule"]).agg(tp=("tp", "sum"), fp=("fp", "sum"), n_pos=("n_pos", "sum"), n_neg=("n_neg", "sum"),
                                              thr_min=("threshold", "min"), thr_max=("threshold", "max"))
            .assign(sens=lambda d: d.tp / d.n_pos, fpr=lambda d: d.fp / d.n_neg).reset_index())
    # post-hoc reference: the same rule chosen on the pooled test scores themselves (Table 1b)
    ref = []
    for m in TRAINED + UNTRAINED:
        for rule in RULES:
            t = pick(y, oof[f"prob_{m}"].values, rule)
            r = realised(y, oof[f"prob_{m}"].values, t)
            ref.append({"model": m, "rule": rule, "posthoc_threshold": t, "posthoc_sens": r["sens"], "posthoc_fpr": r["fpr"]})
    summ = summ.merge(pd.DataFrame(ref), on=["model", "rule"])
    summ.to_csv(SUM_OUT, index=False)
    print(summ[["model", "rule", "sens", "fpr", "posthoc_sens", "posthoc_fpr", "thr_min", "thr_max"]].round(3).to_string(index=False))


def transfer():
    oof = pd.read_csv(OOF)
    y_b = oof["label"].values
    cinc = pd.read_csv(RESULTS / "cinc2011_expanded_predictions.csv")
    draws = pd.read_csv(RESULTS / "cinc_random_draws_predictions.csv")
    mem = pd.read_csv(PROJECT_ROOT / "cache" / "datasets" / "cinc2011" / "cinc2011_random_draws_manifest.csv")
    win = pd.read_csv(RESULTS / "external_window_predictions.csv")
    d1 = pd.read_csv(RESULTS / "d1_external_predictions.csv")
    c17 = win[win.dataset == "CinC2017"].copy()
    full17 = RESULTS / "d1_cinc2017_full_predictions.csv"  # d1-3B on all CinC 2017 windows (analysis plan A2)
    if full17.exists():
        c17["LiquidAI_d1_3B_Neural"] = c17["segment_id"].map(pd.read_csv(full17).set_index("segment_id")["prob_unusable"])
    else:
        c17["LiquidAI_d1_3B_Neural"] = c17["segment_id"].map(d1[d1.dataset == "CinC2017_subset"].set_index("segment_id")["prob_unusable"])
    rows = []
    for m in TRAINED + UNTRAINED:
        for rule in ["target1_fpr5", "target2_sens80", "youden", "fixed_0.5"]:
            t = pick(y_b, oof[f"prob_{m}"].values, rule)
            rows.append({"model": m, "rule": rule, "threshold": t, "dataset": "BUT_QDB pooled OOF (selection set)",
                         **realised(y_b, oof[f"prob_{m}"].values, t)})
            rows.append({"model": m, "rule": rule, "threshold": t, "dataset": "CinC2011 first-N (lead II)",
                         **realised(cinc["label"].values, cinc[f"prob_{m}"].values, t)})
            for draw, mm in mem[mem.draw.str.startswith("random")].groupby("draw"):
                p = draws.drop_duplicates("record_name").set_index("record_name").loc[mm.record_name, m].values
                rows.append({"model": m, "rule": rule, "threshold": t, "dataset": f"CinC2011 {draw} (lead II)",
                             **realised(mm.label.values, p, t)})
            rec = c17.groupby("record_name").agg(label=("label", "first"), p=(m, "max")).dropna()
            name = ("CinC2017 balanced d1 subset (record max)" if m == "LiquidAI_d1_3B_Neural" and not full17.exists()
                    else "CinC2017 all (record max)")
            rows.append({"model": m, "rule": rule, "threshold": t, "dataset": name,
                         **realised(rec.label.values, rec.p.values, t)})
    out = pd.DataFrame(rows)
    out.to_csv(TRANSFER_OUT, index=False)
    print(out[out.rule.isin(["target1_fpr5", "target2_sens80"]) & ~out.dataset.str.contains("random")]
          [["model", "rule", "dataset", "threshold", "sens", "fpr"]].round(3).to_string(index=False))
    print(f"[Saved] {TRANSFER_OUT}")


if __name__ == "__main__":
    if "--transfer-only" not in sys.argv:
        nested()
    transfer()
