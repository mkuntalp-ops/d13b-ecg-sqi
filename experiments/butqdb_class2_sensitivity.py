"""
Sensitivity analysis: relabel BUT QDB class-2 windows as unusable.

Two label schemes are trained with identical folds, settings and seed (42), in the same run, so the
comparison isolates the effect of the labels:
- original:        class 3 = unusable (main analysis; 297 unusable windows)
- class2_unusable: classes 2 and 3 = unusable (490 unusable windows)

The rule heuristic and the zero-shot d1-3B do not use training labels, so their scores are taken from
the main consolidated predictions and only the evaluation labels change.

Outputs:
- results/butqdb_class2_oof_predictions.csv  (out-of-fold scores for both schemes)
- results/butqdb_class2_sensitivity.csv      (AUROC with subject-level bootstrap CI, AP, Brier, sens/FR at 0.5)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_subject_level_bootstrap_ci
from features.sqi_extractor import SQI_FEATURE_NAMES
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import IndependentSQIClassifier, SinglePassMultimodalClassifier
from models.tabular_baselines import SQIGradientBoosting, SQILogisticRegression

RESULTS = PROJECT_ROOT / "results"
MANIFEST = RESULTS / "manifest_butqdb_1000_cohort.csv"
OOF_OUT = RESULTS / "butqdb_class2_oof_predictions.csv"
SUMMARY_OUT = RESULTS / "butqdb_class2_sensitivity.csv"

TRAINED = ["Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG", "2D_CNN_Image", "Multimodal_Fusion"]
REUSED = {"Rule_Based_SQI_Heuristic": "prob_Rule_Based_SQI_Heuristic", "LiquidAI_d1_3B_Neural": "prob_LiquidAI_d1_3B_Neural"}
SCHEMES = {"original": lambda c: (c == 3).astype(int), "class2_unusable": lambda c: (c >= 2).astype(int)}


def train_oof(df: pd.DataFrame, y: np.ndarray) -> dict:
    """5-fold subject-independent OOF predictions with fixed threshold 0.5 (no threshold tuning needed)."""
    X = np.nan_to_num(df[SQI_FEATURE_NAMES].values.astype(float))
    sigs = [np.load(p) for p in df["signal_path"]]
    imgs = df["image_path"].tolist()
    fold = df["fold_id"].values
    oof = {m: np.zeros(len(df)) for m in TRAINED}
    for k in range(5):
        tr = np.where(fold != k)[0]
        te = np.where(fold == k)[0]
        lr = SQILogisticRegression().fit(X[tr], y[tr])
        oof["Logistic_Regression_SQI"][te] = lr.predict_proba(X[te])
        lgb = SQIGradientBoosting().fit(X[tr], y[tr])
        oof["LightGBM_SQI"][te] = lgb.predict_proba(X[te])
        mlp = IndependentSQIClassifier(epochs=35, lr=1e-3).fit(X[tr], y[tr])
        oof["Independent_SQI_MLP"][te] = mlp.predict_proba(X[te])
        c1 = ECG1DCNNClassifier(epochs=25, lr=1e-3).fit([sigs[i] for i in tr], y[tr])
        oof["1D_CNN_RawECG"][te] = c1.predict_proba([sigs[i] for i in te])
        c2 = ECG2DCNNClassifier(epochs=25, lr=1e-3).fit([imgs[i] for i in tr], y[tr])
        oof["2D_CNN_Image"][te] = c2.predict_proba([imgs[i] for i in te])
        cm = SinglePassMultimodalClassifier(epochs=25, lr=1e-3).fit([imgs[i] for i in tr], X[tr], y[tr])
        oof["Multimodal_Fusion"][te] = cm.predict_proba([imgs[i] for i in te], X[te])
        print(f"[fold {k}] done", flush=True)
    return oof


def metrics(df: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1]).ravel()
    boot = pd.DataFrame({"subject_id": df["subject_id"].values, "label": y, "prob": p})
    ci = compute_subject_level_bootstrap_ci(boot, pred_col="prob", target_col="label", subject_col="subject_id",
                                            n_bootstraps=1000, seed=42)["auroc"]
    return {
        "n_unusable": int(y.sum()),
        "auroc": roc_auc_score(y, p),
        "auroc_ci_low": ci["ci_low"],
        "auroc_ci_high": ci["ci_high"],
        "ap": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "sens_at_0.5": tp / (tp + fn),
        "fr_at_0.5": fp / (tn + fp),
    }


def main():
    df = pd.read_csv(MANIFEST)
    main_run = pd.read_csv(RESULTS / "cohort1000_consolidated_oof_predictions.csv").set_index("segment_id")
    df = df.set_index("segment_id", drop=False).loc[main_run.index].reset_index(drop=True)
    orig = df["original_quality_class"].values

    oof_rows, summary = [], []
    for scheme, make_y in SCHEMES.items():
        y = make_y(orig)
        print(f"\n=== label scheme: {scheme} (unusable = {y.sum()}) ===", flush=True)
        oof = train_oof(df, y)
        for m in TRAINED:
            p = oof[m]
            summary.append({"scheme": scheme, "model": m, **metrics(df, y, p)})
            oof_rows.append(pd.DataFrame({"scheme": scheme, "segment_id": df["segment_id"], "model": m, "label": y, "prob": p}))
        for m, col in REUSED.items():
            p = main_run.loc[df["segment_id"], col].values.astype(float)
            summary.append({"scheme": scheme, "model": m, **metrics(df, y, p)})
            oof_rows.append(pd.DataFrame({"scheme": scheme, "segment_id": df["segment_id"], "model": m, "label": y, "prob": p}))

    pd.concat(oof_rows, ignore_index=True).to_csv(OOF_OUT, index=False)
    out = pd.DataFrame(summary)
    out.to_csv(SUMMARY_OUT, index=False)
    print(out[["scheme", "model", "n_unusable", "auroc", "auroc_ci_low", "auroc_ci_high", "sens_at_0.5", "fr_at_0.5"]].round(3).to_string(index=False))
    print(f"[Saved] {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
