"""
Subject-Independent 5-Fold Cross-Validation for Supervised & Rule Baselines on 1,000-Cohort
Evaluates:
1. Logistic Regression (SQI)
2. LightGBM (SQI)
3. Independent SQI MLP
4. 1D CNN (Raw Signal)
5. 2D CNN (Rendered Image)
6. SinglePass Multimodal Net (Image + SQI)
7. Rule-Based SQI Heuristic (Expert Rule Baseline, 0 parameters)

Saves raw out-of-fold predictions to results/raw_supervised_cohort1000_oof_predictions.csv
with fold_id, inner validation chosen thresholds, and segment_ids.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, TARGET_FS
from features.sqi_extractor import SQI_FEATURE_NAMES
from models.tabular_baselines import SQILogisticRegression, SQIGradientBoosting
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier, IndependentSQIClassifier
from models.rule_heuristic_baseline import RuleBasedSQIHeuristic

OUT_CSV = RESULTS_DIR / "raw_supervised_cohort1000_oof_predictions.csv"


def select_threshold_on_inner_val(y_val: np.ndarray, p_val: np.ndarray) -> float:
    if len(np.unique(y_val)) < 2:
        return 0.5
    thresholds = np.linspace(0.1, 0.9, 81)
    best_j = -1.0
    best_thresh = 0.5
    for t in thresholds:
        preds = (p_val >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_thresh = float(t)
    return best_thresh


def run_supervised_cv():
    print("=" * 80)
    print("TRAINING SUPERVISED BASELINES ON 1,000-COHORT (5-FOLD SUBJECT CV)")
    print("=" * 80)

    manifest_path = RESULTS_DIR / "manifest_butqdb_1000_cohort.csv"
    df = pd.read_csv(manifest_path)
    n_samples = len(df)

    X_sqi = np.nan_to_num(df[SQI_FEATURE_NAMES].values, nan=0.0)
    raw_signals = [np.load(p) for p in df["signal_path"]]
    image_paths = df["image_path"].tolist()
    y = df["label"].values
    subject_ids = df["subject_id"].values
    fold_ids = df["fold_id"].values

    model_names = [
        "Logistic_Regression_SQI",
        "LightGBM_SQI",
        "Independent_SQI_MLP",
        "1D_CNN_RawECG",
        "2D_CNN_Image",
        "Multimodal_Fusion",
        "Rule_Based_SQI_Heuristic"
    ]

    oof_probs = {m: np.zeros(n_samples, dtype=np.float32) for m in model_names}
    inner_thresholds = {m: [] for m in model_names}

    # Evaluate Rule Heuristic
    print("\nEvaluating Rule-Based SQI Heuristic...")
    rule_model = RuleBasedSQIHeuristic()
    for i in range(n_samples):
        img = Image.open(image_paths[i]).convert("RGB")
        sqi_d = {name: float(df[name].iloc[i]) for name in SQI_FEATURE_NAMES}
        dec = rule_model.decide_single(img, sqi_d, modality="multimodal")
        oof_probs["Rule_Based_SQI_Heuristic"][i] = dec["prob_unusable"]

    for fold in range(5):
        test_mask = (fold_ids == fold)
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        train_subjects = np.unique(subject_ids[train_mask])
        val_subj = train_subjects[:max(1, len(train_subjects)//4)]
        inner_val_mask = np.isin(subject_ids, val_subj) & train_mask
        inner_train_mask = train_mask & (~inner_val_mask)

        print(f"\n>>> Fold {fold}: Train={sum(train_mask)} (InnerTrain={sum(inner_train_mask)}, InnerVal={sum(inner_val_mask)}), Test={sum(test_mask)}")

        # A. Logistic Regression
        lr_inner = SQILogisticRegression()
        lr_inner.fit(X_sqi[inner_train_mask], y[inner_train_mask])
        p_val_lr = lr_inner.predict_proba(X_sqi[inner_val_mask])
        t_lr = select_threshold_on_inner_val(y[inner_val_mask], p_val_lr)
        lr_full = SQILogisticRegression()
        lr_full.fit(X_sqi[train_mask], y[train_mask])
        oof_probs["Logistic_Regression_SQI"][test_idx] = lr_full.predict_proba(X_sqi[test_mask])
        inner_thresholds["Logistic_Regression_SQI"].append(t_lr)

        # B. LightGBM
        lgbm_inner = SQIGradientBoosting()
        lgbm_inner.fit(X_sqi[inner_train_mask], y[inner_train_mask])
        p_val_lgbm = lgbm_inner.predict_proba(X_sqi[inner_val_mask])
        t_lgbm = select_threshold_on_inner_val(y[inner_val_mask], p_val_lgbm)
        lgbm_full = SQIGradientBoosting()
        lgbm_full.fit(X_sqi[train_mask], y[train_mask])
        oof_probs["LightGBM_SQI"][test_idx] = lgbm_full.predict_proba(X_sqi[test_mask])
        inner_thresholds["LightGBM_SQI"].append(t_lgbm)

        # C. Independent SQI MLP
        mlp = IndependentSQIClassifier(epochs=35, lr=1e-3)
        mlp.fit(X_sqi[inner_train_mask], y[inner_train_mask])
        p_val_mlp = mlp.predict_proba(X_sqi[inner_val_mask])
        t_mlp = select_threshold_on_inner_val(y[inner_val_mask], p_val_mlp)
        mlp_full = IndependentSQIClassifier(epochs=35, lr=1e-3)
        mlp_full.fit(X_sqi[train_mask], y[train_mask])
        oof_probs["Independent_SQI_MLP"][test_idx] = mlp_full.predict_proba(X_sqi[test_mask])
        inner_thresholds["Independent_SQI_MLP"].append(t_mlp)

        # D. 1D CNN
        train_sigs = [raw_signals[i] for i in train_idx]
        test_sigs = [raw_signals[i] for i in test_idx]
        cnn1d = ECG1DCNNClassifier(epochs=25, lr=1e-3)
        cnn1d.fit(train_sigs, y[train_mask])
        oof_probs["1D_CNN_RawECG"][test_idx] = cnn1d.predict_proba(test_sigs)
        inner_thresholds["1D_CNN_RawECG"].append(0.5)

        # E. 2D CNN
        train_imgs = [image_paths[i] for i in train_idx]
        test_imgs = [image_paths[i] for i in test_idx]
        cnn2d = ECG2DCNNClassifier(epochs=25, lr=1e-3)
        cnn2d.fit(train_imgs, y[train_mask])
        oof_probs["2D_CNN_Image"][test_idx] = cnn2d.predict_proba(test_imgs)
        inner_thresholds["2D_CNN_Image"].append(0.5)

        # F. SinglePass Multimodal Net
        mm = SinglePassMultimodalClassifier(epochs=25, lr=1e-3)
        mm.fit(train_imgs, X_sqi[train_mask], y[train_mask])
        oof_probs["Multimodal_Fusion"][test_idx] = mm.predict_proba(test_imgs, X_sqi[test_mask])
        inner_thresholds["Multimodal_Fusion"].append(0.5)

        inner_thresholds["Rule_Based_SQI_Heuristic"].append(0.5)

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "segment_id": df["segment_id"],
        "subject_id": df["subject_id"],
        "record_name": df["record_name"],
        "label": y,
        "fold_id": fold_ids,
        **{f"prob_{m}": oof_probs[m] for m in model_names}
    })
    oof_df.to_csv(OUT_CSV, index=False)
    print(f"\n[Saved] Supervised OOF predictions saved to {OUT_CSV}")


if __name__ == "__main__":
    run_supervised_cv()
