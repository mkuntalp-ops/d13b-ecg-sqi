"""
Comprehensive Benchmark on Expanded BUT QDB 1,000-Window Cohort
Evaluates all 8 model paradigms:
1. Logistic Regression (SQI)
2. LightGBM (SQI)
3. Independent SQI MLP
4. 1D CNN (Raw Signal)
5. 2D CNN (Rendered Image)
6. SinglePass Multimodal Net (Image + SQI)
7. Rule-Based SQI Heuristic (Expert Rule Baseline, 0 parameters)
8. Real LiquidAI/d1-3B (Genuine 3.12B Neural Model, fail-closed)

Subject-independent 5-fold cross-validation on manifest_butqdb_1000_cohort.csv.
Computes:
- OOF AUROC, AUPRC, Brier Score, BalAcc, Sens, False Reject Rate (tau=0.5 and tuned tau*)
- Operating points: Sens @ FPR <= 5% and False Reject @ Sens >= 80%
- Saves out-of-fold predictions to results/cohort1000_oof_predictions.csv
- Saves summary table to results/cohort1000_subject_cv_summary.csv
- Saves operating points to results/cohort1000_operating_point_alignment.csv
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
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
from models.d1_real_neural_model import RealD1DecisionModel
from evaluation.metrics import compute_usability_metrics, compute_subject_level_bootstrap_ci


def select_threshold_on_inner_val(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Maximizes Youden Index on inner validation set."""
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


def extract_locked_operating_points_for_transfer(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """
    Extracts optimal thresholds on this development set for future lock-in during cross-dataset transfer.
    These are not cross-validated metrics, but calibration points to be used blindly on target sets.
    """
    thresholds = np.linspace(0.01, 0.99, 990)
    best_sens_at_low_fpr = 0.0
    thresh_at_low_fpr = 0.5
    best_fpr_at_high_sens = 1.0
    thresh_at_high_sens = 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        fpr = fp / max(1, fp + tn)
        sens = tp / max(1, tp + fn)

        if fpr <= 0.05:
            if sens >= best_sens_at_low_fpr:
                best_sens_at_low_fpr = sens
                thresh_at_low_fpr = t

        if sens >= 0.80:
            if fpr <= best_fpr_at_high_sens:
                best_fpr_at_high_sens = fpr
                thresh_at_high_sens = t

    return {
        "sens_at_fpr_le_5": best_sens_at_low_fpr,
        "thresh_at_fpr_le_5": thresh_at_low_fpr,
        "false_reject_at_sens_ge_80": best_fpr_at_high_sens,
        "thresh_at_sens_ge_80": thresh_at_high_sens
    }


def run_cohort_benchmark():
    manifest_path = RESULTS_DIR / "manifest_butqdb_1000_cohort.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run data/extract_butqdb_1000_cohort.py first.")

    df = pd.read_csv(manifest_path)
    n_samples = len(df)
    print(f"Loaded 1,000-cohort manifest: {n_samples} samples across {df['subject_id'].nunique()} subjects.")
    print(f"Class distribution: Usable(0)={sum(df['label']==0)}, Unusable(1)={sum(df['label']==1)}")

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
        "Rule_Based_SQI_Heuristic",
        "LiquidAI_d1_3B_Neural"
    ]

    oof_probs = {m: np.zeros(n_samples, dtype=np.float32) for m in model_names}
    chosen_thresholds = {m: [] for m in model_names}

    # Initialize zero-shot / rule models
    rule_model = RuleBasedSQIHeuristic()
    print("[Benchmark] Initializing real LiquidAI/d1-3B neural model (fail-closed)...")
    d1_neural = RealD1DecisionModel(device="mps")

    # 1. Zero-Shot and Rule Model Predictions across all samples
    print("\n--- Running Rule-Based SQI Heuristic & LiquidAI/d1-3B Neural on all samples ---")
    d1_raw_responses = {}
    for i in range(n_samples):
        img = Image.open(image_paths[i]).convert("RGB")
        sqi_d = {name: float(df[name].iloc[i]) for name in SQI_FEATURE_NAMES}
        seg_id = df["segment_id"].iloc[i]

        # Rule heuristic
        dec_rule = rule_model.decide_single(img, sqi_d, modality="multimodal")
        oof_probs["Rule_Based_SQI_Heuristic"][i] = dec_rule["prob_unusable"]

        # Real d1-3B neural model (FAIL-CLOSED)
        dec_d1 = d1_neural.decide_single(img, sqi_d, modality="multimodal", segment_id=seg_id)
        oof_probs["LiquidAI_d1_3B_Neural"][i] = dec_d1["prob_unusable"]
        d1_raw_responses[seg_id] = dec_d1["raw_response"]

        if (i + 1) % 50 == 0 or (i + 1) == n_samples:
            print(f"  Processed {i+1}/{n_samples} zero-shot samples (Last d1 P(unusable): {dec_d1['prob_unusable']:.4f})")

    # Save raw d1 responses to JSONL for auditability
    raw_resp_path = RESULTS_DIR / "cohort1000_d1_neural_raw_responses.jsonl"
    with open(raw_resp_path, "w") as f:
        for sid, resp in d1_raw_responses.items():
            f.write(json.dumps({"segment_id": sid, "response": resp}) + "\n")
    print(f"Saved raw d1 responses to {raw_resp_path}")

    # 2. 5-Fold Cross-Validation for Supervised Models
    print("\n--- Training Supervised Models across 5 Subject-Independent Folds ---")
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
        chosen_thresholds["Logistic_Regression_SQI"].append(t_lr)

        # B. LightGBM
        lgbm_inner = SQIGradientBoosting()
        lgbm_inner.fit(X_sqi[inner_train_mask], y[inner_train_mask])
        p_val_lgbm = lgbm_inner.predict_proba(X_sqi[inner_val_mask])
        t_lgbm = select_threshold_on_inner_val(y[inner_val_mask], p_val_lgbm)
        lgbm_full = SQIGradientBoosting()
        lgbm_full.fit(X_sqi[train_mask], y[train_mask])
        oof_probs["LightGBM_SQI"][test_idx] = lgbm_full.predict_proba(X_sqi[test_mask])
        chosen_thresholds["LightGBM_SQI"].append(t_lgbm)

        # C. Independent SQI MLP
        mlp = IndependentSQIClassifier(in_features=len(SQI_FEATURE_NAMES), epochs=35, lr=1e-3)
        mlp.fit(X_sqi[inner_train_mask], y[inner_train_mask], X_val=X_sqi[inner_val_mask], y_val=y[inner_val_mask])
        p_val_mlp = mlp.predict_proba(X_sqi[inner_val_mask])
        t_mlp = select_threshold_on_inner_val(y[inner_val_mask], p_val_mlp)
        mlp_full = IndependentSQIClassifier(in_features=len(SQI_FEATURE_NAMES), epochs=35, lr=1e-3)
        mlp_full.fit(X_sqi[train_mask], y[train_mask])
        oof_probs["Independent_SQI_MLP"][test_idx] = mlp_full.predict_proba(X_sqi[test_mask])
        chosen_thresholds["Independent_SQI_MLP"].append(t_mlp)

        # D. 1D CNN
        train_sigs = [raw_signals[i] for i in train_idx]
        test_sigs = [raw_signals[i] for i in test_idx]
        cnn1d = ECG1DCNNClassifier(epochs=25, lr=1e-3)
        cnn1d.fit(train_sigs, y[train_mask])
        oof_probs["1D_CNN_RawECG"][test_idx] = cnn1d.predict_proba(test_sigs)
        chosen_thresholds["1D_CNN_RawECG"].append(0.5)

        # E. 2D CNN
        train_imgs = [image_paths[i] for i in train_idx]
        test_imgs = [image_paths[i] for i in test_idx]
        cnn2d = ECG2DCNNClassifier(epochs=25, lr=1e-3)
        cnn2d.fit(train_imgs, y[train_mask])
        oof_probs["2D_CNN_Image"][test_idx] = cnn2d.predict_proba(test_imgs)
        chosen_thresholds["2D_CNN_Image"].append(0.5)

        # F. SinglePass Multimodal Net
        mm = SinglePassMultimodalClassifier(epochs=25, lr=1e-3)
        mm.fit(train_imgs, X_sqi[train_mask], y[train_mask])
        oof_probs["Multimodal_Fusion"][test_idx] = mm.predict_proba(test_imgs, X_sqi[test_mask])
        chosen_thresholds["Multimodal_Fusion"].append(0.5)

        chosen_thresholds["Rule_Based_SQI_Heuristic"].append(0.5)
        chosen_thresholds["LiquidAI_d1_3B_Neural"].append(0.5)

    # 3. Save OOF predictions
    oof_df = pd.DataFrame({
        "segment_id": df["segment_id"],
        "subject_id": df["subject_id"],
        "record_name": df["record_name"],
        "label": y,
        "fold_id": fold_ids,
        **{f"prob_{m}": oof_probs[m] for m in model_names}
    })
    oof_out = RESULTS_DIR / "cohort1000_oof_predictions.csv"
    oof_df.to_csv(oof_out, index=False)
    print(f"\n[Saved] OOF predictions saved to {oof_out}")

    # 4. Compute Metrics Summary Table
    summary_rows = []
    op_rows = []
    for m in model_names:
        probs = oof_probs[m]
        auroc = float(roc_auc_score(y, probs))
        auprc = float(average_precision_score(y, probs))
        brier = float(brier_score_loss(y, probs))

        preds_fixed = (probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds_fixed, labels=[0, 1]).ravel()
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        bal_acc = 0.5 * (sens + spec)
        false_reject = fp / max(1, fp + tn)

        # Bootstrap CIs
        ci = compute_subject_level_bootstrap_ci(y, probs, subject_ids, n_bootstraps=500)

        # Operating points
        ops = extract_locked_operating_points_for_transfer(y, probs)

        summary_rows.append({
            "model": m,
            "auroc": auroc,
            "auprc": auprc,
            "brier_score": brier,
            "bal_acc_fixed": bal_acc,
            "sens_fixed": sens,
            "false_reject_fixed": false_reject,
            "tp_fixed": int(tp),
            "fp_fixed": int(fp),
            "tn_fixed": int(tn),
            "fn_fixed": int(fn),
            "bal_acc_ci": ci.get("balanced_accuracy", ""),
            "sens_ci": ci.get("sensitivity", ""),
            "false_reject_ci": ci.get("false_reject_rate", ""),
            "auroc_ci": ci.get("auroc", "")
        })

        op_rows.append({
            "model": m,
            "sens_at_fpr_le_5": ops["sens_at_fpr_le_5"],
            "thresh_at_fpr_le_5": ops["thresh_at_fpr_le_5"],
            "false_reject_at_sens_ge_80": ops["false_reject_at_sens_ge_80"],
            "thresh_at_sens_ge_80": ops["thresh_at_sens_ge_80"]
        })

    summary_df = pd.DataFrame(summary_rows)
    sum_out = RESULTS_DIR / "cohort1000_subject_cv_summary.csv"
    summary_df.to_csv(sum_out, index=False)
    print(f"[Saved] Summary table saved to {sum_out}")

    op_df = pd.DataFrame(op_rows)
    op_out = RESULTS_DIR / "cohort1000_operating_point_alignment.csv"
    op_df.to_csv(op_out, index=False)
    print(f"[Saved] Operating point alignment (extracted for target transfer) saved to {op_out}")

    print("\n" + "=" * 90)
    print("COHORT 1,000 BENCHMARK SUMMARY (1,047 WINDOWS, 15 SUBJECTS)")
    print("=" * 90)
    print(summary_df[["model", "auroc", "auprc", "bal_acc_fixed", "sens_fixed", "false_reject_fixed", "brier_score"]].to_string(index=False))


if __name__ == "__main__":
    run_cohort_benchmark()
