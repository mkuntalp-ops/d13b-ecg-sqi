"""
Comprehensive Evaluation Metrics for ECG Usability Assessment.
Strict adherence to study design:
- Positive class = Unusable (1)
- Primary metrics:
    * Unusable Sensitivity (TPR for unusable)
    * Usable False Reject Rate (FPR, 1 - Specificity)
    * Balanced Accuracy, Macro F1
    * AUROC, AUPRC
    * Brier Score & Expected Calibration Error (ECE)
    * Subject/Record-level 95% Bootstrap Confidence Intervals
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    balanced_accuracy_score,
    f1_score,
    brier_score_loss,
    confusion_matrix
)

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        bin_mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if i == 0:
            bin_mask = bin_mask | (y_prob == 0.0)
            
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            avg_confidence = np.mean(y_prob[bin_mask])
            avg_accuracy = np.mean(y_true[bin_mask])
            ece += (bin_size / n) * np.abs(avg_confidence - avg_accuracy)
            
    return float(ece)


def compute_usability_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes all clinical and statistical quality metrics.
    Positive Class = 1 (Unusable).
    Negative Class = 0 (Usable).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=float), nan=0.5)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion matrix:
    # tn: true usable, fp: false unusable (unnecessary rejection)
    # fn: false usable (missed unusable), tp: true unusable
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    unusable_sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    usable_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    usable_false_reject_rate = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    unusable_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    # AUROC & AUPRC
    try:
        if len(np.unique(y_true)) > 1:
            auroc = float(roc_auc_score(y_true, y_prob))
            precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
            auprc = float(auc(recalls, precisions))
        else:
            auroc = float(np.nan)
            auprc = float(np.nan)
    except Exception:
        auroc = float(np.nan)
        auprc = float(np.nan)
        
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob)
    
    return {
        "unusable_sensitivity": unusable_sensitivity,
        "usable_false_reject_rate": usable_false_reject_rate,
        "usable_specificity": usable_specificity,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "unusable_f1": unusable_f1,
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": brier,
        "ece": ece,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_samples": int(len(y_true))
    }


def compute_subject_level_bootstrap_ci(
    df: pd.DataFrame,
    pred_col: str = "prob_unusable",
    target_col: str = "label",
    subject_col: str = "subject_id",
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Computes 95% Confidence Intervals via Subject-Level Non-Parametric Bootstrap.
    Resamples whole subjects to preserve within-subject correlation.
    """
    rng = np.random.RandomState(seed)
    unique_subjects = df[subject_col].unique()
    n_subj = len(unique_subjects)
    
    boot_metrics: Dict[str, List[float]] = {
        "unusable_sensitivity": [],
        "usable_false_reject_rate": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "auroc": [],
        "auprc": [],
        "brier_score": []
    }
    
    for _ in range(n_bootstraps):
        sampled_subjs = rng.choice(unique_subjects, size=n_subj, replace=True)
        # Concatenate rows of sampled subjects
        sampled_dfs = [df[df[subject_col] == s] for s in sampled_subjs]
        boot_df = pd.concat(sampled_dfs, ignore_index=True)
        
        y_true = boot_df[target_col].values
        y_prob = boot_df[pred_col].values
        
        if len(np.unique(y_true)) < 2:
            continue
            
        m = compute_usability_metrics(y_true, y_prob)
        for k in boot_metrics:
            if not np.isnan(m[k]):
                boot_metrics[k].append(m[k])
                
    ci_results = {}
    for k, values in boot_metrics.items():
        if len(values) > 0:
            low = float(np.percentile(values, 2.5))
            high = float(np.percentile(values, 97.5))
            mean_val = float(np.mean(values))
            ci_results[k] = {"mean": mean_val, "ci_low": low, "ci_high": high}
        else:
            ci_results[k] = {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            
    return ci_results
