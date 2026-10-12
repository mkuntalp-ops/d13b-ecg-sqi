"""
Consolidated Metrics Generator from Raw Verified Predictions
Strict Reproducibility Standards:
- Merges raw predictions with manifest strictly on 'segment_id' (validate='one_to_one').
- Zero manual patching or duplicated rows.
- Computes:
  * AUROC with 95% subject-level bootstrap CI
  * AUPRC
  * Brier Score
  * Fixed threshold (tau=0.50) confusion matrix & balanced accuracy
  * Operating point alignments:
    - Target 1: Sensitivity for Unusable signals at False Reject Rate <= 5%
    - Target 2: False Reject Rate at Unusable Sensitivity >= 80%
- Saves:
  * results/cohort1000_consolidated_oof_predictions.csv
  * results/cohort1000_consolidated_summary.csv
  * results/cohort1000_consolidated_operating_points.csv
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR
from evaluation.metrics import compute_subject_level_bootstrap_ci


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
                thresh_at_low_fpr = float(t)

        if sens >= 0.80:
            if fpr <= best_fpr_at_high_sens:
                best_fpr_at_high_sens = fpr
                thresh_at_high_sens = float(t)

    return {
        "sens_at_fpr_le_5": best_sens_at_low_fpr,
        "thresh_at_fpr_le_5": thresh_at_low_fpr,
        "false_reject_at_sens_ge_80": best_fpr_at_high_sens,
        "thresh_at_sens_ge_80": thresh_at_high_sens
    }


def compute_consolidated_benchmark():
    print("=" * 80)
    print("CONSOLIDATING 1,000-COHORT BENCHMARK METRICS FROM RAW AUDIT PREDICTIONS")
    print("=" * 80)

    manifest_path = RESULTS_DIR / "manifest_butqdb_1000_cohort.csv"
    manifest = pd.read_csv(manifest_path)
    n = len(manifest)
    print(f"Loaded manifest: {n} segments across {manifest['subject_id'].nunique()} subjects.")

    sup_path = RESULTS_DIR / "raw_supervised_cohort1000_oof_predictions.csv"
    d1_path = RESULTS_DIR / "raw_d1_neural_cohort1000_predictions.csv"

    if not sup_path.exists():
        raise FileNotFoundError(f"Missing supervised predictions: {sup_path}")
    if not d1_path.exists():
        raise FileNotFoundError(f"Missing d1 neural predictions: {d1_path}")

    sup_df = pd.read_csv(sup_path)
    d1_df = pd.read_csv(d1_path)

    # Validate one-to-one merges strictly on segment_id
    m1 = manifest.merge(sup_df[["segment_id"] + [c for c in sup_df.columns if c.startswith("prob_")]],
                        on="segment_id", how="inner", validate="one_to_one")
    assert len(m1) == n, f"Supervised predictions merge failed: {len(m1)} vs {n}"

    m2 = m1.merge(d1_df[["segment_id", "prob_unusable"]].rename(columns={"prob_unusable": "prob_LiquidAI_d1_3B_Neural"}),
                  on="segment_id", how="inner", validate="one_to_one")
    assert len(m2) == n, f"d1 predictions merge failed: {len(m2)} vs {n}"

    # Save consolidated OOF predictions
    oof_out = RESULTS_DIR / "cohort1000_consolidated_oof_predictions.csv"
    m2.to_csv(oof_out, index=False)
    print(f"[Saved] Consolidated OOF predictions saved to {oof_out}")

    y = m2["label"].values
    subject_ids = m2["subject_id"].values
    prob_cols = [c for c in m2.columns if c.startswith("prob_")]

    summary_rows = []
    op_rows = []

    for col in prob_cols:
        model_name = col.replace("prob_", "")
        probs = m2[col].values

        auroc = float(roc_auc_score(y, probs))
        auprc = float(average_precision_score(y, probs))
        brier = float(brier_score_loss(y, probs))

        preds_50 = (probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds_50, labels=[0, 1]).ravel()
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        bal_acc = 0.5 * (sens + spec)
        false_reject = fp / max(1, fp + tn)

        # Bootstrap CIs
        sub_df = pd.DataFrame({"label": y, col: probs, "subject_id": subject_ids})
        ci = compute_subject_level_bootstrap_ci(sub_df, pred_col=col, target_col="label", subject_col="subject_id", n_bootstraps=500)

        def fmt_ci(k, pct=False):
            if k in ci and not np.isnan(ci[k]["mean"]):
                if pct:
                    return f"{ci[k]['mean']*100:.1f}% [{ci[k]['ci_low']*100:.1f} - {ci[k]['ci_high']*100:.1f}%]"
                return f"{ci[k]['mean']:.3f} [{ci[k]['ci_low']:.3f} - {ci[k]['ci_high']:.3f}]"
            return ""

        # Operating point alignment
        ops = extract_locked_operating_points_for_transfer(y, probs)

        summary_rows.append({
            "model": model_name,
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
            "bal_acc_ci": fmt_ci("balanced_accuracy", pct=True),
            "sens_ci": fmt_ci("unusable_sensitivity", pct=True),
            "false_reject_ci": fmt_ci("usable_false_reject_rate", pct=True),
            "auroc_ci": fmt_ci("auroc", pct=False)
        })

        op_rows.append({
            "model": model_name,
            "sens_at_fpr_le_5": ops["sens_at_fpr_le_5"],
            "thresh_at_fpr_le_5": ops["thresh_at_fpr_le_5"],
            "false_reject_at_sens_ge_80": ops["false_reject_at_sens_ge_80"],
            "thresh_at_sens_ge_80": ops["thresh_at_sens_ge_80"]
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_out = RESULTS_DIR / "cohort1000_consolidated_summary.csv"
    sum_df.to_csv(sum_out, index=False)
    print(f"[Saved] Consolidated summary saved to {sum_out}")

    op_df = pd.DataFrame(op_rows)
    op_out = RESULTS_DIR / "cohort1000_consolidated_operating_points.csv"
    op_df.to_csv(op_out, index=False)
    print(f"[Saved] Consolidated operating points (extracted for target transfer) saved to {op_out}")

    print("\n" + "=" * 90)
    print("CONSOLIDATED 1,000-COHORT BENCHMARK SUMMARY (1,047 SAMPLES)")
    print("=" * 90)
    print(sum_df[["model", "auroc", "auprc", "bal_acc_fixed", "sens_fixed", "false_reject_fixed", "brier_score"]].to_string(index=False))

    print("\n" + "=" * 90)
    print("EXTRACTED OPERATING POINTS FOR CROSS-DATASET TRANSFER (CALIBRATION ONLY, NOT TEST METRICS)")
    print("=" * 90)
    print(op_df.to_string(index=False))


if __name__ == "__main__":
    compute_consolidated_benchmark()
