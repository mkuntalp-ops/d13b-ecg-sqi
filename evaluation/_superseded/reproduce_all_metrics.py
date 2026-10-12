"""
Standalone Reproducibility & Metric Audit Script
Recomputes all discrimination, calibration, fixed-threshold, and exploratory ROC operating-point metrics
directly from raw prediction files without data leakage or hidden heuristics.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix

RESULTS_DIR = Path("results")

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    fpr = fp / max(1, fp + tn)
    bal_acc = (sens + spec) / 2.0
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "sensitivity": sens, "specificity": spec, "fpr": fpr, "balanced_accuracy": bal_acc
    }

def audit_exploratory_roc_operating_points(y_true: np.ndarray, probs: np.ndarray):
    """
    Computes exact ROC operating points using all unique probability thresholds.
    Explicitly tracks whether Target 1 (FPR <= 5%) and Target 2 (Sens >= 80%) are reachable.
    """
    from sklearn.metrics import roc_curve
    fprs, senss, thresholds = roc_curve(y_true, probs, pos_label=1, drop_intermediate=False)

    # Target 1: FPR <= 5%
    le_5_idx = np.where(fprs <= 0.05)[0]
    if len(le_5_idx) > 0:
        best_sens = np.max(senss[le_5_idx])
        # Among thresholds achieving best_sens with FPR <= 0.05, take the lowest threshold
        # In roc_curve, thresholds are decreasing, fprs and senss are increasing.
        # We want the lowest threshold that gives this max sensitivity while keeping FPR <= 0.05.
        best_idx_candidates = np.where((fprs <= 0.05) & (senss == best_sens))[0]
        best_idx = best_idx_candidates[-1] # lowest threshold is at the end of candidates
        t1_thresh = float(thresholds[best_idx])
        t1_actual_fpr = float(fprs[best_idx])
        t1_status = "Achievable"
    else:
        best_sens = float("nan")
        t1_thresh = float("nan")
        t1_actual_fpr = float(np.min(fprs))
        t1_status = f"Unachievable (Min FPR: {t1_actual_fpr*100:.1f}%)"

    # Target 2: Sens >= 80%
    ge_80_idx = np.where(senss >= 0.80)[0]
    if len(ge_80_idx) > 0:
        best_fpr = np.min(fprs[ge_80_idx])
        best_idx_candidates = np.where((senss >= 0.80) & (fprs == best_fpr))[0]
        # Highest threshold that gives this minimum FPR while Sens >= 0.80
        best_idx = best_idx_candidates[0] # highest threshold is at the beginning of candidates
        t2_thresh = float(thresholds[best_idx])
        t2_actual_sens = float(senss[best_idx])
        t2_status = "Achievable"
    else:
        best_fpr = float("nan")
        t2_thresh = float("nan")
        t2_actual_sens = float(np.max(senss))
        t2_status = f"Unachievable (Max Sens: {t2_actual_sens*100:.1f}%)"

    return {
        "t1_status": t1_status,
        "t1_sens": best_sens,
        "t1_thresh": t1_thresh,
        "t1_actual_fpr": t1_actual_fpr,
        "t2_status": t2_status,
        "t2_fpr": best_fpr,
        "t2_thresh": t2_thresh,
        "t2_actual_sens": t2_actual_sens,
    }


def main():
    print("=" * 80)
    print("STANDALONE METRIC REPRODUCIBILITY AUDIT")
    print("=" * 80)

    # 1. BUT QDB Audit (1,047 windows)
    but_path = RESULTS_DIR / "cohort1000_consolidated_oof_predictions.csv"
    if not but_path.exists():
        print(f"Error: {but_path} not found.")
        sys.exit(1)
    
    but_df = pd.read_csv(but_path)
    y_but = but_df["label"].values
    print(f"\n[BUT QDB Development Cohort] Loaded {len(but_df)} samples. Prevalence: {np.mean(y_but)*100:.1f}% ({np.sum(y_but)} unusable, {len(y_but)-np.sum(y_but)} usable)")
    
    prob_cols_but = [c for c in but_df.columns if c.startswith("prob_")]
    but_summary = []
    but_op_rows = []

    for col in prob_cols_but:
        model_name = col.replace("prob_", "")
        probs = but_df[col].values
        auroc = roc_auc_score(y_but, probs)
        auprc = average_precision_score(y_but, probs)
        brier = brier_score_loss(y_but, probs)
        
        # Fixed threshold 0.50
        fixed_preds = (probs >= 0.50).astype(int)
        cm_fixed = compute_binary_metrics(y_but, fixed_preds)

        but_summary.append({
            "model": model_name,
            "auroc": round(auroc, 3),
            "auprc": round(auprc, 3),
            "brier": round(brier, 3),
            "cm_fixed": f"[{cm_fixed['tp']} / {cm_fixed['fp']} / {cm_fixed['tn']} / {cm_fixed['fn']}]",
            "sens_fixed": f"{cm_fixed['sensitivity']*100:.1f}%",
            "false_reject_fixed": f"{cm_fixed['fpr']*100:.1f}%",
            "bal_acc_fixed": f"{cm_fixed['balanced_accuracy']*100:.1f}%"
        })

        # Exploratory ROC operating points
        ops = audit_exploratory_roc_operating_points(y_but, probs)
        but_op_rows.append({
            "model": model_name,
            "sens_at_fpr_le_5": f"{ops['t1_sens']*100:.2f}%" if not np.isnan(ops['t1_sens']) else "N/A",
            "t1_actual_fpr": f"{ops['t1_actual_fpr']*100:.2f}%",
            "t1_thresh": f"{ops['t1_thresh']:.6f}" if not np.isnan(ops['t1_thresh']) else "N/A",
            "false_reject_at_sens_ge_80": f"{ops['t2_fpr']*100:.2f}%" if not np.isnan(ops['t2_fpr']) else "N/A",
            "t2_actual_sens": f"{ops['t2_actual_sens']*100:.2f}%",
            "t2_thresh": f"{ops['t2_thresh']:.6f}" if not np.isnan(ops['t2_thresh']) else "N/A",
            "t1_status": ops["t1_status"],
            "t2_status": ops["t2_status"]
        })

    print("\n--- BUT QDB Performance Summary (Fixed tau = 0.50) ---")
    print(pd.DataFrame(but_summary).to_string(index=False))

    print("\n--- BUT QDB Exploratory ROC Operating Points ---")
    print(pd.DataFrame(but_op_rows).to_string(index=False))

    # 2. CinC 2011 Set A Audit (150 records)
    cinc_path = RESULTS_DIR / "cinc2011_expanded_predictions.csv"
    if not cinc_path.exists():
        print(f"Error: {cinc_path} not found.")
        sys.exit(1)

    cinc_df = pd.read_csv(cinc_path)
    y_cinc = cinc_df["label"].values
    print(f"\n[CinC 2011 Set A Target Domain] Loaded {len(cinc_df)} records. Prevalence: {np.mean(y_cinc)*100:.1f}% ({np.sum(y_cinc)} unacceptable, {len(y_cinc)-np.sum(y_cinc)} acceptable)")

    prob_cols_cinc = [c for c in cinc_df.columns if c.startswith("prob_")]
    cinc_summary = []
    cinc_op_rows = []

    for col in prob_cols_cinc:
        model_name = col.replace("prob_", "")
        probs = cinc_df[col].values
        auroc = roc_auc_score(y_cinc, probs)
        auprc = average_precision_score(y_cinc, probs)
        brier = brier_score_loss(y_cinc, probs)

        # Fixed threshold 0.50
        fixed_preds = (probs >= 0.50).astype(int)
        cm_fixed = compute_binary_metrics(y_cinc, fixed_preds)

        cinc_summary.append({
            "model": model_name,
            "auroc": round(auroc, 3),
            "auprc": round(auprc, 3),
            "brier": round(brier, 3),
            "cm_fixed": f"[{cm_fixed['tp']} / {cm_fixed['fp']} / {cm_fixed['tn']} / {cm_fixed['fn']}]",
            "sens_fixed": f"{cm_fixed['sensitivity']*100:.1f}%",
            "false_reject_fixed": f"{cm_fixed['fpr']*100:.1f}%",
            "bal_acc_fixed": f"{cm_fixed['balanced_accuracy']*100:.1f}%"
        })

        # Exploratory ROC operating points
        ops = audit_exploratory_roc_operating_points(y_cinc, probs)
        cinc_op_rows.append({
            "model": model_name,
            "sens_at_fpr_le_5": f"{ops['t1_sens']*100:.2f}%" if not np.isnan(ops['t1_sens']) else "N/A",
            "t1_actual_fpr": f"{ops['t1_actual_fpr']*100:.2f}%",
            "t1_thresh": f"{ops['t1_thresh']:.6f}" if not np.isnan(ops['t1_thresh']) else "N/A",
            "t1_status": ops["t1_status"],
            "false_reject_at_sens_ge_80": f"{ops['t2_fpr']*100:.2f}%" if not np.isnan(ops['t2_fpr']) else "N/A",
            "t2_actual_sens": f"{ops['t2_actual_sens']*100:.2f}%",
            "t2_thresh": f"{ops['t2_thresh']:.6f}" if not np.isnan(ops['t2_thresh']) else "N/A",
            "t2_status": ops["t2_status"]
        })

    print("\n--- CinC 2011 Set A Transfer Summary (Fixed tau = 0.50) ---")
    print(pd.DataFrame(cinc_summary).to_string(index=False))

    print("\n--- CinC 2011 Set A Exploratory ROC Operating Points ---")
    print(pd.DataFrame(cinc_op_rows).to_string(index=False))

    # Save audited tables
    pd.DataFrame(but_summary).to_csv(RESULTS_DIR / "audited_butqdb_summary.csv", index=False)
    pd.DataFrame(but_op_rows).to_csv(RESULTS_DIR / "audited_butqdb_operating_points.csv", index=False)
    pd.DataFrame(cinc_summary).to_csv(RESULTS_DIR / "audited_cinc2011_summary.csv", index=False)
    pd.DataFrame(cinc_op_rows).to_csv(RESULTS_DIR / "audited_cinc2011_operating_points.csv", index=False)
    print("\n[Audit Complete] All 4 tables re-calculated and saved to results/audited_*.csv")

if __name__ == "__main__":
    main()
