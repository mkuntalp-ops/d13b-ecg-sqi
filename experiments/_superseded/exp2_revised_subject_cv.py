"""
Revised Experiment 2: Leakage-Free Subject-Independent Cross-Validation on BUT QDB
Key Methodological Standards (Audit Revisions):
1. Inner-validation threshold selection: for the SQI models (Logistic Regression, LightGBM, Independent SQI MLP)
   tau* is chosen on an inner subject-independent validation split and applied frozen to the outer test fold.
   The CNNs and the multimodal net use a fixed tau = 0.5 (their "tuned" columns equal the fixed-threshold columns).
2. Independent unimodal baselines:
   - Independent SQI MLP
   - Independent 2D CNN
   - 1D CNN
   - Logistic Regression
   - LightGBM
3. Test-time modality removal kept as a distinct, explicitly labeled degradation experiment.
4. Evaluation of the real d1-3B decision model (RealD1DecisionModel, fail-closed: no heuristic fallback)
   on the exact same benchmark folds. The model is zero-shot, so its predictions do not depend on the fold.
5. Exact binomial upper bound reported for 0-count bootstrap outcomes.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from config import RESULTS_DIR, TARGET_FS
from features.sqi_extractor import SQI_FEATURE_NAMES
from models.tabular_baselines import SQILogisticRegression, SQIGradientBoosting
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier, IndependentSQIClassifier
from models.d1_real_neural_model import RealD1DecisionModel
from evaluation.metrics import compute_usability_metrics, compute_subject_level_bootstrap_ci

def select_threshold_on_inner_val(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """
    Selects threshold maximizing Youden's Index (Sensitivity + Specificity - 1)
    on inner validation set. If only one class present or degenerate, defaults to 0.5.
    """
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

def run_revised_subject_cv(verbose: bool = True):
    print("=" * 80)
    print("REVISED EXPERIMENT 2: SUBJECT-INDEPENDENT 5-FOLD CV (AUDITED PIPELINE)")
    print("=" * 80)
    
    manifest_path = RESULTS_DIR / "manifest_butqdb_subject_folds.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(df)} samples across {df['subject_id'].nunique()} subjects.")
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
        "TestTime_Vision_Removal",
        "TestTime_SQI_Removal",
        "LiquidAI_d1_3B"
    ]
    
    oof_probs = {m: np.zeros(len(df), dtype=np.float32) for m in model_names}
    oof_preds_fixed = {m: np.zeros(len(df), dtype=int) for m in model_names}
    oof_preds_tuned = {m: np.zeros(len(df), dtype=int) for m in model_names}
    chosen_thresholds = {m: [] for m in model_names}
    
    # Outer 5-fold loop
    for fold in range(5):
        test_mask = (fold_ids == fold)
        train_mask = ~test_mask
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        print(f"\n--- Outer Fold {fold} ---")
        print(f"Train subjects: {np.unique(subject_ids[train_idx])} (n={len(train_idx)})")
        print(f"Test subjects:  {np.unique(subject_ids[test_idx])} (n={len(test_idx)}, pos={sum(y[test_idx])}, neg={sum(y[test_idx]==0)})")
        
        # Inner validation split within train_idx
        inner_gkf = GroupKFold(n_splits=3)
        inner_tr_sub, inner_val_sub = next(inner_gkf.split(train_idx, groups=subject_ids[train_idx]))
        inner_tr_idx = train_idx[inner_tr_sub]
        inner_val_idx = train_idx[inner_val_sub]
        
        # Train & tune candidate models
        # 1. Logistic Regression
        lr = SQILogisticRegression()
        lr.fit(X_sqi[inner_tr_idx], y[inner_tr_idx])
        p_val_lr = lr.predict_proba(X_sqi[inner_val_idx])
        t_lr = select_threshold_on_inner_val(y[inner_val_idx], p_val_lr)
        # Refit on full outer train
        lr.fit(X_sqi[train_idx], y[train_idx])
        p_test_lr = lr.predict_proba(X_sqi[test_idx])
        oof_probs["Logistic_Regression_SQI"][test_idx] = p_test_lr
        oof_preds_fixed["Logistic_Regression_SQI"][test_idx] = (p_test_lr >= 0.5).astype(int)
        oof_preds_tuned["Logistic_Regression_SQI"][test_idx] = (p_test_lr >= t_lr).astype(int)
        chosen_thresholds["Logistic_Regression_SQI"].append(t_lr)
        
        # 2. LightGBM
        lgb = SQIGradientBoosting()
        lgb.fit(X_sqi[inner_tr_idx], y[inner_tr_idx])
        p_val_lgb = lgb.predict_proba(X_sqi[inner_val_idx])
        t_lgb = select_threshold_on_inner_val(y[inner_val_idx], p_val_lgb)
        lgb.fit(X_sqi[train_idx], y[train_idx])
        p_test_lgb = lgb.predict_proba(X_sqi[test_idx])
        oof_probs["LightGBM_SQI"][test_idx] = p_test_lgb
        oof_preds_fixed["LightGBM_SQI"][test_idx] = (p_test_lgb >= 0.5).astype(int)
        oof_preds_tuned["LightGBM_SQI"][test_idx] = (p_test_lgb >= t_lgb).astype(int)
        chosen_thresholds["LightGBM_SQI"].append(t_lgb)
        
        # 3. Independent SQI MLP
        sqi_mlp = IndependentSQIClassifier(epochs=35)
        sqi_mlp.fit(X_sqi[inner_tr_idx], y[inner_tr_idx])
        p_val_mlp = sqi_mlp.predict_proba(X_sqi[inner_val_idx])
        t_mlp = select_threshold_on_inner_val(y[inner_val_idx], p_val_mlp)
        sqi_mlp.fit(X_sqi[train_idx], y[train_idx])
        p_test_mlp = sqi_mlp.predict_proba(X_sqi[test_idx])
        oof_probs["Independent_SQI_MLP"][test_idx] = p_test_mlp
        oof_preds_fixed["Independent_SQI_MLP"][test_idx] = (p_test_mlp >= 0.5).astype(int)
        oof_preds_tuned["Independent_SQI_MLP"][test_idx] = (p_test_mlp >= t_mlp).astype(int)
        chosen_thresholds["Independent_SQI_MLP"].append(t_mlp)
        
        # 4. 1D CNN
        train_sigs = [raw_signals[i] for i in train_idx]
        test_sigs = [raw_signals[i] for i in test_idx]
        cnn1d = ECG1DCNNClassifier(epochs=15, batch_size=32)
        cnn1d.fit(train_sigs, y[train_idx])
        p_test_1d = cnn1d.predict_proba(test_sigs)
        t_1d = 0.5 # Default stable threshold
        oof_probs["1D_CNN_RawECG"][test_idx] = p_test_1d
        oof_preds_fixed["1D_CNN_RawECG"][test_idx] = (p_test_1d >= 0.5).astype(int)
        oof_preds_tuned["1D_CNN_RawECG"][test_idx] = (p_test_1d >= t_1d).astype(int)
        chosen_thresholds["1D_CNN_RawECG"].append(t_1d)
        
        # 5. 2D CNN
        train_imgs = [image_paths[i] for i in train_idx]
        test_imgs = [image_paths[i] for i in test_idx]
        cnn2d = ECG2DCNNClassifier(epochs=15, batch_size=32)
        cnn2d.fit(train_imgs, y[train_idx])
        p_test_2d = cnn2d.predict_proba(test_imgs)
        t_2d = 0.5
        oof_probs["2D_CNN_Image"][test_idx] = p_test_2d
        oof_preds_fixed["2D_CNN_Image"][test_idx] = (p_test_2d >= 0.5).astype(int)
        oof_preds_tuned["2D_CNN_Image"][test_idx] = (p_test_2d >= t_2d).astype(int)
        chosen_thresholds["2D_CNN_Image"].append(t_2d)
        
        # Save CNN2D checkpoint
        ckpt_dir = config.CACHE_DIR / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(cnn2d.net.state_dict(), ckpt_dir / f"cnn2d_fold_{fold}.pt")
        
        # 6. Multimodal Decision Net
        mm_clf = SinglePassMultimodalClassifier(epochs=15, batch_size=32, mode="multimodal")
        mm_clf.fit(train_imgs, X_sqi[train_idx], y[train_idx])
        p_test_mm = mm_clf.predict_proba(test_imgs, X_sqi[test_idx])
        t_mm = 0.5
        oof_probs["Multimodal_Fusion"][test_idx] = p_test_mm
        oof_preds_fixed["Multimodal_Fusion"][test_idx] = (p_test_mm >= 0.5).astype(int)
        oof_preds_tuned["Multimodal_Fusion"][test_idx] = (p_test_mm >= t_mm).astype(int)
        chosen_thresholds["Multimodal_Fusion"].append(t_mm)
        
        # Save Multimodal checkpoint
        torch.save(mm_clf.net.state_dict(), ckpt_dir / f"mm_fold_{fold}.pt")
        
        # 7 & 8: Test-Time Modality Removal
        p_test_no_vis = mm_clf.predict_proba(test_imgs, X_sqi[test_idx], override_mode="sqi_only")
        p_test_no_sqi = mm_clf.predict_proba(test_imgs, X_sqi[test_idx], override_mode="image_only")
        oof_probs["TestTime_Vision_Removal"][test_idx] = p_test_no_vis
        oof_preds_fixed["TestTime_Vision_Removal"][test_idx] = (p_test_no_vis >= 0.5).astype(int)
        oof_preds_tuned["TestTime_Vision_Removal"][test_idx] = (p_test_no_vis >= t_mm).astype(int)
        
        oof_probs["TestTime_SQI_Removal"][test_idx] = p_test_no_sqi
        oof_preds_fixed["TestTime_SQI_Removal"][test_idx] = (p_test_no_sqi >= 0.5).astype(int)
        oof_preds_tuned["TestTime_SQI_Removal"][test_idx] = (p_test_no_sqi >= t_mm).astype(int)
        
        # 9. Liquid AI d1-3B Decision Model (real weights; raises instead of falling back to a heuristic)
        d1 = RealD1DecisionModel()
        p_d1 = []
        for img_p, sqi_vec in zip(test_imgs, X_sqi[test_idx]):
            from PIL import Image
            sqi_dict = dict(zip(SQI_FEATURE_NAMES, sqi_vec))
            with Image.open(img_p) as pil_img:
                dec = d1.decide_single(pil_img, sqi_dict, modality="multimodal", segment_id=img_p)
                p_d1.append(dec["prob_unusable"])
        p_test_d1 = np.array(p_d1, dtype=np.float32)
        oof_probs["LiquidAI_d1_3B"][test_idx] = p_test_d1
        oof_preds_fixed["LiquidAI_d1_3B"][test_idx] = (p_test_d1 >= 0.5).astype(int)
        oof_preds_tuned["LiquidAI_d1_3B"][test_idx] = (p_test_d1 >= 0.5).astype(int)
        chosen_thresholds["LiquidAI_d1_3B"].append(0.5)

    print("\n" + "=" * 80)
    print("CALCULATING AUDITED OUT-OF-FOLD METRICS (FIXED & INNER-TUNED THRESHOLDS)")
    print("=" * 80)
    
    summary_rows = []
    
    for m in model_names:
        probs = oof_probs[m]
        p_fixed = oof_preds_fixed[m]
        p_tuned = oof_preds_tuned[m]
        
        auc = roc_auc_score(y, probs)
        auprc = average_precision_score(y, probs)
        brier = brier_score_loss(y, probs)
        
        # Confusion matrix for fixed threshold (0.5)
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(y, p_fixed, labels=[0, 1]).ravel()
        sens_f = tp_f / max(1, tp_f + fn_f)
        spec_f = tn_f / max(1, tn_f + fp_f)
        bal_acc_f = (sens_f + spec_f) / 2.0
        false_rej_f = fp_f / max(1, tn_f + fp_f)
        
        # Confusion matrix for inner-tuned threshold
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y, p_tuned, labels=[0, 1]).ravel()
        sens_t = tp_t / max(1, tp_t + fn_t)
        spec_t = tn_t / max(1, tn_t + fp_t)
        bal_acc_t = (sens_t + spec_t) / 2.0
        false_rej_t = fp_t / max(1, tn_t + fp_t)
        
        # Bootstrap CI on subject-level
        df_mod = df.copy()
        df_mod["prob"] = probs
        ci = compute_subject_level_bootstrap_ci(df_mod, pred_col="prob", target_col="label", subject_col="subject_id")
        
        # Exact one-sided 95% binomial upper bound for a 0-count false-rejection outcome: 1 - 0.05**(1/n_usable)
        n_usable = int(tn_f + fp_f)
        exact_ub = 1.0 - 0.05 ** (1.0 / n_usable) if n_usable > 0 else float("nan")
        exact_ub_str = f" (exact 95% UB: {exact_ub*100:.1f}%)" if fp_f == 0 else ""
        
        row = {
            "model": m,
            "mean_inner_threshold": np.mean(chosen_thresholds.get(m, [0.5])),
            "bal_acc_fixed": bal_acc_f,
            "sens_fixed": sens_f,
            "false_reject_fixed": false_rej_f,
            "tp_fixed": tp_f,
            "fp_fixed": fp_f,
            "tn_fixed": tn_f,
            "fn_fixed": fn_f,
            "bal_acc_tuned": bal_acc_t,
            "sens_tuned": sens_t,
            "false_reject_tuned": false_rej_t,
            "auroc": auc,
            "auprc": auprc,
            "brier_score": brier,
            "bal_acc_ci": f"{bal_acc_f*100:.1f}% [{ci['balanced_accuracy']['ci_low']*100:.1f} - {ci['balanced_accuracy']['ci_high']*100:.1f}%]",
            "sens_ci": f"{sens_f*100:.1f}% [{ci['unusable_sensitivity']['ci_low']*100:.1f} - {ci['unusable_sensitivity']['ci_high']*100:.1f}%]",
            "false_reject_ci": f"{false_rej_f*100:.1f}% [{ci['usable_false_reject_rate']['ci_low']*100:.1f} - {ci['usable_false_reject_rate']['ci_high']*100:.1f}%]{exact_ub_str}",
            "auroc_ci": f"{auc:.3f} [{ci['auroc']['ci_low']:.3f} - {ci['auroc']['ci_high']:.3f}]"
        }
        summary_rows.append(row)
        
        print(f"\nModel: {m}")
        print(f"  AUROC:               {row['auroc_ci']}")
        print(f"  Fixed Thresh (0.5):  BalAcc={bal_acc_f*100:.1f}%, Sens={sens_f*100:.1f}%, FalseReject={false_rej_f*100:.1f}% (TP={tp_f}, FP={fp_f}, TN={tn_f}, FN={fn_f})")
        print(f"  Inner-Tuned Thresh:  BalAcc={bal_acc_t*100:.1f}%, Sens={sens_t*100:.1f}%, FalseReject={false_rej_t*100:.1f}%")
        print(f"  Brier Score:         {brier:.3f}")
        
    df_summary = pd.DataFrame(summary_rows)
    out_csv = RESULTS_DIR / "exp2_revised_subject_cv_summary.csv"
    df_summary.to_csv(out_csv, index=False)
    
    # Save revised OOF predictions
    oof_df = df[["segment_id", "subject_id", "record_name", "label", "fold_id"]].copy()
    for m in model_names:
        oof_df[f"prob_{m}"] = oof_probs[m]
        oof_df[f"pred_fixed_{m}"] = oof_preds_fixed[m]
        oof_df[f"pred_tuned_{m}"] = oof_preds_tuned[m]
    oof_df.to_csv(RESULTS_DIR / "exp2_revised_oof_predictions.csv", index=False)
    
    print(f"\n[Completed] Audited subject-independent CV saved to {out_csv}")
    return df_summary

if __name__ == "__main__":
    run_revised_subject_cv()
