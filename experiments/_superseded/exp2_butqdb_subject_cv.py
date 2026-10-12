"""
Experiment 2: Leakage-Free Subject-Independent Cross-Validation on BUT QDB
(Study Requirement #5.A & #4)

- 5-Fold GroupKFold stratified across 15 distinct subjects.
- All windows of a patient belong strictly to either train or test fold.
- Head-to-head comparison of:
    1. Logistic Regression (SQIs)
    2. Gradient Boosting / LightGBM (SQIs)
    3. 1D CNN (Raw waveform)
    4. 2D CNN (Waveform image)
    5. Multimodal Decision Model (Image + SQIs)
    6. Multimodal Ablation (Image Only)
    7. Multimodal Ablation (SQI Only)
- Outputs full metrics table and 95% subject-level bootstrap confidence intervals.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR, RANDOM_SEED, CLASS_USABLE, CLASS_UNUSABLE
from data.butqdb_dataset import BUTQDBManager
from features.sqi_extractor import SQI_FEATURE_NAMES, sqi_dict_to_vector
from models.tabular_baselines import SQILogisticRegression, SQIGradientBoosting
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier
from evaluation.metrics import compute_usability_metrics, compute_subject_level_bootstrap_ci

def run_butqdb_subject_cv(n_splits: int = 5, verbose: bool = True):
    print("=" * 80)
    print("EXPERIMENT 2: LEAKAGE-FREE SUBJECT-INDEPENDENT CROSS-VALIDATION ON BUT QDB")
    print("=" * 80)
    
    but_mgr = BUTQDBManager()
    df = but_mgr.extract_segments(max_segments_per_subject=40, verbose=verbose)
    
    print(f"Total extracted segments: {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()} -> {list(df['subject_id'].unique())}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Prepare data arrays
    X_sqi = df[SQI_FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].values.astype(int)
    groups = df["subject_id"].values
    image_paths = df["image_path"].tolist()
    
    # Load raw signals for 1D CNN
    raw_signals = np.array([np.load(p) for p in df["signal_path"]])
    
    gkf = GroupKFold(n_splits=n_splits)
    
    # Containers for out-of-fold predictions
    oof_predictions = {
        "Logistic_Regression_SQI": np.zeros(len(df), dtype=np.float32),
        "LightGBM_SQI": np.zeros(len(df), dtype=np.float32),
        "1D_CNN_RawECG": np.zeros(len(df), dtype=np.float32),
        "2D_CNN_Image": np.zeros(len(df), dtype=np.float32),
        "Multimodal_Image_Plus_SQI": np.zeros(len(df), dtype=np.float32),
        "SinglePass_Image_Only": np.zeros(len(df), dtype=np.float32),
        "SinglePass_SQI_Only": np.zeros(len(df), dtype=np.float32),
    }
    
    print(f"\nStarting {n_splits}-Fold GroupKFold Cross-Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_sqi, y, groups=groups), 1):
        train_subjs = np.unique(groups[train_idx])
        test_subjs = np.unique(groups[test_idx])
        print(f"\n--- Fold {fold}/{n_splits} ---")
        print(f"  Train subjects ({len(train_subjs)}): {list(train_subjs)}")
        print(f"  Test subjects ({len(test_subjs)}): {list(test_subjs)}")
        print(f"  Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        # 1. Logistic Regression
        lr = SQILogisticRegression(random_state=RANDOM_SEED)
        lr.fit(X_sqi[train_idx], y[train_idx])
        oof_predictions["Logistic_Regression_SQI"][test_idx] = lr.predict_proba(X_sqi[test_idx])
        
        # 2. LightGBM
        lgb_clf = SQIGradientBoosting(random_state=RANDOM_SEED)
        lgb_clf.fit(X_sqi[train_idx], y[train_idx])
        oof_predictions["LightGBM_SQI"][test_idx] = lgb_clf.predict_proba(X_sqi[test_idx])
        
        # 3. 1D CNN (Raw signal)
        cnn1d = ECG1DCNNClassifier(epochs=15, batch_size=32)
        cnn1d.fit(raw_signals[train_idx], y[train_idx])
        oof_predictions["1D_CNN_RawECG"][test_idx] = cnn1d.predict_proba(raw_signals[test_idx])
        
        # 4. 2D CNN (Image)
        train_imgs = [image_paths[i] for i in train_idx]
        test_imgs = [image_paths[i] for i in test_idx]
        cnn2d = ECG2DCNNClassifier(epochs=12, batch_size=32)
        cnn2d.fit(train_imgs, y[train_idx])
        oof_predictions["2D_CNN_Image"][test_idx] = cnn2d.predict_proba(test_imgs)
        
        # 5. Multimodal (Image + SQI)
        mm_clf = SinglePassMultimodalClassifier(epochs=15, batch_size=32, mode="multimodal")
        mm_clf.fit(train_imgs, X_sqi[train_idx], y[train_idx])
        oof_predictions["Multimodal_Image_Plus_SQI"][test_idx] = mm_clf.predict_proba(test_imgs, X_sqi[test_idx])
        
        # 6 & 7. Multimodal Ablations (evaluated with image_only and sqi_only forward pass)
        oof_predictions["SinglePass_Image_Only"][test_idx] = mm_clf.predict_proba(test_imgs, X_sqi[test_idx], override_mode="image_only")
        oof_predictions["SinglePass_SQI_Only"][test_idx] = mm_clf.predict_proba(test_imgs, X_sqi[test_idx], override_mode="sqi_only")

    # Compute overall out-of-fold metrics and subject-level bootstrap CIs
    print("\n" + "=" * 80)
    print("OUT-OF-FOLD PERFORMANCE EVALUATION (SUBJECT-INDEPENDENT)")
    print("=" * 80)
    
    metrics_summary = []
    
    for model_name, probs in oof_predictions.items():
        df_model = df.copy()
        df_model["prob_unusable"] = probs
        
        m = compute_usability_metrics(y, probs)
        m["model"] = model_name
        
        # Subject-level bootstrap CI
        ci = compute_subject_level_bootstrap_ci(df_model, pred_col="prob_unusable", target_col="label", subject_col="subject_id")
        
        m["bal_acc_ci"] = f"{m['balanced_accuracy']*100:.1f}% [{ci['balanced_accuracy']['ci_low']*100:.1f} - {ci['balanced_accuracy']['ci_high']*100:.1f}%]"
        m["sens_ci"] = f"{m['unusable_sensitivity']*100:.1f}% [{ci['unusable_sensitivity']['ci_low']*100:.1f} - {ci['unusable_sensitivity']['ci_high']*100:.1f}%]"
        m["false_reject_ci"] = f"{m['usable_false_reject_rate']*100:.1f}% [{ci['usable_false_reject_rate']['ci_low']*100:.1f} - {ci['usable_false_reject_rate']['ci_high']*100:.1f}%]"
        m["auroc_ci"] = f"{m['auroc']:.3f} [{ci['auroc']['ci_low']:.3f} - {ci['auroc']['ci_high']:.3f}]"
        
        metrics_summary.append(m)
        
        print(f"\nModel: {model_name}")
        print(f"  Balanced Accuracy:     {m['bal_acc_ci']}")
        print(f"  Unusable Sensitivity:  {m['sens_ci']}")
        print(f"  Usable False Reject:   {m['false_reject_ci']}")
        print(f"  Macro F1:              {m['macro_f1']:.3f}")
        print(f"  AUROC:                 {m['auroc_ci']}")
        print(f"  AUPRC:                 {m['auprc']:.3f}")
        print(f"  Brier Score:           {m['brier_score']:.3f}")
        print(f"  ECE (Calibration):     {m['ece']:.3f}")

    df_summary = pd.DataFrame(metrics_summary)
    out_path = RESULTS_DIR / "exp2_butqdb_subject_cv_summary.csv"
    df_summary.to_csv(out_path, index=False)
    
    # Save OOF predictions
    oof_df = df[["segment_id", "subject_id", "record_name", "label"]].copy()
    for m_name, probs in oof_predictions.items():
        oof_df[f"prob_{m_name}"] = probs
    oof_df.to_csv(RESULTS_DIR / "exp2_oof_predictions.csv", index=False)
    
    print(f"\n[Completed] Subject-independent CV results saved to {out_path}")
    return df_summary

if __name__ == "__main__":
    run_butqdb_subject_cv()
