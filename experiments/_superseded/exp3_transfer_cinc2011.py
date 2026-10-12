"""
Experiment 3: Independent Cross-Dataset Transfer Benchmark on CinC Challenge 2011 Set A
(Study Requirement #5.B)

- Models trained exclusively on BUT QDB development set.
- Evaluated on CinC 2011 Set A with ZERO threshold retuning and ZERO fine-tuning.
- Analyzes out-of-distribution transfer under altered quality criteria (lead placement & mobile recording).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR, RANDOM_SEED, CLASS_USABLE, CLASS_UNUSABLE
from data.butqdb_dataset import BUTQDBManager
from data.cinc2011_dataset import CinC2011Manager
from features.sqi_extractor import SQI_FEATURE_NAMES
from models.tabular_baselines import SQILogisticRegression, SQIGradientBoosting
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier
from evaluation.metrics import compute_usability_metrics

def run_transfer_cinc2011(max_transfer_records: int = 150, verbose: bool = True):
    print("=" * 80)
    print("EXPERIMENT 3: INDEPENDENT CROSS-DATASET TRANSFER TO CinC 2011 SET A")
    print("Zero-shot evaluation with fixed decision thresholds")
    print("=" * 80)
    
    # 1. Load BUT QDB (Source Domain Training Set)
    print("\n[Step 1] Loading BUT QDB training data...")
    but_mgr = BUTQDBManager()
    but_df = but_mgr.extract_segments(max_segments_per_subject=40, verbose=verbose)
    
    X_train_sqi = but_df[SQI_FEATURE_NAMES].values.astype(np.float32)
    y_train = but_df["label"].values.astype(int)
    train_imgs = but_df["image_path"].tolist()
    train_signals = np.array([np.load(p) for p in but_df["signal_path"]])
    
    # 2. Load CinC 2011 (Target Domain Evaluation Set)
    print(f"\n[Step 2] Loading CinC 2011 Set A target transfer set (N={max_transfer_records})...")
    cinc_mgr = CinC2011Manager()
    cinc_df = cinc_mgr.extract_segments(max_records=max_transfer_records, verbose=verbose)
    
    X_test_sqi = cinc_df[SQI_FEATURE_NAMES].values.astype(np.float32)
    y_test = cinc_df["label"].values.astype(int)
    test_imgs = cinc_df["image_path"].tolist()
    test_signals = np.array([np.load(p) for p in cinc_df["signal_path"]])
    
    print(f"Target set class balance: {cinc_df['label'].value_counts().to_dict()}")
    
    # 3. Train models on source domain
    print("\n[Step 3] Training candidate models on BUT QDB source data...")
    
    # Logistic Regression
    lr = SQILogisticRegression(random_state=RANDOM_SEED)
    lr.fit(X_train_sqi, y_train)
    
    # LightGBM
    lgb_clf = SQIGradientBoosting(random_state=RANDOM_SEED)
    lgb_clf.fit(X_train_sqi, y_train)
    
    # 1D CNN
    cnn1d = ECG1DCNNClassifier(epochs=15, batch_size=32)
    cnn1d.fit(train_signals, y_train)
    
    # 2D CNN
    cnn2d = ECG2DCNNClassifier(epochs=12, batch_size=32)
    cnn2d.fit(train_imgs, y_train)
    
    # Multimodal Decision Network
    mm_clf = SinglePassMultimodalClassifier(epochs=15, batch_size=32, mode="multimodal")
    mm_clf.fit(train_imgs, X_train_sqi, y_train)
    
    # 4. Predict on target domain (Zero-Shot Transfer)
    print("\n[Step 4] Evaluating out-of-distribution transfer performance on CinC 2011 Set A...")
    
    models_eval = {
        "Logistic_Regression_SQI": lr.predict_proba(X_test_sqi),
        "LightGBM_SQI": lgb_clf.predict_proba(X_test_sqi),
        "1D_CNN_RawECG": cnn1d.predict_proba(test_signals),
        "2D_CNN_Image": cnn2d.predict_proba(test_imgs),
        "Multimodal_Image_Plus_SQI": mm_clf.predict_proba(test_imgs, X_test_sqi),
        "SinglePass_Image_Only": mm_clf.predict_proba(test_imgs, X_test_sqi, override_mode="image_only"),
        "SinglePass_SQI_Only": mm_clf.predict_proba(test_imgs, X_test_sqi, override_mode="sqi_only"),
    }
    
    transfer_results = []
    for model_name, probs in models_eval.items():
        m = compute_usability_metrics(y_test, probs, threshold=0.5)
        m["model"] = model_name
        m["source_dataset"] = "BUT_QDB"
        m["target_dataset"] = "CinC_2011_SetA"
        transfer_results.append(m)
        
        print(f"\nModel: {model_name:25s}")
        print(f"  Balanced Accuracy:     {m['balanced_accuracy']*100:.1f}%")
        print(f"  Unusable Sensitivity:  {m['unusable_sensitivity']*100:.1f}%")
        print(f"  Usable False Reject:   {m['usable_false_reject_rate']*100:.1f}%")
        print(f"  Macro F1:              {m['macro_f1']:.3f}")
        print(f"  AUROC:                 {m['auroc']:.3f}")
        print(f"  AUPRC:                 {m['auprc']:.3f}")
        print(f"  Brier Score:           {m['brier_score']:.3f}")
        print(f"  ECE (Calibration):     {m['ece']:.3f}")
        
    df_res = pd.DataFrame(transfer_results)
    out_csv = RESULTS_DIR / "exp3_cinc2011_transfer_summary.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n[Completed] Cross-dataset transfer summary saved to {out_csv}")
    return df_res

if __name__ == "__main__":
    run_transfer_cinc2011()
