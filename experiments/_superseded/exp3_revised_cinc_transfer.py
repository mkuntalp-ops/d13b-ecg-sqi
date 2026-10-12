"""
Revised Experiment 3: Zero-Shot Cross-Dataset Transfer to PhysioNet CinC 2011 Set A
Methodological Standards:
- Models are trained on full BUT QDB development set.
- Thresholds are strictly locked from BUT QDB (fixed tau=0.5 and inner-tuned tau*).
- No post-hoc threshold tuning on CinC 2011 Set A test labels.
- Continuous metrics (AUROC, AUPRC) reported alongside fixed-threshold clinical metrics.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from config import RESULTS_DIR, TARGET_FS
from features.sqi_extractor import SQI_FEATURE_NAMES
from data.butqdb_dataset import BUTQDBManager
from data.cinc2011_dataset import CinC2011Manager
from models.tabular_baselines import SQILogisticRegression, SQIGradientBoosting
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier, IndependentSQIClassifier
from models.d1_decision_interface import D1DecisionModelWrapper

def run_revised_cinc_transfer(verbose: bool = True):
    print("=" * 80)
    print("REVISED EXPERIMENT 3: INDEPENDENT TRANSFER TO CINC 2011 SET A (LOCKED THRESHOLDS)")
    print("=" * 80)
    
    # 1. Load Source (BUT QDB) Data
    manifest_path = RESULTS_DIR / "manifest_butqdb_subject_folds.csv"
    but_df = pd.read_csv(manifest_path)
    X_train_sqi = np.nan_to_num(but_df[SQI_FEATURE_NAMES].values, nan=0.0)
    y_train = but_df["label"].values
    train_sigs = [np.load(p) for p in but_df["signal_path"]]
    train_imgs = but_df["image_path"].tolist()
    
    # 2. Load Target (CinC 2011 Set A) Data
    cinc_mgr = CinC2011Manager()
    cinc_df = cinc_mgr.extract_segments(max_records=150, verbose=verbose)
    X_test_sqi = np.nan_to_num(cinc_df[SQI_FEATURE_NAMES].values, nan=0.0)
    y_test = cinc_df["label"].values
    test_sigs = [np.load(p) for p in cinc_df["signal_path"]]
    test_imgs = cinc_df["image_path"].tolist()
    
    print(f"Transfer Target: {len(cinc_df)} records (75 Acceptable=0, 75 Unacceptable=1)")
    
    # Load inner-tuned thresholds from exp2 summary
    cv_sum_path = RESULTS_DIR / "exp2_revised_subject_cv_summary.csv"
    thresh_map = {}
    if cv_sum_path.exists():
        df_cv_sum = pd.read_csv(cv_sum_path)
        for _, r in df_cv_sum.iterrows():
            thresh_map[r["model"]] = float(r["mean_inner_threshold"])
            
    models = {
        "Logistic_Regression_SQI": SQILogisticRegression(),
        "LightGBM_SQI": SQIGradientBoosting(),
        "Independent_SQI_MLP": IndependentSQIClassifier(epochs=35),
        "1D_CNN_RawECG": ECG1DCNNClassifier(epochs=15, batch_size=32),
        "2D_CNN_Image": ECG2DCNNClassifier(epochs=15, batch_size=32),
        "Multimodal_Fusion": SinglePassMultimodalClassifier(epochs=15, batch_size=32, mode="multimodal"),
        "TestTime_Vision_Removal": None, # Evaluated on Multimodal
        "TestTime_SQI_Removal": None,
        "LiquidAI_d1_3B": D1DecisionModelWrapper(load_weights=False)
    }
    
    print("\nTraining models on full BUT QDB development set...")
    models["Logistic_Regression_SQI"].fit(X_train_sqi, y_train)
    models["LightGBM_SQI"].fit(X_train_sqi, y_train)
    models["Independent_SQI_MLP"].fit(X_train_sqi, y_train)
    models["1D_CNN_RawECG"].fit(train_sigs, y_train)
    models["2D_CNN_Image"].fit(train_imgs, y_train)
    models["Multimodal_Fusion"].fit(train_imgs, X_train_sqi, y_train)
    
    print("Evaluating zero-shot transfer on CinC 2011 Set A...")
    test_probs = {}
    test_probs["Logistic_Regression_SQI"] = models["Logistic_Regression_SQI"].predict_proba(X_test_sqi)
    test_probs["LightGBM_SQI"] = models["LightGBM_SQI"].predict_proba(X_test_sqi)
    test_probs["Independent_SQI_MLP"] = models["Independent_SQI_MLP"].predict_proba(X_test_sqi)
    test_probs["1D_CNN_RawECG"] = models["1D_CNN_RawECG"].predict_proba(test_sigs)
    test_probs["2D_CNN_Image"] = models["2D_CNN_Image"].predict_proba(test_imgs)
    test_probs["Multimodal_Fusion"] = models["Multimodal_Fusion"].predict_proba(test_imgs, X_test_sqi)
    test_probs["TestTime_Vision_Removal"] = models["Multimodal_Fusion"].predict_proba(test_imgs, X_test_sqi, override_mode="sqi_only")
    test_probs["TestTime_SQI_Removal"] = models["Multimodal_Fusion"].predict_proba(test_imgs, X_test_sqi, override_mode="image_only")
    
    p_d1 = []
    from PIL import Image
    for img_p, sqi_vec in zip(test_imgs, X_test_sqi):
        sqi_dict = dict(zip(SQI_FEATURE_NAMES, sqi_vec))
        with Image.open(img_p) as pil_img:
            dec = models["LiquidAI_d1_3B"].decide_single(pil_img, sqi_dict, modality="multimodal")
            p_d1.append(dec["prob_unusable"])
    test_probs["LiquidAI_d1_3B"] = np.array(p_d1, dtype=np.float32)
    
    summary_rows = []
    
    for m, probs in test_probs.items():
        auc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        
        # Fixed threshold (0.5)
        p_fixed = (probs >= 0.5).astype(int)
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_test, p_fixed, labels=[0, 1]).ravel()
        sens_f = tp_f / max(1, tp_f + fn_f)
        spec_f = tn_f / max(1, tn_f + fp_f)
        bal_acc_f = (sens_f + spec_f) / 2.0
        false_rej_f = fp_f / max(1, tn_f + fp_f)
        
        # Locked threshold from BUT QDB development
        tau_dev = thresh_map.get(m, 0.5)
        p_dev = (probs >= tau_dev).astype(int)
        tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_test, p_dev, labels=[0, 1]).ravel()
        sens_d = tp_d / max(1, tp_d + fn_d)
        spec_d = tn_d / max(1, tn_d + fp_d)
        bal_acc_d = (sens_d + spec_d) / 2.0
        false_rej_d = fp_d / max(1, tn_d + fp_d)
        
        row = {
            "model": m,
            "locked_tau_from_dev": tau_dev,
            "bal_acc_fixed_0.5": bal_acc_f,
            "sens_fixed": sens_f,
            "false_reject_fixed": false_rej_f,
            "tp_fixed": tp_f,
            "fp_fixed": fp_f,
            "tn_fixed": tn_f,
            "fn_fixed": fn_f,
            "bal_acc_locked_tau": bal_acc_d,
            "sens_locked_tau": sens_d,
            "false_reject_locked_tau": false_rej_d,
            "tp_locked": tp_d,
            "fp_locked": fp_d,
            "tn_locked": tn_d,
            "fn_locked": fn_d,
            "auroc": auc,
            "auprc": auprc,
            "brier_score": brier
        }
        summary_rows.append(row)
        
        print(f"\nModel: {m}")
        print(f"  AUROC:               {auc:.3f}, AUPRC: {auprc:.3f}, Brier: {brier:.3f}")
        print(f"  Fixed Thresh (0.5):  BalAcc={bal_acc_f*100:.1f}%, Sens={sens_f*100:.1f}%, FalseReject={false_rej_f*100:.1f}% (TP={tp_f}, FP={fp_f}, TN={tn_f}, FN={fn_f})")
        print(f"  Locked Dev Thresh:   tau={tau_dev:.2f} -> BalAcc={bal_acc_d*100:.1f}%, Sens={sens_d*100:.1f}%, FalseReject={false_rej_d*100:.1f}% (TP={tp_d}, FP={fp_d}, TN={tn_d}, FN={fn_d})")

    df_sum = pd.DataFrame(summary_rows)
    out_csv = RESULTS_DIR / "exp3_revised_cinc2011_transfer_summary.csv"
    df_sum.to_csv(out_csv, index=False)
    
    # Save transfer predictions
    col_name = "record_name" if "record_name" in cinc_df.columns else "segment_id"
    pred_df = cinc_df[[col_name, "label"]].copy()
    for m, probs in test_probs.items():
        pred_df[f"prob_{m}"] = probs
    pred_df.to_csv(RESULTS_DIR / "exp3_revised_cinc2011_predictions.csv", index=False)
    
    print(f"\n[Completed] Revised CinC transfer summary saved to {out_csv}")
    return df_sum

if __name__ == "__main__":
    run_revised_cinc_transfer()
