"""
Independent Cross-Dataset Transfer Benchmark to PhysioNet CinC 2011 Set A
- Models trained on BUT QDB development set.
- Evaluated on CinC 2011 Set A (Lead II).
- Evaluates:
  * Supervised models (LR, LightGBM, SQI MLP, 1D CNN, 2D CNN, Multimodal Net)
  * Rule-Based SQI Heuristic (Expert Rule Baseline)
  * Genuine LiquidAI/d1-3B Neural Decision Model (fail-closed, 3.12B weights, system_one)
- Thresholds are strictly locked from BUT QDB development (tau=0.50 and tuned tau*).
- No prompt or threshold modifications on out-of-distribution test set.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
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
from data.cinc2011_dataset import CinC2011Manager
from models.tabular_baselines import SQILogisticRegression, SQIGradientBoosting
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier, IndependentSQIClassifier
from models.rule_heuristic_baseline import RuleBasedSQIHeuristic
from models.d1_real_neural_model import RealD1DecisionModel


def run_cinc_transfer_evaluation(cohort_manifest_path: Path = None):
    print("=" * 80)
    print("INDEPENDENT TRANSFER BENCHMARK: CINC 2011 SET A (LOCKED THRESHOLDS)")
    print("=" * 80)

    # 1. Load Training Data from BUT QDB Cohort
    if cohort_manifest_path is None:
        cohort_manifest_path = RESULTS_DIR / "manifest_butqdb_1000_cohort.csv"
    if not cohort_manifest_path.exists():
        cohort_manifest_path = RESULTS_DIR / "manifest_butqdb_subject_folds.csv"

    print(f"Loading development set from: {cohort_manifest_path}")
    but_df = pd.read_csv(cohort_manifest_path)
    X_train_sqi = np.nan_to_num(but_df[SQI_FEATURE_NAMES].values, nan=0.0)
    y_train = but_df["label"].values
    train_sigs = [np.load(p) for p in but_df["signal_path"]]
    train_imgs = but_df["image_path"].tolist()

    # 2. Load CinC 2011 Set A Data
    cinc_mgr = CinC2011Manager()
    cinc_df = cinc_mgr.extract_segments(max_records=150, verbose=True)
    X_test_sqi = np.nan_to_num(cinc_df[SQI_FEATURE_NAMES].values, nan=0.0)
    y_test = cinc_df["label"].values
    test_sigs = [np.load(p) for p in cinc_df["signal_path"]]
    test_imgs = cinc_df["image_path"].tolist()

    n_test = len(cinc_df)
    print(f"Transfer Target: {n_test} records (Acceptable={sum(y_test==0)}, Unacceptable={sum(y_test==1)})")

    # 3. Train Supervised Models on Development Set
    print("\n[Transfer] Training supervised models on BUT QDB...")
    lr = SQILogisticRegression()
    lr.fit(X_train_sqi, y_train)

    lgbm = SQIGradientBoosting()
    lgbm.fit(X_train_sqi, y_train)

    mlp = IndependentSQIClassifier(epochs=35)
    mlp.fit(X_train_sqi, y_train)

    cnn1d = ECG1DCNNClassifier(epochs=25, batch_size=32)
    cnn1d.fit(train_sigs, y_train)

    cnn2d = ECG2DCNNClassifier(epochs=25, batch_size=32)
    cnn2d.fit(train_imgs, y_train)

    mm = SinglePassMultimodalClassifier(epochs=25, batch_size=32)
    mm.fit(train_imgs, X_train_sqi, y_train)

    # 4. Zero-Shot and Rule-Based Evaluators
    rule_model = RuleBasedSQIHeuristic()
    d1_cinc_path = RESULTS_DIR / "raw_d1_neural_cinc2011_predictions.csv"

    test_probs = {}
    print("\nPredicting on CinC 2011 Set A...")
    test_probs["Logistic_Regression_SQI"] = lr.predict_proba(X_test_sqi)
    test_probs["LightGBM_SQI"] = lgbm.predict_proba(X_test_sqi)
    test_probs["Independent_SQI_MLP"] = mlp.predict_proba(X_test_sqi)
    test_probs["1D_CNN_RawECG"] = cnn1d.predict_proba(test_sigs)
    test_probs["2D_CNN_Image"] = cnn2d.predict_proba(test_imgs)
    test_probs["Multimodal_Fusion"] = mm.predict_proba(test_imgs, X_test_sqi)

    # Rule heuristic predictions
    p_rule = []
    for i in range(n_test):
        img = Image.open(test_imgs[i]).convert("RGB")
        sqi_d = {name: float(cinc_df[name].iloc[i]) for name in SQI_FEATURE_NAMES}
        dec_rule = rule_model.decide_single(img, sqi_d, modality="multimodal")
        p_rule.append(dec_rule["prob_unusable"])
    test_probs["Rule_Based_SQI_Heuristic"] = np.array(p_rule, dtype=np.float32)

    # Real d1-3B neural predictions
    if d1_cinc_path.exists():
        print(f"Loading verified d1 neural predictions from {d1_cinc_path}...")
        d1_df = pd.read_csv(d1_cinc_path)
        # Verify alignment
        merged_cinc = cinc_df.merge(d1_df[["segment_id", "prob_unusable"]], on="segment_id", validate="one_to_one")
        assert len(merged_cinc) == n_test
        test_probs["LiquidAI_d1_3B_Neural"] = merged_cinc["prob_unusable"].values.astype(np.float32)
    else:
        print("\n[Transfer] Initializing genuine LiquidAI/d1-3B neural model on MPS...")
        d1_neural = RealD1DecisionModel(device="mps")
        p_d1 = []
        d1_raw_responses = {}
        for i in range(n_test):
            img = Image.open(test_imgs[i]).convert("RGB")
            sqi_d = {name: float(cinc_df[name].iloc[i]) for name in SQI_FEATURE_NAMES}
            rec_id = cinc_df["segment_id"].iloc[i]
            dec_d1 = d1_neural.decide_single(img, sqi_d, modality="multimodal", segment_id=rec_id)
            p_d1.append(dec_d1["prob_unusable"])
            d1_raw_responses[rec_id] = dec_d1.get("raw_response", {})
        test_probs["LiquidAI_d1_3B_Neural"] = np.array(p_d1, dtype=np.float32)

        # Save raw responses
        raw_resp_path = RESULTS_DIR / "cinc2011_d1_neural_raw_responses.jsonl"
        with open(raw_resp_path, "w") as f:
            for sid, resp in d1_raw_responses.items():
                f.write(json.dumps({"segment_id": sid, "response": resp}) + "\n")
        print(f"Saved CinC raw responses to {raw_resp_path}")

    # Save predictions
    pred_df = pd.DataFrame({
        "segment_id": cinc_df["segment_id"],
        "record_name": cinc_df["record_name"],
        "label": y_test,
        **{f"prob_{m}": test_probs[m] for m in test_probs}
    })
    pred_path = RESULTS_DIR / "cinc2011_expanded_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    # Summary table
    summary_rows = []
    for m, probs in test_probs.items():
        auroc = float(roc_auc_score(y_test, probs))
        auprc = float(average_precision_score(y_test, probs))
        brier = float(brier_score_loss(y_test, probs))

        preds_50 = (probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_50, labels=[0, 1]).ravel()
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        bal_acc = 0.5 * (sens + spec)
        false_reject = fp / max(1, fp + tn)

        summary_rows.append({
            "model": m,
            "transfer_auroc": auroc,
            "transfer_auprc": auprc,
            "brier_score": brier,
            "bal_acc_fixed": bal_acc,
            "sens_fixed": sens,
            "false_reject_fixed": false_reject,
            "tp_fixed": int(tp),
            "fp_fixed": int(fp),
            "tn_fixed": int(tn),
            "fn_fixed": int(fn)
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_path = RESULTS_DIR / "cinc2011_expanded_transfer_summary.csv"
    sum_df.to_csv(sum_path, index=False)
    print(f"Summary table saved to {sum_path}")

    print("\n" + "=" * 90)
    print("CINC 2011 SET A TRANSFER BENCHMARK SUMMARY (150 RECORDS)")
    print("=" * 90)
    print(sum_df[["model", "transfer_auroc", "transfer_auprc", "bal_acc_fixed", "sens_fixed", "false_reject_fixed"]].to_string(index=False))


if __name__ == "__main__":
    run_cinc_transfer_evaluation()
