"""
Empirical Latency and Memory Profiling with Explicit MPS Synchronization
Methodology:
1. Uses n_samples diverse, verified ECG recordings from manifest_butqdb_1000_cohort.csv (25 in the reported runs).
2. For each sample and each model, measures:
   - Raw preprocessing duration (resampling, SQI extraction, waveform rendering, image decoding).
   - Forward pass duration with strict synchronization (torch.mps.synchronize() before & after).
   - Total end-to-end duration measured directly on that sample: t_total = t_pre + t_fwd.
3. Computes true empirical distribution statistics: median and p95 directly from the raw total series
   (NOT by summing component medians or component p95s).
4. Memory labeled strictly with accurate PyTorch definitions (MiB and GiB):
   - Model loaded MPS tensor memory
   - Post-inference MPS tensor memory
   - Post-inference Metal driver memory
5. Saves raw per-sample timings to results/empirical_latency_raw_measurements.csv
   and summary table to results/empirical_computational_profile_summary.csv.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, TARGET_FS, DEFAULT_WINDOW_SEC
from features.sqi_extractor import SQI_FEATURE_NAMES, extract_sqi_features
from data.ecg_rendering import render_ecg_image
from data.units import to_millivolts
from models.tabular_baselines import SQILogisticRegression, SQIGradientBoosting
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier, IndependentSQIClassifier
from models.rule_heuristic_baseline import RuleBasedSQIHeuristic
from models.d1_real_neural_model import RealD1DecisionModel


def run_empirical_profiling(n_samples: int = 25):
    print("=" * 80)
    print("EMPIRICAL LATENCY & MEMORY PROFILING (STRICT MPS SYNCHRONIZATION)")
    print("=" * 80)

    manifest_path = RESULTS_DIR / "manifest_butqdb_1000_cohort.csv"
    df = pd.read_csv(manifest_path)
    
    # Pick n_samples diverse recordings across different subjects
    selected_indices = np.linspace(0, len(df) - 1, n_samples, dtype=int)
    eval_df = df.iloc[selected_indices].reset_index(drop=True)
    print(f"Profiling across {len(eval_df)} diverse ECG samples from {eval_df['subject_id'].nunique()} subjects.")

    raw_measurements = []
    summary_stats = []

    # -------------------------------------------------------------
    # 1. Rule-Based SQI Heuristic (0 Parameters)
    # -------------------------------------------------------------
    print("\n[1/7] Profiling Rule-Based SQI Heuristic...")
    rule_model = RuleBasedSQIHeuristic()
    fwd_times = []
    pre_times = []
    total_times = []

    for idx, row in eval_df.iterrows():
        raw_sig = to_millivolts(np.load(row["signal_path"]), "uV").astype(np.float32)  # BUT QDB cache is in uV
        
        # Preprocessing: SQI extraction + Image load & ink density array conversion
        t0_pre = time.perf_counter()
        sqis = extract_sqi_features(raw_sig, TARGET_FS)
        img = Image.open(row["image_path"]).convert("L")
        _ = np.array(img, dtype=np.float32)
        t_pre_ms = (time.perf_counter() - t0_pre) * 1000.0

        # Forward pass (log-odds calculation)
        t0_fwd = time.perf_counter()
        _ = rule_model.decide_single(img, sqis, modality="multimodal")
        t_fwd_ms = (time.perf_counter() - t0_fwd) * 1000.0

        t_tot_ms = t_pre_ms + t_fwd_ms
        pre_times.append(t_pre_ms)
        fwd_times.append(t_fwd_ms)
        total_times.append(t_tot_ms)

        raw_measurements.append({
            "model": "Rule_Based_SQI_Heuristic",
            "sample_idx": idx,
            "segment_id": row["segment_id"],
            "pre_ms": t_pre_ms,
            "fwd_ms": t_fwd_ms,
            "total_ms": t_tot_ms
        })

    summary_stats.append({
        "Model": "Rule-Based SQI Heuristic",
        "Parameters": 0,
        "Weights_MiB": 0.0,
        "Loaded_MPS_Tensor_MiB": "N/A (CPU)",
        "Post_Infer_MPS_Tensor_MiB": "N/A (CPU)",
        "Post_Infer_Driver_MiB": "N/A (CPU)",
        "Preprocess_Median_ms": float(np.median(pre_times)),
        "Preprocess_p95_ms": float(np.percentile(pre_times, 95)),
        "Forward_Median_ms": float(np.median(fwd_times)),
        "Forward_p95_ms": float(np.percentile(fwd_times, 95)),
        "Total_End_to_End_Median_ms": float(np.median(total_times)),
        "Total_End_to_End_p95_ms": float(np.percentile(total_times, 95)),
        "Pipeline_Description": "SQI extraction + image array conversion + log-odds (0 params)"
    })

    # -------------------------------------------------------------
    # 2. Logistic Regression (SQI)
    # -------------------------------------------------------------
    print("[2/7] Profiling Logistic Regression (SQI)...")
    lr = SQILogisticRegression()
    # Fit dummy on first 10
    X_init = np.nan_to_num(eval_df[SQI_FEATURE_NAMES].values[:10])
    y_init = eval_df["label"].values[:10]
    lr.fit(X_init, y_init)

    fwd_times, pre_times, total_times = [], [], []
    for idx, row in eval_df.iterrows():
        raw_sig = to_millivolts(np.load(row["signal_path"]), "uV").astype(np.float32)  # BUT QDB cache is in uV
        t0_pre = time.perf_counter()
        sqis = extract_sqi_features(raw_sig, TARGET_FS)
        x_vec = np.array([[sqis[k] for k in SQI_FEATURE_NAMES]])
        t_pre_ms = (time.perf_counter() - t0_pre) * 1000.0

        t0_fwd = time.perf_counter()
        _ = lr.predict_proba(x_vec)
        t_fwd_ms = (time.perf_counter() - t0_fwd) * 1000.0

        t_tot_ms = t_pre_ms + t_fwd_ms
        pre_times.append(t_pre_ms)
        fwd_times.append(t_fwd_ms)
        total_times.append(t_tot_ms)

        raw_measurements.append({
            "model": "Logistic_Regression_SQI",
            "sample_idx": idx,
            "segment_id": row["segment_id"],
            "pre_ms": t_pre_ms,
            "fwd_ms": t_fwd_ms,
            "total_ms": t_tot_ms
        })

    summary_stats.append({
        "Model": "Logistic Regression (SQI)",
        "Parameters": 14,
        "Weights_MiB": 0.01,
        "Loaded_MPS_Tensor_MiB": "N/A (CPU)",
        "Post_Infer_MPS_Tensor_MiB": "N/A (CPU)",
        "Post_Infer_Driver_MiB": "N/A (CPU)",
        "Preprocess_Median_ms": float(np.median(pre_times)),
        "Preprocess_p95_ms": float(np.percentile(pre_times, 95)),
        "Forward_Median_ms": float(np.median(fwd_times)),
        "Forward_p95_ms": float(np.percentile(fwd_times, 95)),
        "Total_End_to_End_Median_ms": float(np.median(total_times)),
        "Total_End_to_End_p95_ms": float(np.percentile(total_times, 95)),
        "Pipeline_Description": "SQI extraction + sklearn logistic regression"
    })

    # -------------------------------------------------------------
    # 2b. LightGBM (SQI)
    # -------------------------------------------------------------
    print("[2b/7] Profiling LightGBM (SQI)...")
    lgb = SQIGradientBoosting()
    lgb.fit(X_init, y_init)

    # Parameter count and weight size from the cohort-fitted model (the 10-sample fit above is latency-only)
    cohort_df = pd.read_csv(manifest_path)
    lgb_full = SQIGradientBoosting().fit(np.nan_to_num(cohort_df[SQI_FEATURE_NAMES].values), cohort_df["label"].values)
    tdf = lgb_full.model.booster_.trees_to_dataframe()
    lgb_split_nodes = int(tdf["split_feature"].notna().sum())
    import pickle
    lgb_weights_mib = len(pickle.dumps(lgb_full.model)) / (1024 * 1024)

    fwd_times, pre_times, total_times = [], [], []
    for idx, row in eval_df.iterrows():
        raw_sig = to_millivolts(np.load(row["signal_path"]), "uV").astype(np.float32)  # BUT QDB cache is in uV
        t0_pre = time.perf_counter()
        sqis = extract_sqi_features(raw_sig, TARGET_FS)
        x_vec = np.array([[sqis[k] for k in SQI_FEATURE_NAMES]])
        t_pre_ms = (time.perf_counter() - t0_pre) * 1000.0

        t0_fwd = time.perf_counter()
        _ = lgb.predict_proba(x_vec)
        t_fwd_ms = (time.perf_counter() - t0_fwd) * 1000.0

        t_tot_ms = t_pre_ms + t_fwd_ms
        pre_times.append(t_pre_ms)
        fwd_times.append(t_fwd_ms)
        total_times.append(t_tot_ms)

        raw_measurements.append({
            "model": "LightGBM_SQI",
            "sample_idx": idx,
            "segment_id": row["segment_id"],
            "pre_ms": t_pre_ms,
            "fwd_ms": t_fwd_ms,
            "total_ms": t_tot_ms
        })

    summary_stats.append({
        "Model": "LightGBM (SQI)",
        "Parameters": lgb_split_nodes,
        "Weights_MiB": round(lgb_weights_mib, 2),
        "Loaded_MPS_Tensor_MiB": "N/A (CPU)",
        "Post_Infer_MPS_Tensor_MiB": "N/A (CPU)",
        "Post_Infer_Driver_MiB": "N/A (CPU)",
        "Preprocess_Median_ms": float(np.median(pre_times)),
        "Preprocess_p95_ms": float(np.percentile(pre_times, 95)),
        "Forward_Median_ms": float(np.median(fwd_times)),
        "Forward_p95_ms": float(np.percentile(fwd_times, 95)),
        "Total_End_to_End_Median_ms": float(np.median(total_times)),
        "Total_End_to_End_p95_ms": float(np.percentile(total_times, 95)),
        "Pipeline_Description": "SQI extraction + LightGBM (100 trees; Parameters = split nodes of the cohort-fitted model)"
    })

    # -------------------------------------------------------------
    # 3. Independent SQI MLP
    # -------------------------------------------------------------
    print("[3/7] Profiling Independent SQI MLP...")
    mlp = IndependentSQIClassifier(epochs=1)
    mlp.fit(X_init, y_init)

    fwd_times, pre_times, total_times = [], [], []
    for idx, row in eval_df.iterrows():
        raw_sig = to_millivolts(np.load(row["signal_path"]), "uV").astype(np.float32)  # BUT QDB cache is in uV
        t0_pre = time.perf_counter()
        sqis = extract_sqi_features(raw_sig, TARGET_FS)
        x_vec = np.array([[sqis[k] for k in SQI_FEATURE_NAMES]])
        t_pre_ms = (time.perf_counter() - t0_pre) * 1000.0

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t0_fwd = time.perf_counter()
        _ = mlp.predict_proba(x_vec)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t_fwd_ms = (time.perf_counter() - t0_fwd) * 1000.0

        t_tot_ms = t_pre_ms + t_fwd_ms
        pre_times.append(t_pre_ms)
        fwd_times.append(t_fwd_ms)
        total_times.append(t_tot_ms)

        raw_measurements.append({
            "model": "Independent_SQI_MLP",
            "sample_idx": idx,
            "segment_id": row["segment_id"],
            "pre_ms": t_pre_ms,
            "fwd_ms": t_fwd_ms,
            "total_ms": t_tot_ms
        })

    summary_stats.append({
        "Model": "Independent SQI MLP",
        "Parameters": 3201,
        "Weights_MiB": 0.01,
        "Loaded_MPS_Tensor_MiB": "N/A",
        "Post_Infer_MPS_Tensor_MiB": "N/A",
        "Post_Infer_Driver_MiB": "N/A",
        "Preprocess_Median_ms": float(np.median(pre_times)),
        "Preprocess_p95_ms": float(np.percentile(pre_times, 95)),
        "Forward_Median_ms": float(np.median(fwd_times)),
        "Forward_p95_ms": float(np.percentile(fwd_times, 95)),
        "Total_End_to_End_Median_ms": float(np.median(total_times)),
        "Total_End_to_End_p95_ms": float(np.percentile(total_times, 95)),
        "Pipeline_Description": "SQI extraction + PyTorch 2-layer MLP"
    })

    # -------------------------------------------------------------
    # 4. 1D CNN (Raw Signal)
    # -------------------------------------------------------------
    print("[4/7] Profiling 1D CNN (Raw Signal)...")
    cnn1d = ECG1DCNNClassifier(epochs=1)
    cnn1d.fit([np.load(p) for p in eval_df["signal_path"][:6]], y_init[:6])

    fwd_times, pre_times, total_times = [], [], []
    for idx, row in eval_df.iterrows():
        raw_sig = to_millivolts(np.load(row["signal_path"]), "uV").astype(np.float32)  # BUT QDB cache is in uV
        t0_pre = time.perf_counter()
        # Direct array slicing / normalization
        inp_sig = [raw_sig]
        t_pre_ms = (time.perf_counter() - t0_pre) * 1000.0

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t0_fwd = time.perf_counter()
        _ = cnn1d.predict_proba(inp_sig)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t_fwd_ms = (time.perf_counter() - t0_fwd) * 1000.0

        t_tot_ms = t_pre_ms + t_fwd_ms
        pre_times.append(t_pre_ms)
        fwd_times.append(t_fwd_ms)
        total_times.append(t_tot_ms)

        raw_measurements.append({
            "model": "1D_CNN_RawECG",
            "sample_idx": idx,
            "segment_id": row["segment_id"],
            "pre_ms": t_pre_ms,
            "fwd_ms": t_fwd_ms,
            "total_ms": t_tot_ms
        })

    summary_stats.append({
        "Model": "1D CNN (Raw Signal)",
        "Parameters": sum(p.numel() for p in ECG1DCNNClassifier().net.parameters()),
        "Weights_MiB": 0.64,
        "Loaded_MPS_Tensor_MiB": "N/A",
        "Post_Infer_MPS_Tensor_MiB": "N/A",
        "Post_Infer_Driver_MiB": "N/A",
        "Preprocess_Median_ms": float(np.median(pre_times)),
        "Preprocess_p95_ms": float(np.percentile(pre_times, 95)),
        "Forward_Median_ms": float(np.median(fwd_times)),
        "Forward_p95_ms": float(np.percentile(fwd_times, 95)),
        "Total_End_to_End_Median_ms": float(np.median(total_times)),
        "Total_End_to_End_p95_ms": float(np.percentile(total_times, 95)),
        "Pipeline_Description": "Direct signal array pass (5000 samples, 500 Hz)"
    })

    # -------------------------------------------------------------
    # 5. 2D CNN (Image)
    # -------------------------------------------------------------
    print("[5/7] Profiling 2D CNN (Image)...")
    cnn2d = ECG2DCNNClassifier(epochs=1)
    cnn2d.fit(eval_df["image_path"][:6].tolist(), y_init[:6])

    fwd_times, pre_times, total_times = [], [], []
    for idx, row in eval_df.iterrows():
        raw_sig = to_millivolts(np.load(row["signal_path"]), "uV").astype(np.float32)  # BUT QDB cache is in uV
        t0_pre = time.perf_counter()
        # Full rendering from raw signal into PIL image
        img = render_ecg_image(raw_sig, TARGET_FS, dpi=100)
        t_pre_ms = (time.perf_counter() - t0_pre) * 1000.0

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t0_fwd = time.perf_counter()
        _ = cnn2d.predict_proba([row["image_path"]])
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t_fwd_ms = (time.perf_counter() - t0_fwd) * 1000.0

        t_tot_ms = t_pre_ms + t_fwd_ms
        pre_times.append(t_pre_ms)
        fwd_times.append(t_fwd_ms)
        total_times.append(t_tot_ms)

        raw_measurements.append({
            "model": "2D_CNN_Image",
            "sample_idx": idx,
            "segment_id": row["segment_id"],
            "pre_ms": t_pre_ms,
            "fwd_ms": t_fwd_ms,
            "total_ms": t_tot_ms
        })

    summary_stats.append({
        "Model": "2D CNN (Rendered Image)",
        "Parameters": sum(p.numel() for p in ECG2DCNNClassifier().net.parameters()),
        "Weights_MiB": 0.94,
        "Loaded_MPS_Tensor_MiB": "N/A",
        "Post_Infer_MPS_Tensor_MiB": "N/A",
        "Post_Infer_Driver_MiB": "N/A",
        "Preprocess_Median_ms": float(np.median(pre_times)),
        "Preprocess_p95_ms": float(np.percentile(pre_times, 95)),
        "Forward_Median_ms": float(np.median(fwd_times)),
        "Forward_p95_ms": float(np.percentile(fwd_times, 95)),
        "Total_End_to_End_Median_ms": float(np.median(total_times)),
        "Total_End_to_End_p95_ms": float(np.percentile(total_times, 95)),
        "Pipeline_Description": "Matplotlib rendering (100 DPI) + 4-block 2D CNN"
    })

    # -------------------------------------------------------------
    # 6. SinglePass Multimodal Net
    # -------------------------------------------------------------
    print("[6/7] Profiling SinglePass Multimodal Net...")
    mm = SinglePassMultimodalClassifier(epochs=1)
    mm.fit(eval_df["image_path"][:6].tolist(), X_init[:6], y_init[:6])

    fwd_times, pre_times, total_times = [], [], []
    for idx, row in eval_df.iterrows():
        raw_sig = to_millivolts(np.load(row["signal_path"]), "uV").astype(np.float32)  # BUT QDB cache is in uV
        t0_pre = time.perf_counter()
        img = render_ecg_image(raw_sig, TARGET_FS, dpi=100)
        sqis = extract_sqi_features(raw_sig, TARGET_FS)
        x_vec = np.array([[sqis[k] for k in SQI_FEATURE_NAMES]])
        t_pre_ms = (time.perf_counter() - t0_pre) * 1000.0

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t0_fwd = time.perf_counter()
        _ = mm.predict_proba([row["image_path"]], x_vec)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t_fwd_ms = (time.perf_counter() - t0_fwd) * 1000.0

        t_tot_ms = t_pre_ms + t_fwd_ms
        pre_times.append(t_pre_ms)
        fwd_times.append(t_fwd_ms)
        total_times.append(t_tot_ms)

        raw_measurements.append({
            "model": "Multimodal_Fusion",
            "sample_idx": idx,
            "segment_id": row["segment_id"],
            "pre_ms": t_pre_ms,
            "fwd_ms": t_fwd_ms,
            "total_ms": t_tot_ms
        })

    summary_stats.append({
        "Model": "SinglePass Multimodal Net",
        "Parameters": sum(p.numel() for p in SinglePassMultimodalClassifier().net.parameters()),
        "Weights_MiB": 0.47,
        "Loaded_MPS_Tensor_MiB": "N/A",
        "Post_Infer_MPS_Tensor_MiB": "N/A",
        "Post_Infer_Driver_MiB": "N/A",
        "Preprocess_Median_ms": float(np.median(pre_times)),
        "Preprocess_p95_ms": float(np.percentile(pre_times, 95)),
        "Forward_Median_ms": float(np.median(fwd_times)),
        "Forward_p95_ms": float(np.percentile(fwd_times, 95)),
        "Total_End_to_End_Median_ms": float(np.median(total_times)),
        "Total_End_to_End_p95_ms": float(np.percentile(total_times, 95)),
        "Pipeline_Description": "Matplotlib rendering + SQI extraction + dual-branch fusion"
    })

    # -------------------------------------------------------------
    # 7. Real LiquidAI/d1-3B (Genuine 3.12B Neural Model on MPS)
    # -------------------------------------------------------------
    print("[7/7] Profiling Real LiquidAI/d1-3B Neural Decision Model with torch.mps.synchronize()...")
    torch.mps.empty_cache()
    t_start_load = time.time()
    d1 = RealD1DecisionModel(device="mps")
    load_duration_s = time.time() - t_start_load

    m_loaded_tensor_mib = torch.mps.current_allocated_memory() / (1024**2)
    m_loaded_driver_mib = torch.mps.driver_allocated_memory() / (1024**2)

    # Warmup on sample 0
    w_row = eval_df.iloc[0]
    w_img = Image.open(w_row["image_path"]).convert("RGB")
    w_sqi = {k: float(w_row[k]) for k in SQI_FEATURE_NAMES}
    torch.mps.synchronize()
    _ = d1.decide_single(w_img, w_sqi, modality="multimodal", segment_id="warmup")
    torch.mps.synchronize()

    fwd_times, pre_times, total_times = [], [], []
    for idx, row in eval_df.iterrows():
        raw_sig = to_millivolts(np.load(row["signal_path"]), "uV").astype(np.float32)  # BUT QDB cache is in uV
        
        # Preprocessing: Matplotlib render + SQI prompt format
        t0_pre = time.perf_counter()
        img = render_ecg_image(raw_sig, TARGET_FS, dpi=100)
        sqis = extract_sqi_features(raw_sig, TARGET_FS)
        t_pre_ms = (time.perf_counter() - t0_pre) * 1000.0

        # Synchronized forward pass
        torch.mps.synchronize()
        t0_fwd = time.perf_counter()
        _ = d1.decide_single(img, sqis, modality="multimodal", segment_id=row["segment_id"])
        torch.mps.synchronize()
        t_fwd_ms = (time.perf_counter() - t0_fwd) * 1000.0

        t_tot_ms = t_pre_ms + t_fwd_ms
        pre_times.append(t_pre_ms)
        fwd_times.append(t_fwd_ms)
        total_times.append(t_tot_ms)

        raw_measurements.append({
            "model": "LiquidAI_d1_3B_Neural",
            "sample_idx": idx,
            "segment_id": row["segment_id"],
            "pre_ms": t_pre_ms,
            "fwd_ms": t_fwd_ms,
            "total_ms": t_tot_ms
        })

    m_post_tensor_mib = torch.mps.current_allocated_memory() / (1024**2)
    m_post_driver_mib = torch.mps.driver_allocated_memory() / (1024**2)

    summary_stats.append({
        "Model": "Liquid AI d1-3B (Real Neural Model)",
        "Parameters": 3120000000,
        "Weights_MiB": 5958.0,
        "Loaded_MPS_Tensor_MiB": round(m_loaded_tensor_mib, 1),
        "Post_Infer_MPS_Tensor_MiB": round(m_post_tensor_mib, 1),
        "Post_Infer_Driver_MiB": round(m_post_driver_mib, 1),
        "Preprocess_Median_ms": float(np.median(pre_times)),
        "Preprocess_p95_ms": float(np.percentile(pre_times, 95)),
        "Forward_Median_ms": float(np.median(fwd_times)),
        "Forward_p95_ms": float(np.percentile(fwd_times, 95)),
        "Total_End_to_End_Median_ms": float(np.median(total_times)),
        "Total_End_to_End_p95_ms": float(np.percentile(total_times, 95)),
        "Pipeline_Description": f"Matplotlib rendering + SQI text prompt + synchronized system_one MPS forward pass"
    })

    # Save raw measurements
    raw_df = pd.DataFrame(raw_measurements)
    raw_path = RESULTS_DIR / "empirical_latency_raw_measurements.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\n[Saved] Raw sample-by-sample measurements ({len(raw_df)} rows) saved to {raw_path}")

    # Save summary
    sum_df = pd.DataFrame(summary_stats)
    sum_path = RESULTS_DIR / "empirical_computational_profile_summary.csv"
    sum_df.to_csv(sum_path, index=False)
    print(f"[Saved] Empirical summary saved to {sum_path}")

    print("\n" + "=" * 90)
    print(f"EMPIRICAL COMPUTATIONAL PROFILE ({n_samples} DIVERSE REAL ECG RECORDINGS, SYNCHRONIZED MPS)")
    print("=" * 90)
    print(sum_df[["Model", "Parameters", "Forward_Median_ms", "Forward_p95_ms", "Total_End_to_End_Median_ms", "Total_End_to_End_p95_ms", "Post_Infer_MPS_Tensor_MiB", "Post_Infer_Driver_MiB"]].to_string(index=False))


if __name__ == "__main__":
    run_empirical_profiling(n_samples=25)
