"""
Experiment 4: Controlled Noise Stress Degradation Benchmark on MIT-BIH NSTDB
(Study Requirement #5.C)

- Evaluates models across calibrated signal-to-noise ratios: 24, 18, 12, 6, 0, and -6 dB.
- Analyzes transition dynamics: How does predicted P(Usable) diminish and P(Unusable) escalate as SNR degrades?
- Generates publication-ready degradation curves and inflection point analysis.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR, FIGURES_DIR, RANDOM_SEED, NSTDB_SNR_LEVELS
from data.butqdb_dataset import BUTQDBManager
from data.nstdb_dataset import NSTDBManager
from features.sqi_extractor import SQI_FEATURE_NAMES
from models.tabular_baselines import SQILogisticRegression, SQIGradientBoosting
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier

def run_nstdb_noise_stress(base_record: str = "118", verbose: bool = True):
    print("=" * 80)
    print(f"EXPERIMENT 4: CONTROLLED NOISE DEGRADATION ON NSTDB (RECORD {base_record})")
    print(f"Calibrated SNR levels: {NSTDB_SNR_LEVELS} dB")
    print("=" * 80)
    
    # 1. Train models on BUT QDB development set
    print("\n[Step 1] Training candidate models on BUT QDB...")
    but_mgr = BUTQDBManager()
    but_df = but_mgr.extract_segments(max_segments_per_subject=40, verbose=verbose)
    
    X_train_sqi = but_df[SQI_FEATURE_NAMES].values.astype(np.float32)
    y_train = but_df["label"].values.astype(int)
    train_imgs = but_df["image_path"].tolist()
    
    lr = SQILogisticRegression(random_state=RANDOM_SEED).fit(X_train_sqi, y_train)
    lgb_clf = SQIGradientBoosting(random_state=RANDOM_SEED).fit(X_train_sqi, y_train)
    cnn2d = ECG2DCNNClassifier(epochs=12, batch_size=32).fit(train_imgs, y_train)
    mm_clf = SinglePassMultimodalClassifier(epochs=15, batch_size=32, mode="multimodal").fit(train_imgs, X_train_sqi, y_train)
    
    # 2. Extract NSTDB SNR segments
    print(f"\n[Step 2] Extracting NSTDB segments across calibrated SNR levels...")
    nst_mgr = NSTDBManager()
    nst_df = nst_mgr.extract_snr_series(base_record=base_record, num_windows_per_snr=15, verbose=verbose)
    
    X_nst_sqi = nst_df[SQI_FEATURE_NAMES].values.astype(np.float32)
    nst_imgs = nst_df["image_path"].tolist()
    
    # 3. Predict across SNR levels
    print("\n[Step 3] Evaluating model decisions across SNR degradation...")
    nst_df["prob_unusable_lr"] = lr.predict_proba(X_nst_sqi)
    nst_df["prob_unusable_lgb"] = lgb_clf.predict_proba(X_nst_sqi)
    nst_df["prob_unusable_cnn2d"] = cnn2d.predict_proba(nst_imgs)
    nst_df["prob_unusable_multimodal"] = mm_clf.predict_proba(nst_imgs, X_nst_sqi)
    
    # Calculate P(Usable) = 1 - P(Unusable)
    models_to_plot = {
        "Logistic Regression (SQI)": "prob_unusable_lr",
        "LightGBM (SQI)": "prob_unusable_lgb",
        "2D CNN (Image)": "prob_unusable_cnn2d",
        "Multimodal (Image + SQI)": "prob_unusable_multimodal"
    }
    
    snr_summary = []
    for snr in NSTDB_SNR_LEVELS:
        sub = nst_df[nst_df["snr_db"] == snr]
        entry = {"snr_db": snr, "n_windows": len(sub)}
        for m_label, col in models_to_plot.items():
            entry[f"{m_label}_p_usable_mean"] = float(1.0 - sub[col].mean())
            entry[f"{m_label}_p_usable_std"] = float(sub[col].std())
        snr_summary.append(entry)
        
    df_snr_summary = pd.DataFrame(snr_summary)
    out_csv = RESULTS_DIR / f"exp4_nstdb_{base_record}_snr_degradation_summary.csv"
    df_snr_summary.to_csv(out_csv, index=False)
    
    print("\nDegradation Table: P(Usable) across SNR levels:")
    print(df_snr_summary[["snr_db"] + [f"{m}_p_usable_mean" for m in models_to_plot.keys()]])
    
    # 4. Generate Degradation Plot
    print("\n[Step 4] Generating publication-quality degradation curves...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    colors = ["#2b5c8f", "#e66101", "#5e3c99", "#d01c8b"]
    markers = ["o", "s", "^", "D"]
    
    for (m_label, col), color, marker in zip(models_to_plot.items(), colors, markers):
        means = df_snr_summary[f"{m_label}_p_usable_mean"]
        stds = df_snr_summary[f"{m_label}_p_usable_std"]
        snrs = df_snr_summary["snr_db"]
        ax.plot(snrs, means, label=m_label, color=color, marker=marker, linewidth=2.2, markersize=7)
        ax.fill_between(snrs, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1), color=color, alpha=0.15)
        
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Usability Threshold (0.5)")
    ax.set_xlabel("Signal-to-Noise Ratio (SNR in dB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Probability of Usability P(Usable)", fontsize=12, fontweight="bold")
    ax.set_title(f"Controlled ECG Usability Degradation under Noise Stress (NSTDB Record {base_record})", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(26, -8) # High SNR on left, Low SNR on right
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower left")
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / f"figure_nstdb_{base_record}_degradation_curve.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[Completed] Degradation plot saved to {fig_path}")
    
    return df_snr_summary

if __name__ == "__main__":
    run_nstdb_noise_stress()
