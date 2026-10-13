"""
Revised Experiment 4: Calibrated Noise Stress Degradation on MIT-BIH NSTDB
Methodological Standards:
- Standard 118 and 119 records evaluate electrode motion ('em') noise.
- Noisy windows start inside the noise segments only (305-415s, 545-655s); see nstdb_paired_extraction.py.
- Each noisy window is paired with the clean base-record window at the same start time; the paired
  change (noisy minus clean) is reported per SNR label alongside the means.
- Evaluates Record 118, Record 119 separately, and combined.
- Clean reference windows (base records, no SNR label) are summarised in one row.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from config import RESULTS_DIR, FIGURES_DIR
from features.sqi_extractor import SQI_FEATURE_NAMES
from models.tabular_baselines import SQILogisticRegression, SQIGradientBoosting
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier
from models.d1_real_neural_model import RealD1DecisionModel

def run_revised_nstdb_experiment(verbose: bool = True):
    print("=" * 80)
    print("REVISED EXPERIMENT 4: CALIBRATED NOISE STRESS TEST (NSTDB 118 & 119)")
    print("=" * 80)
    
    # 1. Load source models trained on BUT QDB
    manifest_path = RESULTS_DIR / "manifest_butqdb_1000_cohort.csv"
    but_df = pd.read_csv(manifest_path)
    X_train_sqi = np.nan_to_num(but_df[SQI_FEATURE_NAMES].values, nan=0.0)
    y_train = but_df["label"].values
    train_sigs = [np.load(p) for p in but_df["signal_path"]]
    train_imgs = but_df["image_path"].tolist()
    
    lr = SQILogisticRegression().fit(X_train_sqi, y_train)
    lgb = SQIGradientBoosting().fit(X_train_sqi, y_train)
    cnn1d = ECG1DCNNClassifier(epochs=25, batch_size=32).fit(train_sigs, y_train)
    cnn2d = ECG2DCNNClassifier(epochs=25, batch_size=32).fit(train_imgs, y_train)
    mm = SinglePassMultimodalClassifier(epochs=25, batch_size=32, mode="multimodal").fit(train_imgs, X_train_sqi, y_train)
    d1 = RealD1DecisionModel()  # real weights, fail-closed (no heuristic fallback)
    
    # 2. Load calibrated NSTDB dataset
    nstdb_csv = config.DATA_DIR / "nstdb" / "nstdb_paired_windows.csv"  # experiments/nstdb_paired_extraction.py
    if not nstdb_csv.exists():
        raise FileNotFoundError(f"Calibrated NSTDB CSV not found at {nstdb_csv}")
    df_nstdb = pd.read_csv(nstdb_csv)
    print(f"Loaded calibrated NSTDB dataset: {len(df_nstdb)} windows.")
    
    X_nstdb_sqi = np.nan_to_num(df_nstdb[SQI_FEATURE_NAMES].values, nan=0.0)
    sigs_nstdb = [np.load(p) for p in df_nstdb["signal_path"]]
    imgs_nstdb = df_nstdb["image_path"].tolist()
    
    # Predict unusable probabilities
    p_unusable_lr = lr.predict_proba(X_nstdb_sqi)
    p_unusable_lgb = lgb.predict_proba(X_nstdb_sqi)
    p_unusable_1d = cnn1d.predict_proba(sigs_nstdb)
    p_unusable_2d = cnn2d.predict_proba(imgs_nstdb)
    p_unusable_mm = mm.predict_proba(imgs_nstdb, X_nstdb_sqi)
    
    p_d1 = []
    for img_p, sqi_vec in zip(imgs_nstdb, X_nstdb_sqi):
        sqi_dict = dict(zip(SQI_FEATURE_NAMES, sqi_vec))
        with Image.open(img_p) as pil_img:
            dec = d1.decide_single(pil_img, sqi_dict, modality="multimodal", segment_id=img_p)
            p_d1.append(dec["prob_unusable"])
    p_unusable_d1 = np.array(p_d1, dtype=np.float32)
    
    # P(Usable) = 1 - P(Unusable)
    df_nstdb["p_usable_LR"] = 1.0 - p_unusable_lr
    df_nstdb["p_usable_LGBM"] = 1.0 - p_unusable_lgb
    df_nstdb["p_usable_1D_CNN"] = 1.0 - p_unusable_1d
    df_nstdb["p_usable_2D_CNN"] = 1.0 - p_unusable_2d
    df_nstdb["p_usable_Multimodal"] = 1.0 - p_unusable_mm
    df_nstdb["p_usable_d1_3B"] = 1.0 - p_unusable_d1
    
    out_preds_csv = RESULTS_DIR / "exp4_revised_nstdb_predictions.csv"
    df_nstdb.to_csv(out_preds_csv, index=False)
    
    # Aggregate summaries:
    # 1. Active Noise Only (118 vs 119 vs Combined)
    df_active = df_nstdb[df_nstdb["condition"] == "noisy"].copy()
    
    snr_levels = [24, 18, 12, 6, 0, -6]
    model_keys = ["LR", "LGBM", "1D_CNN", "2D_CNN", "Multimodal", "d1_3B"]
    # Paired change: each noisy window minus its clean base-record partner (same start time, same beats)
    clean_idx = df_nstdb[df_nstdb["condition"] == "clean_reference"].set_index("segment_id")
    for m in model_keys:
        partner_p = clean_idx.loc[df_active["partner_segment_id"], f"p_usable_{m}"].values
        df_active[f"delta_{m}"] = df_active[f"p_usable_{m}"].values - partner_p
    
    summary_list = []
    for subset_name, sub_df in [
        ("Record_118_ActiveNoise", df_active[df_active["base_record"].astype(str) == "118"]),
        ("Record_119_ActiveNoise", df_active[df_active["base_record"].astype(str) == "119"]),
        ("Combined_118_119_ActiveNoise", df_active),
    ]:
        for snr in snr_levels:
            snr_df = sub_df[sub_df["snr_db"] == snr]
            row = {
                "subset": subset_name,
                "snr_db": snr,
                "n_windows": len(snr_df)
            }
            for m in model_keys:
                row[f"{m}_mean"] = snr_df[f"p_usable_{m}"].mean()
                row[f"{m}_std"] = snr_df[f"p_usable_{m}"].std()
                row[f"{m}_delta_mean"] = snr_df[f"delta_{m}"].mean()
                row[f"{m}_delta_std"] = snr_df[f"delta_{m}"].std()
            summary_list.append(row)
    # Clean reference windows come from the MIT-BIH base records and carry no SNR label: one row for all of them
    clean_all = df_nstdb[df_nstdb["condition"] == "clean_reference"]
    row = {"subset": "Clean_Control_Windows", "snr_db": np.nan, "n_windows": len(clean_all)}
    for m in model_keys:
        row[f"{m}_mean"] = clean_all[f"p_usable_{m}"].mean()
        row[f"{m}_std"] = clean_all[f"p_usable_{m}"].std()
    summary_list.append(row)
            
    df_summary = pd.DataFrame(summary_list)
    out_summary_csv = RESULTS_DIR / "exp4_revised_nstdb_summary.csv"
    df_summary.to_csv(out_summary_csv, index=False)
    
    print("\n[NSTDB Degradation Summary - Combined Active Noise (Records 118 & 119)]")
    comb_df = df_summary[df_summary["subset"] == "Combined_118_119_ActiveNoise"]
    print(comb_df[["snr_db", "n_windows", "LR_mean", "LGBM_mean", "1D_CNN_mean", "2D_CNN_mean", "Multimodal_mean", "d1_3B_mean"]])
    
    print("\n[NSTDB Clean Control Windows (Integrity Check)]")
    clean_df = df_summary[df_summary["subset"] == "Clean_Control_Windows"]
    print(clean_df[["snr_db", "n_windows", "LR_mean", "LGBM_mean", "1D_CNN_mean", "2D_CNN_mean", "Multimodal_mean", "d1_3B_mean"]])
    
    # Plot degradation curves
    plt.figure(figsize=(10, 6), dpi=300)
    for m, col, label in [
        ("LR", "#1f77b4", "Logistic Regression (SQI)"),
        ("LGBM", "#ff7f0e", "LightGBM (SQI)"),
        ("1D_CNN", "#9467bd", "1D CNN (Raw Signal)"),
        ("2D_CNN", "#d62728", "2D CNN (Image)"),
        ("Multimodal", "#2ca02c", "Multimodal (Image + SQI)"),
        ("d1_3B", "#8c564b", "Liquid AI d1-3B (Decision Net)")
    ]:
        means = comb_df[f"{m}_mean"].values
        stds = comb_df[f"{m}_std"].values
        snrs = comb_df["snr_db"].values
        plt.plot(snrs, means, "o-", label=label, color=col, linewidth=2)
        plt.fill_between(snrs, means - stds, means + stds, color=col, alpha=0.15)
        
    plt.axhline(0.5, color="black", linestyle="--", alpha=0.5, label="Decision Threshold (tau = 0.5)")
    plt.gca().invert_xaxis()
    plt.xlabel("Electrode Motion Noise SNR (dB, degrading ->)", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Predicted Usability Probability P(Usable)", fontsize=11, fontweight="bold")
    plt.title("Calibrated Noise Stress Degradation (NSTDB Records 118 & 119 Active Noise)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(frameon=True, facecolor="white", loc="best", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_revised_nstdb_degradation_curve.png")
    plt.close()
    
    print(f"\n[Completed] Revised NSTDB analysis saved to {out_summary_csv}")
    print(f"Chart saved to {FIGURES_DIR / 'figure_revised_nstdb_degradation_curve.png'}")
    return df_summary

if __name__ == "__main__":
    run_revised_nstdb_experiment()
