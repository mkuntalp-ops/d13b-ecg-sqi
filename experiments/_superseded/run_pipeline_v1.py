"""
Main Experimental Pipeline & Research Orchestrator
"Cross-dataset ECG usability assessment with a single-pass multimodal decision model"

Usage:
    python run_pipeline.py --exp pilot        # Run Experiment 1 (Feasibility Pilot)
    python run_pipeline.py --exp cv           # Run Experiment 2 (BUT QDB Subject CV)
    python run_pipeline.py --exp transfer     # Run Experiment 3 (CinC 2011 Transfer)
    python run_pipeline.py --exp nstdb        # Run Experiment 4 (NSTDB Noise Degradation)
    python run_pipeline.py --exp robustness   # Run Experiment 5 (Imaging Perturbations)
    python run_pipeline.py --exp profile      # Run Experiment 6 (Computational Cost)
    python run_pipeline.py --exp all          # Execute complete scientific pipeline
"""

import argparse
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from config import RESULTS_DIR, FIGURES_DIR
from experiments.exp1_feasibility_pilot import run_feasibility_pilot
from experiments.exp2_butqdb_subject_cv import run_butqdb_subject_cv
from experiments.exp3_transfer_cinc2011 import run_transfer_cinc2011
from experiments.exp4_noise_stress_nstdb import run_nstdb_noise_stress
from experiments.exp5_imaging_robustness import run_imaging_robustness_test
from experiments.exp6_computational_profile import run_computational_profile

def generate_consolidated_figures():
    """Generates consolidated publication figures from results."""
    print("\n[Figures] Generating consolidated study figures...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    
    # 1. Figure: Feasibility Pilot Modality Comparison
    pilot_csv = RESULTS_DIR / "exp1_feasibility_pilot_summary.csv"
    if pilot_csv.exists():
        df_p = pd.read_csv(pilot_csv)
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        
        palette = {"image_only": "#1f77b4", "sqi_only": "#ff7f0e", "multimodal": "#2ca02c"}
        labels = {"image_only": "Image Only", "sqi_only": "SQI Only", "multimodal": "Multimodal (Image + SQI)"}
        
        x = np.arange(len(df_p["dataset"].unique()))
        width = 0.25
        
        for i, mod in enumerate(["image_only", "sqi_only", "multimodal"]):
            sub = df_p[df_p["modality"] == mod]
            vals = sub["balanced_accuracy"].values * 100.0
            ax.bar(x + (i - 1) * width, vals, width, label=labels[mod], color=palette[mod], alpha=0.9)
            
        ax.set_ylabel("Balanced Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title("Feasibility Pilot: Modality Contribution across Datasets", fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(df_p["dataset"].unique(), fontsize=10)
        ax.set_ylim(40, 105)
        ax.legend(frameon=True, facecolor="white")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "figure_pilot_modality_comparison.png")
        plt.close()
        
    # 2. Figure: Cross-Validation Comparison on BUT QDB
    cv_csv = RESULTS_DIR / "exp2_butqdb_subject_cv_summary.csv"
    if cv_csv.exists():
        df_cv = pd.read_csv(cv_csv)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        
        # Sort by balanced accuracy
        df_cv = df_cv.sort_values(by="balanced_accuracy", ascending=False)
        colors = ["#2ca02c" if "Multimodal" in m else "#1f77b4" for m in df_cv["model"]]
        
        y_pos = np.arange(len(df_cv))
        ax.barh(y_pos, df_cv["balanced_accuracy"] * 100.0, color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_cv["model"].str.replace("_", " "), fontsize=9)
        ax.set_xlabel("Balanced Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title("Subject-Independent 5-Fold CV on BUT QDB (Leakage-Free)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim(50, 105)
        
        for i, v in enumerate(df_cv["balanced_accuracy"] * 100.0):
            ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
            
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "figure_butqdb_cv_comparison.png")
        plt.close()
        
    # 3. Figure: Cross-Dataset Transfer Comparison
    trans_csv = RESULTS_DIR / "exp3_cinc2011_transfer_summary.csv"
    if trans_csv.exists():
        df_t = pd.read_csv(trans_csv)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        
        df_t = df_t.sort_values(by="balanced_accuracy", ascending=False)
        colors = ["#2ca02c" if "Multimodal" in m else "#ff7f0e" for m in df_t["model"]]
        
        y_pos = np.arange(len(df_t))
        ax.barh(y_pos, df_t["balanced_accuracy"] * 100.0, color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_t["model"].str.replace("_", " "), fontsize=9)
        ax.set_xlabel("Transfer Balanced Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title("Zero-Shot Cross-Dataset Transfer to CinC 2011 Set A", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim(40, 105)
        
        for i, v in enumerate(df_t["balanced_accuracy"] * 100.0):
            ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
            
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "figure_cinc2011_transfer_comparison.png")
        plt.close()
        
    # 4. Figure: Latency vs Balanced Accuracy Tradeoff
    prof_csv = RESULTS_DIR / "exp6_computational_profile_summary.csv"
    if cv_csv.exists() and prof_csv.exists():
        df_p = pd.read_csv(prof_csv)
        df_c = pd.read_csv(cv_csv)
        
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        # Match names
        plot_pairs = [
            ("Logistic Regression (SQI)", "Logistic_Regression_SQI"),
            ("LightGBM (SQI)", "LightGBM_SQI"),
            ("1D CNN (Raw Signal)", "1D_CNN_RawECG"),
            ("2D CNN (Image)", "2D_CNN_Image"),
            ("SinglePass Multimodal Net (Fusion)", "Multimodal_Image_Plus_SQI"),
        ]
        
        for p_name, c_name in plot_pairs:
            p_row = df_p[df_p["Model"] == p_name]
            c_row = df_c[df_c["model"] == c_name]
            if len(p_row) > 0 and len(c_row) > 0:
                lat = p_row["Latency_ms"].values[0]
                acc = c_row["balanced_accuracy"].values[0] * 100.0
                ax.scatter(lat, acc, s=120, label=p_name)
                ax.annotate(p_name.split()[0], (lat, acc), textcoords="offset points", xytext=(5, 5), fontsize=8)
                
        ax.set_xscale("log")
        ax.set_xlabel("Inference Latency (ms, log-scale)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Balanced Accuracy on BUT QDB (%)", fontsize=11, fontweight="bold")
        ax.set_title("Performance vs Computational Latency Frontier", fontsize=12, fontweight="bold", pad=12)
        ax.legend(frameon=True, facecolor="white", loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "figure_accuracy_latency_frontier.png")
        plt.close()
        
    print(f"[Figures] All figures saved to {FIGURES_DIR}")

def main():
    parser = argparse.ArgumentParser(description="ECG Usability Assessment Multimodal Research Pipeline")
    parser.add_argument("--exp", type=str, default="pilot",
                        choices=["pilot", "cv", "transfer", "nstdb", "robustness", "profile", "all"],
                        help="Experiment to execute")
    args = parser.parse_args()
    
    t_start = time.time()
    
    if args.exp in ["pilot", "all"]:
        run_feasibility_pilot()
        
    if args.exp in ["cv", "all"]:
        run_butqdb_subject_cv()
        
    if args.exp in ["transfer", "all"]:
        run_transfer_cinc2011()
        
    if args.exp in ["nstdb", "all"]:
        run_nstdb_noise_stress()
        
    if args.exp in ["robustness", "all"]:
        run_imaging_robustness_test()
        
    if args.exp in ["profile", "all"]:
        run_computational_profile()
        
    generate_consolidated_figures()
    
    print("\n" + "=" * 80)
    print(f"PIPELINE EXECUTION COMPLETED IN {time.time() - t_start:.1f} SECONDS")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
