"""
Audited Publication-Grade Figure Generation Script
Implements all specific revisions:
- Figure 1: Combined 2-panel figure (Panel A: ROC, Panel B: PR) for BUT QDB.
- Figure 2: Combined 2-panel figure (Panel A: ROC, Panel B: PR) for CinC 2011 Set A.
- Figure 3: Combined 2-panel figure for NSTDB (Panel A: Clean Control Baseline, Panel B: Calibrated SNR Degradation with ±1 SD bands).
- Figure 4: Log-scale latency vs. AUROC trade-off frontier matching Table 4.
- Generates 300 DPI PNG and vector PDF for all figures.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font="sans-serif", font_scale=1.1)

# Standardized Model Palette & Display Names across all 8 models
MODELS_META = {
    "1D_CNN_RawECG": {
        "display_name": "1D CNN (Raw Signal)",
        "color": "#d62728",
        "linestyle": "-",
        "profile_name": "1D CNN (Raw Signal)"
    },
    "2D_CNN_Image": {
        "display_name": "2D CNN (Rendered Image)",
        "color": "#9467bd",
        "linestyle": "-",
        "profile_name": "2D CNN (Rendered Image)"
    },
    "Multimodal_Fusion": {
        "display_name": "Multimodal Net",
        "color": "#8c564b",
        "linestyle": "-",
        "profile_name": "SinglePass Multimodal Net"
    },
    "LiquidAI_d1_3B_Neural": {
        "display_name": "Liquid AI d1-3B (Zero-Shot)",
        "color": "#7f7f7f",
        "linestyle": "-.",
        "profile_name": "Liquid AI d1-3B (Real Neural Model)"
    },
    "LightGBM_SQI": {
        "display_name": "LightGBM (SQI)",
        "color": "#ff7f0e",
        "linestyle": ":",
        "profile_name": None
    },
    "Logistic_Regression_SQI": {
        "display_name": "Logistic Regression (SQI)",
        "color": "#1f77b4",
        "linestyle": ":",
        "profile_name": "Logistic Regression (SQI)"
    },
    "Independent_SQI_MLP": {
        "display_name": "Independent SQI MLP",
        "color": "#2ca02c",
        "linestyle": "--",
        "profile_name": "Independent SQI MLP"
    },
    "Rule_Based_SQI_Heuristic": {
        "display_name": "Rule-Based Heuristic",
        "color": "#e377c2",
        "linestyle": "--",
        "profile_name": "Rule-Based SQI Heuristic"
    }
}


def generate_combined_roc_pr_figure(csv_path: Path, label_col: str, title_main: str, file_prefix: str, pos_label_name: str = "Unusable"):
    """
    Generates a single 2-panel figure containing Panel A (ROC) and Panel B (PR).
    """
    df = pd.read_csv(csv_path)
    y_true = df[label_col].values.astype(int)
    n_samples = len(df)
    prevalence = np.mean(y_true)

    print(f"\n[Figure Generation] {file_prefix}: {n_samples} samples, {pos_label_name} prevalence: {prevalence*100:.1f}%")

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

    # Panel A Baselines
    ax_roc.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.4, label="Chance (AUROC = 0.500)")
    # Panel B Baselines
    ax_pr.axhline(prevalence, color="black", linestyle="--", alpha=0.4, label=f"Prevalence Baseline (AP = {prevalence:.3f})")

    for m_key, meta in MODELS_META.items():
        prob_col = None
        for cand in [f"prob_{m_key}", f"p_unusable_{m_key}", m_key]:
            if cand in df.columns:
                prob_col = cand
                break
        if prob_col is None:
            raise KeyError(f"Missing column for {m_key} in {csv_path.name}")

        probs = df[prob_col].values

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(
            fpr, tpr,
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=2.0,
            label=f"{meta['display_name']} ({roc_auc:.3f})"
        )

        # PR Curve
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap_score = average_precision_score(y_true, probs)
        ax_pr.plot(
            rec, prec,
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=2.0,
            label=f"{meta['display_name']} ({ap_score:.3f})"
        )

    # Styling Panel A
    ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    ax_roc.set_ylabel(f"True Positive Rate (Sensitivity for {pos_label_name})", fontsize=11, fontweight="bold")
    ax_roc.set_title("A. Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold", pad=8)
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([-0.01, 1.02])
    ax_roc.legend(loc="lower right", fontsize=8.2, frameon=True, facecolor="white", framealpha=0.9)

    # Styling Panel B
    ax_pr.set_xlabel(f"Recall (Sensitivity for {pos_label_name})", fontsize=11, fontweight="bold")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax_pr.set_title("B. Precision-Recall Curve (Average Precision)", fontsize=12, fontweight="bold", pad=8)
    ax_pr.set_xlim([-0.01, 1.01])
    ax_pr.set_ylim([-0.01, 1.02])
    ax_pr.legend(loc="lower left", fontsize=8.2, frameon=True, facecolor="white", framealpha=0.9)

    fig.suptitle(title_main, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = FIGURES_DIR / f"{file_prefix}.png"
    pdf_path = FIGURES_DIR / f"{file_prefix}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved combined 2-panel figure to {png_path.name} and {pdf_path.name}")


def generate_figure3_nstdb_degradation():
    """
    Generates Figure 3:
    Panel A: Clean Control Baseline Distributions (Mean ± 1 SD)
    Panel B: Calibrated SNR Degradation Curves (+24 dB to -6 dB) with ±1 SD shading.
    """
    summary_csv = RESULTS_DIR / "exp4_revised_nstdb_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing {summary_csv}")

    df_summary = pd.read_csv(summary_csv)
    comb_df = df_summary[df_summary["subset"] == "Combined_118_119_ActiveNoise"]
    clean_df = df_summary[df_summary["subset"] == "Clean_Control_Windows"]

    fig, (ax_clean, ax_deg) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300, gridspec_kw={'width_ratios': [1, 2.5]})

    model_plot_order = [
        ("LR", "#1f77b4", "Logistic Regression (SQI)"),
        ("LGBM", "#ff7f0e", "LightGBM (SQI)"),
        ("1D_CNN", "#d62728", "1D CNN (Raw Signal)"),
        ("2D_CNN", "#9467bd", "2D CNN (Rendered Image)"),
        ("Multimodal", "#8c564b", "Multimodal Net"),
        ("d1_3B", "#7f7f7f", "Liquid AI d1-3B (Zero-Shot)")
    ]

    # Panel A: Clean Control Baseline
    # Clean control represents baseline non-corrupted segments
    clean_means = [clean_df[f"{m}_mean"].values[0] for m, _, _ in model_plot_order]
    clean_stds = [clean_df[f"{m}_std"].values[0] for m, _, _ in model_plot_order]
    y_pos = np.arange(len(model_plot_order))

    ax_clean.barh(y_pos, clean_means, xerr=clean_stds, align='center', color=[c for _, c, _ in model_plot_order], alpha=0.85, capsize=4, edgecolor='black', linewidth=0.8)
    ax_clean.set_yticks(y_pos)
    ax_clean.set_yticklabels([l for _, _, l in model_plot_order], fontsize=9.5, fontweight='bold')
    ax_clean.invert_yaxis()
    ax_clean.set_xlabel("Mean P(Usable) ± 1 SD", fontsize=10.5, fontweight='bold')
    ax_clean.set_title("A. Clean Reference Baseline (N=88)", fontsize=11.5, fontweight='bold', pad=8)
    ax_clean.set_xlim([0.0, 1.05])
    ax_clean.axvline(0.5, color="black", linestyle="--", alpha=0.5)

    # Panel B: Calibrated SNR Degradation
    snrs = comb_df["snr_db"].values
    for m, col, label in model_plot_order:
        means = comb_df[f"{m}_mean"].values
        stds = comb_df[f"{m}_std"].values
        ax_deg.plot(snrs, means, "o-", label=label, color=col, linewidth=2.0, markersize=6)
        ax_deg.fill_between(snrs, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1), color=col, alpha=0.15)

    ax_deg.axhline(0.5, color="black", linestyle="--", alpha=0.5, label="Default Triage Threshold (τ = 0.50)")
    ax_deg.invert_xaxis()
    ax_deg.set_xlabel("Electrode Motion (em) Noise SNR (dB, degrading ->)", fontsize=11, fontweight="bold")
    ax_deg.set_ylabel("Mean Predicted Usability Probability P(Usable)", fontsize=11, fontweight="bold")
    ax_deg.set_title("B. Active Noise Degradation Across SNR Tiers (N=44 windows per SNR tier, ±1 SD shaded)", fontsize=11.5, fontweight="bold", pad=8)
    ax_deg.set_ylim([-0.02, 1.05])
    ax_deg.legend(frameon=True, facecolor="white", loc="best", fontsize=8.5)

    fig.suptitle("Figure 3: Predicted Usability Probability on MIT-BIH Noise Stress Test Database (NSTDB Records 118 & 119)", fontsize=12.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = FIGURES_DIR / "figure_revised_nstdb_degradation_curve.png"
    pdf_path = FIGURES_DIR / "figure_revised_nstdb_degradation_curve.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved Figure 3 to {png_path.name} and {pdf_path.name}")


def generate_figure4_latency_tradeoff():
    """
    Generates Figure 4:
    Log-scale measured end-to-end latency vs. BUT QDB AUROC.
    """
    lat_csv = RESULTS_DIR / "empirical_computational_profile_summary.csv"
    perf_csv = RESULTS_DIR / "cohort1000_consolidated_summary.csv"

    df_lat = pd.read_csv(lat_csv)
    df_perf = pd.read_csv(perf_csv).set_index("model")

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=300)

    # Latency from the profiling run; AUROC from the regenerated BUT QDB cohort summary (no hard-coded values)
    # (display name, latency-table name substring, AUROC key in the cohort summary, colour)
    profile_models = [
        ("1D CNN (Raw Signal)", "1D CNN", "1D_CNN_RawECG", "#d62728"),
        ("2D CNN (Rendered Image)", "2D CNN", "2D_CNN_Image", "#9467bd"),
        ("SinglePass Multimodal Net", "Multimodal Net", "Multimodal_Fusion", "#8c564b"),
        ("d1-3B (Zero-Shot)", "d1-3B", "LiquidAI_d1_3B_Neural", "#7f7f7f"),
    ]

    for name, lat_key, auroc_key, color in profile_models:
        lat_rows = df_lat[df_lat["Model"].str.contains(lat_key, regex=False)]
        if len(lat_rows) != 1:
            raise ValueError(f"Expected one latency row matching '{lat_key}', found {len(lat_rows)}")
        lat = float(lat_rows["Total_End_to_End_Median_ms"].iloc[0])
        auroc = float(df_perf.loc[auroc_key, "auroc"])
        ax.scatter(lat, auroc, s=200, color=color, edgecolors="black", linewidth=1.5, zorder=5)
        ax.text(lat, auroc + 0.012, f"{name}\n({lat:.2f} ms, AUROC {auroc:.3f})", fontsize=9.5, fontweight="bold", ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("Measured End-to-End Latency per 10-s Window (ms) [Log Scale]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Patient-Independent Development AUROC (BUT QDB)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 4: Computational Latency vs. Diagnostic Discrimination Trade-Off", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim([0.80, 1.02])
    ax.set_xlim([0.5, 2500.0])
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    # Explanatory text annotation
    ax.text(0.03, 0.05, 
            "* Latencies measured on Apple M2 (24 GB RAM) via synchronized torch.mps.synchronize().\n"
            "  1D CNN operates on standardized cached signal; 2D and multimodal models include visual rendering (Table 4).",
            transform=ax.transAxes, fontsize=8.5, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9))

    fig.tight_layout()
    png_path = FIGURES_DIR / "latency_vs_auroc.png"
    pdf_path = FIGURES_DIR / "latency_vs_auroc.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved Figure 4 to {png_path.name} and {pdf_path.name}")


def main():
    print("=" * 80)
    print("REGENERATING AUDITED PUBLICATION FIGURES (300 DPI PNG & VECTOR PDF)")
    print("=" * 80)

    # 1. Figure 1: BUT QDB Combined 2-panel (ROC & PR)
    but_csv = RESULTS_DIR / "cohort1000_consolidated_oof_predictions.csv"
    generate_combined_roc_pr_figure(
        csv_path=but_csv,
        label_col="label",
        title_main="Figure 1: Patient-Independent Cross-Validation on BUT QDB (1,047 Windows, 15 Subjects)",
        file_prefix="figure_1_butqdb_combined_roc_pr",
        pos_label_name="Unusable"
    )

    # 2. Figure 2: CinC 2011 Set A Combined 2-panel (ROC & PR)
    cinc_csv = RESULTS_DIR / "cinc2011_expanded_predictions.csv"
    generate_combined_roc_pr_figure(
        csv_path=cinc_csv,
        label_col="label",
        title_main="Figure 2: Out-of-Distribution Transfer on PhysioNet/CinC 2011 Set A (150 Records, Lead II)",
        file_prefix="figure_2_cinc2011_combined_roc_pr",
        pos_label_name="Unacceptable"
    )

    # 3. Figure 3: NSTDB Degradation with Clean Control Panel
    generate_figure3_nstdb_degradation()

    # 4. Figure 4: Measured Latency vs AUROC Trade-Off Frontier
    generate_figure4_latency_tradeoff()

    print("\n[ALL 4 REVISED PUBLICATION FIGURES GENERATED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
