"""
Experiment 5: Imaging Perturbation & Rendering Style Robustness Benchmark
(Study Requirement #5.D)

- Renders the EXACT same ECG signals under 7 distinct visual styles:
    1. Baseline standard clinical (100 DPI, 1.0 pt, pink grid)
    2. Low resolution (50 DPI)
    3. High resolution (150 DPI)
    4. Thin line stroke (0.5 pt)
    5. Thick line stroke (2.5 pt)
    6. No-grid plain background
    7. Monochrome gray grid
- Evaluates whether pure image models (2D CNN) vs Multimodal models (Vision + SQI) learn
  actual ECG signal quality or overfit to plotting artifacts/styles.
- Measures prediction stability (MAE vs baseline) and decision flip rates.
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

from config import RESULTS_DIR, FIGURES_DIR, RANDOM_SEED, TARGET_FS
from data.butqdb_dataset import BUTQDBManager
from data.ecg_rendering import render_perturbed_ecg_variants
from features.sqi_extractor import SQI_FEATURE_NAMES
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier
from evaluation.metrics import compute_usability_metrics

def run_imaging_robustness_test(num_eval_samples: int = 50, verbose: bool = True):
    print("=" * 80)
    print("EXPERIMENT 5: IMAGING PERTURBATION & RENDERING ROBUSTNESS TEST")
    print("=" * 80)
    
    # 1. Load BUT QDB data
    but_mgr = BUTQDBManager()
    but_df = but_mgr.extract_segments(max_segments_per_subject=40, verbose=verbose)
    but_df["subject_id"] = but_df["subject_id"].astype(str)
    
    # Split train and test by subject to keep clean test set
    test_subjs = ["122", "124", "125"] # subjects with both classes
    train_df = but_df[~but_df["subject_id"].isin(test_subjs)]
    test_df = but_df[but_df["subject_id"].isin(test_subjs)].head(num_eval_samples)
    
    print(f"Train samples: {len(train_df)}, Perturbation Test samples: {len(test_df)}")
    
    # Train 2D CNN and Multimodal on baseline styles
    X_train_sqi = train_df[SQI_FEATURE_NAMES].values.astype(np.float32)
    y_train = train_df["label"].values.astype(int)
    train_imgs = train_df["image_path"].tolist()
    
    print("\nTraining models on standard clinical rendered ECGs...")
    cnn2d = ECG2DCNNClassifier(epochs=12, batch_size=32).fit(train_imgs, y_train)
    mm_clf = SinglePassMultimodalClassifier(epochs=15, batch_size=32, mode="multimodal").fit(train_imgs, X_train_sqi, y_train)
    
    # 2. Generate and evaluate perturbed images on test set
    temp_dir = RESULTS_DIR / "temp_perturbations"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    X_test_sqi = test_df[SQI_FEATURE_NAMES].values.astype(np.float32)
    y_test = test_df["label"].values.astype(int)
    
    print("\nRendering perturbed image sets across 7 styles for each test signal...")
    variant_images = {
        "baseline_clinical": [],
        "low_dpi_50": [],
        "high_dpi_150": [],
        "thin_line_0.5": [],
        "thick_line_2.5": [],
        "no_grid": [],
        "gray_grid": []
    }
    
    for idx, row in test_df.iterrows():
        raw_sig = np.load(row["signal_path"])
        variants = render_perturbed_ecg_variants(raw_sig, fs=TARGET_FS)
        seg_id = row["segment_id"]
        
        for v_name, v_img in variants.items():
            img_p = temp_dir / f"{seg_id}_{v_name}.png"
            v_img.save(img_p)
            variant_images[v_name].append(str(img_p))
            
    # 3. Predict across each visual style
    print("\nEvaluating 2D CNN (Image-Only) vs Multimodal (Image + SQI) under perturbations...")
    robustness_results = []
    
    # Baseline predictions
    base_probs_cnn2d = cnn2d.predict_proba(variant_images["baseline_clinical"])
    base_probs_mm = mm_clf.predict_proba(variant_images["baseline_clinical"], X_test_sqi)
    
    for v_name, img_list in variant_images.items():
        probs_cnn2d = cnn2d.predict_proba(img_list)
        probs_mm = mm_clf.predict_proba(img_list, X_test_sqi)
        
        # Stability metrics relative to baseline
        mae_cnn2d = np.mean(np.abs(probs_cnn2d - base_probs_cnn2d))
        mae_mm = np.mean(np.abs(probs_mm - base_probs_mm))
        
        flip_cnn2d = np.mean((probs_cnn2d >= 0.5) != (base_probs_cnn2d >= 0.5)) * 100.0
        flip_mm = np.mean((probs_mm >= 0.5) != (base_probs_mm >= 0.5)) * 100.0
        
        m_cnn = compute_usability_metrics(y_test, probs_cnn2d)
        m_mm = compute_usability_metrics(y_test, probs_mm)
        
        entry = {
            "style": v_name,
            "cnn2d_bal_acc": m_cnn["balanced_accuracy"] * 100.0,
            "cnn2d_auroc": m_cnn["auroc"],
            "cnn2d_mae_from_base": mae_cnn2d,
            "cnn2d_decision_flip_pct": flip_cnn2d,
            "multimodal_bal_acc": m_mm["balanced_accuracy"] * 100.0,
            "multimodal_auroc": m_mm["auroc"],
            "multimodal_mae_from_base": mae_mm,
            "multimodal_decision_flip_pct": flip_mm
        }
        robustness_results.append(entry)
        
        print(f"\nStyle: {v_name}")
        print(f"  2D CNN:     BalAcc: {m_cnn['balanced_accuracy']*100:.1f}%, MAE: {mae_cnn2d:.3f}, Flips: {flip_cnn2d:.1f}%")
        print(f"  Multimodal: BalAcc: {m_mm['balanced_accuracy']*100:.1f}%, MAE: {mae_mm:.3f}, Flips: {flip_mm:.1f}%")
        
    df_rob = pd.DataFrame(robustness_results)
    out_csv = RESULTS_DIR / "exp5_imaging_robustness_summary.csv"
    df_rob.to_csv(out_csv, index=False)
    
    # 4. Generate Robustness Comparison Chart
    print("\nGenerating visual robustness chart...")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    x = np.arange(len(df_rob))
    width = 0.35
    
    ax.bar(x - width/2, df_rob["cnn2d_decision_flip_pct"], width, label="2D CNN (Image Only)", color="#d95f02", alpha=0.9)
    ax.bar(x + width/2, df_rob["multimodal_decision_flip_pct"], width, label="Multimodal (Image + SQI)", color="#7570b3", alpha=0.9)
    
    ax.set_ylabel("Decision Flip Rate vs Baseline (%)", fontsize=11, fontweight="bold")
    ax.set_title("Vulnerability to Visual Perturbations: Image-Only vs Multimodal Decision Model", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df_rob["style"], rotation=25, ha="right")
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "figure_imaging_robustness_flip_rate.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[Completed] Robustness chart saved to {fig_path}")
    
    return df_rob

if __name__ == "__main__":
    run_imaging_robustness_test()
