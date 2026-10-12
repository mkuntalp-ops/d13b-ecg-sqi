"""
Revised Experiment 5: Rendering Perturbation & Rasterization Sensitivity on All OOF Windows
Methodological Standards:
- Evaluates ALL 124 out-of-fold test windows using their corresponding fold models.
- Measures observed empirical effects without unproven mechanistic speculations:
    * Mean Absolute Error (MAE) from baseline probability
    * Decision flip rate (%)
    * Balanced accuracy under each perturbation style
    * Class- and subject-level impacts
    * True pixel dimensions before and after network resize (128x256)
- Styles tested:
    1. baseline_clinical: 100 DPI, 1.0 pt, red grid
    2. low_dpi_50: 50 DPI rasterization (coarse grid/waveform)
    3. high_dpi_150: 150 DPI rasterization
    4. thin_line_0.5: 0.5 pt stroke width
    5. thick_line_2.5: 2.5 pt stroke width
    6. no_grid: Grid lines removed entirely
    7. gray_grid: Neutral gray grid
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from config import RESULTS_DIR, FIGURES_DIR, TARGET_FS
from features.sqi_extractor import SQI_FEATURE_NAMES
from data.ecg_rendering import render_perturbed_ecg_variants
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier

def run_revised_imaging_robustness():
    print("=" * 80)
    print("REVISED EXPERIMENT 5: FULL OUT-OF-FOLD RENDERING & RASTERIZATION SENSITIVITY")
    print("=" * 80)
    
    manifest_path = RESULTS_DIR / "manifest_butqdb_subject_folds.csv"
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(df)} segments across {df['subject_id'].nunique()} subjects.")
    
    X_sqi = np.nan_to_num(df[SQI_FEATURE_NAMES].values, nan=0.0)
    raw_signals = [np.load(p) for p in df["signal_path"]]
    baseline_img_paths = df["image_path"].tolist()
    y = df["label"].values
    fold_ids = df["fold_id"].values
    
    perturb_dir = config.CACHE_DIR / "datasets" / "butqdb" / "full_oof_perturbations"
    perturb_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load exact fold model checkpoints saved by Experiment 2
    print("\nLoading exact fold model checkpoints saved by Experiment 2...")
    ckpt_dir = config.CACHE_DIR / "checkpoints"
    fold_models_2d = {}
    fold_models_mm = {}
    
    for fold in range(5):
        m_2d = ECG2DCNNClassifier()
        ckpt_2d = ckpt_dir / f"cnn2d_fold_{fold}.pt"
        if not ckpt_2d.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_2d}. Run Exp 2 first.")
        m_2d.net.load_state_dict(torch.load(ckpt_2d, map_location=m_2d.device))
        m_2d.net.eval()
        
        m_mm = SinglePassMultimodalClassifier(mode="multimodal")
        ckpt_mm = ckpt_dir / f"mm_fold_{fold}.pt"
        if not ckpt_mm.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_mm}. Run Exp 2 first.")
        m_mm.net.load_state_dict(torch.load(ckpt_mm, map_location=m_mm.device))
        m_mm.net.eval()
        
        fold_models_2d[fold] = m_2d
        fold_models_mm[fold] = m_mm
        
    # 2. Render all 124 segments across the 7 perturbation styles
    print("\nRendering all 124 OOF test signals across all 7 styles...")
    style_names = ["baseline_clinical", "low_dpi_50", "high_dpi_150", "thin_line_0.5", "thick_line_2.5", "no_grid", "gray_grid"]
    style_img_paths = {style: [] for style in style_names}
    
    for i, sig in enumerate(raw_signals):
        seg_id = df.loc[i, "segment_id"]
        # Check if all 7 exist
        all_exist = all((perturb_dir / f"{seg_id}_{st}.png").exists() for st in style_names)
        if not all_exist:
            variants = render_perturbed_ecg_variants(sig, TARGET_FS)
            for st, img in variants.items():
                img.save(perturb_dir / f"{seg_id}_{st}.png")
        for st in style_names:
            style_img_paths[st].append(str(perturb_dir / f"{seg_id}_{st}.png"))
            
    # Check rendered pixel dimensions vs network input
    sample_img_50 = Image.open(style_img_paths["low_dpi_50"][0])
    sample_img_100 = Image.open(style_img_paths["baseline_clinical"][0])
    sample_img_150 = Image.open(style_img_paths["high_dpi_150"][0])
    
    print("\nRendered pixel dimensions (Canvas Width x Height before 128x256 resize):")
    print(f"  low_dpi_50:         {sample_img_50.size} -> Resized to (256, 128)")
    print(f"  baseline_clinical:  {sample_img_100.size} -> Resized to (256, 128)")
    print(f"  high_dpi_150:       {sample_img_150.size} -> Resized to (256, 128)")
    
    # 3. Predict out-of-fold for baseline and each perturbation style
    print("\nEvaluating OOF predictions across all 124 segments for each style...")
    oof_probs_2d = {}
    oof_probs_mm = {}
    
    for style_name, img_paths in style_img_paths.items():
        p_2d = np.zeros(len(df), dtype=np.float32)
        p_mm = np.zeros(len(df), dtype=np.float32)
        
        for fold in range(5):
            test_mask = (fold_ids == fold)
            test_idx = np.where(test_mask)[0]
            
            fold_imgs = [img_paths[idx] for idx in test_idx]
            fold_sqi = X_sqi[test_idx]
            
            p_2d[test_idx] = fold_models_2d[fold].predict_proba(fold_imgs)
            p_mm[test_idx] = fold_models_mm[fold].predict_proba(fold_imgs, fold_sqi)
            
        oof_probs_2d[style_name] = p_2d
        oof_probs_mm[style_name] = p_mm
        
    # Baseline probabilities
    base_p_2d = oof_probs_2d["baseline_clinical"]
    base_p_mm = oof_probs_mm["baseline_clinical"]
    base_pred_2d = (base_p_2d >= 0.5).astype(int)
    base_pred_mm = (base_p_mm >= 0.5).astype(int)
    
    summary_data = []
    
    for style_name in style_names:
        p_2d = oof_probs_2d[style_name]
        p_mm = oof_probs_mm[style_name]
        pred_2d = (p_2d >= 0.5).astype(int)
        pred_mm = (p_mm >= 0.5).astype(int)
        
        # 2D CNN metrics
        mae_2d = float(np.mean(np.abs(p_2d - base_p_2d)))
        flip_2d = float(np.mean(pred_2d != base_pred_2d) * 100.0)
        bal_acc_2d = float(balanced_accuracy_score(y, pred_2d) * 100.0)
        auc_2d = float(roc_auc_score(y, p_2d))
        
        # Multimodal metrics
        mae_mm = float(np.mean(np.abs(p_mm - base_p_mm)))
        flip_mm = float(np.mean(pred_mm != base_pred_mm) * 100.0)
        bal_acc_mm = float(balanced_accuracy_score(y, pred_mm) * 100.0)
        auc_mm = float(roc_auc_score(y, p_mm))
        
        row = {
            "style": style_name,
            "cnn2d_bal_acc": bal_acc_2d,
            "cnn2d_auroc": auc_2d,
            "cnn2d_mae_from_base": mae_2d,
            "cnn2d_decision_flip_pct": flip_2d,
            "multimodal_bal_acc": bal_acc_mm,
            "multimodal_auroc": auc_mm,
            "multimodal_mae_from_base": mae_mm,
            "multimodal_decision_flip_pct": flip_mm
        }
        summary_data.append(row)
        
        print(f"\nStyle: {style_name}")
        print(f"  2D CNN:     BalAcc={bal_acc_2d:.1f}%, AUROC={auc_2d:.3f}, MAE={mae_2d:.3f}, Flips={flip_2d:.1f}%")
        print(f"  Multimodal: BalAcc={bal_acc_mm:.1f}%, AUROC={auc_mm:.3f}, MAE={mae_mm:.3f}, Flips={flip_mm:.1f}%")
        
    df_summary = pd.DataFrame(summary_data)
    out_csv = RESULTS_DIR / "exp5_revised_imaging_robustness_summary.csv"
    df_summary.to_csv(out_csv, index=False)
    
    # Export sample-by-sample checkpoint verification against Exp 2 OOF
    oof_exp2_path = RESULTS_DIR / "exp2_revised_oof_predictions.csv"
    if oof_exp2_path.exists():
        oof_exp2_df = pd.read_csv(oof_exp2_path)
        verif_df = pd.DataFrame({
            "segment_id": df["segment_id"],
            "fold_id": df["fold_id"],
            "checkpoint_id": [f"mm_fold_{f}.pt" for f in df["fold_id"]],
            "threshold": 0.5,
            "original_probability": oof_exp2_df["prob_Multimodal_Fusion"],
            "reference_render_probability": base_p_mm,
            "original_prediction": oof_exp2_df["pred_fixed_Multimodal_Fusion"],
            "reference_render_prediction": base_pred_mm
        })
        verif_csv = RESULTS_DIR / "perturbation_checkpoint_verification.csv"
        verif_df.to_csv(verif_csv, index=False)
        max_prob_diff = np.max(np.abs(verif_df["original_probability"] - verif_df["reference_render_probability"]))
        pred_mismatches = np.sum(verif_df["original_prediction"] != verif_df["reference_render_prediction"])
        print(f"\n[Verification Audit] Max probability diff vs Exp 2: {max_prob_diff:.6f}")
        print(f"[Verification Audit] Total prediction mismatches vs Exp 2: {pred_mismatches} / {len(verif_df)}")
        print(f"[Verification Audit] Verification table saved to {verif_csv}")
    
    # Generate visualization
    plt.figure(figsize=(10, 5), dpi=300)
    x = np.arange(len(df_summary))
    width = 0.35
    
    plt.bar(x - width/2, df_summary["cnn2d_decision_flip_pct"], width, label="2D CNN (Image Only)", color="#d62728", alpha=0.85)
    plt.bar(x + width/2, df_summary["multimodal_decision_flip_pct"], width, label="Multimodal (Image + SQI)", color="#2ca02c", alpha=0.85)
    
    plt.xlabel("Rendering Perturbation & Rasterization Style", fontsize=11, fontweight="bold")
    plt.ylabel("Decision Flip Rate vs Baseline (%)", fontsize=11, fontweight="bold")
    plt.title("Empirical Decision Flip Rate Across All 124 OOF ECG Windows", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(x, [s.replace("_", "\n") for s in df_summary["style"]], fontsize=9)
    plt.legend(frameon=True, facecolor="white")
    plt.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_revised_imaging_robustness_flip_rate.png")
    plt.close()
    
    print(f"\n[Completed] Full OOF perturbation summary saved to {out_csv}")
    print(f"Chart saved to {FIGURES_DIR / 'figure_revised_imaging_robustness_flip_rate.png'}")
    return df_summary

if __name__ == "__main__":
    run_revised_imaging_robustness()
