"""
Experiment 1: Small Phased Feasibility Pilot (Study Requirement #7)
Objective:
- Prepare separate pilot groups:
    * BUT QDB segments from distinct subjects: clearly high quality (Class 1) vs clearly unusable (Class 3).
    * CinC 2011 Set A independent pilot transfer subset.
- Evaluate Image-only, SQI-only, and Multimodal (Image + SQI) on the EXACT same samples using the fixed decision interface.
- Determine if multimodal input provides genuine discriminative gain before scaling to full experiments.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR, FIGURES_DIR, CLASS_USABLE, CLASS_UNUSABLE
from data.butqdb_dataset import BUTQDBManager
from data.cinc2011_dataset import CinC2011Manager
from features.sqi_extractor import SQI_FEATURE_NAMES, sqi_dict_to_vector
from models.d1_decision_interface import D1DecisionModelWrapper
from evaluation.metrics import compute_usability_metrics

def run_feasibility_pilot(num_pilot_samples_per_class: int = 15, verbose: bool = True):
    print("=" * 80)
    print("PHASE 1: FEASIBILITY PILOT EXPERIMENT")
    print("Assessing modality contribution (Image vs SQI vs Multimodal) on BUT QDB & CinC 2011")
    print("=" * 80)
    
    # 1. Prepare BUT QDB Pilot Subset
    print("\n[Step 1] Preparing BUT QDB pilot segments across distinct subjects...")
    but_mgr = BUTQDBManager()
    but_df = but_mgr.extract_segments(max_segments_per_subject=20, verbose=verbose)
    
    # Select clean (Class 1) and unusable (Class 3) across distinct subjects
    clean_pool = but_df[(but_df["original_quality_class"] == 1) & (but_df["label"] == CLASS_USABLE)]
    unusable_pool = but_df[(but_df["original_quality_class"] == 3) & (but_df["label"] == CLASS_UNUSABLE)]
    
    # Stratify by subject
    clean_sample = clean_pool.drop_duplicates(subset=["subject_id"]).head(num_pilot_samples_per_class)
    if len(clean_sample) < num_pilot_samples_per_class:
        clean_sample = clean_pool.head(num_pilot_samples_per_class)
        
    unusable_sample = unusable_pool.head(num_pilot_samples_per_class)
    pilot_but = pd.concat([clean_sample, unusable_sample], ignore_index=True)
    print(f"  BUT QDB pilot set size: {len(pilot_but)} (Usable: {sum(pilot_but['label']==0)}, Unusable: {sum(pilot_but['label']==1)})")
    print(f"  Distinct subjects in pilot: {pilot_but['subject_id'].nunique()}")
    
    # 2. Prepare CinC 2011 Pilot Transfer Subset
    print("\n[Step 2] Preparing CinC 2011 Set A pilot transfer subset...")
    cinc_mgr = CinC2011Manager()
    cinc_df = cinc_mgr.extract_segments(max_records=30, verbose=verbose)
    pilot_cinc = cinc_df.head(2 * num_pilot_samples_per_class)
    print(f"  CinC 2011 pilot set size: {len(pilot_cinc)} (Usable: {sum(pilot_cinc['label']==0)}, Unusable: {sum(pilot_cinc['label']==1)})")

    # 3. Decision Model Initialization
    print("\n[Step 3] Initializing Decision Model Interface with standard schema...")
    decision_model = D1DecisionModelWrapper()
    
    # 4. Evaluate 3 Modalities on BUT QDB Pilot
    modalities = ["image_only", "sqi_only", "multimodal"]
    pilot_results = []
    
    for dataset_name, df_eval in [("BUT_QDB_Pilot", pilot_but), ("CinC2011_Pilot_Transfer", pilot_cinc)]:
        print(f"\n--- Evaluating on {dataset_name} ({len(df_eval)} samples) ---")
        y_true = df_eval["label"].values
        
        for mod in modalities:
            probs = []
            for _, row in df_eval.iterrows():
                img = Image.open(row["image_path"])
                sqi_dict = {k: float(row[k]) for k in SQI_FEATURE_NAMES if k in row}
                
                dec = decision_model.decide_single(img=img, sqi_dict=sqi_dict, modality=mod)
                probs.append(dec["prob_unusable"])
                
            probs = np.array(probs)
            m = compute_usability_metrics(y_true, probs)
            m["dataset"] = dataset_name
            m["modality"] = mod
            pilot_results.append(m)
            
            print(f"  [{mod.upper():12s}] BalAcc: {m['balanced_accuracy']*100:.1f}% | "
                  f"Unusable Sens: {m['unusable_sensitivity']*100:.1f}% | "
                  f"False Reject Rate: {m['usable_false_reject_rate']*100:.1f}% | "
                  f"AUROC: {m['auroc']:.3f} | Brier: {m['brier_score']:.3f}")
                  
    df_pilot_res = pd.DataFrame(pilot_results)
    out_csv = RESULTS_DIR / "exp1_feasibility_pilot_summary.csv"
    df_pilot_res.to_csv(out_csv, index=False)
    print(f"\n[Pilot Completed] Summary saved to {out_csv}")
    
    return df_pilot_res

if __name__ == "__main__":
    run_feasibility_pilot()
