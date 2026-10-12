"""
Independent Zero-Shot Neural Inference for LiquidAI/d1-3B
Evaluates genuine 3.12B weights on:
1. BUT QDB 1,047-Window Cohort (results/manifest_butqdb_1000_cohort.csv)
2. CinC 2011 Set A (cache/datasets/cinc2011/cinc2011_segments_lead1_max150.csv)

Strict Fail-Closed Architecture:
- No heuristic fallback.
- Incremental checkpointing per sample (resumes seamlessly if interrupted).
- Explicit metadata: segment_id, subject_id, backend="neural", model_name="LiquidAI/d1-3B",
  prob_unusable, prob_usable, choice, confidence, input_tokens, output_tokens (must be 0), latency_ms.
- Raw system_one response dictionary saved per sample to JSONL for complete auditability.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR
from features.sqi_extractor import SQI_FEATURE_NAMES
from models.d1_real_neural_model import RealD1DecisionModel

BUT_OUT_CSV = RESULTS_DIR / "raw_d1_neural_cohort1000_predictions.csv"
BUT_RAW_JSONL = RESULTS_DIR / "raw_d1_neural_cohort1000_responses.jsonl"
CINC_OUT_CSV = RESULTS_DIR / "raw_d1_neural_cinc2011_predictions.csv"
CINC_RAW_JSONL = RESULTS_DIR / "raw_d1_neural_cinc2011_responses.jsonl"


def run_inference_on_dataset(
    manifest_path: Path,
    out_csv: Path,
    out_jsonl: Path,
    dataset_name: str,
    device: str = "mps"
):
    print(f"\n=======================================================")
    print(f"RUNNING REAL d1-3B NEURAL INFERENCE ON: {dataset_name}")
    print(f"=======================================================")

    df = pd.read_csv(manifest_path)
    n_total = len(df)
    print(f"Dataset total samples: {n_total}")

    # Check already processed samples (Resumption support)
    processed_ids = set()
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        processed_ids = set(existing_df["segment_id"].unique())
        print(f"Found {len(processed_ids)} already completed samples. Resuming...")

    # Load Model (Fail-Closed)
    d1 = RealD1DecisionModel(device=device)

    # Open CSV for appending if exists, else write header
    csv_exists = out_csv.exists()
    f_csv = open(out_csv, "a", buffering=1) # line buffered
    if not csv_exists:
        header = (
            "segment_id,dataset,subject_id,record_name,label,fold_id,backend,model_name,"
            "prob_unusable,prob_usable,choice,confidence,input_tokens,output_tokens,latency_ms\n"
        )
        f_csv.write(header)

    f_jsonl = open(out_jsonl, "a", buffering=1)

    t_start = time.time()
    n_done_session = 0

    for idx, row in df.iterrows():
        seg_id = str(row["segment_id"])
        if seg_id in processed_ids:
            continue

        img_path = row["image_path"]
        with Image.open(img_path) as pil_img:
            img = pil_img.convert("RGB")

        sqi_d = {name: float(row[name]) for name in SQI_FEATURE_NAMES if name in row}
        subj_id = row.get("subject_id", row.get("record_name", "unknown"))
        rec_name = row.get("record_name", "unknown")
        label = int(row["label"])
        fold_id = row.get("fold_id", -1)

        # Fail-closed execution
        dec = d1.decide_single(img, sqi_d, modality="multimodal", segment_id=seg_id)

        # Write to CSV
        csv_line = (
            f"{seg_id},{dataset_name},{subj_id},{rec_name},{label},{fold_id},"
            f"{dec['backend']},{dec['model_name']},{dec['prob_unusable']:.6f},{dec['prob_usable']:.6f},"
            f"{dec['choice']},{dec['confidence']:.6f},{dec['input_tokens']},{dec['output_tokens']},"
            f"{dec['latency_ms']:.2f}\n"
        )
        f_csv.write(csv_line)

        # Write to JSONL
        jsonl_entry = {
            "segment_id": seg_id,
            "dataset": dataset_name,
            "raw_response": dec["raw_response"],
            "timestamp": time.time()
        }
        f_jsonl.write(json.dumps(jsonl_entry) + "\n")

        processed_ids.add(seg_id)
        n_done_session += 1

        if n_done_session % 25 == 0 or len(processed_ids) == n_total:
            elapsed = time.time() - t_start
            rate = n_done_session / max(1e-3, elapsed)
            rem = (n_total - len(processed_ids)) / max(1e-3, rate)
            print(
                f"[{dataset_name}] Processed {len(processed_ids)}/{n_total} "
                f"({len(processed_ids)/n_total*100:.1f}%) | "
                f"Rate: {rate:.1f} samp/s | Est remaining: {rem/60:.1f} min | "
                f"Last P(unusable): {dec['prob_unusable']:.4f}"
            )

    f_csv.close()
    f_jsonl.close()
    print(f"Finished inference on {dataset_name}. Results saved to {out_csv}")


def main():
    # 1. BUT QDB 1,047 Cohort
    but_manifest = RESULTS_DIR / "manifest_butqdb_1000_cohort.csv"
    run_inference_on_dataset(but_manifest, BUT_OUT_CSV, BUT_RAW_JSONL, dataset_name="BUT_QDB_1000")

    # 2. CinC 2011 Set A Transfer Cohort
    cinc_manifest = PROJECT_ROOT / "cache" / "datasets" / "cinc2011" / "cinc2011_segments_lead1_max150.csv"
    if cinc_manifest.exists():
        run_inference_on_dataset(cinc_manifest, CINC_OUT_CSV, CINC_RAW_JSONL, dataset_name="CinC_2011_SetA")


if __name__ == "__main__":
    main()
