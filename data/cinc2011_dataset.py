"""
PhysioNet / Computing in Cardiology Challenge 2011 (Set A) Loader
- 1,000 10-second 12-lead ECGs (fs = 500 Hz)
- Gold standard reference labels:
    * RECORDS-acceptable -> Usable (0)
    * RECORDS-unacceptable -> Unusable (1)
- Evaluated as an independent cross-dataset transfer benchmark
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from config import DATA_DIR, TARGET_FS, DEFAULT_WINDOW_SEC, CLASS_USABLE, CLASS_UNUSABLE, CINC2011_URL, CINC2011_DEFAULT_LEAD
from features.sqi_extractor import extract_sqi_features
from data.ecg_rendering import render_ecg_image

CINC_DIR = DATA_DIR / "cinc2011"
CINC_DIR.mkdir(parents=True, exist_ok=True)

class CinC2011Manager:
    def __init__(self, cache_dir: Path = CINC_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_label_lists(self) -> Tuple[List[str], List[str]]:
        """Fetches acceptable and unacceptable record lists from PhysioNet."""
        acc_file = self.cache_dir / "RECORDS-acceptable.txt"
        unacc_file = self.cache_dir / "RECORDS-unacceptable.txt"
        
        if not acc_file.exists():
            url = f"{CINC2011_URL}RECORDS-acceptable"
            r = requests.get(url, timeout=15)
            with open(acc_file, "w") as f:
                f.write(r.text)
                
        if not unacc_file.exists():
            url = f"{CINC2011_URL}RECORDS-unacceptable"
            r = requests.get(url, timeout=15)
            with open(unacc_file, "w") as f:
                f.write(r.text)
                
        with open(acc_file) as f:
            acc_records = [line.strip() for line in f if line.strip()]
        with open(unacc_file) as f:
            unacc_records = [line.strip() for line in f if line.strip()]
            
        return acc_records, unacc_records

    def extract_segments(
        self,
        max_records: Optional[int] = 100,
        lead_idx: int = CINC2011_DEFAULT_LEAD, # Lead II by default
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Extracts segments and features from Challenge 2011 Set A.
        """
        cache_csv = self.cache_dir / f"cinc2011_segments_lead{lead_idx}_max{max_records}.csv"
        cache_signals_dir = self.cache_dir / "signals"
        cache_images_dir = self.cache_dir / "images"
        cache_signals_dir.mkdir(parents=True, exist_ok=True)
        cache_images_dir.mkdir(parents=True, exist_ok=True)
        
        if cache_csv.exists():
            if verbose:
                print(f"[CinC 2011] Loading cached segments from {cache_csv}")
            return pd.read_csv(cache_csv)
            
        acc_records, unacc_records = self.fetch_label_lists()
        
        # Sample or take up to max_records (balanced between classes)
        if max_records is not None:
            half = max_records // 2
            selected_acc = acc_records[:half]
            selected_unacc = unacc_records[:half]
        else:
            selected_acc = acc_records
            selected_unacc = unacc_records
            
        all_targets = [(r, CLASS_USABLE) for r in selected_acc] + [(r, CLASS_UNUSABLE) for r in selected_unacc]
        
        if verbose:
            print(f"[CinC 2011] Extracting {len(all_targets)} records (Lead idx={lead_idx})...")
            
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_record(item):
            rec_id, label = item
            seg_id = f"cinc2011_{rec_id}_lead{lead_idx}"
            sig_path = cache_signals_dir / f"{seg_id}.npy"
            img_path = cache_images_dir / f"{seg_id}.png"
            
            if sig_path.exists() and img_path.exists():
                raw_lead = np.load(sig_path)
                sqis = extract_sqi_features(raw_lead, 500)
                return {
                    "segment_id": seg_id,
                    "dataset": "CinC_2011_SetA",
                    "record_name": rec_id,
                    "lead_idx": lead_idx,
                    "label": int(label),
                    "signal_path": str(sig_path),
                    "image_path": str(img_path),
                    **sqis
                }
                
            try:
                rec = wfdb.rdrecord(rec_id, pn_dir="challenge-2011/1.0.0/set-a")
                raw_lead = rec.p_signal[:, lead_idx]
                fs = rec.fs
            except Exception:
                return None
                
            sqis = extract_sqi_features(raw_lead, fs)
            np.save(sig_path, raw_lead.astype(np.float32))
            img = render_ecg_image(raw_lead, fs, dpi=100)
            img.save(img_path)
            
            return {
                "segment_id": seg_id,
                "dataset": "CinC_2011_SetA",
                "record_name": rec_id,
                "lead_idx": lead_idx,
                "label": int(label),
                "signal_path": str(sig_path),
                "image_path": str(img_path),
                **sqis
            }

        segment_records = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_record, item) for item in all_targets]
            for f in tqdm(as_completed(futures), total=len(all_targets), disable=not verbose):
                res = f.result()
                if res is not None:
                    segment_records.append(res)
            
        df = pd.DataFrame(segment_records)
        df.to_csv(cache_csv, index=False)
        if verbose:
            print(f"[CinC 2011] Completed extraction. Records: {len(df)}")
            print(f"  Class distribution: {df['label'].value_counts().to_dict()}")
        return df
