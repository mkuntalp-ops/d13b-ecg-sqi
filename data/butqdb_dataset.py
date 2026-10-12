"""
BUT ECG Quality Database (BUT QDB) Loader & Processor
- 18 long single-lead ECG recordings from 15 subjects (fs = 1000 Hz)
- Extracts ONLY consensus-annotated intervals (columns 9, 10, 11 of *_ANN.csv)
- Binary classification mapping:
    * Classes 1 & 2 -> Usable (0)
    * Class 3 -> Unusable (1)
- Strict subject tracking (GroupKFold) for leakage-free validation
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
import io
import numpy as np
import pandas as pd
from scipy import signal
import wfdb
from tqdm import tqdm

from config import DATA_DIR, CACHE_DIR, TARGET_FS, DEFAULT_WINDOW_SEC, CLASS_USABLE, CLASS_UNUSABLE, BUTQDB_URL, get_subject_id
from features.sqi_extractor import extract_sqi_features
from data.units import to_millivolts
from data.ecg_rendering import render_ecg_image

BUTQDB_DIR = DATA_DIR / "butqdb"
BUTQDB_DIR.mkdir(parents=True, exist_ok=True)

class BUTQDBManager:
    def __init__(self, cache_dir: Path = BUTQDB_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.records_url = f"{BUTQDB_URL}RECORDS"
        
    def get_ecg_record_list(self) -> List[Tuple[str, str]]:
        """Returns list of (subject_folder, ecg_record_name) tuples."""
        resp = requests.get(self.records_url, timeout=15)
        lines = resp.text.strip().split("\n")
        ecg_records = []
        for line in lines:
            line = line.strip()
            if line and "ECG" in line:
                parts = line.split("/")
                if len(parts) == 2:
                    ecg_records.append((parts[0], parts[1]))
        return ecg_records
        
    def fetch_consensus_annotations(self, subject: str, record_name: str) -> pd.DataFrame:
        """
        Parses the consensus annotations for a record.
        Consensus is stored in columns 9 (start), 10 (end), 11 (quality class).
        Only quality classes {1, 2, 3} are returned.
        """
        ann_name = record_name.replace("_ECG", "_ANN")
        local_ann_file = self.cache_dir / f"{ann_name}.csv"
        
        if not local_ann_file.exists():
            ann_url = f"{BUTQDB_URL}{subject}/{ann_name}.csv"
            resp = requests.get(ann_url, timeout=20)
            if resp.status_code == 200:
                with open(local_ann_file, "w") as f:
                    f.write(resp.text)
            else:
                raise RuntimeError(f"Failed to fetch annotations from {ann_url}")
                
        df = pd.read_csv(local_ann_file, header=None)
        # Consensus columns: 9=start, 10=end, 11=class
        consensus = df.iloc[:, [9, 10, 11]].dropna()
        consensus.columns = ["start_sample", "end_sample", "quality_class"]
        consensus["start_sample"] = consensus["start_sample"].astype(int)
        consensus["end_sample"] = consensus["end_sample"].astype(int)
        consensus["quality_class"] = consensus["quality_class"].astype(int)
        
        # Filter for valid classes 1, 2, 3
        valid_consensus = consensus[consensus["quality_class"].isin([1, 2, 3])].copy()
        valid_consensus["binary_label"] = valid_consensus["quality_class"].apply(
            lambda c: CLASS_USABLE if c in [1, 2] else CLASS_UNUSABLE
        )
        return valid_consensus

    def extract_segments(
        self,
        max_segments_per_subject: Optional[int] = 50,
        window_sec: float = DEFAULT_WINDOW_SEC,
        target_fs: int = TARGET_FS,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Extracts standardized 10-second segments strictly within annotated intervals.
        Caches features, raw signal, labels, and metadata.
        """
        cache_csv = self.cache_dir / f"butqdb_segments_fs{target_fs}_win{int(window_sec)}s.csv"
        cache_signals_dir = self.cache_dir / "signals"
        cache_images_dir = self.cache_dir / "images"
        cache_signals_dir.mkdir(parents=True, exist_ok=True)
        cache_images_dir.mkdir(parents=True, exist_ok=True)
        
        if cache_csv.exists():
            if verbose:
                print(f"[BUT QDB] Loading pre-extracted segments from {cache_csv}")
            return pd.read_csv(cache_csv)
            
        ecg_records = self.get_ecg_record_list()
        segment_records = []
        
        if verbose:
            print(f"[BUT QDB] Extracting segments across {len(ecg_records)} recordings...")
            
        for subject, rec_name in tqdm(ecg_records, disable=not verbose):
            subj_id = get_subject_id(subject)
            try:
                ann_df = self.fetch_consensus_annotations(subject, rec_name)
            except Exception as e:
                if verbose:
                    print(f"Skipping {rec_name}: {e}")
                continue
                
            native_fs = 1000 # BUT QDB native fs
            win_samples = int(window_sec * native_fs)
            
            # Group intervals by binary class to balance extraction
            usable_intervals = ann_df[ann_df["binary_label"] == CLASS_USABLE]
            unusable_intervals = ann_df[ann_df["binary_label"] == CLASS_UNUSABLE]
            
            # Determine quota per class
            quota_per_class = max_segments_per_subject // 2 if max_segments_per_subject else None
            
            for label, intervals in [(CLASS_USABLE, usable_intervals), (CLASS_UNUSABLE, unusable_intervals)]:
                extracted_for_class = 0
                for _, row in intervals.iterrows():
                    if quota_per_class and extracted_for_class >= quota_per_class:
                        break
                        
                    start_s = row["start_sample"]
                    end_s = row["end_sample"]
                    interval_len = end_s - start_s
                    
                    if interval_len < win_samples:
                        continue # Interval is shorter than analysis window
                        
                    # Calculate how many windows we need from this interval
                    num_windows = interval_len // win_samples
                    needed_windows = num_windows
                    if quota_per_class:
                        needed_windows = min(num_windows, quota_per_class - extracted_for_class)
                    if needed_windows <= 0:
                        break
                        
                    req_start = start_s
                    req_end = start_s + needed_windows * win_samples
                    
                    # Read entire needed range in ONE single request
                    try:
                        rec_slice = wfdb.rdrecord(
                            rec_name,
                            pn_dir=f"butqdb/{subject}",
                            sampfrom=req_start,
                            sampto=req_end
                        )
                        block_signal = rec_slice.p_signal[:, 0]
                    except Exception as e:
                        continue
                        
                    # Slice individual 10-second windows in-memory instantly
                    for w_idx in range(needed_windows):
                        w_start = req_start + w_idx * win_samples
                        w_end = w_start + win_samples
                        raw_signal = block_signal[w_idx * win_samples : (w_idx + 1) * win_samples]
                        
                        # Resample to target_fs (500 Hz)
                        target_len = int(window_sec * target_fs)
                        resampled_signal = signal.resample(raw_signal, target_len)
                        
                        # Compute SQIs on millivolts (BUT QDB is stored in uV; see data/units.py)
                        sqis = extract_sqi_features(to_millivolts(resampled_signal, "uV"), target_fs)
                        
                        # Save signal to disk
                        seg_id = f"butqdb_{subj_id}_{rec_name}_{w_start}_{w_end}"
                        sig_path = cache_signals_dir / f"{seg_id}.npy"
                        np.save(sig_path, resampled_signal.astype(np.float32))
                        
                        # Render and save image
                        img_path = cache_images_dir / f"{seg_id}.png"
                        img = render_ecg_image(resampled_signal, target_fs, dpi=100)
                        img.save(img_path)
                        
                        record_entry = {
                            "segment_id": seg_id,
                            "dataset": "BUT_QDB",
                            "subject_id": subj_id,
                            "record_name": rec_name,
                            "start_sample": w_start,
                            "end_sample": w_end,
                            "original_quality_class": int(row["quality_class"]),
                            "label": int(label),
                            "signal_path": str(sig_path),
                            "image_path": str(img_path),
                            **sqis
                        }
                        segment_records.append(record_entry)
                        extracted_for_class += 1
                        
        df_segments = pd.DataFrame(segment_records)
        df_segments.to_csv(cache_csv, index=False)
        if verbose:
            print(f"[BUT QDB] Finished extraction. Total segments: {len(df_segments)}")
            print(f"  Class distribution: {df_segments['label'].value_counts().to_dict()}")
            print(f"  Subjects count: {df_segments['subject_id'].nunique()}")
        return df_segments
