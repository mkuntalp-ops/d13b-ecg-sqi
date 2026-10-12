"""
MIT-BIH Noise Stress Test Database (NSTDB) Loader & Degradation Benchmark
- Controlled SNR levels: 24, 18, 12, 6, 0, -6 dB
- Evaluates model behavior under controlled electrode motion ('em') noise degradation.
- Strictly separates active noise intervals from clean intervals.
- Supports both source records 118 and 119.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal
import wfdb
from tqdm import tqdm

from config import DATA_DIR, TARGET_FS, DEFAULT_WINDOW_SEC, NSTDB_SNR_LEVELS
from features.sqi_extractor import extract_sqi_features
from data.ecg_rendering import render_ecg_image

NSTDB_DIR = DATA_DIR / "nstdb"
NSTDB_DIR.mkdir(parents=True, exist_ok=True)

class NSTDBManager:
    def __init__(self, cache_dir: Path = NSTDB_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Record naming in NSTDB: e.g. 118e24 (24 dB), 118e18, 118e12, 118e06, 118e00, 118e_6 (-6 dB)
        self.snr_map = {
            24: "24",
            18: "18",
            12: "12",
            6: "06",
            0: "00",
            -6: "_6"
        }
        
    def extract_snr_series(
        self,
        base_records: List[str] = ["118", "119"],
        window_sec: float = DEFAULT_WINDOW_SEC,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Extracts 10-second segments strictly within active noise and clean intervals
        for both records 118 and 119 across all 6 SNR levels.
        """
        cache_csv = self.cache_dir / "nstdb_calibrated_active_noise_series.csv"
        cache_signals_dir = self.cache_dir / "signals"
        cache_images_dir = self.cache_dir / "images"
        cache_signals_dir.mkdir(parents=True, exist_ok=True)
        cache_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Intervals in seconds:
        # In NSTDB: 0-300s is clean, 300-420s is noise, 420-540s is clean, 540-660s is noise.
        # We define:
        # - Active noise windows: 305 to 415 (11 windows) and 545 to 655 (11 windows)
        # - Clean control windows: 65 to 175 (11 windows) and 425 to 535 (11 windows)
        noise_window_starts = [305.0 + i * 10.0 for i in range(11)] + [545.0 + i * 10.0 for i in range(11)]
        clean_window_starts = [65.0 + i * 10.0 for i in range(11)] + [425.0 + i * 10.0 for i in range(11)]
        
        segment_records = []
        
        for base_record in base_records:
            if verbose:
                print(f"[NSTDB] Processing base record {base_record} across 6 SNR levels...")
                
            for snr_val, snr_suffix in self.snr_map.items():
                rec_name = f"{base_record}e{snr_suffix}"
                try:
                    # Read first 12 minutes (720 seconds)
                    rec = wfdb.rdrecord(rec_name, pn_dir="nstdb", sampfrom=0, sampto=int(720 * 360))
                    ecg_lead = rec.p_signal[:, 0]
                    native_fs = rec.fs
                except Exception as e:
                    if verbose:
                        print(f"Error loading {rec_name}: {e}")
                    continue
                    
                target_len = int(window_sec * TARGET_FS)
                
                # 1. Process active noise windows
                for idx, start_sec in enumerate(noise_window_starts):
                    start_s = int(start_sec * native_fs)
                    end_s = start_s + int(window_sec * native_fs)
                    if end_s > len(ecg_lead):
                        continue
                    raw_slice = ecg_lead[start_s:end_s]
                    resampled = signal.resample(raw_slice, target_len)
                    sqis = extract_sqi_features(resampled, TARGET_FS)
                    
                    seg_id = f"nstdb_{base_record}_em_{snr_val}dB_noise_w{idx}_{int(start_sec)}s"
                    sig_path = cache_signals_dir / f"{seg_id}.npy"
                    np.save(sig_path, resampled.astype(np.float32))
                    
                    img_path = cache_images_dir / f"{seg_id}.png"
                    img = render_ecg_image(resampled, TARGET_FS, dpi=100)
                    img.save(img_path)
                    
                    segment_records.append({
                        "segment_id": seg_id,
                        "dataset": "NSTDB",
                        "base_record": base_record,
                        "noise_type": "electrode_motion_em",
                        "condition": "active_noise",
                        "snr_db": snr_val,
                        "window_idx": idx,
                        "start_sec": start_sec,
                        "end_sec": start_sec + window_sec,
                        "signal_path": str(sig_path),
                        "image_path": str(img_path),
                        **sqis
                    })
                    
                # 2. Process clean control windows (only once per base record, or at each SNR for verification)
                for idx, start_sec in enumerate(clean_window_starts):
                    start_s = int(start_sec * native_fs)
                    end_s = start_s + int(window_sec * native_fs)
                    if end_s > len(ecg_lead):
                        continue
                    raw_slice = ecg_lead[start_s:end_s]
                    resampled = signal.resample(raw_slice, target_len)
                    sqis = extract_sqi_features(resampled, TARGET_FS)
                    
                    seg_id = f"nstdb_{base_record}_em_{snr_val}dB_clean_w{idx}_{int(start_sec)}s"
                    sig_path = cache_signals_dir / f"{seg_id}.npy"
                    np.save(sig_path, resampled.astype(np.float32))
                    
                    img_path = cache_images_dir / f"{seg_id}.png"
                    img = render_ecg_image(resampled, TARGET_FS, dpi=100)
                    img.save(img_path)
                    
                    segment_records.append({
                        "segment_id": seg_id,
                        "dataset": "NSTDB",
                        "base_record": base_record,
                        "noise_type": "electrode_motion_em",
                        "condition": "clean_interval",
                        "snr_db": snr_val,
                        "window_idx": idx,
                        "start_sec": start_sec,
                        "end_sec": start_sec + window_sec,
                        "signal_path": str(sig_path),
                        "image_path": str(img_path),
                        **sqis
                    })
                    
        df = pd.DataFrame(segment_records)
        df.to_csv(cache_csv, index=False)
        if verbose:
            print(f"[NSTDB] Calibrated extraction complete: {len(df)} total windows across {base_records}")
            print(f"  Active noise: {(df['condition'] == 'active_noise').sum()}, Clean: {(df['condition'] == 'clean_interval').sum()}")
        return df
