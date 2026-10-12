"""
Paired NSTDB windows with a true clean reference (replaces the earlier clean-control definition).

Finding (verified 2026-10-12): the NSTDB noisy records (118e24 ... 119e_6) are the MIT-BIH base records
118 and 119 (pn_dir 'mitdb') with electrode-motion noise added over the whole record. The residual
noisy - base is non-zero in every second (median per-second rms 0.139 mV at the 24 dB record, 4.40 mV at
the -6 dB record; the ratio is 30 dB as expected). The earlier "clean control" windows (65-175 s and
425-535 s) were therefore noisy and could not serve as a clean baseline.

This script builds:
- clean reference windows from the MIT-BIH base records (lead MLII), and
- noisy windows from the NSTDB records at each SNR label,
at the same 44 start times per record, so that every noisy window has a paired clean window.

Start times (seconds): 65+10i, 425+10i, 305+10i, 545+10i for i = 0..10 (the earlier window grid).
The nominal SNR labels are used as published. The 30 dB spacing between the labels was verified; the
absolute values depend on NSTDB's SNR definition, which was not reproduced here.

Output: cache/datasets/nstdb/nstdb_paired_windows.csv (+ signals and images under cache/datasets/nstdb/paired/)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.ecg_rendering import render_ecg_image
from features.sqi_extractor import extract_sqi_features

NSTDB_DIR = PROJECT_ROOT / "cache" / "datasets" / "nstdb"
PAIRED_DIR = NSTDB_DIR / "paired"
SIG_DIR = PAIRED_DIR / "signals"
IMG_DIR = PAIRED_DIR / "images"
OUT = NSTDB_DIR / "nstdb_paired_windows.csv"

NATIVE_FS = 360
TARGET_FS = 500
WINDOW_SEC = 10
BASE_RECORDS = ["118", "119"]
SNR_SUFFIX = {24: "24", 18: "18", 12: "12", 6: "06", 0: "00", -6: "_6"}
STARTS = [65 + 10 * i for i in range(11)] + [425 + 10 * i for i in range(11)] \
       + [305 + 10 * i for i in range(11)] + [545 + 10 * i for i in range(11)]
RECORD_SEC = 720


def window(sig: np.ndarray, start_sec: int) -> np.ndarray:
    a = start_sec * NATIVE_FS
    raw = sig[a: a + WINDOW_SEC * NATIVE_FS]
    return signal.resample(raw, WINDOW_SEC * TARGET_FS)


def record_rows(rec_name: str, sig: np.ndarray, condition: str, snr, source: str):
    rows = []
    for s in STARTS:
        x = window(sig, s)
        sqis = extract_sqi_features(x, TARGET_FS)
        tag = "clean" if condition == "clean_reference" else f"snr{snr}"
        seg_id = f"nstdb_{rec_name}_{tag}_{s}s"
        sig_path = SIG_DIR / f"{seg_id}.npy"
        img_path = IMG_DIR / f"{seg_id}.png"
        np.save(sig_path, x.astype(np.float32))
        render_ecg_image(x, TARGET_FS, dpi=100).save(img_path)
        rows.append({
            "segment_id": seg_id, "dataset": "NSTDB", "base_record": rec_name, "condition": condition,
            "snr_db": snr if snr is not None else np.nan, "start_sec": s, "end_sec": s + WINDOW_SEC,
            "source_record": source, "signal_unit": "mV",
            "signal_path": str(sig_path), "image_path": str(img_path), **sqis,
        })
    return rows


def main():
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    n = RECORD_SEC * NATIVE_FS
    rows = []
    for base in BASE_RECORDS:
        clean = wfdb.rdrecord(base, pn_dir="mitdb", sampfrom=0, sampto=n)
        assert clean.units[0] == "mV" and clean.sig_name[0] == "MLII"
        rows += record_rows(base, clean.p_signal[:, 0], "clean_reference", None, f"mitdb/{base}")
        for snr, suffix in SNR_SUFFIX.items():
            name = f"{base}e{suffix}"
            noisy = wfdb.rdrecord(name, pn_dir="nstdb", sampfrom=0, sampto=n)
            assert noisy.sig_name[0] == "MLII"
            rows += record_rows(base, noisy.p_signal[:, 0], "noisy", snr, f"nstdb/{name}")
            print(f"[paired] {name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.groupby(["condition"]).size().to_string())
    print(f"[Saved] {OUT} ({len(df)} windows)")


if __name__ == "__main__":
    main()
