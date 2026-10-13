"""
Paired NSTDB windows: each electrode-motion noisy window has a clean partner from its MIT-BIH base record.

Verified 2026-10-12 against the NSTDB documentation (nstdb.txt, nst.1) and the cached signals:
- The noisy records 118e24 ... 119e_6 are MIT-BIH 118 and 119 (lead MLII) with electrode-motion noise added
  in two-minute segments after the first 5 minutes: 300-420 s and 540-660 s of each 720 s record. Outside
  those segments the noisy record equals the base record up to a constant offset (residual std below 1e-6 mV
  after resampling).
- Noisy windows start at 305+10i and 545+10i s (i = 0..10), inside the noise segments. Each noisy window is
  paired with the base-record window at the same start (identical beats, no DC offset). The clean intervals
  of the noisy records (65-175 s, 425-535 s) are noise-free but carry the offset, so they are not used.

SNR labels: the nominal NSTDB values are used as published. NSTDB defines the signal power S from the
peak-to-peak amplitude of the QRS complexes and the noise power N as a frequency-weighted measurement over the
first 300 s of the noise record (nst.1). That definition is not reproduced here. As a diagnostic, the full-band
SNR of each window (clean power over residual power, mean removed) is stored in measured_snr_db. It lies about
14 dB below the nominal label for every label; the labels step by 6 dB (30 dB from +24 to -6).

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
STARTS = [305 + 10 * i for i in range(11)] + [545 + 10 * i for i in range(11)]
RECORD_SEC = 720
MIN_RESIDUAL_MV = 1e-3  # a noisy window must differ from its clean partner by more than this


def window(sig: np.ndarray, start_sec: int) -> np.ndarray:
    a = start_sec * NATIVE_FS
    raw = sig[a: a + WINDOW_SEC * NATIVE_FS]
    return signal.resample(raw, WINDOW_SEC * TARGET_FS)


def save_window(base: str, x: np.ndarray, condition: str, snr, start_sec: int, source: str,
                partner_id, residual_std: float, measured_snr: float) -> dict:
    tag = "clean" if condition == "clean_reference" else f"snr{snr}"
    seg_id = f"nstdb_{base}_{tag}_{start_sec}s"
    sig_path = SIG_DIR / f"{seg_id}.npy"
    img_path = IMG_DIR / f"{seg_id}.png"
    np.save(sig_path, x.astype(np.float32))
    render_ecg_image(x, TARGET_FS, dpi=100).save(img_path)
    sqis = extract_sqi_features(x, TARGET_FS)
    return {
        "segment_id": seg_id, "dataset": "NSTDB", "base_record": base, "condition": condition,
        "snr_db": snr if snr is not None else np.nan, "start_sec": start_sec, "end_sec": start_sec + WINDOW_SEC,
        "source_record": source, "signal_unit": "mV", "partner_segment_id": partner_id,
        "residual_std_mV": residual_std, "measured_snr_db": measured_snr,
        "signal_path": str(sig_path), "image_path": str(img_path), **sqis,
    }


def main():
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    n = RECORD_SEC * NATIVE_FS
    rows = []
    for base in BASE_RECORDS:
        clean = wfdb.rdrecord(base, pn_dir="mitdb", sampfrom=0, sampto=n)
        assert clean.units[0] == "mV" and clean.sig_name[0] == "MLII"
        clean_sig = clean.p_signal[:, 0]
        clean_windows = {s: window(clean_sig, s) for s in STARTS}
        for s in STARTS:
            rows.append(save_window(base, clean_windows[s], "clean_reference", None, s, f"mitdb/{base}",
                                    partner_id=None, residual_std=np.nan, measured_snr=np.nan))
        for snr, suffix in SNR_SUFFIX.items():
            name = f"{base}e{suffix}"
            noisy = wfdb.rdrecord(name, pn_dir="nstdb", sampfrom=0, sampto=n)
            assert noisy.sig_name[0] == "MLII"
            noisy_sig = noisy.p_signal[:, 0]
            for s in STARTS:
                x_n = window(noisy_sig, s)
                x_c = clean_windows[s]
                resid = x_n - x_c
                resid_std = float(resid.std())
                assert resid_std > MIN_RESIDUAL_MV, f"{name} at {s} s has no noise (residual std {resid_std:.2e} mV)"
                measured = float(10 * np.log10(np.var(x_c) / np.var(resid)))
                rows.append(save_window(base, x_n, "noisy", snr, s, f"nstdb/{name}",
                                        partner_id=f"nstdb_{base}_clean_{s}s",
                                        residual_std=resid_std, measured_snr=measured))
            print(f"[paired] {name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.groupby(["condition", "base_record"]).size().to_string())
    print(f"[Saved] {OUT} ({len(df)} windows)")


if __name__ == "__main__":
    main()
