"""
External single-lead cohort: PhysioNet/CinC Challenge 2017 training set (AliveCor handheld, lead I equivalent,
300 Hz). Record labels: N (normal), A (AF), O (other rhythm), ~ (noisy: "too noisy to classify").
The '~' label is a rhythm-classifiability judgement on a single lead, so it is close to the BUT QDB Class 3
definition (QRS-based rhythm analysis not possible). It is used here as the "unusable" target at its natural
prevalence.

Windows: non-overlapping 10-s windows from the start of each recording, at most 3 per recording (30 s; the
median recording length is 30 s). Each window is resampled 300 -> 500 Hz with scipy.signal.resample (as for
BUT QDB), SQIs are computed on millivolts and the strip is rendered with the corrected renderer.

Input:  $ECG_DATASETS_ROOT/cinc2017/training2017 (local copy of the PhysioNet training set) and REFERENCE.csv
Output: cache/datasets/cinc2017/cinc2017_windows.csv (+ signals/ and images/)
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import LOCAL_DATASETS_DIR
from data.ecg_rendering import render_ecg_image
from data.units import to_millivolts
from features.sqi_extractor import extract_sqi_features

SRC = LOCAL_DATASETS_DIR / "cinc2017"
OUT_DIR = PROJECT_ROOT / "cache" / "datasets" / "cinc2017"
SIG_DIR, IMG_DIR = OUT_DIR / "signals", OUT_DIR / "images"
OUT = OUT_DIR / "cinc2017_windows.csv"
WIN_SEC, TARGET_FS, MAX_WIN = 10, 500, 3


def process(rec_label):
    rec, lab = rec_label
    r = wfdb.rdrecord(str(SRC / "training2017" / rec))
    fs = float(r.fs)
    x_all = to_millivolts(r.p_signal[:, 0], r.units[0])
    n_win = min(MAX_WIN, int(len(x_all) // (WIN_SEC * fs)))
    rows = []
    for w in range(n_win):
        a = int(w * WIN_SEC * fs)
        x = signal.resample(x_all[a: a + int(WIN_SEC * fs)], WIN_SEC * TARGET_FS).astype(np.float32)
        seg = f"cinc2017_{rec}_w{w}"
        sig_path, img_path = SIG_DIR / f"{seg}.npy", IMG_DIR / f"{seg}.png"
        np.save(sig_path, x)
        render_ecg_image(x, TARGET_FS, dpi=100).save(img_path)
        rows.append({"segment_id": seg, "record_name": rec, "label_code": lab, "label": int(lab == "~"),
                     "window_idx": w, "start_sec": w * WIN_SEC, "record_sec": len(x_all) / fs, "native_fs": fs,
                     "signal_unit": "mV", "signal_path": str(sig_path), "image_path": str(img_path),
                     **extract_sqi_features(x, TARGET_FS)})
    return rows


def main():
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(SRC / "REFERENCE.csv", header=None, names=["rec", "lab"])
    ref["rec"] = ref["rec"].str.split("/").str[-1]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, rs in enumerate(ex.map(process, ref[["rec", "lab"]].itertuples(index=False, name=None), chunksize=16)):
            rows.extend(rs)
            if i % 1000 == 0:
                print(f"[cinc2017] {i}/{len(ref)} recordings", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    rec = df.groupby("record_name").agg(label=("label", "first"), n=("label", "size"))
    print(f"[Saved] {OUT}: {len(df)} windows from {len(rec)} recordings "
          f"({len(ref) - len(rec)} recordings shorter than 10 s); noisy recordings {int(rec.label.sum())} "
          f"({100 * rec.label.mean():.1f}%)")


if __name__ == "__main__":
    main()
