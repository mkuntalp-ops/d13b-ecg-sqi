"""
All 12 leads of the 998 labelled CinC 2011 Set A records (v2; v1 used the 565 records of the transfer analyses and
wrote cinc2011_twelve_lead_windows.csv, which is left in place). The reference label is a whole-record (12-lead)
acceptability judgement, while the main analysis scores lead II only. Scoring every lead and aggregating per record tests how much of
the transfer gap is the single-lead vs 12-lead target mismatch.

Signals: local copy of Set A ($ECG_DATASETS_ROOT/challenge-2011/set-a), 500 Hz, mV, 10 s; no resampling.
SQIs on mV; strips rendered with the corrected renderer. Signals and images already cached are reused after checking
that the cached signal equals the local PhysioNet copy sample by sample.

Output: cache/datasets/cinc2011/twelve_lead/cinc2011_twelve_lead_windows_full.csv (+ signals/ and images/)
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CINC2011_LEADII_FULL, CINC2011_TWELVE_LEAD_FULL, LOCAL_DATASETS_DIR
from data.ecg_rendering import render_ecg_image
from data.units import to_millivolts
from features.sqi_extractor import extract_sqi_features

SRC = LOCAL_DATASETS_DIR / "challenge-2011" / "set-a"
CINC_DIR = PROJECT_ROOT / "cache" / "datasets" / "cinc2011"
OUT_DIR = CINC_DIR / "twelve_lead"
SIG_DIR, IMG_DIR = OUT_DIR / "signals", OUT_DIR / "images"
OUT = CINC2011_TWELVE_LEAD_FULL


def process(rec_label):
    rec, label = rec_label
    r = wfdb.rdrecord(str(SRC / rec))
    assert int(r.fs) == 500 and r.p_signal.shape[1] == 12, rec
    rows = []
    for li, lead in enumerate(r.sig_name):
        x = to_millivolts(r.p_signal[:, li], r.units[li]).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0)
        seg = f"cinc2011_{rec}_lead{li}"
        sig_path, img_path = SIG_DIR / f"{seg}.npy", IMG_DIR / f"{seg}.png"
        if sig_path.exists():
            cached = np.load(sig_path)
            if not np.array_equal(cached, x):
                raise ValueError(f"{seg}: cached signal differs from the local PhysioNet copy")
        else:
            np.save(sig_path, x)
        if not img_path.exists():
            render_ecg_image(x, 500, dpi=100).save(img_path)
        rows.append({"segment_id": seg, "record_name": rec, "label": int(label), "lead_idx": li, "lead": lead,
                     "signal_unit": "mV", "signal_path": str(sig_path), "image_path": str(img_path),
                     **extract_sqi_features(x, 500)})
    return rows


def main():
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    recs = pd.read_csv(CINC2011_LEADII_FULL, usecols=["record_name", "label"]).astype({"record_name": str})
    if len(recs) != 998 or recs.record_name.duplicated().any():
        raise ValueError(f"expected 998 unique records, found {len(recs)}")
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for rs in ex.map(process, recs.itertuples(index=False, name=None), chunksize=8):
            rows.extend(rs)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"[Saved] {OUT}: {len(df)} lead windows from {df.record_name.nunique()} records")


if __name__ == "__main__":
    main()
