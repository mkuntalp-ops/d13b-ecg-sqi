"""
All 12 leads of the CinC 2011 Set A records used in the transfer analyses (the reported first-N subset and the
five random subsets: 565 records). The reference label is a whole-record (12-lead) acceptability judgement,
while the main analysis scores lead II only. Scoring every lead and aggregating per record tests how much of
the transfer gap is the single-lead vs 12-lead target mismatch.

Signals: local copy of Set A (/Volumes/KIOXIA/Datasets/challenge-2011/set-a), 500 Hz, mV, 10 s; no resampling.
SQIs on mV; strips rendered with the corrected renderer.

Output: cache/datasets/cinc2011/twelve_lead/cinc2011_twelve_lead_windows.csv (+ signals/ and images/)
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.ecg_rendering import render_ecg_image
from data.units import to_millivolts
from features.sqi_extractor import extract_sqi_features

SRC = Path("/Volumes/KIOXIA/Datasets/challenge-2011/set-a")
CINC_DIR = PROJECT_ROOT / "cache" / "datasets" / "cinc2011"
OUT_DIR = CINC_DIR / "twelve_lead"
SIG_DIR, IMG_DIR = OUT_DIR / "signals", OUT_DIR / "images"
OUT = OUT_DIR / "cinc2011_twelve_lead_windows.csv"
MEMBERSHIP = CINC_DIR / "cinc2011_random_draws_manifest.csv"


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
        np.save(sig_path, x)
        render_ecg_image(x, 500, dpi=100).save(img_path)
        rows.append({"segment_id": seg, "record_name": rec, "label": int(label), "lead_idx": li, "lead": lead,
                     "signal_unit": "mV", "signal_path": str(sig_path), "image_path": str(img_path),
                     **extract_sqi_features(x, 500)})
    return rows


def main():
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    mem = pd.read_csv(MEMBERSHIP)
    recs = mem.drop_duplicates("record_name")[["record_name", "label"]]
    recs["record_name"] = recs["record_name"].astype(str)
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for rs in ex.map(process, recs.itertuples(index=False, name=None), chunksize=8):
            rows.extend(rs)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"[Saved] {OUT}: {len(df)} lead windows from {df.record_name.nunique()} records")


if __name__ == "__main__":
    main()
