"""
CinC 2011 Set A, lead II, all 998 labelled records (revision of 2026-11-05, plan item 3.2).

Earlier analyses scored a 565-record union of balanced 75 + 75 draws. This script extracts every labelled Set A
record (773 acceptable, 225 unacceptable; the two unlabelled records are not used) with the conventions of
experiments/cinc_random_draws.py: lead II (channel index 1), native 500 Hz, 10 s (5,000 samples), signal stored in
the WFDB physical unit (mV), image rendered with data.ecg_rendering.render_ecg_image at dpi=100, SQIs computed on
the signal in mV. Records already in the cache are reused after checking that the cached signal equals the local
PhysioNet copy sample by sample.

Labels: 1 = unacceptable (RECORDS-unacceptable.txt), 0 = acceptable. No model is fitted here.

Input:  $ECG_DATASETS_ROOT/challenge-2011/set-a (local PhysioNet copy), cache/datasets/cinc2011/RECORDS-*.txt
Output: cache/datasets/cinc2011/cinc2011_setA_leadII_full.csv (one row per record)
Usage:  python experiments/cinc2011_full_extraction.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import LOCAL_DATASETS_DIR  # noqa: E402
from data.ecg_rendering import render_ecg_image  # noqa: E402
from data.units import to_millivolts  # noqa: E402
from features.sqi_extractor import SQI_FEATURE_NAMES, extract_sqi_features  # noqa: E402

CINC_DIR = PROJECT_ROOT / "cache" / "datasets" / "cinc2011"
SIG_DIR, IMG_DIR = CINC_DIR / "signals", CINC_DIR / "images"
SET_A = LOCAL_DATASETS_DIR / "challenge-2011" / "set-a"
OUT = CINC_DIR / "cinc2011_setA_leadII_full.csv"
LEAD = 1
FS = 500.0
N_SAMPLES = 5000


def read_labels() -> pd.DataFrame:
    acc = [l.strip() for l in open(CINC_DIR / "RECORDS-acceptable.txt") if l.strip()]
    unacc = [l.strip() for l in open(CINC_DIR / "RECORDS-unacceptable.txt") if l.strip()]
    if set(acc) & set(unacc):
        raise ValueError("a record is listed as both acceptable and unacceptable")
    lab = pd.DataFrame({"record_name": acc + unacc, "label": [0] * len(acc) + [1] * len(unacc)})
    return lab.sort_values("record_name").reset_index(drop=True)


def load_lead(rec_id: str) -> np.ndarray:
    rec = wfdb.rdrecord(str(SET_A / rec_id))
    if rec.fs != FS or rec.sig_len != N_SAMPLES or rec.units[LEAD] != "mV" or rec.sig_name[LEAD] != "II":
        raise ValueError(f"{rec_id}: unexpected header (fs={rec.fs}, len={rec.sig_len}, unit={rec.units[LEAD]}, "
                         f"lead={rec.sig_name[LEAD]})")
    return rec.p_signal[:, LEAD].astype(np.float32)


def main():
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    labels = read_labels()
    rows, n_cached, n_new = [], 0, 0
    for i, (rec_id, label) in enumerate(zip(labels.record_name, labels.label)):
        seg_id = f"cinc2011_{rec_id}_lead1"
        sig_path, img_path = SIG_DIR / f"{seg_id}.npy", IMG_DIR / f"{seg_id}.png"
        raw = load_lead(rec_id)
        if sig_path.exists() and img_path.exists():
            cached = np.load(sig_path)
            if cached.shape != raw.shape or not np.array_equal(cached, raw):
                raise ValueError(f"{rec_id}: cached signal differs from the local PhysioNet copy")
            n_cached += 1
        else:
            np.save(sig_path, raw)
            render_ecg_image(raw, FS, dpi=100).save(img_path)
            n_new += 1
        sqis = extract_sqi_features(to_millivolts(raw, "mV"), FS)
        rows.append({"record_name": rec_id, "segment_id": seg_id, "label": int(label), "signal_path": str(sig_path),
                     "image_path": str(img_path), "signal_unit": "mV", **{k: sqis[k] for k in SQI_FEATURE_NAMES}})
        if (i + 1) % 100 == 0:
            print(f"[extract] {i + 1}/{len(labels)}", flush=True)
    out = pd.DataFrame(rows)
    counts = out.label.value_counts().to_dict()
    if len(out) != 998 or counts != {0: 773, 1: 225} or out.segment_id.duplicated().any():
        raise ValueError(f"unexpected output: n={len(out)}, counts={counts}")
    out.to_csv(OUT, index=False)
    print(f"[done] {len(out)} records ({counts}); reused {n_cached} cached, extracted {n_new} new -> {OUT}")


if __name__ == "__main__":
    main()
