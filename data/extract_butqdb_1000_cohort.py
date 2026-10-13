"""
Extract BUT QDB 1,000-Window Cohort (Stratified by Subject, Class & Time)
- Expands BUT QDB from 124-window pilot to ~1,000 non-overlapping windows.
- Stratified sampling across all 15 subjects (18 records):
  * Usable (Classes 1 & 2): up to 50 windows per subject, evenly spaced across the recording.
  * Unusable (Class 3): for the 7 subjects with Class 3, up to 60 windows per subject, evenly spaced.
  * Long recordings (Subject 100, 105, 111) are capped to prevent sample domination.
  * Windows are non-overlapping and spread across time (not clustered in first 50s).
- Preprocesses signals: resampled to 500 Hz, SQIs extracted, standard 100 DPI images rendered.
- Deterministic 5-fold subject-independent split (GroupKFold on subject_id).
- Saves manifest to results/manifest_butqdb_1000_cohort.csv.
"""

import os
import sys
import glob
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import signal
import wfdb
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from config import DATA_DIR, TARGET_FS, DEFAULT_WINDOW_SEC, CLASS_USABLE, CLASS_UNUSABLE, RESULTS_DIR
from features.sqi_extractor import extract_sqi_features
from data.ecg_rendering import render_ecg_image
from data.units import to_millivolts

BUT_NATIVE_UNIT = "uV"  # BUT QDB header unit (verified: wfdb units ['uV'])

BUTQDB_DIR = DATA_DIR / "butqdb"
COHORT_DIR = BUTQDB_DIR / "cohort1000"
SIGNALS_DIR = COHORT_DIR / "signals"
IMAGES_DIR = COHORT_DIR / "images"

for d in [COHORT_DIR, SIGNALS_DIR, IMAGES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MANIFEST_OUT = RESULTS_DIR / "manifest_butqdb_1000_cohort.csv"


def build_candidate_windows() -> pd.DataFrame:
    """Finds all valid, non-overlapping 10s windows within consensus annotations."""
    ann_files = sorted(glob.glob(str(BUTQDB_DIR / "*_ANN.csv")))
    all_windows = []

    for f in ann_files:
        rec_name = os.path.basename(f).replace("_ANN.csv", "_ECG")
        subj_id = rec_name.split("_")[0][:3]
        df = pd.read_csv(f, header=None)
        # Consensus columns: 9=start, 10=end, 11=class
        c_df = df.iloc[:, [9, 10, 11]].dropna()
        c_df.columns = ["start", "end", "cls"]
        c_df["start"] = c_df["start"].astype(int)
        c_df["end"] = c_df["end"].astype(int)
        c_df["cls"] = c_df["cls"].astype(int)
        c_df = c_df[c_df["cls"].isin([1, 2, 3])]

        for _, row in c_df.iterrows():
            dur_samples = row["end"] - row["start"]
            n_win = dur_samples // 10000  # 10s at 1000 Hz
            for w in range(n_win):
                w_start = row["start"] + w * 10000
                w_end = w_start + 10000
                all_windows.append({
                    "record_name": rec_name,
                    "subject_id": subj_id,
                    "original_quality_class": int(row["cls"]),
                    "label": CLASS_USABLE if row["cls"] in [1, 2] else CLASS_UNUSABLE,
                    "start_sample": w_start,
                    "end_sample": w_end,
                    "mid_sample": (w_start + w_end) // 2
                })

    return pd.DataFrame(all_windows)


def sample_stratified_cohort(win_df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects ~1,000 non-overlapping windows:
    - Up to 50 usable per subject, evenly spaced across the recording timeline.
    - Up to 60 unusable per subject, evenly spaced across the recording timeline.
    """
    selected = []
    for subj_id, s_group in win_df.groupby("subject_id"):
        # Usable (Classes 1 & 2)
        u_group = s_group[s_group["label"] == CLASS_USABLE].sort_values("mid_sample")
        n_u = len(u_group)
        target_u = min(50, n_u)
        if target_u > 0:
            idx_u = np.linspace(0, n_u - 1, target_u, dtype=int)
            selected.append(u_group.iloc[idx_u])

        # Unusable (Class 3)
        un_group = s_group[s_group["label"] == CLASS_UNUSABLE].sort_values("mid_sample")
        n_un = len(un_group)
        target_un = min(60, n_un)
        if target_un > 0:
            idx_un = np.linspace(0, n_un - 1, target_un, dtype=int)
            selected.append(un_group.iloc[idx_un])

    cohort = pd.concat(selected, ignore_index=True)
    cohort["segment_id"] = [
        f"butqdb_{row['subject_id']}_{row['record_name']}_{row['start_sample']}_{row['end_sample']}"
        for _, row in cohort.iterrows()
    ]
    return cohort


def process_single_window(row: dict) -> dict:
    """Extracts, resamples, computes SQIs and renders image for a single window."""
    seg_id = row["segment_id"]
    sig_path = SIGNALS_DIR / f"{seg_id}.npy"
    img_path = IMAGES_DIR / f"{seg_id}.png"

    rec_name = row["record_name"]
    subj_folder = rec_name.split("_")[0]
    w_start = int(row["start_sample"])
    w_end = int(row["end_sample"])

    # Signals are cached in their native header unit (BUT QDB: uV). SQIs are always computed
    # on millivolts so that they are comparable with CinC 2011 and NSTDB (see data/units.py).
    if sig_path.exists() and img_path.exists():
        resampled_signal = np.load(sig_path)
        sqis = extract_sqi_features(to_millivolts(resampled_signal, BUT_NATIVE_UNIT), TARGET_FS)
        return {
            **row,
            "signal_path": str(sig_path),
            "image_path": str(img_path),
            "signal_unit": BUT_NATIVE_UNIT,
            **sqis
        }

    # Fetch slice from WFDB PhysioNet
    rec_slice = wfdb.rdrecord(
        rec_name,
        pn_dir=f"butqdb/{subj_folder}",
        sampfrom=w_start,
        sampto=w_end
    )
    raw_signal = rec_slice.p_signal[:, 0]
    unit = rec_slice.units[0]
    if unit != BUT_NATIVE_UNIT:
        raise ValueError(f"{rec_name}: expected BUT QDB unit '{BUT_NATIVE_UNIT}', got '{unit}'")

    # Resample to TARGET_FS (500 Hz)
    target_len = int(DEFAULT_WINDOW_SEC * TARGET_FS)
    resampled_signal = signal.resample(raw_signal, target_len).astype(np.float32)

    # Compute SQIs on millivolts
    sqis = extract_sqi_features(to_millivolts(resampled_signal, unit), TARGET_FS)

    # Save signal in native units
    np.save(sig_path, resampled_signal)

    # Render & save image
    img = render_ecg_image(to_millivolts(resampled_signal, unit).astype(np.float32), TARGET_FS, dpi=100)
    img.save(img_path)

    return {
        **row,
        "signal_path": str(sig_path),
        "image_path": str(img_path),
        "signal_unit": unit,
        **sqis
    }


def main():
    print("[1/4] Scanning all annotated BUT QDB intervals...")
    win_df = build_candidate_windows()
    print(f"Total available candidate 10s windows: {len(win_df)}")

    print("[2/4] Sampling stratified 1,000-window cohort across 15 subjects...")
    cohort_df = sample_stratified_cohort(win_df)
    print(f"Selected cohort size: {len(cohort_df)} windows")
    print(f"Class distribution: {cohort_df['label'].value_counts().to_dict()}")
    print(f"Original quality classes: {cohort_df['original_quality_class'].value_counts().to_dict()}")

    print("[3/4] Extracting signals, computing SQIs, and rendering ECG images...")
    rows_list = cohort_df.to_dict(orient="records")
    processed_records = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_single_window, r): r["segment_id"] for r in rows_list}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            res = fut.result()
            processed_records.append(res)

    df_final = pd.DataFrame(processed_records)

    print("[4/4] Generating leak-free subject GroupKFold splits...")
    gkf = GroupKFold(n_splits=5)
    df_final["fold_id"] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_final, groups=df_final["subject_id"])):
        df_final.loc[val_idx, "fold_id"] = fold

    # Sort deterministically
    df_final = df_final.sort_values(["subject_id", "record_name", "start_sample"]).reset_index(drop=True)
    df_final.to_csv(MANIFEST_OUT, index=False)
    print(f"[Done] Cohort manifest successfully saved to {MANIFEST_OUT}")
    print(df_final.groupby(["fold_id", "label"]).size().unstack().fillna(0).astype(int))


if __name__ == "__main__":
    main()
