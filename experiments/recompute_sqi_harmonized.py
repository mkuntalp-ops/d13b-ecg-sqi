"""
Recompute SQI features for all cached segments with the harmonized pipeline:
- amplitude units harmonized to millivolts (data/units.py); BUT QDB cached signals are in uV
- one-to-one bSQI matching (features/qrs_detectors.py, calculate_bsqi)
- absolute image/signal paths rebased to this project directory (old prefix from another machine)

Signals on disk are NOT modified; the unit is applied at SQI time. Idempotent: re-running
yields the same CSVs. Backups of the input CSVs are written to results/_backup_pre_fix_2026-10-12/
the first time this script touches a file.

Usage:
    python experiments/recompute_sqi_harmonized.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.sqi_extractor import SQI_FEATURE_NAMES, extract_sqi_features
from data.units import to_millivolts

OLD_PREFIX = "/Volumes/KIOXIA/projects/db-3/"
NEW_PREFIX = str(PROJECT_ROOT) + "/"
BACKUP_DIR = PROJECT_ROOT / "results" / "_backup_pre_fix_2026-10-12"
TARGET_FS = 500  # all sources are resampled to / natively at 500 Hz

MANIFESTS = [
    PROJECT_ROOT / "results" / "manifest_butqdb_1000_cohort.csv",
    PROJECT_ROOT / "results" / "manifest_butqdb_subject_folds.csv",
    PROJECT_ROOT / "cache" / "datasets" / "cinc2011" / "cinc2011_segments_lead1_max150.csv",
    PROJECT_ROOT / "cache" / "datasets" / "nstdb" / "nstdb_calibrated_active_noise_series.csv",
]


def unit_for(segment_id: str) -> str:
    """BUT QDB segments are in uV (header unit); CinC 2011 and NSTDB are in mV."""
    if segment_id.startswith("butqdb_"):
        return "uV"
    if segment_id.startswith(("cinc2011_", "nstdb_")):
        return "mV"
    raise ValueError(f"Cannot infer amplitude unit for segment '{segment_id}'")


def backup_once(path: Path) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = BACKUP_DIR / path.name
    if not dst.exists():
        shutil.copy2(path, dst)


def recompute(path: Path) -> None:
    backup_once(path)
    df = pd.read_csv(path)
    df["signal_path"] = df["signal_path"].astype(str).str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)
    df["image_path"] = df["image_path"].astype(str).str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)

    rows = []
    for _, row in df.iterrows():
        unit = unit_for(str(row["segment_id"]))
        sig = np.load(row["signal_path"])
        sqis = extract_sqi_features(to_millivolts(sig, unit), TARGET_FS)
        rows.append({"signal_unit": unit, **sqis})

    new_feats = pd.DataFrame(rows, index=df.index)
    for col in SQI_FEATURE_NAMES:
        df[col] = new_feats[col]
    df["signal_unit"] = new_feats["signal_unit"]
    df.to_csv(path, index=False)
    print(f"[recompute] {path.name}: {len(df)} rows, units={df['signal_unit'].value_counts().to_dict()}")


def main() -> None:
    for m in MANIFESTS:
        recompute(m)


if __name__ == "__main__":
    main()
