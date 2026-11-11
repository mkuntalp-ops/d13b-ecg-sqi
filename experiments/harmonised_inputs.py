"""
Inputs of the harmonised-pipeline analysis (reviewer finding PAPER-002; pre-stated in
REVISION_PLAN_2026-11-05_post_desk_rejection.md on 2026-11-11, before any harmonised score was computed).

Only numpy and pandas are imported, so the TabPFN environment can use this module too.

G2: flatline_fraction is replaced by flatline_fraction_robust (definition of 2026-11-06, unchanged; values from
    flatline_robust_features.csv, written by experiments/flatline_robust_exploratory.py).
G3: inputs are z-scored with the training statistics and clipped to [-CLIP, CLIP] (CLIP = 3, the bound of the "Z-clip"
    scaler of Table S15, in the code since v1).
G1 (constant-input rule) is applied to the scores, in experiments/harmonised_pipeline.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ORIGINAL = ["bsqi_detector_agreement", "baseline_wander_ratio", "high_frequency_noise_ratio", "saturation_fraction",
            "flatline_fraction", "qrs_energy_ratio", "kurtosis_ksqi", "skewness_ssqi", "zero_crossing_rate",
            "powerline_noise_ratio", "signal_std"]
HARMONISED = [("flatline_fraction_robust" if f == "flatline_fraction" else f) for f in ORIGINAL]
CLIP = 3.0


def zclip_fit(X_train: np.ndarray):
    """Mean and SD of the training matrix (SD 0 is replaced by 1, so a constant training column stays 0)."""
    X_train = np.nan_to_num(np.asarray(X_train, dtype=float))
    mean, sd = X_train.mean(axis=0), X_train.std(axis=0)
    return mean, np.where(sd > 0, sd, 1.0)


def zscore(X: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.nan_to_num(np.asarray(X, dtype=float)) - mean) / sd


def zclip_apply(X: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.clip(zscore(X, mean, sd), -CLIP, CLIP)


def load_tables(root: Path, results_dir: Path) -> dict:
    """BUT QDB cohort, CinC 2011 lead II (998 records) and CinC 2017 windows, each with flatline_fraction_robust,
    a 'record' column (subject on BUT QDB) and 'constant' (signal SD of the window is 0)."""
    root, results_dir = Path(root), Path(results_dir)
    feats = pd.read_csv(results_dir / "flatline_robust_features.csv")
    tables = {"BUT_QDB": pd.read_csv(results_dir / "manifest_butqdb_1000_cohort.csv"),
              "CinC2011": pd.read_csv(root / "cache" / "datasets" / "cinc2011" / "cinc2011_setA_leadII_full.csv"),
              "CinC2017": pd.read_csv(root / "cache" / "datasets" / "cinc2017" / "cinc2017_windows.csv")}
    out = {}
    for name, df in tables.items():
        f = feats[feats.dataset == name].set_index("segment_id")
        if f.index.duplicated().any() or not set(df.segment_id) <= set(f.index):
            raise ValueError(f"{name}: robust flatline values missing or duplicated")
        df = df.reset_index(drop=True).copy()
        if not np.array_equal(f.loc[df.segment_id, "label"].values.astype(int), df.label.values.astype(int)):
            raise ValueError(f"{name}: labels differ between the window table and flatline_robust_features.csv")
        df["flatline_fraction_robust"] = f.loc[df.segment_id, "flatline_fraction_robust"].values
        df["record"] = (df.subject_id if name == "BUT_QDB" else df.record_name).astype(str)
        df["constant"] = df.signal_std.values == 0
        out[name] = df
    return out
