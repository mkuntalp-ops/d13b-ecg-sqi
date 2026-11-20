"""
PTB-XL strat_fold of the records in the frozen PTB-XL test (round V20, 2026-11-20; review finding PAPER-010). Descriptive
only; no model is scored.

The HuBERT-ECG developers list PTB-XL among the self-supervised pre-training data and follow the original PTB-XL split,
testing on partitions not used in pre-training (preprint version 4, Methods). The recommended split uses strat_fold 10
as the test fold, so records of folds 1-9 were probably seen in pre-training (without labels) and records of fold 10
probably not. This script counts the sampled records per stratum and fold.

Inputs: <LOCAL_DATASETS_DIR>/ptb-xl/ptbxl_database.csv (PTB-XL v1.0.3 metadata), cache/datasets/ptbxl/ptbxl_windows.csv.
Output: <results>/ptbxl_strat_fold_counts.csv
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/ptbxl_fold_counts.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, LOCAL_DATASETS_DIR, RESULTS_DIR  # noqa: E402

OUT = RESULTS_DIR / "ptbxl_strat_fold_counts.csv"


def main():
    meta = pd.read_csv(LOCAL_DATASETS_DIR / "ptb-xl" / "ptbxl_database.csv", usecols=["ecg_id", "strat_fold"])
    w = pd.read_csv(DATA_DIR / "ptbxl" / "ptbxl_windows.csv", usecols=["record_name", "stratum"]).drop_duplicates()
    m = w.merge(meta, left_on="record_name", right_on="ecg_id", how="left", validate="one_to_one")
    if m.strat_fold.isna().any() or len(m) != 900:
        raise ValueError("every one of the 900 sampled records must have a strat_fold")
    out = m.groupby(["stratum", "strat_fold"]).size().rename("records").reset_index()
    out.to_csv(OUT, index=False)
    print(pd.crosstab(m.stratum, m.strat_fold, margins=True))
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
