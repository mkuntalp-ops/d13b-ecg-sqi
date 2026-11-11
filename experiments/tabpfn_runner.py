"""
TabPFN v3 on the 13 SQI features (Q1 revision, 2026-10-25): a strong tuning-free tabular baseline.

Purpose: the LR / LightGBM / MLP baselines use fixed hyperparameters. TabPFN is a pre-trained tabular foundation
model that needs no tuning, so it tests whether the tabular results are limited by the baselines or by the
features. Settings mirror the main analysis exactly:
- BUT QDB 5-fold subject-independent CV (the manifest's fold_id);
- trained on the full BUT QDB cohort, applied to CinC 2011 lead II (565 records), CinC 2017 (all windows) and
  NSTDB (paired windows);
- the three leave-one-dataset-out splits of experiments/lodo_multisource.py (same CinC 2017 training subset).

This script runs in a separate environment that has TabPFN installed (no download), on the CPU:
    /Volumes/KIOXIA/AI/tabpfn_env/bin/python experiments/tabpfn_runner.py
Checkpoint: $TABPFN_CKPT (default: /Volumes/KIOXIA/AI Models/tabpfn/tabpfn-v3-classifier-v3_default.ckpt)
Output: results/tabpfn_predictions.csv (setting, dataset, segment_id, label, prob_unusable)

--official-split (v2, plan item 3.1, analysis F1b): fit on the BUT QDB windows of the official train records and score
those of the official test records, as listed in results/butqdb_official_split_assignment.csv (written by
experiments/butqdb_official_split.py --split-only). Output: results/tabpfn_official_split_predictions.csv.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from config import RESULTS_DIR as _CFG_RESULTS  # noqa: E402  (ECG_RESULTS_DIR overrides)
RESULTS = _CFG_RESULTS
CKPT = os.environ.get("TABPFN_CKPT", "/Volumes/KIOXIA/AI Models/tabpfn/tabpfn-v3-classifier-v3_default.ckpt")
OUT = RESULTS / "tabpfn_predictions.csv"
# identical to features.sqi_extractor.MODEL_INPUT_FEATURE_NAMES (this environment does not import the project package)
FEATURES = ["bsqi_detector_agreement", "baseline_wander_ratio", "high_frequency_noise_ratio", "saturation_fraction",
            "flatline_fraction", "qrs_energy_ratio", "kurtosis_ksqi", "skewness_ssqi", "zero_crossing_rate",
            "powerline_noise_ratio", "signal_std"]
SEED = 42


def model():
    return TabPFNClassifier(model_path=CKPT, device="cpu", random_state=SEED)


def X(df):
    return np.nan_to_num(df[FEATURES].values.astype(float))


def sources():
    but = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv").assign(dataset="BUT_QDB")
    c11 = pd.read_csv(ROOT / "cache/datasets/cinc2011/cinc2011_setA_leadII_full.csv")  # all 998 labelled records (v2)
    c11 = c11.assign(dataset="CinC2011", record=c11["record_name"].astype(str))
    c17 = pd.read_csv(ROOT / "cache/datasets/cinc2017/cinc2017_windows.csv").assign(dataset="CinC2017")
    c17["record"] = c17["record_name"].astype(str)
    nst = pd.read_csv(ROOT / "cache/datasets/nstdb/nstdb_paired_windows.csv").assign(dataset="NSTDB", label=np.nan)
    return but, c11, c17, nst


def cinc17_training_subset(c17):
    """Same rule as lodo_multisource.cinc17_training_subset (all noisy + 1,000 random non-noisy recordings)."""
    rec = c17.drop_duplicates("record")[["record", "label"]]
    keep = pd.concat([rec[rec.label == 1].record, rec[rec.label == 0].sample(1000, random_state=SEED).record])
    return c17[c17.record.isin(keep)].reset_index(drop=True)


def rows(setting, df, p):
    return pd.DataFrame({"setting": setting, "dataset": df["dataset"].values, "segment_id": df["segment_id"].values,
                         "label": df["label"].values, "prob_unusable": p})


def official_split():
    split = pd.read_csv(RESULTS / "butqdb_official_split_assignment.csv")
    but = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv").assign(dataset="BUT_QDB")
    if list(split.segment_id) != list(but.segment_id) or (split.label.values != but.label.values).any():
        raise ValueError("assignment file does not match the cohort manifest; rerun butqdb_official_split.py --split-only")
    tr, te = split.split.values == "train", split.split.values == "test"
    p = model().fit(X(but[tr]), but.label.values[tr]).predict_proba(X(but[te]))[:, 1]
    out = RESULTS / "tabpfn_official_split_predictions.csv"
    rows("butqdb_official_split", but[te], p).to_csv(out, index=False)
    print(f"[tabpfn] official split: train {tr.sum()}, test {te.sum()} windows. [Saved] {out}")


def main():
    but, c11, c17, nst = sources()
    parts = []
    # 1. BUT QDB 5-fold subject CV
    oof = np.zeros(len(but))
    for k in range(5):
        tr, te = but.fold_id.values != k, but.fold_id.values == k
        oof[te] = model().fit(X(but[tr]), but.label.values[tr]).predict_proba(X(but[te]))[:, 1]
        print(f"[tabpfn] CV fold {k} done", flush=True)
    parts.append(rows("butqdb_cv", but, oof))
    # 2. Trained on BUT QDB, applied to the external sets
    full = model().fit(X(but), but.label.values)
    for df in [c11, c17, nst]:
        parts.append(rows("but_only_external", df, full.predict_proba(X(df))[:, 1]))
        print(f"[tabpfn] external {df.dataset.iloc[0]} done", flush=True)
    pd.concat(parts).to_csv(OUT, index=False)
    # 3. Leave-one-dataset-out
    c17_train = cinc17_training_subset(c17)
    pool = {"BUT_QDB": but, "CinC2011": c11, "CinC2017": c17_train}
    tests = {"BUT_QDB": but, "CinC2011": c11, "CinC2017": c17}
    for test_name, test in tests.items():
        train = pd.concat([v for k, v in pool.items() if k != test_name], ignore_index=True)
        m = model().fit(X(train), train.label.values.astype(int))
        parts.append(rows(f"lodo_test_{test_name}", test, m.predict_proba(X(test))[:, 1]))
        pd.concat(parts).to_csv(OUT, index=False)
        print(f"[tabpfn] LODO test={test_name} done (train {len(train)})", flush=True)
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    if "--official-split" in sys.argv:
        official_split()
    else:
        main()
