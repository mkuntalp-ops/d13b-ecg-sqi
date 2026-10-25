"""
TabPFN latency (Q1 revision, 2026-10-26). Runs in the TabPFN environment, on the CPU, alone:
    /Volumes/KIOXIA/AI/tabpfn_env/bin/python experiments/latency_tabpfn.py

TabPFN predicts in context: every prediction is a forward pass over the training set (1,047 BUT QDB windows, 13 SQIs)
together with the query. The fit (preprocessing of the training set) is timed once per run; prediction is timed for one
window at a time on the same 25 windows and three runs as the other latency scripts (first window of each run is a
warm-up and excluded). SQI extraction is not included (Table 4b reports it separately).

Output: results/latency_tabpfn_raw.csv, results/latency_tabpfn_summary.csv
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CKPT = os.environ.get("TABPFN_CKPT", "/Volumes/KIOXIA/AI Models/tabpfn/tabpfn-v3-classifier-v3_default.ckpt")
FEATURES = ["bsqi_detector_agreement", "baseline_wander_ratio", "high_frequency_noise_ratio", "saturation_fraction",
            "flatline_fraction", "qrs_energy_ratio", "kurtosis_ksqi", "skewness_ssqi", "zero_crossing_rate",
            "powerline_noise_ratio", "signal_std", "duration_sec", "sampling_rate"]
N_WINDOWS, N_RUNS = 25, 3  # as in experiments/latency_extended.py


def main():
    man = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv")
    X = np.nan_to_num(man[FEATURES].values.astype(float))
    y = man.label.values
    idx = np.linspace(0, len(man) - 1, N_WINDOWS, dtype=int)  # same selection as latency_extended.select_windows
    rows = []
    for run in range(1, N_RUNS + 1):
        t0 = time.perf_counter()
        clf = TabPFNClassifier(model_path=CKPT, device="cpu", random_state=42).fit(X, y)
        fit_ms = (time.perf_counter() - t0) * 1000.0
        for i, k in enumerate(idx):
            t0 = time.perf_counter()
            clf.predict_proba(X[k:k + 1])
            rows.append({"run": run, "window": i, "warmup": i == 0, "fit_ms": fit_ms,
                         "tabpfn_predict_cpu_ms": (time.perf_counter() - t0) * 1000.0})
        print(f"[latency_tabpfn] run {run} done (fit {fit_ms:.0f} ms)", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "latency_tabpfn_raw.csv", index=False)
    meas = ["fit_ms", "tabpfn_predict_cpu_ms"]
    per_run = raw[~raw.warmup].groupby("run")[meas].median()
    summ = pd.DataFrame({"median_over_runs_ms": per_run.median(), "min_run_ms": per_run.min(), "max_run_ms": per_run.max(),
                         "p95_median_over_runs_ms": raw[~raw.warmup].groupby("run")[meas].quantile(0.95).median()})
    summ.index.name = "measurement"
    summ.to_csv(RESULTS / "latency_tabpfn_summary.csv")
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
