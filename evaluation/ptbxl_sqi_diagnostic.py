"""
Diagnostic of the SQI inputs on the frozen PTB-XL test (round V19, 2026-11-19; review finding PAPER-004 of the V17
review). Descriptive and post hoc; no model is fitted or rescored.

Question: why are several SQI-based scores inverted on PTB-XL global noise versus clean (Supplementary Table S38)? The
review proposed the quantization artefact in the flatline fraction already seen on CinC 2011. For lead II and for the
record mean over the 12 leads, the script reports, per stratum (global noise, clean), the median of each stored SQI, its
univariate AUROC for global noise versus clean (values below 0.5: higher feature value associated with clean), and the
standardised value of the flatline fraction relative to the BUT QDB development windows (z = (x - mean) / SD), the
quantity that the tabular models receive.

Inputs: cache/datasets/ptbxl/ptbxl_windows.csv (stored SQIs of the frozen sample), the BUT QDB cohort SQIs in
<results>/cohort1000_consolidated_oof_predictions.csv.
Output: <results>/ptbxl_sqi_diagnostic.csv
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/ptbxl_sqi_diagnostic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, RESULTS_DIR  # noqa: E402

SQIS = ["bsqi_detector_agreement", "baseline_wander_ratio", "high_frequency_noise_ratio", "saturation_fraction",
        "flatline_fraction", "qrs_energy_ratio", "kurtosis_ksqi", "skewness_ssqi", "zero_crossing_rate",
        "powerline_noise_ratio", "signal_std"]
OUT = RESULTS_DIR / "ptbxl_sqi_diagnostic.csv"


def main():
    w = pd.read_csv(DATA_DIR / "ptbxl" / "ptbxl_windows.csv")
    w = w[w.stratum.isin(["global", "clean"])]
    but = pd.read_csv(RESULTS_DIR / "cohort1000_consolidated_oof_predictions.csv")
    if len(but) != 988:
        raise ValueError(f"expected 988 BUT QDB windows, got {len(but)}")
    mu, sd = but.flatline_fraction.mean(), but.flatline_fraction.std(ddof=0)
    lead2 = w[w.lead_idx == 1]
    rec = w.groupby(["record_name", "stratum"], as_index=False)[SQIS].mean()
    if lead2.record_name.nunique() != 600 or len(rec) != 600:
        raise ValueError("expected 600 records (300 global, 300 clean)")
    rows = []
    for unit, d in (("lead II", lead2), ("record mean over 12 leads", rec)):
        y = (d.stratum == "global").astype(int).values
        for k in SQIS:
            rows.append({"unit": unit, "sqi": k, "median_global": d.loc[y == 1, k].median(),
                         "median_clean": d.loc[y == 0, k].median(), "auroc_global_vs_clean": roc_auc_score(y, d[k])})
        z = (d.flatline_fraction - mu) / sd
        rows.append({"unit": unit, "sqi": "flatline_fraction, z relative to BUT QDB", "median_global": z[y == 1].median(),
                     "median_clean": z[y == 0].median(), "auroc_global_vs_clean": roc_auc_score(y, z)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"BUT QDB flatline fraction: mean {mu:.6f}, SD {sd:.6f}")
    print(out.to_string(index=False))
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
