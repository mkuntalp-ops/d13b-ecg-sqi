"""
PTB-XL AUROC by PTB-XL fold (round V21, 2026-11-20; review finding PAPER-004). Post hoc and descriptive; the saved
window scores of the frozen PTB-XL test are re-aggregated and no model is scored or fitted.

The HuBERT-ECG developers list PTB-XL among the self-supervised pre-training data and test on the original test fold
(strat_fold 10), which they state was not used in pre-training (preprint version 4, Methods). For global noise versus
clean, the script gives the AUROC of the ten scored models under the 12-lead maximum and mean on the records of fold 10
(probably unseen in HuBERT-ECG pre-training) and on folds 1-9, with 95% percentile bootstrap intervals over records
(2,000 resamples, seed 42). The full-sample AUROCs are checked against the stored summary of the frozen test.

Inputs: <results>/ptbxl_frozen_window_scores.csv, <results>/ptbxl_frozen_summary.csv,
        <LOCAL_DATASETS_DIR>/ptb-xl/ptbxl_database.csv (strat_fold).
Output: <results>/ptbxl_fold10_subset.csv
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/ptbxl_fold10_subset.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import LOCAL_DATASETS_DIR, RESULTS_DIR  # noqa: E402

MODELS = ["Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG", "2D_CNN_Image",
          "Multimodal_Fusion", "Rule_Based_SQI_Heuristic", "LiquidAI_d1_3B_Neural", "d1_image_probe", "HuBERT_ECG_probe"]
OUT = RESULTS_DIR / "ptbxl_fold10_subset.csv"
B, SEED = 2000, 42


def boot_ci(y, s):
    rng = np.random.default_rng(SEED)
    n, vals = len(y), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    w = pd.read_csv(RESULTS_DIR / "ptbxl_frozen_window_scores.csv")
    w = w[w.stratum.isin(["global", "clean"])]
    meta = pd.read_csv(LOCAL_DATASETS_DIR / "ptb-xl" / "ptbxl_database.csv", usecols=["ecg_id", "strat_fold"]).set_index("ecg_id")
    stored = pd.read_csv(RESULTS_DIR / "ptbxl_frozen_summary.csv")
    stored = stored[stored.outcome == "global_vs_clean"].set_index(["aggregation", "model"]).auroc
    rows = []
    for agg in ["max", "mean"]:
        r = w.groupby("record_name").agg({**{m: agg for m in MODELS}, "stratum": "first"})
        if len(r) != 600:
            raise ValueError("expected 600 records")
        r["y"] = (r.stratum == "global").astype(int)
        r["fold"] = meta.loc[r.index, "strat_fold"].values
        for m in MODELS:
            full = roc_auc_score(r.y, r[m])
            if abs(full - stored.loc[(f"{agg}_12", m)]) > 1e-9:
                raise ValueError(f"{agg} {m}: AUROC {full} differs from the stored {stored.loc[(f'{agg}_12', m)]}")
        for subset, d in (("fold 10", r[r.fold == 10]), ("folds 1-9", r[r.fold != 10]), ("all", r)):
            y = d.y.values
            for m in MODELS:
                s = d[m].values
                lo, hi = boot_ci(y, s)
                rows.append({"aggregation": agg, "subset": subset, "n": len(d), "n_global": int(y.sum()), "model": m,
                             "auroc": roc_auc_score(y, s), "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.pivot_table(index="model", columns=["aggregation", "subset"], values="auroc").round(3).to_string())
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
