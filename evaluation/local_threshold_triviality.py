"""
Separates trivial from useful false-reject control in the local-threshold analysis (review PAPER-006, 2026-11-13).
Post hoc, requested in review; reads local_recalibration_draws.csv (experiments/local_recalibration.py) only.

A local threshold can meet the false-reject target by rejecting nothing. Per dataset, model and n, over the 500 draws
with the local threshold (held-out records of each draw):
  zero_rejection   share of draws whose local threshold flags no held-out record (sensitivity 0 and FPR 0);
  useful_control   share of draws with FPR <= 5% and sensitivity > 0;
  sens_nontrivial  median sensitivity over the draws that flag at least one record (NaN if none).
Descriptive; no tests.

Output: $ECG_RESULTS_DIR/local_threshold_triviality.csv
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/local_threshold_triviality.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_comparisons import RESULTS  # noqa: E402

SRC = RESULTS / "local_recalibration_draws.csv"
OUT = RESULTS / "local_threshold_triviality.csv"
TARGET = 0.05


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    for c in ("sens_local", "fpr_local"):
        if d[c].isna().any() or not d[c].between(0, 1).all():
            raise ValueError(f"{c}: values outside [0, 1] or missing")
    zero = (d.sens_local == 0) & (d.fpr_local == 0)
    d = d.assign(zero=zero, useful=(d.fpr_local <= TARGET) & (d.sens_local > 0))
    rows = []
    for (ds, m, n), g in d.groupby(["dataset", "model", "n"], sort=False):
        nt = g.loc[~g.zero, "sens_local"]
        rows.append({"dataset": ds, "model": m, "n": n, "draws": len(g), "zero_rejection": g.zero.mean(),
                     "useful_control": g.useful.mean(), "fpr_le_5": (g.fpr_local <= TARGET).mean(),
                     "sens_nontrivial_median": nt.median() if len(nt) else np.nan, "nontrivial_draws": len(nt)})
    return pd.DataFrame(rows)


def main():
    out = summarise(pd.read_csv(SRC))
    out.to_csv(OUT, index=False)
    print(out.round(3).to_string(index=False))
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
