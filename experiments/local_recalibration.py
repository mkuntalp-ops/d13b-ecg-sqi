"""
Local recalibration with a small labelled target sample (analysis plan A3, 2026-10-25).

How many labelled target recordings are needed to set an operating point on a new dataset? For CinC 2011 (565
records) and CinC 2017 (8,512 recordings), and every model with BUT-only scores on the full set (the score tables of
evaluation/paired_comparisons.py), n target records are drawn by simple random sampling (n = 25, 50, 100, 200;
500 repetitions; seed 42). Evaluation is always on the records not drawn.

Threshold methods (Target 1: maximum sensitivity at FPR <= 5%, experiments/nested_thresholds.pick):
- transferred: chosen on the pooled BUT QDB out-of-fold scores, no target labels;
- local: chosen on the n drawn records; if the draw lacks a class, the transferred threshold is used (counted).
Calibration: Platt scaling (logistic regression on the logit of the score) fitted on the drawn records versus the
uncalibrated BUT-only score (Brier score on the held-out records); single-class draws are skipped (counted).
Descriptive only: no hypothesis tests, no model or n selected on the result.

Outputs: results/local_recalibration_draws.csv (one row per model, n and repetition),
         results/local_recalibration_summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from nested_thresholds import pick  # noqa: E402
from paired_comparisons import families  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
DRAWS_OUT = RESULTS / "local_recalibration_draws.csv"
SUM_OUT = RESULTS / "local_recalibration_summary.csv"
SAMPLE_SIZES = [25, 50, 100, 200]
REPS = 500
SEED = 42
FPR_TARGET, FPR_TOLERANCE = 0.05, 0.075
TARGETS = {"CinC2011 lead II (565 records)": "CinC2011", "CinC2017 (8,512 recordings)": "CinC2017"}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def sens_fpr(y, p, t):
    pred = p >= t
    return pred[y == 1].mean(), pred[y == 0].mean()


def main():
    fam = families()
    but = fam["BUT_QDB (5-fold subject CV)"][1]
    rows = []
    for fam_name, short in TARGETS.items():
        df = fam[fam_name][1]
        y = df["y"].values.astype(int)
        models = [c for c in df.columns if c not in ("unit", "y", "cluster") and c in but.columns]
        rng = np.random.default_rng(SEED)
        draws = {n: [rng.choice(len(df), n, replace=False) for _ in range(REPS)] for n in SAMPLE_SIZES}
        for m in models:
            p = df[m].values.astype(float)
            t_trans = pick(but["y"].values.astype(int), but[m].values.astype(float), "target1_fpr5")
            for n in SAMPLE_SIZES:
                for rep, idx in enumerate(draws[n]):
                    rest = np.ones(len(df), bool)
                    rest[idx] = False
                    ys, ps, yr, pr = y[idx], p[idx], y[rest], p[rest]
                    two_class = ys.min() != ys.max()
                    t_loc = pick(ys, ps, "target1_fpr5") if two_class else t_trans
                    s_tr, f_tr = sens_fpr(yr, pr, t_trans)
                    s_lo, f_lo = sens_fpr(yr, pr, t_loc)
                    brier_raw = np.mean((pr - yr) ** 2)
                    brier_platt = np.nan
                    if two_class:
                        cal = LogisticRegression(C=1e6, max_iter=1000).fit(logit(ps), ys)
                        brier_platt = np.mean((cal.predict_proba(logit(pr))[:, 1] - yr) ** 2)
                    rows.append({"dataset": short, "model": m, "n": n, "rep": rep, "n_pos_sample": int(ys.sum()),
                                 "fallback": not two_class, "t_transferred": t_trans, "t_local": t_loc,
                                 "sens_transferred": s_tr, "fpr_transferred": f_tr, "sens_local": s_lo,
                                 "fpr_local": f_lo, "brier_raw": brier_raw, "brier_platt": brier_platt})
            print(f"[{short}] {m} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(DRAWS_OUT, index=False)

    def q(x, a):
        return float(np.nanpercentile(x, a))
    summ = []
    for (ds, m, n), g in d.groupby(["dataset", "model", "n"], sort=False):
        rec = {"dataset": ds, "model": m, "n": n, "reps": len(g), "fallback_draws": int(g.fallback.sum()),
               "platt_skipped": int(g.brier_platt.isna().sum())}
        for meth in ["transferred", "local"]:
            for k in ["fpr", "sens"]:
                x = g[f"{k}_{meth}"].values
                rec.update({f"{k}_{meth}_median": float(np.median(x)), f"{k}_{meth}_p2.5": q(x, 2.5),
                            f"{k}_{meth}_p97.5": q(x, 97.5)})
            rec[f"share_fpr_le_7.5_{meth}"] = float((g[f"fpr_{meth}"] <= FPR_TOLERANCE).mean())
        rec.update(brier_raw_median=float(g.brier_raw.median()), brier_platt_median=float(g.brier_platt.median()))
        summ.append(rec)
    s = pd.DataFrame(summ)
    s.to_csv(SUM_OUT, index=False)
    cols = ["dataset", "model", "n", "fallback_draws", "fpr_transferred_median", "fpr_local_median",
            "fpr_local_p97.5", "sens_local_median", "share_fpr_le_7.5_local", "brier_raw_median", "brier_platt_median"]
    print(s[s.n == 100][cols].round(3).to_string(index=False))
    print(f"[Saved] {SUM_OUT}")


if __name__ == "__main__":
    main()
