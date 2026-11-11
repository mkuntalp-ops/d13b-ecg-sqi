"""
Decision-level metrics with 95% CIs and calibration intercept/slope for every model and evaluation set
(revision of 2026-11-05, plan items 4.1a and 4.2; TRIPOD+AI 23a).

Inputs: the evaluation sets of paired_comparisons.families() (same units and record aggregation as the AUROC tables:
BUT QDB windows clustered by subject; external sets one unit per recording, recording score = maximum).

Decision metrics at the fixed threshold tau = 0.50 (positive class = unusable):
  sensitivity = TP / (TP + FN); false reject rate = FP / (FP + TN); PPV = TP / (TP + FP); NPV = TN / (TN + FN).
  PPV/NPV are NaN when no unit is predicted positive/negative. PPV and NPV depend on prevalence and are reported at
  each set's own prevalence.
Calibration (Van Calster et al., BMC Med 2019), on z = logit(p) with p clipped to [1e-6, 1 - 1e-6] (pre-stated; several
models output probabilities of exactly 0 or 1, whose logit is infinite):
  calibration intercept (calibration-in-the-large): a in logit P(y=1) = a + z (z as offset); target 0;
    a < 0 means the predicted risks are too high on average;
  calibration slope: b in logit P(y=1) = a' + b z; target 1; b < 1 means predictions are too extreme.
  Both are maximum-likelihood fits (Newton-Raphson, unpenalized).
CIs: percentile bootstrap, 2,000 resamples, seed 42, resampling subjects on BUT QDB and recordings externally;
resamples with one class only are skipped (count reported).

Output: $ECG_RESULTS_DIR/decision_calibration.csv
Usage:  python evaluation/decision_calibration.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_comparisons import RESULTS, families  # noqa: E402

OUT = RESULTS / "decision_calibration.csv"
TAU = 0.5
EPS = 1e-6
B = 2000
SEED = 42
META = {"unit", "y", "cluster"}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _newton(X: np.ndarray, y: np.ndarray, offset: np.ndarray, iters: int = 100) -> np.ndarray:
    """Unpenalized logistic regression with an offset; returns coefficients (NaN if it does not converge)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(offset + X @ beta, -35, 35)
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        H = X.T @ (X * w[:, None])
        g = X.T @ (y - mu)
        try:
            step = np.linalg.solve(H + 1e-12 * np.eye(len(beta)), g)
        except np.linalg.LinAlgError:
            return np.full(X.shape[1], np.nan)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            return beta
    return np.full(X.shape[1], np.nan)


def calibration_intercept(y: np.ndarray, p: np.ndarray) -> float:
    z = logit(p)
    return float(_newton(np.ones((len(y), 1)), np.asarray(y, float), z)[0])


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    z = logit(p)
    return float(_newton(np.column_stack([np.ones(len(y)), z]), np.asarray(y, float), np.zeros(len(y)))[1])


def decision(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= TAU
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "false_reject_rate": fp / (fp + tn) if fp + tn else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan}


STATS = ["sensitivity", "false_reject_rate", "ppv", "npv", "calibration_intercept", "calibration_slope"]


def all_stats(y, p) -> dict:
    d = decision(y, p)
    d["calibration_intercept"] = calibration_intercept(y, p)
    d["calibration_slope"] = calibration_slope(y, p)
    return d


def main():
    rows = []
    for fam, (_, df) in families().items():
        y = df["y"].values.astype(int)
        clusters = df["cluster"].values if "cluster" in df else df["unit"].values
        groups = {c: np.where(clusters == c)[0] for c in np.unique(clusters)}
        keys = list(groups)
        rng = np.random.default_rng(SEED)
        draws = [np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]) for _ in range(B)]
        draws = [i for i in draws if y[i].min() != y[i].max()]
        for m in [c for c in df.columns if c not in META]:
            p = df[m].values.astype(float)
            if np.isnan(p).any():
                continue
            est = all_stats(y, p)
            boot = np.array([[all_stats(y[i], p[i])[s] for s in STATS] for i in draws])
            rec = {"family": fam, "model": m, "n_units": len(y), "n_positive": int(y.sum()), "prevalence": y.mean(),
                   "resampling_unit": "subject" if "cluster" in df else "recording", "valid_resamples": len(draws),
                   "tau": TAU, **{k: est[k] for k in ["tp", "fp", "tn", "fn"]}}
            for j, s in enumerate(STATS):
                col = boot[:, j]
                col = col[np.isfinite(col)]
                rec[s] = est[s]
                rec[f"{s}_lo"], rec[f"{s}_hi"] = (np.percentile(col, [2.5, 97.5]) if len(col) else (np.nan, np.nan))
                rec[f"{s}_n_finite"] = len(col)
            rows.append(rec)
            print(f"[{fam}] {m}: sens {est['sensitivity']:.3f}, FRR {est['false_reject_rate']:.3f}, "
                  f"intercept {est['calibration_intercept']:.2f}, slope {est['calibration_slope']:.2f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
