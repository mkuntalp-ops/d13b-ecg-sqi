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
  Both are unpenalized maximum-likelihood fits. Revised 2026-11-13 (review PAPER-003): the intercept is the root of its
  score equation (Brent's method), finite whenever both classes occur; the slope uses Newton-Raphson with step halving
  and is reported with a status. The previous
  undamped Newton fit returned NaN for estimable intercepts; its output is kept as
  decision_calibration_pre_PAPER003_2026-11-13.csv. V3 (2026-11-13): separation is diagnosed from the score-label
  structure (separation()) before fitting, and only a diagnosed separation is labelled as such; statuses are
  'not identifiable (constant score)', 'not estimable (complete|quasi-complete separation)' and
  'not converged (optimisation)'. Output before this change: decision_calibration_pre_V3_2026-11-13.csv.
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


def _loglik(eta: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(y * eta - np.logaddexp(0, eta)))


def calibration_intercept_status(y: np.ndarray, p: np.ndarray) -> tuple[float, str]:
    """Offset intercept a solving sum(y - expit(a + z)) = 0 (revised 2026-11-13, review PAPER-003).

    The score is strictly decreasing in a, so a unique finite root exists whenever both classes occur, also when
    z is constant. It is found by bracketing and Brent's method instead of the undamped Newton iteration used
    before, which diverged when all scores were at the clipping bound."""
    from scipy.optimize import brentq
    from scipy.special import expit
    y = np.asarray(y, float)
    z = logit(p)
    if y.min() == y.max():
        return np.nan, "one class only"
    f = lambda a: float(np.sum(y - expit(a + z)))
    lo, hi = -1.0, 1.0
    while f(lo) <= 0:
        lo *= 2
    while f(hi) >= 0:
        hi *= 2
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=500)), "estimated"


def separation(y: np.ndarray, z: np.ndarray) -> str | None:
    """Complete or quasi-complete separation of y by the single covariate z (with intercept), or None.

    In one dimension the maximum-likelihood slope does not exist exactly when some cut v leaves only one class on
    one side and only the other class (or nothing) on the other side, with the units at z == v (if any) unrestricted
    (Albert & Anderson, Biometrika 1984). Checked directly on the score-label structure, independently of any
    optimiser (revised 2026-11-13, V3, follow-up to review PAPER-003)."""
    y = np.asarray(y).astype(int)
    lv, inv = np.unique(z, return_inverse=True)
    if len(lv) < 2 or y.min() == y.max():
        return None
    pos = np.bincount(inv, weights=y, minlength=len(lv))
    neg = np.bincount(inv, weights=1 - y, minlength=len(lv))
    cp, cn = np.r_[0, np.cumsum(pos)], np.r_[0, np.cumsum(neg)]   # counts over levels [0, k)
    P, N = cp[-1], cn[-1]

    def separated(pb, nb, pa, na):
        # one class (or nothing) below, the other class (or nothing) above, not both sides empty
        nonempty = (pb + nb + pa + na) > 0
        return nonempty & (((pb == 0) & (na == 0)) | ((nb == 0) & (pa == 0)))
    k = np.arange(1, len(lv))            # cut between levels k-1 and k
    if np.any(separated(cp[k], cn[k], P - cp[k], N - cn[k])):
        return "complete"
    k = np.arange(len(lv))               # cut at level k; units at the level unrestricted
    if np.any(separated(cp[k], cn[k], P - cp[k + 1], N - cn[k + 1])):
        return "quasi-complete"
    return None


def calibration_slope_status(y: np.ndarray, p: np.ndarray, iters: int = 200) -> tuple[float, str]:
    """Slope b in logit P(y=1) = a + b z: Newton-Raphson with step halving on the log-likelihood.

    Status: 'not identifiable (constant score)' when z has a single value; 'not estimable (complete separation)' or
    'not estimable (quasi-complete separation)' when separation() finds it in the data (the maximum-likelihood slope
    does not exist); 'not converged (optimisation)' when no separation is present but the fit fails, which is an
    optimiser failure and is not labelled separation (V3, 2026-11-13)."""
    y = np.asarray(y, float)
    z = logit(p)
    if y.min() == y.max():
        return np.nan, "one class only"
    if np.ptp(z) == 0:
        return np.nan, "not identifiable (constant score)"
    sep = separation(y, z)
    if sep is not None:
        return np.nan, f"not estimable ({sep} separation)"
    X = np.column_stack([np.ones(len(y)), z])
    beta = np.zeros(2)
    ll = _loglik(X @ beta, y)
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        w = mu * (1 - mu)
        H = X.T @ (X * w[:, None])
        g = X.T @ (y - mu)
        try:
            step = np.linalg.solve(H + 1e-12 * np.eye(2), g)
        except np.linalg.LinAlgError:
            return np.nan, "not converged (optimisation)"
        t = 1.0
        while t > 1e-8:
            cand = beta + t * step
            ll_c = _loglik(X @ cand, y)
            if ll_c >= ll - 1e-12:
                break
            t /= 2
        beta, ll = cand, ll_c
        if np.max(np.abs(t * step)) < 1e-10:
            if np.max(np.abs(beta)) > 1e3:
                return np.nan, "not converged (optimisation)"
            return float(beta[1]), "estimated"
    return np.nan, "not converged (optimisation)"


def calibration_intercept(y: np.ndarray, p: np.ndarray) -> float:
    return calibration_intercept_status(y, p)[0]


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    return calibration_slope_status(y, p)[0]


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
    d["calibration_intercept"], d["calibration_intercept_status"] = calibration_intercept_status(y, p)
    d["calibration_slope"], d["calibration_slope_status"] = calibration_slope_status(y, p)
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
                   "tau": TAU, **{k: est[k] for k in ["tp", "fp", "tn", "fn"]},
                   "calibration_intercept_status": est["calibration_intercept_status"],
                   "calibration_slope_status": est["calibration_slope_status"]}
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
