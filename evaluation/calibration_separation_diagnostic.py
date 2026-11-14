"""
Separation diagnostic for the calibration slope (V4, 2026-11-14; follow-up to review PAPER-003). Read-only: uses the
saved predictions through paired_comparisons.families() and the functions of decision_calibration.py; trains nothing.

For every evaluation set and model it writes
  1. calibration_separation_levels.csv: for each full-sample fit whose slope is not estimated, every distinct
     clipped-logit score level z with the number of usable (y = 0) and unusable (y = 1) units at it;
  2. calibration_separation_diagnostic.csv: one row per full-sample fit with
     - the order of checks (one class -> constant score -> separation -> Newton fit) and the resulting status;
     - for a separation, a certificate: the cut value v, whether the cut lies between two levels (complete) or at a
       level (quasi-complete), the class counts below, at and above v, the separating inequalities, whether the
       units at v contain both classes, and the direction in which the likelihood increases without bound;
     - for an estimated slope, the slope from scikit-learn (LogisticRegression, C = inf, i.e. unpenalised, newton-cg,
       tol 1e-10), the absolute difference and the norm of the score (gradient) at our estimate;
  3. calibration_separation_bootstrap.csv: for the same 2,000 resamples as decision_calibration.py (seed 42), the
     number of resamples per slope status, with each separation re-checked by an independent brute-force rule.

Definitions (Albert & Anderson, Biometrika 1984), one covariate z with intercept, y in {0, 1}:
  complete separation: some v with y = 1 for all z > v and y = 0 for all z < v (or the reverse), no unit at v;
  quasi-complete separation: the same with the units at z = v unrestricted (both classes may occur there), at
  least one unit off the cut. In either case the log-likelihood increases without bound as the slope tends to
  +inf (unusable above v) or -inf (unusable below v), so the maximum-likelihood slope does not exist.

Tolerance for the solver comparison (pre-stated here): |b_ours - b_sklearn| <= 1e-4 * max(1, |b|).

Usage: ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/calibration_separation_diagnostic.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_calibration import B, META, SEED, calibration_slope_status, logit, separation  # noqa: E402
from paired_comparisons import RESULTS, families  # noqa: E402

TOL = 1e-4


def certificate(y: np.ndarray, z: np.ndarray) -> dict | None:
    """First separating cut found by direct enumeration (complete cuts first), or None."""
    y = np.asarray(y).astype(int)
    lv = np.unique(z)
    cands = [("complete", (lv[k - 1] + lv[k]) / 2) for k in range(1, len(lv))] + [("quasi-complete", v) for v in lv]
    for kind, v in cands:
        below, at, above = z < v, z == v, z > v
        if not (below.any() or above.any()):
            continue
        nb = (below & (y == 0)).sum(), (below & (y == 1)).sum()
        na = (at & (y == 0)).sum(), (at & (y == 1)).sum()
        nab = (above & (y == 0)).sum(), (above & (y == 1)).sum()
        if nb[1] == 0 and nab[0] == 0:
            direction, ineq = "+inf (unusable above the cut)", "y = 0 for all z < v; y = 1 for all z > v"
        elif nb[0] == 0 and nab[1] == 0:
            direction, ineq = "-inf (unusable below the cut)", "y = 1 for all z < v; y = 0 for all z > v"
        else:
            continue
        return {"separation": kind, "cut_v": float(v), "n_y0_below": int(nb[0]), "n_y1_below": int(nb[1]),
                "n_y0_at": int(na[0]), "n_y1_at": int(na[1]), "n_y0_above": int(nab[0]), "n_y1_above": int(nab[1]),
                "inequalities": ineq, "both_classes_at_cut": bool(na[0] > 0 and na[1] > 0),
                "slope_direction": direction}
    return None


def brute_force_separated(y: np.ndarray, z: np.ndarray) -> str | None:
    c = certificate(y, z)
    return None if c is None else c["separation"]


def profile_loglik_increases(y: np.ndarray, z: np.ndarray, sign: int) -> bool:
    """Profile log-likelihood (intercept maximised) at slopes sign * (1, 4, 16, 64): strictly increasing?"""
    from scipy.optimize import minimize_scalar
    vals = []
    for b in [1, 4, 16, 64]:
        eta0 = sign * b * z
        f = lambda a: -float(np.sum(y * (a + eta0) - np.logaddexp(0, a + eta0)))  # noqa: E731
        vals.append(-minimize_scalar(f, bounds=(-2000, 2000), method="bounded", options={"xatol": 1e-10}).fun)
    return bool(np.all(np.diff(vals) > 0))


def sklearn_slope(y: np.ndarray, z: np.ndarray) -> float:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)   # a convergence warning makes the comparison fail loudly
        # C = inf is the unpenalised fit (scikit-learn >= 1.8 spelling of penalty=None)
        m = LogisticRegression(C=np.inf, solver="newton-cg", tol=1e-10, max_iter=10000).fit(z[:, None], y)
    return float(m.coef_[0, 0])


def score_norm(y: np.ndarray, z: np.ndarray, b: float) -> float:
    """Norm of the score vector at (a_hat, b), with a_hat re-solved for the given b."""
    from scipy.optimize import brentq
    from scipy.special import expit
    a = brentq(lambda a: float(np.sum(y - expit(a + b * z))), -1e3, 1e3, xtol=1e-14)
    r = y - expit(a + b * z)
    return float(np.hypot(r.sum(), (r * z).sum()))


def main():
    lev_rows, diag_rows, boot_rows = [], [], []
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
            z = logit(p)
            b, status = calibration_slope_status(y, p)
            row = {"family": fam, "model": m, "n_units": len(y), "n_y1": int(y.sum()), "n_levels": len(np.unique(z)),
                   "check_order": "one class -> constant score -> separation -> Newton fit", "status": status}
            cert = certificate(y, z)
            if cert is not None:
                assert separation(y, z) == cert["separation"], (fam, m)
                row.update(cert)
                row["profile_loglik_increasing"] = profile_loglik_increases(
                    y, z, 1 if cert["slope_direction"].startswith("+") else -1)
            if status == "estimated":
                bs = sklearn_slope(y, z)
                row.update({"slope_ours": b, "slope_sklearn": bs, "abs_diff": abs(b - bs),
                            "tolerance": TOL * max(1, abs(b)), "within_tolerance": abs(b - bs) <= TOL * max(1, abs(b)),
                            "score_norm_at_ours": score_norm(y, z, b)})
            else:
                lv, inv = np.unique(z, return_inverse=True)
                for j, v in enumerate(lv):
                    lev_rows.append({"family": fam, "model": m, "status": status, "level": j, "z": v,
                                     "p_clipped": 1 / (1 + np.exp(-v)), "n_y0": int(((inv == j) & (y == 0)).sum()),
                                     "n_y1": int(((inv == j) & (y == 1)).sum())})
            diag_rows.append(row)
            counts, mismatch = {}, 0
            for i in draws:
                _, s = calibration_slope_status(y[i], p[i])
                counts[s] = counts.get(s, 0) + 1
                if s.startswith("not estimable"):
                    mismatch += brute_force_separated(y[i], logit(p[i])) is None
            boot_rows.append({"family": fam, "model": m, "valid_resamples": len(draws), **counts,
                              "separation_not_confirmed_by_brute_force": mismatch})
            print(f"[{fam}] {m}: {status}; bootstrap {counts}", flush=True)
    pd.DataFrame(lev_rows).to_csv(RESULTS / "calibration_separation_levels.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(RESULTS / "calibration_separation_diagnostic.csv", index=False)
    pd.DataFrame(boot_rows).fillna(0).to_csv(RESULTS / "calibration_separation_bootstrap.csv", index=False)
    print("[Saved] calibration_separation_{levels,diagnostic,bootstrap}.csv in", RESULTS)


if __name__ == "__main__":
    main()
