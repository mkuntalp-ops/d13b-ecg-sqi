"""
Uncertainty of the in-domain/external rank agreement and of in-domain model selection (review PAPER-001 and
PAPER-002, 2026-11-13). Post hoc, requested in review; reads saved predictions only, no model is refitted.

Inputs: the evaluation sets of paired_comparisons.families() (the units, record aggregation and 11 model scores of
Table 1): BUT QDB out-of-fold window scores, CinC 2011 lead II (998 records), CinC 2017 (8,512 recordings, max).

Procedure: B = 2,000 paired bootstrap resamples, seed 42. Each resample draws BUT QDB subjects with replacement
(windows follow their subject) and, independently, recordings of each external set with replacement; all models are
evaluated on the same drawn units, so their AUROCs stay paired. Resamples with one class only are redrawn.
Per resample and pair of sets:
  rho     Spearman correlation of the model AUROCs between the two sets;
  leader  whether the model with the highest AUROC on the first set also has the highest on the second;
  regret  highest AUROC on the second set minus the second-set AUROC of the first-set leader (the AUROC lost by
          selecting on the first set).
Reported: point estimate on the full data, bootstrap median, 2.5th and 97.5th percentiles, and the proportion of
resamples with rho > 0 or with the same leader. The intervals are conditional on the fitted models: training
variability (fold assignment, seeds) is not resampled.
Sensitivity (PAPER-002): the same with the rule-based heuristic excluded (10 models), because its coefficients were set
during prototyping and their independence from BUT QDB cannot be guaranteed.

Ties (V6, 2026-11-14, review PAPER-013). Two models can have exactly the same AUROC in a resample. The primary rule
is stated here: among models tied for the highest AUROC, the leader is the first in the fixed model order of
paired_comparisons.families() (numpy.argmax), on the selection set and on the target set alike. Because that rule
depends on the column order, the script also reports, per pair of sets,
  ties_selection / ties_target   number of resamples in which the highest AUROC is shared (|difference| <= 1e-12);
  same_leader_any                some model is co-best on both sets (most favourable tie-break);
  same_leader_all                the co-best sets of the two sets are identical (least favourable);
  regret_best_case / regret_worst_case   smallest and largest regret over the co-best models of the selection set.
The primary rows are unchanged by this addition; the output before it is rank_uncertainty_pre_V6_2026-11-14.csv.

Output: $ECG_RESULTS_DIR/rank_uncertainty.csv
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/rank_uncertainty.py
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_comparisons import RESULTS, families  # noqa: E402

OUT = RESULTS / "rank_uncertainty.csv"
B = 2000
SEED = 42
META = ("unit", "y", "cluster")
SETS = {"BUT_QDB (5-fold subject CV)": "BUT QDB", "CinC2011 lead II (998 records)": "CinC 2011",
        "CinC2017 (8,512 recordings)": "CinC 2017"}
HEURISTIC = "Rule_Based_SQI_Heuristic"


def auroc_matrix(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """AUROC of every column of S (Mann-Whitney with midranks); y in {0, 1}."""
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    r = np.apply_along_axis(rankdata, 0, S)
    return (r[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def resample_index(rng, groups: list) -> np.ndarray:
    return np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])


def boot_aurocs(df: pd.DataFrame, models: list, rng) -> tuple[np.ndarray, np.ndarray]:
    y = df["y"].values.astype(int)
    S = df[models].values.astype(float)
    clusters = df["cluster"].values if "cluster" in df else np.arange(len(df))
    groups = [np.where(clusters == c)[0] for c in pd.unique(clusters)]
    point = auroc_matrix(y, S)
    out = np.empty((B, len(models)))
    b = 0
    while b < B:
        i = resample_index(rng, groups)
        if y[i].min() == y[i].max():
            continue
        out[b] = auroc_matrix(y[i], S[i])
        b += 1
    return point, out


def selection(a: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    """rho, same-leader indicator and regret for AUROC vectors a (selection set) and c (target set)."""
    lead = int(np.argmax(a))
    return spearmanr(a, c)[0], float(lead == int(np.argmax(c))), float(c.max() - c[lead])


TIE_ATOL = 1e-12


def co_best(a: np.ndarray) -> np.ndarray:
    """Boolean mask of the models sharing the highest AUROC."""
    return np.isclose(a, a.max(), rtol=0, atol=TIE_ATOL)


def selection_ties(a: np.ndarray, c: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Tie-aware bounds: ties on a, ties on c, same leader (any / all), regret (best / worst case)."""
    A, C = co_best(a), co_best(c)
    regrets = c.max() - c[A]
    return (float(A.sum() > 1), float(C.sum() > 1), float((A & C).any()), float(np.array_equal(A, C)),
            float(regrets.min()), float(regrets.max()))


def summarise(models: list, point: dict, boot: dict, label: str) -> list:
    rows = []
    for fa, fb in combinations(SETS, 2):
        pa, pb = point[fa], point[fb]
        rho0, same0, reg0 = selection(pa, pb)
        stats = np.array([selection(boot[fa][k], boot[fb][k]) for k in range(B)])
        for j, (name, est) in enumerate([("spearman_rho", rho0), ("same_leader", same0), ("regret_auroc", reg0)]):
            col = stats[:, j]
            rows.append({"model_set": label, "n_models": len(models), "set_a": SETS[fa], "set_b": SETS[fb],
                         "statistic": name, "estimate": est, "boot_median": np.median(col),
                         "ci_lo": np.percentile(col, 2.5), "ci_hi": np.percentile(col, 97.5),
                         "prop_positive": float(np.mean(col > 0)), "boot_mean": float(np.mean(col)),
                         "leader_a": models[int(np.argmax(pa))], "leader_b": models[int(np.argmax(pb))],
                         "resamples": B, "seed": SEED})
        tie0 = selection_ties(pa, pb)
        ties = np.array([selection_ties(boot[fa][k], boot[fb][k]) for k in range(B)])
        names = ["ties_selection", "ties_target", "same_leader_any", "same_leader_all", "regret_best_case",
                 "regret_worst_case"]
        for j, name in enumerate(names):
            col = ties[:, j]
            rows.append({"model_set": label, "n_models": len(models), "set_a": SETS[fa], "set_b": SETS[fb],
                         "statistic": name, "estimate": tie0[j], "boot_median": np.median(col),
                         "ci_lo": np.percentile(col, 2.5), "ci_hi": np.percentile(col, 97.5),
                         "prop_positive": float(np.mean(col > 0)), "boot_mean": float(np.mean(col)),
                         "n_resamples_positive": int(np.sum(col > 0)),
                         "leader_a": models[int(np.argmax(pa))], "leader_b": models[int(np.argmax(pb))],
                         "resamples": B, "seed": SEED})
    return rows


def main():
    fam = families()
    data = {f: fam[f][1] for f in SETS}
    models = [c for c in data["BUT_QDB (5-fold subject CV)"].columns if c not in META]
    for f, df in data.items():
        missing = set(models) - set(df.columns)
        if missing or df[models].isna().any().any():
            raise ValueError(f"{f}: missing scores for {missing or 'some units'}")
    if len(models) != 11:
        raise ValueError(f"expected 11 models, got {len(models)}")
    rows = []
    for label, ms in [("all 11 models", models), ("heuristic excluded", [m for m in models if m != HEURISTIC])]:
        rng = np.random.default_rng(SEED)
        point, boot = {}, {}
        for f in SETS:
            point[f], boot[f] = boot_aurocs(data[f], ms, rng)
        rows += summarise(ms, point, boot, label)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out[["model_set", "set_a", "set_b", "statistic", "estimate", "boot_median", "ci_lo", "ci_hi",
               "prop_positive", "leader_a", "leader_b"]].round(3).to_string(index=False))
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
