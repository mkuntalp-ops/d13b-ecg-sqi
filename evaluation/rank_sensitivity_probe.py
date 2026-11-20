"""
Rank agreement without the d1-3B image-token probe (round V23, 2026-11-20; review finding PAPER-006). Post hoc and
descriptive; saved predictions only, no model is refitted.

Two d1-3B representations were probed, and the image-token probe was carried forward after both results were known
(Methods, Deviations). Because it is one of the 11 models whose ranks are compared, this script repeats the rank
analysis of evaluation/rank_uncertainty.py (same 2,000 paired bootstrap resamples of BUT QDB subjects and external
recordings, seed 42, same statistics) with the probe excluded (10 models) and with the probe and the rule heuristic
excluded (9 models). Before that it re-runs the two stored model sets ("all 11 models", "heuristic excluded") and
stops unless every statistic reproduces <results>/rank_uncertainty.csv, which is not modified.

Output: <results>/rank_sensitivity_probe.csv (the two new model sets)
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/rank_sensitivity_probe.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rank_uncertainty as ru  # noqa: E402

OUT = ru.RESULTS / "rank_sensitivity_probe.csv"
PROBE = "d1_image_probe"
STATS = ["estimate", "boot_median", "ci_lo", "ci_hi", "prop_positive", "boot_mean"]


def run(data: dict, models: list, label: str) -> pd.DataFrame:
    rng = np.random.default_rng(ru.SEED)
    point, boot = {}, {}
    for f in ru.SETS:
        point[f], boot[f] = ru.boot_aurocs(data[f], models, rng)
    return pd.DataFrame(ru.summarise(models, point, boot, label))


def main():
    fam = ru.families()
    data = {f: fam[f][1] for f in ru.SETS}
    models = [c for c in data["BUT_QDB (5-fold subject CV)"].columns if c not in ru.META]
    if len(models) != 11 or PROBE not in models or ru.HEURISTIC not in models:
        raise ValueError(f"expected 11 models including {PROBE} and {ru.HEURISTIC}, got {models}")
    stored = pd.read_csv(ru.RESULTS / "rank_uncertainty.csv")
    for label, ms in [("all 11 models", models), ("heuristic excluded", [m for m in models if m != ru.HEURISTIC])]:
        new, old = run(data, ms, label), stored[stored.model_set == label].reset_index(drop=True)
        if len(new) != len(old) or not np.allclose(new[STATS].values, old[STATS].values, rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"{label}: re-run does not reproduce rank_uncertainty.csv")
    out = pd.concat([run(data, [m for m in models if m != PROBE], "d1-3B image-token probe excluded"),
                     run(data, [m for m in models if m not in (PROBE, ru.HEURISTIC)], "probe and heuristic excluded")],
                    ignore_index=True)
    out.to_csv(OUT, index=False)
    show = out[out.statistic.isin(["spearman_rho", "same_leader", "regret_auroc"])]
    print(show[["model_set", "n_models", "set_a", "set_b", "statistic", "estimate", "ci_lo", "ci_hi", "prop_positive",
                "leader_a", "leader_b"]].round(3).to_string(index=False))
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
