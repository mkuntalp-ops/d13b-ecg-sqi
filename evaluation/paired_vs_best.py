"""
Sensitivity analysis of the multiplicity family (fourth review, 2026-10-26): Holm adjustment over the ten comparisons
with the best model of each evaluation set, instead of over all 55 pairwise comparisons. Reads the stored paired
tests (evaluation/paired_comparisons.py); the best model is the highest-AUROC model of each set, so this family is
chosen after seeing the point estimates and is reported only as a sensitivity analysis.

Output: results/paired_vs_best.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_comparisons import holm  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    p = pd.read_csv(RESULTS / "paired_comparisons.csv")
    rows = []
    for fam, g in p.groupby("family", sort=False):
        auc = pd.concat([g[["model_a", "auroc_a"]].set_axis(["m", "a"], axis=1),
                         g[["model_b", "auroc_b"]].set_axis(["m", "a"], axis=1)]).drop_duplicates("m").set_index("m").a
        best = auc.idxmax()
        v = g[(g.model_a == best) | (g.model_b == best)].copy()
        v["p_holm_vs_best"] = holm(v.p.values)
        for r in v.itertuples():
            other = r.model_b if r.model_a == best else r.model_a
            rows.append({"family": fam, "best": best, "model": other, "auroc": auc[other], "auroc_best": auc[best],
                         "p_unadjusted": r.p, "p_holm_all_pairs": r.p_holm, "p_holm_vs_best": r.p_holm_vs_best})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "paired_vs_best.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
