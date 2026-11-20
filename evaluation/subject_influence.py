"""
Subject influence on the in-domain BUT QDB comparison (round V21, 2026-11-20; review finding PAPER-003). Post hoc and
descriptive; evaluation only: the saved out-of-fold scores are re-pooled without one subject at a time, and no model is
refitted, so the analysis shows how much the pooled estimates depend on each subject's windows, not how a model trained
without that subject would behave.

For the full cohort and for each of the 15 subjects left out of the evaluation, the script records the pooled AUROC of
the 11 models, the model with the highest AUROC, the rank of the 2D CNN, the AUROC differences of logistic regression
and the MLP from the 2D CNN, and the Spearman correlation of the 11 AUROCs with those of the full cohort.

Inputs: the BUT QDB family of evaluation/paired_comparisons.families() (saved out-of-fold scores, primary cohort).
Output: <results>/butqdb_subject_influence.csv
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/subject_influence.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
import paired_comparisons as pc  # noqa: E402

OUT = pc.RESULTS / "butqdb_subject_influence.csv"
BEST, LR, MLP = "2D_CNN_Image", "Logistic_Regression_SQI", "Independent_SQI_MLP"


def aurocs(d: pd.DataFrame, models: list) -> pd.Series:
    return pd.Series({m: roc_auc_score(d.y, d[m]) for m in models})


def main():
    _, but = pc.families()["BUT_QDB (5-fold subject CV)"]
    models = [c for c in but.columns if c not in ("unit", "y", "cluster")]
    if len(models) != 11 or len(but) != 988 or but.cluster.nunique() != 15:
        raise ValueError("expected 11 models, 988 windows and 15 subjects")
    full = aurocs(but, models)
    rows = []
    for subj in [None] + sorted(but.cluster.unique()):
        d = but if subj is None else but[but.cluster != subj]
        a = aurocs(d, models)
        rank = a.rank(ascending=False, method="min")
        row = {"left_out": "none" if subj is None else int(subj), "n_windows": len(d), "n_unusable": int(d.y.sum()),
               "left_out_unusable": 0 if subj is None else int(but[but.cluster == subj].y.sum()),
               "best_model": a.idxmax(), "rank_2D_CNN": int(rank[BEST]), "delta_LR_minus_2D_CNN": a[LR] - a[BEST],
               "delta_MLP_minus_2D_CNN": a[MLP] - a[BEST], "spearman_vs_full": spearmanr(a.values, full.values)[0]}
        row.update({f"auroc_{m}": a[m] for m in models})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out[["left_out", "left_out_unusable", "best_model", "rank_2D_CNN", "delta_LR_minus_2D_CNN",
               "delta_MLP_minus_2D_CNN", "spearman_vs_full"]].round(3).to_string(index=False))
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
