"""
AUROC and average precision with 95% bootstrap confidence intervals for every model in each primary evaluation set
(fourth review, TRIPOD+AI item 23a; 2026-10-26).

Score tables are those of evaluation/paired_comparisons.py (models trained on BUT QDB only; external sets scored with
the pre-specified recording maximum). Resampling unit: subject on BUT QDB (cluster bootstrap), recording externally.
2,000 resamples, seed 42, percentile intervals; resamples with one class are skipped (counted).

Output: results/model_cis.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_comparisons import families  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "model_cis.csv"
B, SEED = 2000, 42


def main():
    rows = []
    for fam, (_, df) in families().items():
        y = df["y"].values.astype(int)
        units = df["cluster"].values if "cluster" in df.columns else df["unit"].values
        groups = {u: np.where(units == u)[0] for u in np.unique(units)}
        keys = list(groups)
        rng = np.random.default_rng(SEED)
        draws = [np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]) for _ in range(B)]
        draws = [i for i in draws if y[i].min() != y[i].max()]
        for m in [c for c in df.columns if c not in ("unit", "y", "cluster")]:
            p = df[m].values.astype(float)
            auc = np.array([roc_auc_score(y[i], p[i]) for i in draws])
            ap = np.array([average_precision_score(y[i], p[i]) for i in draws])
            rows.append({"family": fam, "model": m, "n_units": len(y), "n_positive": int(y.sum()),
                         "resampling_unit": "subject" if "cluster" in df.columns else "recording",
                         "valid_resamples": len(draws), "auroc": roc_auc_score(y, p),
                         "auroc_lo": np.percentile(auc, 2.5), "auroc_hi": np.percentile(auc, 97.5),
                         "ap": average_precision_score(y, p), "ap_lo": np.percentile(ap, 2.5), "ap_hi": np.percentile(ap, 97.5)})
        print(f"[{fam}] done ({len(draws)} valid resamples)", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
