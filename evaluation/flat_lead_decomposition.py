"""
Decomposition of the CinC 2011 lead II AUROC into the contributions of flat and non-flat unacceptable records (round
V21, 2026-11-20; review finding PAPER-006). Post hoc and descriptive; saved scores only.

All 73 records with a constant lead II are unacceptable, so every negative (acceptable record) is non-flat. The AUROC is
the share of positive-negative pairs ranked correctly (ties counted 1/2), and it splits exactly into
    AUROC = w_flat * A_flat + w_nonflat * A_nonflat,
where A_g is the AUROC of the positives of group g against all 773 negatives and w_g is the share of positives in g
(73/225 and 152/225). A_nonflat equals the AUROC without the constant-lead records (Supplementary Table S40). The
difference between two models splits in the same way.

Inputs: the CinC 2011 family of evaluation/paired_comparisons.families(); signal SD from
        cache/datasets/cinc2011/cinc2011_setA_leadII_full.csv.
Output: <results>/cinc2011_flat_lead_decomposition.csv
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/flat_lead_decomposition.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
import paired_comparisons as pc  # noqa: E402
from config import DATA_DIR  # noqa: E402

OUT = pc.RESULTS / "cinc2011_flat_lead_decomposition.csv"


def main():
    _, c = pc.families()["CinC2011 lead II (998 records)"]
    meta = pd.read_csv(DATA_DIR / "cinc2011" / "cinc2011_setA_leadII_full.csv", usecols=["record_name", "signal_std", "label"])
    meta = meta.assign(record_name=meta.record_name.astype(str)).set_index("record_name")
    flat = (meta.loc[c.unit, "signal_std"] == 0).values
    y = c.y.values.astype(int)
    if flat.sum() != 73 or y[flat].min() != 1 or (meta.loc[c.unit, "label"].values != y).any():
        raise ValueError("expected 73 constant-lead records, all unacceptable, with matching labels")
    pos_f, pos_n, neg = flat & (y == 1), ~flat & (y == 1), y == 0
    w_f, w_n = pos_f.sum() / (y == 1).sum(), pos_n.sum() / (y == 1).sum()
    rows = []
    for m in [k for k in c.columns if k not in ("unit", "y")]:
        s = c[m].values.astype(float)
        a_all = roc_auc_score(y, s)
        a_f = roc_auc_score(y[pos_f | neg], s[pos_f | neg])
        a_n = roc_auc_score(y[pos_n | neg], s[pos_n | neg])
        if abs(w_f * a_f + w_n * a_n - a_all) > 1e-12:
            raise ValueError(f"{m}: decomposition does not add up")
        rows.append({"model": m, "auroc_all": a_all, "auroc_flat_vs_negatives": a_f, "auroc_nonflat_vs_negatives": a_n,
                     "weight_flat": w_f, "weight_nonflat": w_n, "contribution_flat": w_f * a_f, "contribution_nonflat": w_n * a_n})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    p, b = out.set_index("model").loc["d1_image_probe"], out.set_index("model").loc["2D_CNN_Image"]
    print(out.round(3).to_string(index=False))
    print(f"probe - 2D CNN: total {p.auroc_all - b.auroc_all:.4f} = flat {p.contribution_flat - b.contribution_flat:.4f} "
          f"+ non-flat {p.contribution_nonflat - b.contribution_nonflat:.4f}")
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
