"""
CinC 2011 records with a constant lead II (post hoc, descriptive; analysis plan amendment of 2026-10-26).

65 of the 197 unacceptable CinC 2011 records have a constant lead II (signal SD 0). For every model trained on BUT QDB
only, this script counts how many of them are scored >= 0.5 and recomputes the AUROC without them. No tests.
Output: results/cinc2011_constant_lead_check.csv
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "evaluation"))
from lodo_multisource import load_sources  # noqa: E402
from paired_comparisons import families  # noqa: E402

OUT = ROOT / "results" / "cinc2011_constant_lead_check.csv"


def main():
    c11 = load_sources()["CinC2011"]
    constant = set(c11.record[c11.signal_std == 0])
    scores = families()["CinC2011 lead II (565 records)"][1]
    is_const = scores.unit.isin(constant).values
    assert is_const.sum() == len(constant) and scores.y[is_const].eq(1).all()
    rows = []
    for m in [c for c in scores.columns if c not in ("unit", "y", "cluster")]:
        rows.append({"model": m, "n_constant": int(is_const.sum()), "constant_flagged_at_0.5": int((scores.loc[is_const, m] >= 0.5).sum()),
                     "auroc_all": roc_auc_score(scores.y, scores[m]),
                     "auroc_without_constant": roc_auc_score(scores.y[~is_const], scores.loc[~is_const, m])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
