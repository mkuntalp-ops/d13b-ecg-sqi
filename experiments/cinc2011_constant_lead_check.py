"""
CinC 2011 records with a constant lead II (post hoc, descriptive; analysis plan amendment of 2026-10-26).

In the v1 565-record set, 65 of the 197 unacceptable CinC 2011 records had a constant lead II (signal SD 0); v2 uses all
998 labelled records (the counts are printed at run time). For every model trained on BUT QDB
only, this script counts how many of them are scored >= 0.5 and recomputes the AUROC without them. It also gives the
univariate AUROC of every SQI on BUT QDB and on CinC 2011 lead II with and without these records (fourth-review
request to explain the SQI reversals mechanistically). No tests.
Outputs: results/cinc2011_constant_lead_check.csv, results/cinc2011_constant_lead_univariate.csv
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "evaluation"))
from features.sqi_extractor import SQI_FEATURE_NAMES  # noqa: E402
from lodo_multisource import load_sources  # noqa: E402
from paired_comparisons import families  # noqa: E402

from config import RESULTS_DIR  # noqa: E402  (ECG_RESULTS_DIR overrides)
OUT = RESULTS_DIR / "cinc2011_constant_lead_check.csv"
UNI_OUT = RESULTS_DIR / "cinc2011_constant_lead_univariate.csv"


def main():
    c11 = load_sources()["CinC2011"]
    constant = set(c11.record[c11.signal_std == 0])
    scores = families()["CinC2011 lead II (998 records)"][1]
    print(f"[constant] {len(constant)} records with constant lead II, "
          f"{int(c11.label[c11.signal_std == 0].sum())} of them unacceptable; {int(c11.label.sum())} unacceptable in all", flush=True)
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
    but, kept = load_sources()["BUT_QDB"], c11[c11.signal_std > 0]
    uni = pd.DataFrame([{"feature": f, "but_qdb": roc_auc_score(but.label, but[f]), "cinc2011_all": roc_auc_score(c11.label, c11[f]),
                         "cinc2011_without_constant": roc_auc_score(kept.label, kept[f])} for f in SQI_FEATURE_NAMES[:11]])
    uni.to_csv(UNI_OUT, index=False)
    print(uni.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
