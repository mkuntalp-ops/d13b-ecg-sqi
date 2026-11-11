"""
Set up the v2 analysis directory (revision of 2026-11-05; REVISION_PLAN_2026-11-05_post_desk_rejection.md).

v2 differs from the submitted analysis (v1, commit 55095ed, results/) in three pre-stated ways:
1. BUT QDB primary cohort follows the official protocol (Smital et al., Sci Data 2026): the 59 windows of record
   105001 inside the excluded 13-h stretch of missing signal are removed before training, leaving 988 windows
   (15 subjects). Fold assignment of every remaining window is unchanged.
2. The trained models use the 11 informative SQIs (MODEL_INPUT_FEATURE_NAMES); the constant window duration and
   sampling rate are dropped.
3. CinC 2011 Set A is evaluated on all 998 labelled records (lead II).

This script only prepares inputs; it fits nothing. It writes into $ECG_RESULTS_DIR (default results/v2_2026-11):
- manifest_butqdb_1000_cohort.csv   the 988-window cohort (file name kept so the stage scripts run unchanged)
- raw_d1_neural_cohort1000_predictions.csv   stored zero-shot d1-3B scores of the 988 windows (v1 run, unchanged;
  the d1-3B prompt does not change in v2, so the model is not re-run on BUT QDB)
- v2_setup_report.csv   counts before and after
Usage: ECG_RESULTS_DIR=results/v2_2026-11 python experiments/v2_setup.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from config import RESULTS_DIR  # noqa: E402

V1 = ROOT / "results"
N_EXCLUDED_EXPECTED = 59


def official_mask(manifest: pd.DataFrame):
    """Same rule as butqdb_official_split.official_mask (imported lazily: that module pulls in torch models)."""
    from butqdb_official_split import official_mask as _mask
    return _mask(manifest)


def main():
    if RESULTS_DIR == V1.resolve():
        raise SystemExit("Set ECG_RESULTS_DIR to the v2 directory; refusing to write into results/ (v1).")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    v1 = pd.read_csv(V1 / "manifest_butqdb_1000_cohort.csv")
    keep = official_mask(v1)
    dropped = v1[~keep]
    if len(dropped) != N_EXCLUDED_EXPECTED or set(dropped.record_name.str[:6]) != {"105001"}:
        raise ValueError(f"expected {N_EXCLUDED_EXPECTED} excluded windows of 105001, got {len(dropped)} "
                         f"from {sorted(set(dropped.record_name))}")
    v2 = v1[keep].reset_index(drop=True)
    if v2.segment_id.duplicated().any():
        raise ValueError("duplicate segment ids")
    v2.to_csv(RESULTS_DIR / "manifest_butqdb_1000_cohort.csv", index=False)

    d1 = pd.read_csv(V1 / "raw_d1_neural_cohort1000_predictions.csv")
    d1v2 = d1[d1.segment_id.isin(v2.segment_id)]
    if len(d1v2) != len(v2):
        raise ValueError(f"d1 predictions cover {len(d1v2)} of {len(v2)} v2 windows")
    d1v2.to_csv(RESULTS_DIR / "raw_d1_neural_cohort1000_predictions.csv", index=False)

    rows = []
    for name, df in [("v1", v1), ("v2", v2), ("excluded", dropped)]:
        rows.append({"set": name, "windows": len(df), "unusable": int(df.label.sum()),
                     "usable": int((df.label == 0).sum()), "subjects": df.subject_id.nunique(),
                     "windows_by_fold": dict(df.fold_id.value_counts().sort_index())})
    rep = pd.DataFrame(rows)
    rep.to_csv(RESULTS_DIR / "v2_setup_report.csv", index=False)
    print(rep.to_string(index=False))
    print(f"[done] v2 inputs in {RESULTS_DIR}")


if __name__ == "__main__":
    if "ECG_RESULTS_DIR" not in os.environ:
        raise SystemExit("Set ECG_RESULTS_DIR (e.g. results/v2_2026-11).")
    main()
