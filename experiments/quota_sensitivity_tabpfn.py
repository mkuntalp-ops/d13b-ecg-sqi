"""
TabPFN part of the quota sensitivity analysis (ANALYSIS_PLAN_2026-11-16_quota_sensitivity.md).

BUT QDB 5-fold subject cross-validation and BUT-only scoring of CinC 2011 lead II and CinC 2017, with the functions of
experiments/tabpfn_runner.py. NSTDB and the leave-one-dataset-out splits of that script are not in scope and are left out.
Runs in the TabPFN environment:
    ECG_RESULTS_DIR=results/v2_2026-11_quota_sensitivity /Volumes/KIOXIA/AI/tabpfn_env/bin/python experiments/quota_sensitivity_tabpfn.py
    ... --check-primary <scratch dir>   (ECG_RESULTS_DIR=results/v2_2026-11; recomputes the stored primary predictions,
                                         writes only into the scratch directory)
Output: $ECG_RESULTS_DIR/tabpfn_predictions.csv (setting, dataset, segment_id, label, prob_unusable)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabpfn_runner as tr  # noqa: E402

PRIMARY = (tr.ROOT / "results" / "v2_2026-11").resolve()


def predictions() -> pd.DataFrame:
    but, c11, c17, _nst = tr.sources()
    oof = np.zeros(len(but))
    for k in range(5):
        a, b = but.fold_id.values != k, but.fold_id.values == k
        oof[b] = tr.model().fit(tr.X(but[a]), but.label.values[a]).predict_proba(tr.X(but[b]))[:, 1]
        print(f"[tabpfn] CV fold {k} done", flush=True)
    parts = [tr.rows("butqdb_cv", but, oof)]
    full = tr.model().fit(tr.X(but), but.label.values)
    for df in [c11, c17]:
        parts.append(tr.rows("but_only_external", df, full.predict_proba(tr.X(df))[:, 1]))
        print(f"[tabpfn] external {df.dataset.iloc[0]} done", flush=True)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    if "ECG_RESULTS_DIR" not in os.environ:
        raise SystemExit("set ECG_RESULTS_DIR")
    if "--check-primary" in sys.argv:
        if tr.RESULTS != PRIMARY:
            raise SystemExit("--check-primary needs ECG_RESULTS_DIR=results/v2_2026-11")
        scratch = Path(sys.argv[sys.argv.index("--check-primary") + 1])
        scratch.mkdir(parents=True, exist_ok=True)
        P = predictions()
        P.to_csv(scratch / "tabpfn_primary_recomputed.csv", index=False)
        ref = pd.read_csv(tr.OUT)
        ref = ref[ref.setting.isin(["butqdb_cv"]) | ((ref.setting == "but_only_external") & ref.dataset.isin(["CinC2011", "CinC2017"]))]
        ref = ref.set_index(["setting", "dataset", "segment_id"]).prob_unusable
        d = float(np.abs(P.set_index(["setting", "dataset", "segment_id"]).prob_unusable.loc[ref.index].values - ref.values).max())
        print(f"[check-primary] TabPFN: {len(ref)} predictions, max |recomputed - stored| = {d:.3e}")
        sys.exit(0 if d < 1e-6 else 1)
    if tr.RESULTS in (PRIMARY, (tr.ROOT / "results").resolve()):
        raise SystemExit("refusing to write into a primary results directory")
    if tr.OUT.exists():
        raise SystemExit(f"{tr.OUT} exists; not overwriting")
    predictions().to_csv(tr.OUT, index=False)
    print(f"[Saved] {tr.OUT}")
