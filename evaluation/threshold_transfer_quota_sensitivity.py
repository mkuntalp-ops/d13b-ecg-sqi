"""
Threshold transfer repeated with the models of the quota sensitivity cohort (round V21, 2026-11-20; review finding
PAPER-002). Post hoc; uses saved predictions only, no model is refitted.

The transfer step of experiments/nested_thresholds.py (thresholds chosen on the pooled BUT QDB out-of-fold scores,
applied frozen to CinC 2011 lead II and to the CinC 2017 recording maximum; same rules and tie-breaking) is run on the
score tables of evaluation/paired_comparisons.families() for (a) the primary cohort, which must reproduce the stored
<results>/threshold_transfer_summary.csv exactly, and (b) the 1,041-window sensitivity cohort
(<results>_quota_sensitivity), whose nine fitted models were refitted with 53 added unusable windows of subject 105.

Output: <results>/threshold_transfer_quota_sensitivity.csv (cohort column: primary, sensitivity)
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/threshold_transfer_quota_sensitivity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "experiments"))
import paired_comparisons as pc  # noqa: E402
from nested_thresholds import RULES, pick, realised  # noqa: E402

PRIMARY = pc.RESULTS
SENS = PRIMARY.parent / f"{PRIMARY.name}_quota_sensitivity"
OUT = PRIMARY / "threshold_transfer_quota_sensitivity.csv"


def transfer(results_dir: Path) -> pd.DataFrame:
    pc.RESULTS = results_dir  # families() reads the module-level results directory
    fam = pc.families()
    but = fam["BUT_QDB (5-fold subject CV)"][1]
    targets = {"CinC2011 all 998 records (lead II)": fam["CinC2011 lead II (998 records)"][1],
               "CinC2017 all (record max)": fam["CinC2017 (8,512 recordings)"][1]}
    y_b = but["y"].values.astype(int)
    rows = []
    for m in [c for c in but.columns if c not in ("unit", "y", "cluster")]:
        for rule in RULES:
            t = pick(y_b, but[m].values.astype(float), rule)
            rows.append({"model": m, "rule": rule, "threshold": t, "dataset": "BUT_QDB pooled OOF (selection set)",
                         **realised(y_b, but[m].values.astype(float), t)})
            for name, df in targets.items():
                if m in df.columns:
                    rows.append({"model": m, "rule": rule, "threshold": t, "dataset": name,
                                 **realised(df["y"].values.astype(int), df[m].values.astype(float), t)})
    return pd.DataFrame(rows)


def main():
    prim = transfer(PRIMARY)
    stored = pd.read_csv(PRIMARY / "threshold_transfer_summary.csv")
    if prim.shape != stored.shape or not np.allclose(prim[["threshold", "sens", "fpr"]], stored[["threshold", "sens", "fpr"]], atol=1e-6) \
            or not (prim[["model", "rule", "dataset"]].values == stored[["model", "rule", "dataset"]].values).all():
        raise ValueError("primary transfer does not reproduce threshold_transfer_summary.csv")
    sens = transfer(SENS)
    if len(sens[(sens.dataset == "BUT_QDB pooled OOF (selection set)")].query("rule == 'fixed_0.5'")) != 11:
        raise ValueError("expected 11 models in the sensitivity cohort")
    out = pd.concat([prim.assign(cohort="primary"), sens.assign(cohort="sensitivity")], ignore_index=True)
    out.to_csv(OUT, index=False)
    t1 = out[(out.rule == "target1_fpr5") & ~out.dataset.str.startswith("BUT")]
    for c, g in t1.groupby("cohort"):
        met = g[g.fpr <= 0.05]
        print(f"{c}: {len(met)} of {len(g)} Target 1 thresholds met FPR <= 5% externally")
        print(met[["model", "dataset", "fpr", "sens"]].round(3).to_string(index=False))
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
