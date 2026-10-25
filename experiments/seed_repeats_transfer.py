"""
Seed repetition of single-source transfer and LODO (analysis plan A1, 2026-10-25).

For training seeds 1-4 (seed 42 is the stored main run), the models of experiments/lodo_multisource.py are retrained
with the module-level RANDOM_SEED set as in experiments/cinc_draws_by_seed.py:
- BUT-only: trained on the full BUT QDB cohort, scored on CinC 2011 (565 records, lead II) and all CinC 2017 windows;
- LODO: the three leave-one-dataset-out splits, with the same CinC 2017 training subset (seed 42).
Logistic regression and LightGBM use a fixed random_state, so they are identical across seeds; they are retrained
anyway (seconds) and stored for completeness, but only the SQI MLP, 1D CNN, 2D CNN and multimodal net are
seed-varied. Hyperparameters are unchanged.

Resumable: each (seed, setting) block is appended to results/seed_repeats_predictions.csv and skipped on restart.
--analysis-only writes results/seed_repeats_summary.csv: AUROC per seed (42 from the stored predictions), the
seed-matched delta (LODO minus BUT-only) on CinC 2011 and CinC 2017, and the plan's consistency rule (same sign in
all five seeds and |median delta| >= 0.02).

Usage: python experiments/seed_repeats_transfer.py [--analysis-only]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import models.cnn1d as cnn1d_mod  # noqa: E402
import models.cnn2d as cnn2d_mod  # noqa: E402
import models.multimodal_fusion as mm_mod  # noqa: E402
from lodo_multisource import (PRED_OUT as LODO_PRED, TRAINED, cinc17_training_subset, load_sources,  # noqa: E402
                              single_source_predictions, train_and_score)

RESULTS = PROJECT_ROOT / "results"
OUT = RESULTS / "seed_repeats_predictions.csv"
SUM_OUT = RESULTS / "seed_repeats_summary.csv"
NEW_SEEDS = [1, 2, 3, 4]
SEED_VARIED = ["Independent_SQI_MLP", "1D_CNN_RawECG", "2D_CNN_Image", "Multimodal_Fusion"]


def set_seed(seed: int) -> None:
    for mod in (cnn1d_mod, cnn2d_mod, mm_mod):
        mod.RANDOM_SEED = seed


def done_blocks() -> set:
    if not OUT.exists():
        return set()
    d = pd.read_csv(OUT, usecols=["seed", "setting"]).drop_duplicates()
    return set(zip(d.seed, d.setting))


def append(block: pd.DataFrame) -> None:
    block.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


def run():
    src = load_sources()
    c17_train = cinc17_training_subset(src["CinC2017"])
    done = done_blocks()
    for seed in NEW_SEEDS:
        set_seed(seed)
        if (seed, "but_only") not in done:
            ext = pd.concat([src["CinC2011"], src["CinC2017"]], ignore_index=True)
            print(f"[seed {seed}] BUT-only -> CinC 2011 + CinC 2017 ({len(ext)} windows)", flush=True)
            append(train_and_score(src["BUT_QDB"], ext).assign(seed=seed, setting="but_only"))
        for test in ["CinC2017", "CinC2011", "BUT_QDB"]:
            setting = f"lodo_test_{test}"
            if (seed, setting) in done:
                continue
            train = pd.concat([c17_train if k == "CinC2017" else v for k, v in src.items() if k != test],
                              ignore_index=True)
            print(f"[seed {seed}] LODO test={test}: train {len(train)} windows", flush=True)
            append(train_and_score(train, src[test]).assign(seed=seed, setting=setting))


def unit_auroc(df: pd.DataFrame, col: str) -> float:
    if df.dataset.iloc[0] == "BUT_QDB":
        return roc_auc_score(df.label, df[col])
    r = df.groupby("record").agg(y=("label", "first"), p=(col, "max"))
    return roc_auc_score(r.y, r.p)


def analyse():
    new = pd.read_csv(OUT, low_memory=False)
    lodo42 = pd.read_csv(LODO_PRED, low_memory=False)
    single42 = single_source_predictions()
    src = load_sources()
    rows = []
    for test in ["CinC2011", "CinC2017", "BUT_QDB"]:
        base = src[test][["dataset", "segment_id", "record", "label"]]
        for m in TRAINED:
            l42 = base.merge(lodo42[lodo42.dataset == test][["segment_id", m]], on="segment_id")
            rec = {"test": test, "model": m, "seed": 42, "auroc_lodo": unit_auroc(l42, m)}
            if test != "BUT_QDB":
                s42 = base.assign(**{m: single42[test].loc[base.segment_id, m].values})
                rec["auroc_single"] = unit_auroc(s42, m)
            rows.append(rec)
            for seed in NEW_SEEDS:
                g = new[(new.seed == seed) & (new.setting == f"lodo_test_{test}")]
                if g.empty:
                    continue
                rec = {"test": test, "model": m, "seed": seed, "auroc_lodo": unit_auroc(g, m)}
                if test != "BUT_QDB":
                    s = new[(new.seed == seed) & (new.setting == "but_only") & (new.dataset == test)]
                    rec["auroc_single"] = unit_auroc(s, m)
                rows.append(rec)
    out = pd.DataFrame(rows)
    out["delta"] = out.auroc_lodo - out.auroc_single
    out.to_csv(SUM_OUT, index=False)
    agg = []
    for (test, m), g in out.groupby(["test", "model"], sort=False):
        d = g.delta.dropna()
        rec = {"test": test, "model": m, "n_seeds": len(g), "lodo_median": g.auroc_lodo.median(),
               "lodo_min": g.auroc_lodo.min(), "lodo_max": g.auroc_lodo.max(), "seed_varied": m in SEED_VARIED}
        if len(d):
            rec.update(single_median=g.auroc_single.median(), delta_median=d.median(), delta_min=d.min(),
                       delta_max=d.max(), consistent=bool(len(d) == 5 and (np.all(d > 0) or np.all(d < 0))
                                                          and abs(d.median()) >= 0.02))
        agg.append(rec)
    agg = pd.DataFrame(agg)
    agg.to_csv(SUM_OUT.with_name("seed_repeats_consistency.csv"), index=False)
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    if "--analysis-only" not in sys.argv:
        run()
    analyse()
