"""
Training-seed stability of the class-2 relabelling analysis (seeds 1-4; seed 42 is in
results/butqdb_class2_sensitivity.csv and its OOF file).

For each seed the supervised models are retrained with the class-2-as-unusable labels using the same
folds and settings as butqdb_class2_sensitivity.py. The rule heuristic and d1-3B do not depend on the
training seed and are not repeated. LR and LightGBM use random_state=42 in every run.

Outputs:
- results/butqdb_class2_by_seed.csv: seed, model, AUROC, AP, Brier, sensitivity and FR at 0.5
  (class-2-as-unusable labels), for seeds 1-4 and the seed-42 run
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import models.cnn1d as cnn1d_mod
import models.cnn2d as cnn2d_mod
import models.multimodal_fusion as mm_mod
from butqdb_class2_sensitivity import MANIFEST, RESULTS, TRAINED, metrics, train_oof
from features.sqi_extractor import SQI_FEATURE_NAMES

SEEDS = [1, 2, 3, 4]
OUT = RESULTS / "butqdb_class2_by_seed.csv"


def set_seed(seed: int) -> None:
    for mod in (cnn1d_mod, cnn2d_mod, mm_mod):
        mod.RANDOM_SEED = seed


def main():
    df = pd.read_csv(MANIFEST)
    main_run = pd.read_csv(RESULTS / "cohort1000_consolidated_oof_predictions.csv").set_index("segment_id")
    df = df.set_index("segment_id", drop=False).loc[main_run.index].reset_index(drop=True)
    y2 = (df["original_quality_class"].values >= 2).astype(int)

    rows = []
    for seed in SEEDS:
        set_seed(seed)
        oof = train_oof(df, y2)
        for m in TRAINED:
            rows.append({"seed": seed, "model": m, **metrics(df, y2, oof[m])})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"[seed {seed}] done", flush=True)

    # seed 42 from the existing class-2 run (same folds, settings and seed)
    prev = pd.read_csv(RESULTS / "butqdb_class2_sensitivity.csv")
    prev = prev[(prev.scheme == "class2_unusable") & (prev.model.isin(TRAINED))]
    for _, r in prev.iterrows():
        rows.append({"seed": 42, "model": r.model, "auroc": r.auroc, "ap": r.ap,
                     "brier": np.nan, "sens_at_0.5": r["sens_at_0.5"], "fr_at_0.5": r["fr_at_0.5"],
                     "n_unusable": r.n_unusable, "auroc_ci_low": r.auroc_ci_low, "auroc_ci_high": r.auroc_ci_high})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("\nclass-2 scheme, AUROC by seed (mean ± SD over seeds 1-4, with seed 42 shown):")
    for m in TRAINED:
        a = out[(out.model == m) & (out.seed != 42)]["auroc"].values
        a42 = out[(out.model == m) & (out.seed == 42)]["auroc"].values
        print(f"  {m:26s} seed42 {a42[0]:.3f} | seeds 1-4 {a.mean():.3f} ± {a.std(ddof=1):.3f} ({a.min():.3f}–{a.max():.3f})")
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
