"""
Interaction between training seed and CinC subset: each seed retrains the cohort models, then all five
balanced CinC subsets (and the reported first-N subset) are scored with that seed's models.

Seeds 42 (reported) and 1-4 are used. For each seed the CNN and multimodal models use the module-level
RANDOM_SEED, set per run (the constructors seed torch from it). LR and LightGBM use random_state=42 in
every run, so they are not seed-varied; the MLP is seeded like the CNNs.

Outputs:
- results/cinc_draws_by_seed_summary.csv: seed, draw, model, AUROC, AP, sensitivity and FR at 0.5
- printed variance decomposition: variation across seeds within a subset vs across subsets within a seed
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
from cinc_random_draws import CINC_DIR, MANIFEST_OUT, fit_models, metrics
from features.sqi_extractor import SQI_FEATURE_NAMES

RESULTS = PROJECT_ROOT / "results"
SEEDS = [42, 1, 2, 3, 4]
OUT = RESULTS / "cinc_draws_by_seed_summary.csv"
TRAINED = ["Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG", "2D_CNN_Image", "Multimodal_Fusion"]


def set_seed(seed: int) -> None:
    for mod in (cnn1d_mod, cnn2d_mod, mm_mod):
        mod.RANDOM_SEED = seed


def main():
    records = pd.read_csv(CINC_DIR / "cinc2011_random_draws_records.csv")
    membership = pd.read_csv(MANIFEST_OUT)
    rows = []
    for seed in SEEDS:
        set_seed(seed)
        models = fit_models()  # trains LR, LightGBM, MLP, 1D CNN, 2D CNN, multimodal with the current seed
        lr, lgb, mlp, c1, c2, mm = models
        X = np.nan_to_num(records[SQI_FEATURE_NAMES].values.astype(float))
        imgs = records["image_path"].tolist()
        sigs = [np.load(p) for p in records["signal_path"]]
        pred = pd.DataFrame({"record_name": records["record_name"].values})
        pred["Logistic_Regression_SQI"] = lr.predict_proba(X)
        pred["LightGBM_SQI"] = lgb.predict_proba(X)
        pred["Independent_SQI_MLP"] = mlp.predict_proba(X)
        pred["1D_CNN_RawECG"] = c1.predict_proba(sigs)
        pred["2D_CNN_Image"] = c2.predict_proba(imgs)
        pred["Multimodal_Fusion"] = mm.predict_proba(imgs, X)
        for draw in membership["draw"].unique():
            sub = membership[membership.draw == draw].merge(pred, on="record_name", how="left")
            y = sub["label"].values.astype(int)
            for m in TRAINED:
                rows.append({"seed": seed, "draw": draw, "model": m, **metrics(y, sub[m].values.astype(float))})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"[seed {seed}] done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    rnd = out[out.draw.str.startswith("random_seed")]
    print("\nAUROC by model: mean SD across seeds (within a subset), and SD across subsets (within a seed):")
    for m in TRAINED:
        piv = rnd[rnd.model == m].pivot_table(index="seed", columns="draw", values="auroc")
        across_seeds = piv.std(axis=0, ddof=1).mean()
        across_draws = piv.std(axis=1, ddof=1).mean()
        print(f"  {m:26s} seed-SD {across_seeds:.3f} | subset-SD {across_draws:.3f}")
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
