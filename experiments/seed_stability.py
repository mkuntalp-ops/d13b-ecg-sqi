"""
Seed stability of the CNN and multimodal models (seeds 1-4; seed 42 is the reported run).

For each seed:
  (a) 5-fold subject-independent CV on BUT QDB, pooled out-of-fold predictions;
  (b) zero-shot transfer to CinC 2011 Set A after training on the full 1,047-window cohort.
Thresholds are fixed at 0.5. The models read their seed from their module-level RANDOM_SEED,
so setting it per run reproduces the seeded construction without editing the model code.

Output: results/seed_stability_2026-10-12.csv (one row per seed, setting and model).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.sqi_extractor import SQI_FEATURE_NAMES
import models.cnn1d as cnn1d_mod
import models.cnn2d as cnn2d_mod
import models.multimodal_fusion as mm_mod
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier

SEEDS = [1, 2, 3, 4]
RESULTS = PROJECT_ROOT / "results"
COHORT = RESULTS / "manifest_butqdb_1000_cohort.csv"
CINC = PROJECT_ROOT / "cache" / "datasets" / "cinc2011" / "cinc2011_segments_lead1_max150.csv"
OUT = RESULTS / "seed_stability_2026-10-12.csv"
MODELS = ["1D_CNN", "2D_CNN", "Multimodal"]


def set_seed(seed: int) -> None:
    for mod in (cnn1d_mod, cnn2d_mod, mm_mod):
        mod.RANDOM_SEED = seed


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    fr = fp / (tn + fp)
    return {
        "auroc": roc_auc_score(y, p),
        "ap": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "sens_at_0.5": sens,
        "fr_at_0.5": fr,
        "bal_acc_at_0.5": (sens + (1 - fr)) / 2,
    }


def run_cv(seed, df, sigs, imgs, X):
    y = df["label"].values
    fold = df["fold_id"].values
    oof = {m: np.zeros(len(df)) for m in MODELS}
    for k in range(5):
        tr_idx = np.where(fold != k)[0]
        te_idx = np.where(fold == k)[0]
        set_seed(seed)
        c1 = ECG1DCNNClassifier(epochs=25, lr=1e-3)
        c1.fit([sigs[i] for i in tr_idx], y[tr_idx])
        oof["1D_CNN"][te_idx] = c1.predict_proba([sigs[i] for i in te_idx])

        set_seed(seed)
        c2 = ECG2DCNNClassifier(epochs=25, lr=1e-3)
        c2.fit([imgs[i] for i in tr_idx], y[tr_idx])
        oof["2D_CNN"][te_idx] = c2.predict_proba([imgs[i] for i in te_idx])

        set_seed(seed)
        cm = SinglePassMultimodalClassifier(epochs=25, lr=1e-3)
        cm.fit([imgs[i] for i in tr_idx], X[tr_idx], y[tr_idx])
        oof["Multimodal"][te_idx] = cm.predict_proba([imgs[i] for i in te_idx], X[te_idx])
        print(f"[seed {seed}] fold {k} done", flush=True)
    return oof


def run_transfer(seed, df, sigs, imgs, X, cinc_df):
    y = df["label"].values
    cinc_sigs = [np.load(p) for p in cinc_df["signal_path"]]
    cinc_imgs = cinc_df["image_path"].tolist()
    X_cinc = np.nan_to_num(cinc_df[SQI_FEATURE_NAMES].values.astype(float))
    preds = {}
    set_seed(seed)
    c1 = ECG1DCNNClassifier(epochs=25, lr=1e-3)
    c1.fit(sigs, y)
    preds["1D_CNN"] = c1.predict_proba(cinc_sigs)
    set_seed(seed)
    c2 = ECG2DCNNClassifier(epochs=25, lr=1e-3)
    c2.fit(imgs, y)
    preds["2D_CNN"] = c2.predict_proba(cinc_imgs)
    set_seed(seed)
    cm = SinglePassMultimodalClassifier(epochs=25, lr=1e-3)
    cm.fit(imgs, X, y)
    preds["Multimodal"] = cm.predict_proba(cinc_imgs, X_cinc)
    print(f"[seed {seed}] CinC transfer done", flush=True)
    return preds


def main():
    df = pd.read_csv(COHORT)
    cinc_df = pd.read_csv(CINC)
    X = np.nan_to_num(df[SQI_FEATURE_NAMES].values.astype(float))
    sigs = [np.load(p) for p in df["signal_path"]]
    imgs = df["image_path"].tolist()
    rows = []
    for seed in SEEDS:
        oof = run_cv(seed, df, sigs, imgs, X)
        for m in MODELS:
            rows.append({"seed": seed, "setting": "BUT_QDB_CV_pooled", "model": m,
                         **metrics(df["label"].values, oof[m])})
        tr = run_transfer(seed, df, sigs, imgs, X, cinc_df)
        for m in MODELS:
            rows.append({"seed": seed, "setting": "CinC_transfer", "model": m,
                         **metrics(cinc_df["label"].values, tr[m])})
        pd.DataFrame(rows).to_csv(OUT, index=False)  # save after each seed
        print(f"[saved] seed {seed} -> {OUT}", flush=True)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
