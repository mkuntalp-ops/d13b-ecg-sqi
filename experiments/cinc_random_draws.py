"""
Random-draw sensitivity for the CinC 2011 Set A transfer benchmark.

The reported transfer set is the first 75 acceptable and first 75 unacceptable records of the sorted
PhysioNet lists. Here five balanced 75 + 75 subsets are drawn at random (seeds 42, 1, 2, 3, 4) from the
full acceptable (773) and unacceptable (225) lists. The models are trained once on the BUT QDB cohort
with seed 42, exactly as in run_cinc_transfer_benchmark.py, and then scored frozen on each draw.
The original first-N set is scored as well, as a check against Table 2 (seed 42).

Outputs:
- results/cinc_random_draws_predictions.csv  (per record: scores of every model)
- results/cinc_random_draws_summary.csv      (per draw and model: AUROC, AP, Brier, sensitivity/FR at 0.5)
- cache/datasets/cinc2011/cinc2011_random_draws_manifest.csv (draw membership of every record)
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.ecg_rendering import render_ecg_image
from data.units import to_millivolts
from features.sqi_extractor import SQI_FEATURE_NAMES, extract_sqi_features
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.d1_real_neural_model import RealD1DecisionModel
from models.multimodal_fusion import IndependentSQIClassifier, SinglePassMultimodalClassifier
from models.rule_heuristic_baseline import RuleBasedSQIHeuristic
from models.tabular_baselines import SQIGradientBoosting, SQILogisticRegression

CINC_DIR = PROJECT_ROOT / "cache" / "datasets" / "cinc2011"
SIG_DIR = CINC_DIR / "signals"
IMG_DIR = CINC_DIR / "images"
RESULTS = PROJECT_ROOT / "results"
COHORT = RESULTS / "manifest_butqdb_1000_cohort.csv"
DRAW_SEEDS = [42, 1, 2, 3, 4]
N_PER_CLASS = 75
PREDS_OUT = RESULTS / "cinc_random_draws_predictions.csv"
SUMMARY_OUT = RESULTS / "cinc_random_draws_summary.csv"
MANIFEST_OUT = CINC_DIR / "cinc2011_random_draws_manifest.csv"
D1_EXISTING = RESULTS / "raw_d1_neural_cinc2011_predictions.csv"
MODELS = ["Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG",
          "2D_CNN_Image", "Multimodal_Fusion", "Rule_Based_SQI_Heuristic", "LiquidAI_d1_3B_Neural"]


def read_lists():
    acc = [l.strip() for l in open(CINC_DIR / "RECORDS-acceptable.txt") if l.strip()]
    unacc = [l.strip() for l in open(CINC_DIR / "RECORDS-unacceptable.txt") if l.strip()]
    return acc, unacc


def draws(acc, unacc):
    out = {"first_N_original": (acc[:N_PER_CLASS], unacc[:N_PER_CLASS])}
    for s in DRAW_SEEDS:
        rng = np.random.default_rng(s)
        out[f"random_seed{s}"] = (list(rng.choice(acc, N_PER_CLASS, replace=False)),
                                  list(rng.choice(unacc, N_PER_CLASS, replace=False)))
    return out


def extract_record(rec_id: str):
    """Lead II (index 1), same conventions as data/cinc2011_dataset.py; cached if already extracted."""
    seg_id = f"cinc2011_{rec_id}_lead1"
    sig_path, img_path = SIG_DIR / f"{seg_id}.npy", IMG_DIR / f"{seg_id}.png"
    if sig_path.exists() and img_path.exists():
        raw = np.load(sig_path)
        fs = 500.0
        unit = "mV"
    else:
        rec = wfdb.rdrecord(rec_id, pn_dir="challenge-2011/1.0.0/set-a")
        raw = rec.p_signal[:, 1].astype(np.float32)
        fs = float(rec.fs)
        unit = rec.units[1]
        np.save(sig_path, raw)
        render_ecg_image(raw, fs, dpi=100).save(img_path)
    sqis = extract_sqi_features(to_millivolts(raw, unit), fs)
    return {"record_name": rec_id, "segment_id": seg_id, "signal_path": str(sig_path), "image_path": str(img_path),
            "signal_unit": unit, **sqis}


def fit_models():
    df = pd.read_csv(COHORT)
    X = np.nan_to_num(df[SQI_FEATURE_NAMES].values.astype(float))
    y = df["label"].values
    sigs = [np.load(p) for p in df["signal_path"]]
    imgs = df["image_path"].tolist()
    lr = SQILogisticRegression().fit(X, y)
    lgb = SQIGradientBoosting().fit(X, y)
    mlp = IndependentSQIClassifier(epochs=35).fit(X, y)
    c1 = ECG1DCNNClassifier(epochs=25, batch_size=32).fit(sigs, y)
    c2 = ECG2DCNNClassifier(epochs=25, batch_size=32).fit(imgs, y)
    mm = SinglePassMultimodalClassifier(epochs=25, batch_size=32).fit(imgs, X, y)
    print("[fit] cohort models trained (seed 42)", flush=True)
    return lr, lgb, mlp, c1, c2, mm


def score(records: pd.DataFrame, models) -> pd.DataFrame:
    lr, lgb, mlp, c1, c2, mm = models
    X = np.nan_to_num(records[SQI_FEATURE_NAMES].values.astype(float))
    sigs = [np.load(p) for p in records["signal_path"]]
    imgs = records["image_path"].tolist()
    out = pd.DataFrame({"segment_id": records["segment_id"].values, "record_name": records["record_name"].values})
    out["Logistic_Regression_SQI"] = lr.predict_proba(X)
    out["LightGBM_SQI"] = lgb.predict_proba(X)
    out["Independent_SQI_MLP"] = mlp.predict_proba(X)
    out["1D_CNN_RawECG"] = c1.predict_proba(sigs)
    out["2D_CNN_Image"] = c2.predict_proba(imgs)
    out["Multimodal_Fusion"] = mm.predict_proba(imgs, X)
    rule = RuleBasedSQIHeuristic()
    rule_p, d1_p = [], []
    d1 = RealD1DecisionModel(device="mps")
    existing = pd.read_csv(D1_EXISTING).set_index("segment_id")["prob_unusable"].to_dict() if D1_EXISTING.exists() else {}
    for (_, row), sqi in zip(records.iterrows(), X):
        sqi_dict = dict(zip(SQI_FEATURE_NAMES, sqi))
        with Image.open(row["image_path"]) as img:
            rule_p.append(rule.decide_single(img, sqi_dict, modality="multimodal")["prob_unusable"])
            if row["segment_id"] in existing:
                d1_p.append(float(existing[row["segment_id"]]))
            else:
                d1_p.append(d1.decide_single(img, sqi_dict, modality="multimodal", segment_id=row["segment_id"])["prob_unusable"])
    out["Rule_Based_SQI_Heuristic"] = rule_p
    out["LiquidAI_d1_3B_Neural"] = d1_p
    return out


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1]).ravel()
    return {"auroc": roc_auc_score(y, p), "ap": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "sens_at_0.5": tp / (tp + fn), "fr_at_0.5": fp / (tn + fp)}


def main():
    acc, unacc = read_lists()
    print(f"[lists] acceptable {len(acc)}, unacceptable {len(unacc)}", flush=True)
    sets = draws(acc, unacc)
    all_ids = sorted({r for a, u in sets.values() for r in a + u})
    print(f"[draws] {len(sets)} sets, {len(all_ids)} unique records to score", flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(lambda r: _safe(r), all_ids))
    records = pd.DataFrame([r for r in rows if r is not None])
    print(f"[extract] {len(records)} of {len(all_ids)} records available", flush=True)
    records.to_csv(CINC_DIR / "cinc2011_random_draws_records.csv", index=False)

    membership = []
    for name, (a, u) in sets.items():
        for rec, lab in [(r, 0) for r in a] + [(r, 1) for r in u]:
            membership.append({"draw": name, "record_name": rec, "label": lab})
    membership = pd.DataFrame(membership)
    membership.to_csv(MANIFEST_OUT, index=False)

    models = fit_models()
    preds = score(records, models)
    preds.to_csv(PREDS_OUT, index=False)

    summary = []
    for name in sets:
        mem = membership[membership.draw == name]
        sub = mem.merge(preds, on="record_name", how="left")
        y = sub["label"].values.astype(int)
        for m in MODELS:
            summary.append({"draw": name, "model": m, **metrics(y, sub[m].values.astype(float))})
    out = pd.DataFrame(summary)
    out.to_csv(SUMMARY_OUT, index=False)
    print(out.round(3).to_string(index=False))
    print(f"[Saved] {SUMMARY_OUT}")


def _safe(rec):
    try:
        return extract_record(rec)
    except Exception as e:  # missing or unreadable PhysioNet record
        print(f"[skip] {rec}: {e}", flush=True)
        return None


if __name__ == "__main__":
    main()
