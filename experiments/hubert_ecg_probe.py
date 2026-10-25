"""
HuBERT-ECG (base, self-supervised ECG foundation model) frozen-feature probe (Q1 revision, 2026-10-25).

Purpose: an ECG-specific foundation model as a comparator for the general vision-language model (d1-3B) and for
the task-specific networks, under the same probe protocol as experiments/d1_linear_probe.py.

Weights: Edoardo-Coppola/hubert-ecg-base (Hugging Face; CC BY-NC 4.0), downloaded with the user's approval to
$HUBERT_ECG_PATH (default /Volumes/KIOXIA/AI Models/hubert-ecg-base).

Input construction. The model takes 12 leads x 5 s at 500 Hz, flattened lead by lead, decimated by 5 (6,000 samples)
(authors' hubert_ecg/dataset.py). Our windows are single lead: the signal is placed in the lead II slot of a 12-lead
array whose other leads are zero. Each 10-s window gives two 5-s halves; the embedding is the mean of the last hidden
state over time, averaged over the two halves (768-d). Two preprocessing variants (analysis plan A4):
- "authors" (primary since 2026-10-25): the authors' hubert_ecg/utils.py ecg_preprocessing, re-implemented without
  biosppy: FIR band-pass 0.05-47 Hz (biosppy filter_signal: firwin with order int(0.3*fs) made odd = 151 taps,
  forward-backward filtfilt) on the 12-lead 10-s array, then min-max scaling of each lead to [-1, 1] over the 10 s.
  The authors do not specify how absent leads are handled; their scaling maps the zero leads to -1, and this is kept.
- "ours" (sensitivity; the version run first): 4th-order Butterworth 0.05-47 Hz (zero phase) and scaling of each
  5-s half of lead II by its maximum absolute value; absent leads stay 0. Outputs carry the suffix _ourpreproc.

Probe (identical to the d1 probe): standardization + L2 logistic regression, C from {1e-3 .. 1} by inner GroupKFold.
- BUT QDB: the five subject folds (out-of-fold predictions);
- transfer: fit on all BUT QDB windows, applied to CinC 2011 lead II (565 records) and all CinC 2017 windows
  (recording maximum);
- LODO: the three splits of experiments/lodo_multisource.py (same CinC 2017 training subset).

Outputs: cache/hubert_ecg/*.npy (embeddings), results/hubert_probe_predictions.csv, results/hubert_probe_summary.csv
Usage: python experiments/hubert_ecg_probe.py [--analysis-only] [--preproc=ours]
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from lodo_multisource import cinc17_training_subset, load_sources  # noqa: E402

MODEL_DIR = Path(os.environ.get("HUBERT_ECG_PATH", "/Volumes/KIOXIA/AI Models/hubert-ecg-base"))
PREPROC = "ours" if "--preproc=ours" in sys.argv else "authors"
SUFFIX = "_ourpreproc" if PREPROC == "ours" else ""
FEAT = ROOT / "cache" / ("hubert_ecg" if PREPROC == "ours" else "hubert_ecg_authors")
RESULTS = ROOT / "results"
PRED_OUT = RESULTS / f"hubert_probe_predictions{SUFFIX}.csv"
SUM_OUT = RESULTS / f"hubert_probe_summary{SUFFIX}.csv"
FS, HALF = 500, 2500
SOS = signal.butter(4, [0.05, 47.0], btype="bandpass", fs=FS, output="sos")
FIR = signal.firwin(151, [0.05 / (FS / 2), 47.0 / (FS / 2)], pass_zero=False)
C_GRID = {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]}


def load_model(device):
    sys.path.insert(0, str(MODEL_DIR))
    import hubert_ecg  # noqa: F401  (registers the custom config/model classes)
    from hubert_ecg import HuBERTECG
    return HuBERTECG.from_pretrained(str(MODEL_DIR)).to(device).eval()


def model_input_authors(x: np.ndarray) -> np.ndarray:
    """Authors' preprocessing on the zero-filled 12-lead array -> two flattened, decimated inputs of 6,000 samples."""
    twelve = np.zeros((12, 2 * HALF))
    twelve[1] = np.asarray(x, dtype=np.float64)[:2 * HALF]
    twelve = signal.filtfilt(FIR, [1.0], twelve, axis=1)
    lo, hi = twelve.min(axis=1, keepdims=True), twelve.max(axis=1, keepdims=True)
    twelve = 2 * (twelve - lo) / (hi - lo + 1e-8) - 1
    return np.stack([signal.decimate(twelve[:, s:s + HALF].reshape(-1), 5) for s in (0, HALF)]).astype(np.float32)


def model_input(x: np.ndarray) -> np.ndarray:
    """10-s single-lead window (500 Hz) -> two flattened, decimated 12-lead inputs of 6,000 samples."""
    if PREPROC == "authors":
        return model_input_authors(x)
    x = signal.sosfiltfilt(SOS, np.asarray(x, dtype=np.float64))
    halves = []
    for h in (x[:HALF], x[HALF:2 * HALF]):
        m = np.max(np.abs(h))
        lead = h / m if m > 0 else h
        twelve = np.zeros((12, HALF))
        twelve[1] = lead
        halves.append(signal.decimate(twelve.reshape(-1), 5))
    return np.stack(halves).astype(np.float32)


@torch.no_grad()
def embed(model, paths, device, batch=64):
    out = []
    for i in range(0, len(paths), batch):
        xs = np.concatenate([model_input(np.load(p)) for p in paths[i:i + batch]])  # (2B, 6000)
        h = model(torch.from_numpy(xs).to(device)).last_hidden_state.mean(dim=1).float().cpu().numpy()
        out.append(h.reshape(-1, 2, h.shape[-1]).mean(axis=1))
        if (i // batch) % 50 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return np.concatenate(out)


def extract():
    FEAT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(device)
    for name, df in load_sources().items():
        if (FEAT / f"{name}.npy").exists():
            continue
        E = embed(model, df["signal_path"].tolist(), device)
        np.save(FEAT / f"{name}.npy", E.astype(np.float32))
        df[["segment_id"]].to_csv(FEAT / f"{name}_ids.csv", index=False)
        print(f"[hubert] {name}: {E.shape}", flush=True)


def probe():
    return GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), C_GRID,
                        scoring="roc_auc", cv=GroupKFold(n_splits=4))


def analyse():
    src = load_sources()
    F = {k: np.load(FEAT / f"{k}.npy") for k in src}
    for k in src:
        assert (pd.read_csv(FEAT / f"{k}_ids.csv").segment_id.values == src[k].segment_id.values).all()
    rows, preds = [], []
    but = src["BUT_QDB"]
    oof = np.zeros(len(but))
    folds = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv").set_index("segment_id").loc[but.segment_id, "fold_id"].values
    for k in range(5):
        tr, te = folds != k, folds == k
        oof[te] = probe().fit(F["BUT_QDB"][tr], but.label.values[tr], groups=but.unit.values[tr]).predict_proba(F["BUT_QDB"][te])[:, 1]
    preds.append(but[["dataset", "segment_id", "unit", "record", "label"]].assign(setting="butqdb_cv", prob_unusable=oof))
    full = probe().fit(F["BUT_QDB"], but.label.values, groups=but.unit.values)
    for k in ["CinC2011", "CinC2017"]:
        preds.append(src[k][["dataset", "segment_id", "unit", "record", "label"]].assign(
            setting="but_only_external", prob_unusable=full.predict_proba(F[k])[:, 1]))
    c17_train = cinc17_training_subset(src["CinC2017"])
    idx17 = pd.Series(np.arange(len(src["CinC2017"])), index=src["CinC2017"].segment_id).loc[c17_train.segment_id].values
    pool = {"BUT_QDB": (F["BUT_QDB"], but), "CinC2011": (F["CinC2011"], src["CinC2011"]),
            "CinC2017": (F["CinC2017"][idx17], c17_train)}
    for test in ["CinC2017", "CinC2011", "BUT_QDB"]:
        tr = [k for k in pool if k != test]
        X = np.vstack([pool[k][0] for k in tr])
        meta = pd.concat([pool[k][1] for k in tr], ignore_index=True)
        m = probe().fit(X, meta.label.values, groups=(meta.dataset + "_" + meta.unit).values)
        preds.append(src[test][["dataset", "segment_id", "unit", "record", "label"]].assign(
            setting=f"lodo_test_{test}", prob_unusable=m.predict_proba(F[test])[:, 1]))
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(PRED_OUT, index=False)
    for (setting, ds), g in P.groupby(["setting", "dataset"]):
        if ds == "BUT_QDB":
            y, p = g.label.values, g.prob_unusable.values
        else:
            r = g.groupby("record").agg(y=("label", "first"), p=("prob_unusable", "max"))
            y, p = r.y.values, r.p.values
        rows.append({"setting": setting, "dataset": ds, "n_units": len(y), "auroc": roc_auc_score(y, p),
                     "ap": average_precision_score(y, p)})
    out = pd.DataFrame(rows)
    out.to_csv(SUM_OUT, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    if "--analysis-only" not in sys.argv:
        extract()
    analyse()
