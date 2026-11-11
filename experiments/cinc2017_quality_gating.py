"""
Effect of quality gating on rhythm classification, CinC 2017 (analysis B of ANALYSIS_PLAN_2026-11-09_additional_analyses.md).

A 3-class (N, A, O) 1D CNN is trained on the existing 10-s windows of the 8,234 non-noisy recordings (5-fold CV stratified by
rhythm, folds by recording, not tuned). Quality scores are the recording-maximum scores of the primary CinC 2017 analysis
(models fitted on BUT QDB only; they never see rhythm labels). "Too noisy" recordings have no rhythm label and cannot enter.

Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python experiments/cinc2017_quality_gating.py [--analysis-only]
Output: $ECG_RESULTS_DIR/cinc2017_gating_{oof_predictions,summary,differences}.csv (never overwritten)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from config import RESULTS_DIR  # noqa: E402
from evaluation.paired_comparisons import families  # noqa: E402
from models.cnn1d import RawECG1DCNN  # noqa: E402

WINDOWS = ROOT / "cache" / "datasets" / "cinc2017" / "cinc2017_windows.csv"
CLASSES = ["N", "A", "O"]
SEED, EPOCHS, BATCH, N_RANDOM, N_BOOT, BOOT_RANDOM = 42, 25, 64, 1000, 2000, 20
FRACTIONS = [0.05, 0.10, 0.20]
PROBE, CNN2, HUB = "d1_image_probe", "2D_CNN_Image", "HuBERT_ECG_probe"
OOF, SUMMARY, DIFFS = (RESULTS_DIR / f"cinc2017_gating_{k}.csv" for k in ["oof_predictions", "summary", "differences"])


def standardise(x: np.ndarray) -> np.ndarray:
    mean, std = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((x - mean) / std).astype(np.float32)


def train_fold(x_tr, y_tr, x_te, device) -> np.ndarray:
    net = RawECG1DCNN()
    net.classifier[-1] = nn.Linear(32, len(CLASSES))
    net = net.to(device)
    counts = np.bincount(y_tr, minlength=len(CLASSES))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(y_tr) / (len(CLASSES) * counts), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    x_t, y_t = torch.from_numpy(x_tr).unsqueeze(1), torch.from_numpy(y_tr).long()
    net.train()
    for _ in range(EPOCHS):
        order = torch.randperm(len(x_t))
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss_fn(net(x_t[b].to(device)), y_t[b].to(device)).backward()
            opt.step()
        sched.step()
    net.eval()
    out = []
    with torch.no_grad():
        x_e = torch.from_numpy(x_te).unsqueeze(1)
        for i in range(0, len(x_e), 256):
            out.append(torch.softmax(net(x_e[i:i + 256].to(device)), dim=1).cpu().numpy())
    return np.concatenate(out)


def train_oof() -> pd.DataFrame:
    if OOF.exists():
        raise FileExistsError(f"{OOF} exists")
    w = pd.read_csv(WINDOWS, usecols=["record_name", "label_code", "signal_path"])
    w = w[w.label_code.isin(CLASSES)].reset_index(drop=True)
    rec = w.drop_duplicates("record_name").reset_index(drop=True)
    if len(rec) != 8234:
        raise ValueError(f"expected 8,234 non-noisy recordings, got {len(rec)}")
    x = standardise(np.stack([np.load(p) for p in w.signal_path]))
    y = w.label_code.map({c: i for i, c in enumerate(CLASSES)}).values
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(rec.record_name, rec.label_code)):
        torch.manual_seed(SEED + k)
        tr_mask, te_mask = w.record_name.isin(rec.record_name[tr]).values, w.record_name.isin(rec.record_name[te]).values
        if set(w.record_name[tr_mask]) & set(w.record_name[te_mask]):
            raise RuntimeError("a recording is in both partitions")
        p = train_fold(x[tr_mask], y[tr_mask], x[te_mask], device)
        rows.append(w.loc[te_mask, ["record_name", "label_code"]].assign(fold=k, **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
        print(f"[fold {k}] done", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OOF, index=False)
    return out


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 over the three classes; a class with no true or predicted member counts as 0."""
    scores = []
    for c in range(len(CLASSES)):
        tp, fp, fn = np.sum((pred == c) & (y == c)), np.sum((pred == c) & (y != c)), np.sum((pred != c) & (y == c))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(scores))


def kept_after_rejection(score: np.ndarray, frac: float, rng) -> np.ndarray:
    """Boolean mask of the recordings kept after rejecting the given fraction with the highest score (random tie-break)."""
    n_reject = int(round(frac * len(score)))
    order = np.lexsort((rng.random(len(score)), score))
    keep = np.ones(len(score), dtype=bool)
    keep[order[len(score) - n_reject:]] = False
    return keep


def random_change(y, pred, frac, base, rng, reps) -> float:
    n = len(y)
    n_keep = n - int(round(frac * n))
    return float(np.mean([macro_f1(y[i], pred[i]) - base for i in (rng.permutation(n)[:n_keep] for _ in range(reps))]))


def analyse(oof: pd.DataFrame) -> None:
    for path in (SUMMARY, DIFFS):
        if path.exists():
            raise FileExistsError(f"{path} exists; the analysis is run once")
    cols = [f"p_{c}" for c in CLASSES]
    rec = oof.groupby("record_name").agg({**{c: "mean" for c in cols}, "label_code": "first"}).sort_index()
    y = rec.label_code.map({c: i for i, c in enumerate(CLASSES)}).values
    pred = rec[cols].values.argmax(axis=1)
    base = macro_f1(y, pred)
    check = f1_score(y, pred, average="macro", labels=[0, 1, 2])
    print(f"[base] macro-F1 {base:.4f} (sklearn {check:.4f}); n = {len(y)}", flush=True)
    if abs(base - check) > 1e-9:
        raise RuntimeError("macro_f1 does not match scikit-learn")

    table = families()["CinC2017 (8,512 recordings)"][1].set_index("unit")
    models = [c for c in table.columns if c != "y"]
    scores = table.loc[rec.index, models]
    if not (table.loc[rec.index, "y"] == 0).all() or scores.isna().any().any():
        raise ValueError("quality scores missing, or a noisy recording entered")

    rng = np.random.default_rng(SEED)
    rows = []
    boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    deltas = {}
    for frac in FRACTIONS:
        rand = random_change(y, pred, frac, base, rng, N_RANDOM)
        rand_boot = []
        for idx in boot_idx:
            b = macro_f1(y[idx], pred[idx])
            rand_boot.append(random_change(y[idx], pred[idx], frac, b, rng, BOOT_RANDOM))
        rand_boot = np.array(rand_boot)
        for m in models:
            s = scores[m].values
            keep = kept_after_rejection(s, frac, rng)
            change = macro_f1(y[keep], pred[keep]) - base
            boots = np.empty(N_BOOT)
            for b, idx in enumerate(boot_idx):
                k = kept_after_rejection(s[idx], frac, rng)
                boots[b] = macro_f1(y[idx][k], pred[idx][k]) - macro_f1(y[idx], pred[idx]) - rand_boot[b]
            deltas[(frac, m)] = boots
            rejected = rec.label_code.values[~keep]
            rows.append({"fraction": frac, "model": m, "n_all": len(y), "n_kept": int(keep.sum()), "macro_f1_all": base, "macro_f1_kept": base + change,
                         "change": change, "random_change": rand, "change_minus_random": change - rand,
                         "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                         **{f"rejected_{c}": int((rejected == c).sum()) for c in CLASSES}})
        print(f"[fraction {frac}] done", flush=True)
    summary = pd.DataFrame(rows)
    diffs = []
    for frac in FRACTIONS:
        point = summary[summary.fraction == frac].set_index("model").change
        for other in [CNN2, HUB]:
            d = deltas[(frac, PROBE)] - deltas[(frac, other)]
            diffs.append({"fraction": frac, "comparison": f"{PROBE} minus {other}", "difference": point[PROBE] - point[other],
                          "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
    summary.to_csv(SUMMARY, index=False)
    pd.DataFrame(diffs).to_csv(DIFFS, index=False)
    print(summary[["fraction", "model", "macro_f1_kept", "change", "random_change", "change_minus_random", "ci_low", "ci_high", "rejected_N", "rejected_A", "rejected_O"]].round(4).to_string(index=False))
    print(pd.DataFrame(diffs).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    analyse(pd.read_csv(OOF) if "--analysis-only" in sys.argv else train_oof())
