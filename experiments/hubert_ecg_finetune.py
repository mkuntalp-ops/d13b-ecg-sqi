"""
HuBERT-ECG partial fine-tuning (analysis plan F2, 2026-10-25): does adaptation beyond a linear probe help?

Fixed configuration from the plan (no tuning): the last four transformer layers and a linear head are trained, the
rest of the model is frozen; AdamW, learning rate 1e-5 (transformer) and 1e-3 (head); 10 epochs; batch size 16
windows; class-weighted binary cross-entropy (positive weight = usable/unusable ratio of the training data); seed 42.
Input: the authors' preprocessing of experiments/hubert_ecg_probe.py (primary). Each 10-s window gives two 5-s halves;
their time-averaged last hidden states are averaged and passed to the head (as in the probe).
Protocol: the five BUT QDB subject folds (out-of-fold predictions), then training on all of BUT QDB and transfer to
CinC 2011 lead II (565 records) and all CinC 2017 windows (recording maximum).

Feasibility (plan F2: run only if one fold trains in under one hour): --time-fold0 trains one epoch of fold 0 and
reports the time per epoch; the full run is started only if 10 epochs are projected to take under one hour.

Outputs: results/hubert_finetune_predictions.csv, results/hubert_finetune_summary.csv
Usage: python experiments/hubert_ecg_finetune.py --time-fold0
       python experiments/hubert_ecg_finetune.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.argv = [a for a in sys.argv if not a.startswith("--preproc")]  # the primary (authors') preprocessing only
import hubert_ecg_probe as hp  # noqa: E402
from lodo_multisource import load_sources  # noqa: E402

assert hp.PREPROC == "authors"
RESULTS = ROOT / "results"
PRED_OUT = RESULTS / "hubert_finetune_predictions.csv"
SUM_OUT = RESULTS / "hubert_finetune_summary.csv"
SEED, EPOCHS, BATCH, N_TRAINABLE_LAYERS = 42, 10, 16, 4
LR_ENCODER, LR_HEAD = 1e-5, 1e-3
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class FineTuner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = hp.load_model("cpu")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.trainable = self.backbone.encoder.layers[-N_TRAINABLE_LAYERS:]
        for p in self.trainable.parameters():
            p.requires_grad = True
        self.head = torch.nn.Linear(768, 1)

    def forward(self, x):  # x: (B, 2, 6000)
        b = x.shape[0]
        h = self.backbone(x.reshape(2 * b, -1)).last_hidden_state.mean(dim=1)
        return self.head(h.reshape(b, 2, -1).mean(dim=1)).squeeze(-1)


def inputs(paths) -> np.ndarray:
    return np.stack([hp.model_input(np.load(p)) for p in paths])


def fit(X, y, epochs=EPOCHS, max_batches=None):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    net = FineTuner().to(DEVICE)
    opt = torch.optim.AdamW([{"params": net.trainable.parameters(), "lr": LR_ENCODER},
                             {"params": net.head.parameters(), "lr": LR_HEAD}])
    pos_weight = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], dtype=torch.float32, device=DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(y))
        for k, i in enumerate(range(0, len(y), BATCH)):
            if max_batches is not None and k >= max_batches:
                break
            idx = order[i:i + BATCH]
            xb = torch.from_numpy(X[idx]).to(DEVICE)
            yb = torch.from_numpy(y[idx].astype(np.float32)).to(DEVICE)
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
        print(f"  epoch {ep + 1}/{epochs}", flush=True)
    return net


@torch.no_grad()
def predict(net, X, batch=64):
    net.eval()
    out = [torch.sigmoid(net(torch.from_numpy(X[i:i + batch]).to(DEVICE))).cpu().numpy() for i in range(0, len(X), batch)]
    return np.concatenate(out)


def main():
    src = load_sources()
    but = src["BUT_QDB"]
    folds = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv").set_index("segment_id").loc[but.segment_id, "fold_id"].values
    Xb, yb = inputs(but.signal_path), but.label.values.astype(int)
    if "--time-fold0" in sys.argv:
        tr = folds != 0
        t0 = time.time()
        fit(Xb[tr], yb[tr], epochs=1)
        per_epoch = time.time() - t0
        print(f"[feasibility] fold 0: {tr.sum()} training windows, {per_epoch / 60:.1f} min per epoch, "
              f"projected {EPOCHS * per_epoch / 60:.0f} min for {EPOCHS} epochs (device {DEVICE})")
        return
    preds = []
    oof = np.zeros(len(yb))
    for k in range(5):
        tr, te = folds != k, folds == k
        print(f"[fold {k}] train {tr.sum()} windows", flush=True)
        oof[te] = predict(fit(Xb[tr], yb[tr]), Xb[te])
    preds.append(but[["dataset", "segment_id", "unit", "record", "label"]].assign(setting="butqdb_cv", prob_unusable=oof))
    net = fit(Xb, yb)
    for name in ["CinC2011", "CinC2017"]:
        d = src[name]
        p = np.concatenate([predict(net, inputs(d.signal_path.values[i:i + 2000])) for i in range(0, len(d), 2000)])
        preds.append(d[["dataset", "segment_id", "unit", "record", "label"]].assign(setting="but_only_external", prob_unusable=p))
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(PRED_OUT, index=False)
    rows = []
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
    main()
