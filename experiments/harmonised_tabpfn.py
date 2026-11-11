"""
TabPFN in the harmonised-pipeline analysis (PAPER-002; see experiments/harmonised_inputs.py). Runs in the TabPFN
environment: <tabpfn env>/bin/python experiments/harmonised_tabpfn.py
Same checkpoint, defaults and random state as experiments/tabpfn_runner.py; inputs G2 + G3.
Output: $ECG_RESULTS_DIR/harmonised_tabpfn_predictions.csv (scores before the constant-input rule G1)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from harmonised_inputs import HARMONISED, load_tables, zclip_apply, zclip_fit  # noqa: E402

RESULTS = Path(os.environ.get("ECG_RESULTS_DIR", ROOT / "results")).resolve()
CKPT = os.environ.get("TABPFN_CKPT", "/Volumes/KIOXIA/AI Models/tabpfn/tabpfn-v3-classifier-v3_default.ckpt")
SEED = 42


def fit_predict(Xtr, ytr, tests):
    mean, sd = zclip_fit(Xtr)
    m = TabPFNClassifier(model_path=CKPT, device="cpu", random_state=SEED).fit(zclip_apply(Xtr, mean, sd), ytr)
    return [m.predict_proba(zclip_apply(X, mean, sd))[:, 1] for X in tests]


def main():
    T = load_tables(ROOT, RESULTS)
    but = T["BUT_QDB"]
    X, y = but[HARMONISED].values, but.label.values.astype(int)
    oof = np.zeros(len(but))
    for k in range(5):
        tr, te = but.fold_id.values != k, but.fold_id.values == k
        oof[te] = fit_predict(X[tr], y[tr], [X[te]])[0]
        print(f"[tabpfn-h] fold {k} done", flush=True)
    parts = [pd.DataFrame({"dataset": "BUT_QDB", "segment_id": but.segment_id, "model": "TabPFN_SQI", "seed": SEED, "prob_unusable": oof})]
    p11, p17 = fit_predict(X, y, [T["CinC2011"][HARMONISED].values, T["CinC2017"][HARMONISED].values])
    for name, p in [("CinC2011", p11), ("CinC2017", p17)]:
        parts.append(pd.DataFrame({"dataset": name, "segment_id": T[name].segment_id, "model": "TabPFN_SQI", "seed": SEED, "prob_unusable": p}))
    out = RESULTS / "harmonised_tabpfn_predictions.csv"
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    print(f"[Saved] {out}")


if __name__ == "__main__":
    main()
