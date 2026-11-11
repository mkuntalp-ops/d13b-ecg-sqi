"""
Zero-shot d1-3B on all 12 leads of the 998 labelled CinC 2011 Set A records (v2, handoff item 2).

Primary configuration only (strip + SQI text + primary question, unchanged prompt). Scores already computed are reused
from results/v2_2026-11/d1_external_predictions.csv: the 150 first-N records (12 leads) and lead II of all 998 records.
A random sample of reused windows is scored again first; the run stops if d1-3B does not reproduce them.

Resumable: every CHUNK windows the scores are appended to the output and the frozen features are written as a new set in
cache/d1_probe (read by segment id, experiments/feature_cache.py). The features are stored so that the forward pass
need not be repeated; they are not analysed here.

Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python experiments/d1_twelve_lead_full.py
Output: $ECG_RESULTS_DIR/d1_twelve_lead_full_predictions.csv (segment_id, record_name, lead_idx, label, prob_unusable)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from config import CINC2011_TWELVE_LEAD_FULL, RESULTS_DIR  # noqa: E402
from d1_linear_probe import FEAT_DIR, PRED_OUT as D1_EXISTING, Capture  # noqa: E402
from feature_cache import save_set  # noqa: E402
from features.sqi_extractor import SQI_FEATURE_NAMES  # noqa: E402
from models.d1_real_neural_model import RealD1DecisionModel  # noqa: E402

OUT = RESULTS_DIR / "d1_twelve_lead_full_predictions.csv"
CHUNK = 500
N_REPRO = 30
TOL = 1e-3
COLS = ["segment_id", "record_name", "lead_idx", "label", "prob_unusable"]


def score_row(d1, cap, r):
    """One forward pass; returns (probability, decision-state features, image-token features)."""
    sqi = dict(zip(SQI_FEATURE_NAMES, np.nan_to_num(r[SQI_FEATURE_NAMES].values.astype(float), nan=0.0)))
    cap.state = cap.image = None
    with Image.open(r["image_path"]) as img:
        dec = d1.decide_single(img, sqi, modality="multimodal", segment_id=r["segment_id"])
    if cap.state is None or cap.image is None:
        raise RuntimeError(f"hooks did not fire for {r['segment_id']}")
    return dec["prob_unusable"], cap.state, cap.image


def existing_scores() -> pd.Series:
    d = pd.read_csv(D1_EXISTING)
    d = d[d.dataset.isin(["CinC2011_12lead_firstN", "CinC2011_leadII"])]
    both = d[d.segment_id.duplicated(keep=False)].groupby("segment_id").prob_unusable.agg(["min", "max"])
    if len(both) and (both["max"] - both["min"]).max() > TOL:
        raise ValueError("a window has two different stored d1-3B scores")
    return d.drop_duplicates("segment_id").set_index("segment_id")["prob_unusable"]


def main():
    df = pd.read_csv(CINC2011_TWELVE_LEAD_FULL).reset_index(drop=True)
    if df.segment_id.nunique() != 11976 or df.record_name.nunique() != 998:
        raise ValueError("expected 11,976 lead windows from 998 records")
    known = existing_scores()
    reused = df[df.segment_id.isin(known.index)]
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLS)
    if done.segment_id.duplicated().any():
        raise ValueError(f"{OUT}: duplicated segment ids")
    print(f"{len(reused)} windows have stored scores; {len(done)} already in the output", flush=True)

    d1 = RealD1DecisionModel(device="mps")
    cap = Capture(d1.model)
    sample = reused.sample(N_REPRO, random_state=42)
    worst = max(abs(score_row(d1, cap, r)[0] - known[r["segment_id"]]) for _, r in sample.iterrows())
    print(f"[check] max |rescored - stored| on {N_REPRO} windows = {worst:.6f}", flush=True)
    if worst > TOL:
        raise RuntimeError("d1-3B does not reproduce the stored scores")

    if done.empty:
        first = reused.assign(prob_unusable=reused.segment_id.map(known))[COLS]
        first.to_csv(OUT, index=False)
        done = first
    todo = df[~df.segment_id.isin(done.segment_id)].reset_index(drop=True)
    print(f"{len(todo)} windows to score", flush=True)
    part = len(list(FEAT_DIR.glob("CinC2011_12lead_full_part*_ids.csv")))
    for start in range(0, len(todo), CHUNK):
        chunk = todo.iloc[start:start + CHUNK]
        rows, st, im = [], [], []
        for _, r in chunk.iterrows():
            p, s, i = score_row(d1, cap, r)
            rows.append({c: r[c] for c in COLS[:-1]} | {"prob_unusable": p})
            st.append(s)
            im.append(i)
        save_set(FEAT_DIR, f"CinC2011_12lead_full_part{part:03d}", chunk["segment_id"].tolist(),
                 {"state": np.stack(st), "image": np.stack(im)})
        pd.DataFrame(rows)[COLS].to_csv(OUT, mode="a", header=False, index=False)
        part += 1
        print(f"{start + len(chunk)}/{len(todo)}", flush=True)
    final = pd.read_csv(OUT)
    if len(final) != 11976 or final.segment_id.nunique() != 11976 or final.prob_unusable.isna().any():
        raise ValueError(f"{OUT}: incomplete ({len(final)} rows)")
    print(f"[Saved] {OUT}: {len(final)} windows", flush=True)


if __name__ == "__main__":
    main()
