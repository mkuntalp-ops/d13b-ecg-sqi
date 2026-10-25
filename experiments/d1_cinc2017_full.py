"""
Zero-shot d1-3B and the d1-3B image-token features on all of CinC 2017 (analysis plan A2, 2026-10-25).

experiments/d1_linear_probe.py ran d1-3B on the balanced 556-recording subset (1,376 windows). This script runs the
same primary configuration (strip + SQI text + primary question) with the same forward hooks on the remaining
22,609 windows, so that zero-shot d1-3B and the probe can be evaluated on all 8,512 recordings.

Extraction is resumable: windows are processed in chunks of 1,000, each saved as cache/d1_probe/CinC2017_rest/part_XXX.npz
(segment_id, prob_unusable, image tokens, decision state); finished chunks are skipped on restart.
--merge combines the subset and the chunks into cache/d1_probe/CinC2017_full_{image,state}.npy, _ids.csv and
results/d1_cinc2017_full_predictions.csv: per window, the zero-shot probability (prob_unusable) and the image-token
probe score (prob_image_probe; probe fitted on all BUT QDB windows exactly as in d1_linear_probe.analyse). As a check,
the probe's recording maxima on the 556 subset recordings must equal results/d1_probe_transfer_predictions.csv.

Usage: python experiments/d1_cinc2017_full.py            (extract remaining chunks, then merge)
       python experiments/d1_cinc2017_full.py --merge
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from d1_linear_probe import CINC17, FEAT_DIR, PRED_OUT, RESULTS, Capture, probe  # noqa: E402
from features.sqi_extractor import SQI_FEATURE_NAMES  # noqa: E402

PART_DIR = FEAT_DIR / "CinC2017_rest"
FULL_PRED = PROJECT_ROOT / "results" / "d1_cinc2017_full_predictions.csv"
CHUNK = 1000


def remaining_windows() -> pd.DataFrame:
    c17 = pd.read_csv(CINC17)
    done = set(pd.read_csv(FEAT_DIR / "CinC2017_subset_ids.csv").segment_id)
    rest = c17[~c17.segment_id.isin(done)].reset_index(drop=True)
    assert len(c17) == 23985 and len(done) == 1376 and len(rest) == 22609, (len(c17), len(done), len(rest))
    return rest


def extract():
    from models.d1_real_neural_model import RealD1DecisionModel
    PART_DIR.mkdir(parents=True, exist_ok=True)
    rest = remaining_windows()
    d1 = cap = None
    for k in range(0, len(rest), CHUNK):
        out = PART_DIR / f"part_{k // CHUNK:03d}.npz"
        if out.exists():
            continue
        if d1 is None:
            d1 = RealD1DecisionModel(device="mps")
            cap = Capture(d1.model)
        ids, probs, st, im = [], [], [], []
        for _, r in rest.iloc[k:k + CHUNK].iterrows():
            sqi = dict(zip(SQI_FEATURE_NAMES, np.nan_to_num(r[SQI_FEATURE_NAMES].values.astype(float), nan=0.0)))
            cap.state = cap.image = None
            with Image.open(r["image_path"]) as img:
                dec = d1.decide_single(img, sqi, modality="multimodal", segment_id=r["segment_id"])
            if cap.state is None or cap.image is None:
                raise RuntimeError(f"hooks did not fire for {r['segment_id']}")
            ids.append(r["segment_id"]); probs.append(dec["prob_unusable"]); st.append(cap.state); im.append(cap.image)
        np.savez(out, segment_id=np.array(ids), prob=np.array(probs), state=np.stack(st).astype(np.float32),
                 image=np.stack(im).astype(np.float32))
        print(f"[d1 CinC2017] chunk {k // CHUNK + 1}/{-(-len(rest) // CHUNK)} saved", flush=True)


def merge():
    c17 = pd.read_csv(CINC17)
    sub_ids = pd.read_csv(FEAT_DIR / "CinC2017_subset_ids.csv").segment_id.values
    ids = list(sub_ids)
    image = [np.load(FEAT_DIR / "CinC2017_subset_image.npy")]
    state = [np.load(FEAT_DIR / "CinC2017_subset_state.npy")]
    sub_pred = pd.read_csv(PRED_OUT)
    sub_pred = sub_pred[sub_pred.dataset == "CinC2017_subset"].set_index("segment_id").prob_unusable
    probs = list(sub_pred.loc[sub_ids].values)
    n_parts = -(-22609 // CHUNK)
    for k in range(n_parts):
        z = np.load(PART_DIR / f"part_{k:03d}.npz")
        ids += list(z["segment_id"]); probs += list(z["prob"]); image.append(z["image"]); state.append(z["state"])
    assert len(ids) == len(set(ids)) == len(c17) and set(ids) == set(c17.segment_id)
    np.save(FEAT_DIR / "CinC2017_full_image.npy", np.concatenate(image))
    np.save(FEAT_DIR / "CinC2017_full_state.npy", np.concatenate(state))
    pd.Series(ids).to_csv(FEAT_DIR / "CinC2017_full_ids.csv", index=False, header=["segment_id"])
    meta = c17.set_index("segment_id").loc[ids, ["record_name", "label"]].reset_index()
    meta["prob_unusable"] = probs
    but = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv")
    assert (pd.read_csv(FEAT_DIR / "BUT_QDB_ids.csv").segment_id.values == but.segment_id.values).all()
    full = probe(groups=True).fit(np.load(FEAT_DIR / "BUT_QDB_image.npy"), but.label.values, groups=but.subject_id.values)
    meta["prob_image_probe"] = full.predict_proba(np.concatenate(image))[:, 1]
    stored = pd.read_csv(RESULTS / "d1_probe_transfer_predictions.csv")
    stored = stored[(stored.features == "image") & (stored.dataset == "CinC2017")].set_index("record_name").prob_unusable
    mx = meta.groupby(meta.record_name.astype(str)).prob_image_probe.max().loc[stored.index.astype(str)]
    diff = float(np.abs(mx.values - stored.values).max())
    print(f"[check] max |probe recomputed - stored| on the subset recordings = {diff:.2e}")
    assert diff < 1e-6
    meta.to_csv(FULL_PRED, index=False)
    print(f"[Saved] {FULL_PRED} ({len(meta)} windows, {meta.record_name.nunique()} recordings)")


if __name__ == "__main__":
    if "--merge" not in sys.argv:
        extract()
    merge()
