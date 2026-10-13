"""
Re-render cached ECG images with the corrected renderer (data/ecg_rendering.py, 2026-10-13).

The earlier renderer guessed the unit from the amplitude and divided signals with |x| > 20 by 1000.
Millivolt signals with large artefacts (CinC 2011 unacceptable records, NSTDB -6 dB windows) were therefore
drawn as flat lines. This script:
- checks that BUT QDB images are unchanged (their signals are in uV and were converted correctly before),
- re-renders every cached CinC 2011 lead-II image and every NSTDB paired-window image from the stored mV signals,
- reports how many images changed.

Outputs: results/rerender_images_report.csv (one row per image: dataset, segment_id, max |x| in mV, changed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.ecg_rendering import render_ecg_image
from data.units import to_millivolts

RESULTS = PROJECT_ROOT / "results"
CINC_DIR = PROJECT_ROOT / "cache" / "datasets" / "cinc2011"
NSTDB_CSV = PROJECT_ROOT / "cache" / "datasets" / "nstdb" / "nstdb_paired_windows.csv"
OUT = RESULTS / "rerender_images_report.csv"


def same_image(img: Image.Image, path: Path) -> bool:
    with Image.open(path) as old:
        old = old.convert("RGB")
        return old.size == img.size and np.array_equal(np.asarray(old), np.asarray(img))


def main():
    rows = []
    # 1. BUT QDB: re-render a sample in memory and compare (must be identical; nothing is overwritten)
    but = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv")
    sample = but.sample(40, random_state=42)
    for _, r in sample.iterrows():
        x = to_millivolts(np.load(r["signal_path"]), "uV").astype(np.float32)
        img = render_ecg_image(x, 500, dpi=100)
        rows.append({"dataset": "BUT_QDB", "segment_id": r["segment_id"], "max_abs_mV": float(np.abs(x).max()),
                     "changed": not same_image(img, Path(r["image_path"])), "overwritten": False})
    n_but_changed = sum(r["changed"] for r in rows)
    print(f"[BUT QDB] {len(rows)} sampled images re-rendered, {n_but_changed} differ", flush=True)
    if n_but_changed:
        raise RuntimeError("BUT QDB images changed under the corrected renderer; stop and inspect")

    # 2. CinC 2011 lead II: all cached signals (mV, 500 Hz)
    for sig_path in sorted((CINC_DIR / "signals").glob("cinc2011_*_lead1.npy")):
        img_path = CINC_DIR / "images" / sig_path.name.replace(".npy", ".png")
        x = np.load(sig_path)
        img = render_ecg_image(x, 500, dpi=100)
        changed = not img_path.exists() or not same_image(img, img_path)
        if changed:
            img.save(img_path)
        rows.append({"dataset": "CinC2011_leadII", "segment_id": sig_path.stem, "max_abs_mV": float(np.abs(x).max()),
                     "changed": changed, "overwritten": changed})

    # 3. NSTDB paired windows (mV, 500 Hz)
    for _, r in pd.read_csv(NSTDB_CSV).iterrows():
        x = np.load(r["signal_path"])
        img = render_ecg_image(x, 500, dpi=100)
        changed = not same_image(img, Path(r["image_path"]))
        if changed:
            img.save(r["image_path"])
        rows.append({"dataset": "NSTDB_paired", "segment_id": r["segment_id"], "max_abs_mV": float(np.abs(x).max()),
                     "changed": changed, "overwritten": changed})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.groupby("dataset").agg(n=("changed", "size"), changed=("changed", "sum"),
                                     over_20mV=("max_abs_mV", lambda s: int((s > 20).sum()))).to_string())
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
