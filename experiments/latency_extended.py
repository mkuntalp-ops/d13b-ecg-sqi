"""
Extended latency profiling (second review, 2026-10-13). Run alone (no other MPS or CPU-heavy job).

Table 4 times the models on cached 500 Hz windows. The review asked what is left out. This script measures,
on the same 25 BUT QDB windows used by experiments/empirical_latency_benchmark.py, three runs each:

1. Raw-to-decision for the 1D CNN: the native 1000 Hz BUT QDB window (uV, already in memory; disk and
   acquisition time excluded) -> unit conversion -> resampling to 500 Hz (scipy.signal.resample, as in the
   pipeline) -> per-window z-score -> forward pass. On MPS (synchronised) and on CPU.
2. Rendering: the Matplotlib renderer used for every image model vs a direct raster renderer (PIL ImageDraw:
   same 513 x 257 canvas, same per-strip vertical scaling and grid spacing, polyline trace). Only the time is
   compared; the image models were trained on Matplotlib strips and were not evaluated on the raster strips.
3. Forward passes of the 1D CNN, 2D CNN and multimodal net on CPU as well as MPS (untrained weights; the cost of
   a forward pass does not depend on the weights).
4. SQI extraction alone.

Output: results/latency_extended_raw.csv (one row per run, window and measurement),
        results/latency_extended_summary.csv (median over windows per run; median and range over runs)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from PIL import Image, ImageDraw
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.ecg_rendering import render_ecg_image
from data.units import to_millivolts
from features.sqi_extractor import SQI_FEATURE_NAMES, extract_sqi_features
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import SinglePassMultimodalClassifier

BUT_SRC = Path("/Volumes/KIOXIA/Datasets/butqdb/brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0")
RESULTS = PROJECT_ROOT / "results"
N_WINDOWS, N_RUNS = 25, 3


def select_windows():
    """Same selection as empirical_latency_benchmark.py (evenly spaced over the sorted cohort)."""
    df = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv")
    idx = np.linspace(0, len(df) - 1, N_WINDOWS, dtype=int)
    return df.iloc[idx].reset_index(drop=True)


def native_window(row):
    rec = row["record_name"]
    r = wfdb.rdrecord(str(BUT_SRC / rec.split("_")[0] / rec), sampfrom=int(row["start_sample"]), sampto=int(row["end_sample"]))
    return r.p_signal[:, 0].astype(np.float32), r.units[0]


def render_raster(x, fs=500, size=(513, 257)):
    """Direct raster strip: per-strip vertical scaling as in render_ecg_image, grid every 0.5 s / 0.5 mV."""
    w, h = size
    p1, p99 = np.percentile(x, [1, 99])
    margin = max(0.5, (p99 - p1) * 0.2)
    lo, hi = p1 - margin, p99 + margin
    if hi - lo < 1.0:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5, mid + 0.5
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    dur = len(x) / fs
    for s in np.arange(0, dur + 1e-9, 0.5):
        xs = int(s / dur * (w - 1))
        d.line([(xs, 0), (xs, h - 1)], fill=(255, 204, 204))
    step = 0.5 if hi - lo <= 5 else max(1.0, (hi - lo) / 8)
    for v in np.arange(np.ceil(lo / step) * step, hi, step):
        ys = int((hi - v) / (hi - lo) * (h - 1))
        d.line([(0, ys), (w - 1, ys)], fill=(255, 204, 204))
    xs = np.linspace(0, w - 1, len(x))
    ys = (hi - np.clip(x, lo, hi)) / (hi - lo) * (h - 1)
    d.line(list(zip(xs.tolist(), ys.tolist())), fill=(0, 0, 0), width=1)
    return img


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def timed(fn, dev=None):
    sync(dev)
    t0 = time.perf_counter()
    out = fn()
    sync(dev)
    return (time.perf_counter() - t0) * 1000.0, out


def main():
    sel = select_windows()
    natives = [native_window(r) for _, r in sel.iterrows()]
    cached = [to_millivolts(np.load(p), "uV").astype(np.float32) for p in sel["signal_path"]]
    y0 = np.array([0, 1] * 3)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    nets = {}
    for dev in devices:
        c1 = ECG1DCNNClassifier(epochs=1, device=dev).fit(cached[:6], y0)
        c2 = ECG2DCNNClassifier(epochs=1, device=dev).fit(sel["image_path"][:6].tolist(), y0)
        mm = SinglePassMultimodalClassifier(epochs=1, device=dev).fit(sel["image_path"][:6].tolist(),
                                                                      np.nan_to_num(sel[SQI_FEATURE_NAMES].values[:6]), y0)
        nets[dev] = (c1, c2, mm)
    rows = []
    for run in range(1, N_RUNS + 1):
        for i, ((raw, unit), cached_x, (_, r)) in enumerate(zip(natives, cached, sel.iterrows())):
            # warm-up on the first window of each run is excluded below (i == 0 measured but flagged)
            t_conv, x_mv = timed(lambda: to_millivolts(raw, unit).astype(np.float32))
            t_res, x500 = timed(lambda: signal.resample(x_mv, 5000).astype(np.float32))
            t_sqi, sqis = timed(lambda: extract_sqi_features(x500, 500))
            t_mpl, img = timed(lambda: render_ecg_image(cached_x, 500, dpi=100))
            t_ras, img_r = timed(lambda: render_raster(cached_x))
            rec = {"run": run, "window": i, "warmup": i == 0, "convert_ms": t_conv, "resample_ms": t_res, "sqi_ms": t_sqi,
                   "render_matplotlib_ms": t_mpl, "render_raster_ms": t_ras}
            for dev in devices:
                c1, c2, mm = nets[dev]
                rec[f"cnn1d_forward_{dev}_ms"] = timed(lambda: c1.predict_proba([x500]), dev)[0]
                rec[f"cnn2d_forward_{dev}_ms"] = timed(lambda: c2.predict_proba([r["image_path"]]), dev)[0]
                xs = np.nan_to_num(np.array([[sqis[k] for k in SQI_FEATURE_NAMES]]))
                rec[f"mm_forward_{dev}_ms"] = timed(lambda: mm.predict_proba([r["image_path"]], xs), dev)[0]
                rec[f"cnn1d_raw_to_decision_{dev}_ms"] = rec["convert_ms"] + rec["resample_ms"] + rec[f"cnn1d_forward_{dev}_ms"]
            rows.append(rec)
        print(f"[latency_extended] run {run} done", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "latency_extended_raw.csv", index=False)
    meas = [c for c in raw.columns if c.endswith("_ms")]
    per_run = raw[~raw.warmup].groupby("run")[meas].median()
    summ = pd.DataFrame({"median_over_runs_ms": per_run.median(), "min_run_ms": per_run.min(), "max_run_ms": per_run.max(),
                         "p95_median_over_runs_ms": raw[~raw.warmup].groupby("run")[meas].quantile(0.95).median()})
    summ.index.name = "measurement"
    summ.to_csv(RESULTS / "latency_extended_summary.csv")
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
