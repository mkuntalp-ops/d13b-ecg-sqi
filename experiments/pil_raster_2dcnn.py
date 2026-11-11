"""
2D CNN on direct raster strips instead of Matplotlib strips (plan item 3.5; secondary analysis, v2).

The Matplotlib renderer dominates image-model preprocessing time; a raster renderer (PIL ImageDraw,
experiments/latency_extended.render_raster) is about two orders of magnitude cheaper. A different renderer is a different
input distribution, so the 2D CNN is retrained on raster strips rather than applied to them: same five subject
folds, hyperparameters and seed as the primary analysis, then fit on all 988 BUT QDB windows and transferred.

Comparison with the Matplotlib 2D CNN on the same evaluation units: BUT QDB pooled out-of-fold (subject cluster bootstrap),
CinC 2011 lead II 998 records and CinC 2017 8,512 recordings, recording maximum (DeLong). d1-3B and its probe were not
re-run on raster strips (not part of this analysis).

Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python experiments/pil_raster_2dcnn.py [--analysis-only]
Output: cache/datasets/raster_strips/<segment_id>.png, $ECG_RESULTS_DIR/raster_2dcnn_predictions.csv,
        $ECG_RESULTS_DIR/raster_2dcnn_summary.csv
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "experiments"), str(PROJECT_ROOT / "evaluation")]

from config import CINC2011_LEADII_FULL, RESULTS_DIR  # noqa: E402
from data.units import to_millivolts  # noqa: E402
from latency_extended import render_raster  # noqa: E402
from models.cnn2d import ECG2DCNNClassifier  # noqa: E402
from paired_comparisons import cluster_bootstrap_paired, delong_paired, families, holm  # noqa: E402

CINC17 = PROJECT_ROOT / "cache" / "datasets" / "cinc2017" / "cinc2017_windows.csv"
RASTER_DIR = PROJECT_ROOT / "cache" / "datasets" / "raster_strips"
PRED_OUT = RESULTS_DIR / "raster_2dcnn_predictions.csv"
SUM_OUT = RESULTS_DIR / "raster_2dcnn_summary.csv"
FS = 500
MODEL = "2D_CNN_Image"


def render_one(args) -> None:
    sig_path, unit, img_path = args
    if not Path(img_path).exists():
        render_raster(to_millivolts(np.load(sig_path), unit).astype(np.float32), FS).save(img_path)


def add_raster_paths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["raster_path"] = [str(RASTER_DIR / f"{s}.png") for s in df.segment_id]
    with ProcessPoolExecutor(max_workers=6) as ex:
        list(ex.map(render_one, zip(df.signal_path, df.signal_unit, df.raster_path), chunksize=64))
    return df


def run() -> None:
    RASTER_DIR.mkdir(parents=True, exist_ok=True)
    but = add_raster_paths(pd.read_csv(RESULTS_DIR / "manifest_butqdb_1000_cohort.csv"))
    c11 = add_raster_paths(pd.read_csv(CINC2011_LEADII_FULL))
    c17 = add_raster_paths(pd.read_csv(CINC17))
    y = but.label.values
    paths = np.array(but.raster_path)
    oof = np.full(len(but), np.nan)
    for fold in range(5):
        te = but.fold_id.values == fold
        oof[te] = ECG2DCNNClassifier(epochs=25, lr=1e-3).fit(paths[~te].tolist(), y[~te]).predict_proba(paths[te].tolist())
        print(f"[raster] fold {fold} done", flush=True)
    full = ECG2DCNNClassifier(epochs=25, batch_size=32).fit(paths.tolist(), y)
    parts = [pd.DataFrame({"setting": "butqdb_cv", "dataset": "BUT_QDB", "segment_id": but.segment_id, "record_name": but.record_name,
                           "label": y, "prob_unusable": oof})]
    for name, df in [("CinC2011", c11), ("CinC2017", c17)]:
        parts.append(pd.DataFrame({"setting": "but_only_external", "dataset": name, "segment_id": df.segment_id,
                                   "record_name": df.record_name.astype(str), "label": df.label,
                                   "prob_unusable": full.predict_proba(df.raster_path.tolist())}))
        print(f"[raster] {name} scored", flush=True)
    pd.concat(parts, ignore_index=True).to_csv(PRED_OUT, index=False)


def analyse() -> None:
    p = pd.read_csv(PRED_OUT, low_memory=False, dtype={"record_name": str})
    fam = families()
    rows = []
    b = fam["BUT_QDB (5-fold subject CV)"][1]
    r = p[p.dataset == "BUT_QDB"].set_index("segment_id")["prob_unusable"]
    if r.isna().any() or len(b) != 988:
        raise ValueError("BUT QDB: incomplete raster predictions")
    a, m, y = r.loc[b.unit].values, b[MODEL].values, b.y.values
    d, lo, hi, pv = cluster_bootstrap_paired(y, a, m, b.cluster.values)
    rows.append(("BUT QDB (5-fold subject CV)", len(y), int(y.sum()), roc_auc_score(y, m), roc_auc_score(y, a), d, lo, hi, pv))
    for name, fk, agg in [("CinC 2011 lead II", "CinC2011 lead II (998 records)", None), ("CinC 2017 (recording max)", "CinC2017 (8,512 recordings)", "max")]:
        f = fam[fk][1].copy()
        f["unit"] = f.unit.astype(str)
        ds = "CinC2011" if agg is None else "CinC2017"
        s = p[p.dataset == ds].groupby("record_name")["prob_unusable"].max()
        if len(s) != len(f) or set(s.index) != set(f.unit):
            raise ValueError(f"{name}: unit mismatch ({len(s)} vs {len(f)})")
        a, m, y = s.loc[f.unit].values, f[MODEL].values, f.y.values.astype(int)
        d, lo, hi, pv = delong_paired(y, a, m)
        rows.append((name, len(y), int(y.sum()), roc_auc_score(y, m), roc_auc_score(y, a), d, lo, hi, pv))
    out = pd.DataFrame(rows, columns=["dataset", "n_units", "n_pos", "auroc_matplotlib", "auroc_raster", "delta_raster_minus_mpl",
                                      "delta_ci_low", "delta_ci_high", "p"])
    out["p_holm"] = holm(out.p.values)
    out.to_csv(SUM_OUT, index=False)
    print(out.round(4).to_string(index=False))
    print(f"[Saved] {SUM_OUT}")


if __name__ == "__main__":
    if "--analysis-only" not in sys.argv:
        run()
    analyse()
