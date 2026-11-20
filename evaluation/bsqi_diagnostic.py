"""
Diagnostic of the bSQI feature on CinC 2011 lead II and BUT QDB (round V19, 2026-11-19; review finding PAPER-002 of the
V17 review). Descriptive and post hoc; no model is fitted and the stored features are not changed.

Question: why is the median bSQI only 0.545 on acceptable CinC 2011 lead II records (Supplementary Table S29)? For
each window (cached signal, 500 Hz, mV) the script recomputes the two detectors of features/qrs_detectors.py and adds
two validated detectors from the wfdb package (XQRS and GQRS), then records beat counts and agreements:
  - bsqi_orig: Pan-Tompkins-style vs second-difference detector, 50 ms (the stored feature; checked to match);
  - bsqi_orig_150: the same two detectors with the 150 ms tolerance of Li et al. (2008);
  - pt_vs_xqrs, diff_vs_xqrs (50 ms): which of the two detectors departs from a validated detector;
  - xqrs_vs_gqrs_150: a bSQI computed from two validated detectors with a 150 ms tolerance;
  - n_diff_lp40, bsqi_lp40: the second-difference detector after a 40 Hz low-pass filter (4th-order Butterworth,
    zero phase), to test whether high-frequency content inflates its detections.
Per dataset and class it reports median [IQR] and, for CinC 2011, the univariate AUROC (positive class unusable,
lower agreement = more likely unusable) with and without the 73 records with a constant lead II.

Output: <results>/bsqi_diagnostic_windows.csv (per window) and <results>/bsqi_diagnostic_summary.csv
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/bsqi_diagnostic.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb.processing as wp
from scipy import signal
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, RESULTS_DIR  # noqa: E402
from features.qrs_detectors import calculate_bsqi, detect_qrs_difference, detect_qrs_pan_tompkins  # noqa: E402

FS = 500
LP_B, LP_A = signal.butter(4, 40 / (FS / 2), btype="low")
CINC = DATA_DIR / "cinc2011" / "cinc2011_setA_leadII_full.csv"
BUT_OOF = RESULTS_DIR / "cohort1000_consolidated_oof_predictions.csv"
OUT_W = RESULTS_DIR / "bsqi_diagnostic_windows.csv"
OUT_S = RESULTS_DIR / "bsqi_diagnostic_summary.csv"


def xqrs(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # XQRS warns on flat or very short inputs
        return np.asarray(wp.xqrs_detect(sig=x, fs=FS, verbose=False), dtype=int)


def gqrs(x):
    return np.asarray(wp.gqrs_detect(sig=x, fs=FS), dtype=int)


def one(x):
    x = np.asarray(x, dtype=np.float64)
    constant = bool(np.std(x) == 0)
    pt, df = detect_qrs_pan_tompkins(x, FS), detect_qrs_difference(x, FS)
    xq, gq = (np.array([], int), np.array([], int)) if constant else (xqrs(x), gqrs(x))
    dl = detect_qrs_difference(signal.filtfilt(LP_B, LP_A, x), FS)
    return {"constant": constant, "n_pt": len(pt), "n_diff": len(df), "n_xqrs": len(xq), "n_gqrs": len(gq),
            "bsqi_orig": calculate_bsqi(pt, df, FS, 50.0), "bsqi_orig_150": calculate_bsqi(pt, df, FS, 150.0),
            "pt_vs_xqrs": calculate_bsqi(xq, pt, FS, 50.0), "diff_vs_xqrs": calculate_bsqi(xq, df, FS, 50.0),
            "xqrs_vs_gqrs_150": calculate_bsqi(xq, gq, FS, 150.0), "n_diff_lp40": len(dl),
            "bsqi_lp40": calculate_bsqi(pt, dl, FS, 50.0)}


def load():
    c = pd.read_csv(CINC)
    c = c.assign(dataset="CinC 2011 lead II", stored_bsqi=c.bsqi_detector_agreement)[
        ["dataset", "segment_id", "label", "signal_path", "stored_bsqi"]]
    b = pd.read_csv(BUT_OOF)
    b = b.assign(dataset="BUT QDB", stored_bsqi=b.bsqi_detector_agreement)[
        ["dataset", "segment_id", "label", "signal_path", "stored_bsqi"]]
    if len(c) != 998 or len(b) != 988:
        raise ValueError(f"unexpected cohort sizes: CinC 2011 {len(c)}, BUT QDB {len(b)}")
    return pd.concat([c, b], ignore_index=True)


def main():
    d = load()
    rows = [one(np.load(p)) for p in d.signal_path]
    w = pd.concat([d.drop(columns="signal_path"), pd.DataFrame(rows)], axis=1)
    dev = (w.bsqi_orig - w.stored_bsqi).abs().max()
    if dev > 1e-6:
        raise ValueError(f"recomputed bSQI differs from the stored feature by up to {dev}")
    w.to_csv(OUT_W, index=False)
    cols = ["n_pt", "n_diff", "n_diff_lp40", "n_xqrs", "n_gqrs", "bsqi_orig", "bsqi_orig_150", "bsqi_lp40",
            "pt_vs_xqrs", "diff_vs_xqrs", "xqrs_vs_gqrs_150"]
    out = []
    for (ds, y), g in w.groupby(["dataset", "label"]):
        for k in cols:
            q = g[k].quantile([0.25, 0.5, 0.75]).values
            out.append({"dataset": ds, "label": int(y), "n": len(g), "quantity": k, "median": q[1], "q25": q[0],
                        "q75": q[2]})
    for ds, g in w.groupby("dataset"):
        for subset, h in (("all", g), ("without constant lead", g[~g.constant])):
            for k in ["bsqi_orig", "bsqi_orig_150", "bsqi_lp40", "xqrs_vs_gqrs_150"]:
                out.append({"dataset": ds, "label": -1, "n": len(h), "quantity": f"AUROC {k} ({subset})",
                            "median": roc_auc_score(h.label, -h[k]), "q25": np.nan, "q75": np.nan})
    s = pd.DataFrame(out)
    s.to_csv(OUT_S, index=False)
    print(s.to_string(index=False))
    print(f"[Saved] {OUT_W}\n[Saved] {OUT_S}")


if __name__ == "__main__":
    main()
