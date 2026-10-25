"""
Rule heuristic with coefficients learned inside the cross-validation (Q1 revision, 2026-10-25).

The rule heuristic (models/rule_heuristic_baseline.py) combines seven inputs with hand-set coefficients:
bSQI, min(baseline-wander ratio, 3), min(HF ratio, 3), saturation and flatline fractions, and two strip
features (ink density = share of pixels darker than 200; spread = SD over columns of the per-column ink share).
When those coefficients were chosen is not documented, so they may have been informed by BUT QDB. Here the same
seven inputs feed a standardized L2 logistic regression fitted on the training subjects of each fold only
(fixed C = 1, no tuning). Its out-of-fold AUROC is an honest in-domain estimate for this feature set; the gap to
the hand-set rule bounds how much the hand-set coefficients could have profited from BUT QDB. The model fitted on
all of BUT QDB is also applied to CinC 2011 (lead II, 565 records) and CinC 2017 (recording maximum).

Output: results/heuristic_refit_predictions.csv, results/heuristic_refit_summary.csv
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
from paired_comparisons import cluster_bootstrap_paired, delong_paired  # noqa: E402

RESULTS = ROOT / "results"
OUT_PRED = RESULTS / "heuristic_refit_predictions.csv"
OUT_SUM = RESULTS / "heuristic_refit_summary.csv"
INPUTS = ["bsqi", "bw3", "hf3", "sat", "flat", "ink", "spread"]


def strip_features(path: str):
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    ink = arr < 200
    return float(ink.mean()), float(np.std(ink.mean(axis=0)))


def design(df: pd.DataFrame) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=8) as ex:
        vis = list(ex.map(strip_features, df["image_path"]))
    return pd.DataFrame({"bsqi": df["bsqi_detector_agreement"].values,
                         "bw3": np.minimum(df["baseline_wander_ratio"].values, 3.0),
                         "hf3": np.minimum(df["high_frequency_noise_ratio"].values, 3.0),
                         "sat": df["saturation_fraction"].values, "flat": df["flatline_fraction"].values,
                         "ink": [v[0] for v in vis], "spread": [v[1] for v in vis]}).fillna(0.0)


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def main():
    but = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv")
    Xb, yb = design(but), but["label"].values
    oof = np.zeros(len(but))
    for k in range(5):
        tr, te = but.fold_id.values != k, but.fold_id.values == k
        oof[te] = model().fit(Xb[tr], yb[tr]).predict_proba(Xb[te])[:, 1]
    full = model().fit(Xb, yb)
    coefs = dict(zip(INPUTS, full[-1].coef_[0].round(3)))
    print("[refit] standardized coefficients on all BUT QDB:", coefs, flush=True)

    c11 = pd.read_csv(ROOT / "cache/datasets/cinc2011/cinc2011_random_draws_records.csv").drop_duplicates("segment_id")
    lab = pd.read_csv(ROOT / "cache/datasets/cinc2011/cinc2011_random_draws_manifest.csv").drop_duplicates("record_name").set_index("record_name")["label"]
    c11["label"] = lab.loc[c11["record_name"].astype(int)].values
    p11 = full.predict_proba(design(c11))[:, 1]
    c17 = pd.read_csv(ROOT / "cache/datasets/cinc2017/cinc2017_windows.csv")
    p17 = full.predict_proba(design(c17))[:, 1]
    print("[refit] external scored", flush=True)
    pd.concat([pd.DataFrame({"dataset": "BUT_QDB", "segment_id": but.segment_id, "label": yb, "prob_unusable": oof}),
               pd.DataFrame({"dataset": "CinC2011", "segment_id": c11.segment_id, "label": c11.label, "prob_unusable": p11}),
               pd.DataFrame({"dataset": "CinC2017", "segment_id": c17.segment_id, "record_name": c17.record_name,
                             "label": c17.label, "prob_unusable": p17})]).to_csv(OUT_PRED, index=False)

    # paired comparison with the hand-set rule on the same units
    hand_b = pd.read_csv(RESULTS / "cohort1000_consolidated_oof_predictions.csv").set_index("segment_id").loc[but.segment_id, "prob_Rule_Based_SQI_Heuristic"].values
    hand_11 = pd.read_csv(RESULTS / "cinc_random_draws_predictions.csv").drop_duplicates("segment_id").set_index("segment_id").loc[c11.segment_id, "Rule_Based_SQI_Heuristic"].values
    w = pd.read_csv(RESULTS / "external_window_predictions.csv", low_memory=False)
    w = w[w.dataset == "CinC2017"].set_index("segment_id")
    r17 = pd.DataFrame({"record": c17.record_name.values, "y": c17.label.values, "refit": p17,
                        "hand": w.loc[c17.segment_id, "Rule_Based_SQI_Heuristic"].values}).groupby("record").agg(
        y=("y", "first"), refit=("refit", "max"), hand=("hand", "max"))
    rows = []
    for name, y, a, b, cl in [("BUT_QDB (5-fold subject CV)", yb, oof, hand_b, but.subject_id.values),
                              ("CinC2011 lead II (565)", c11.label.values, p11, hand_11, None),
                              ("CinC2017 (8,512, recording max)", r17.y.values, r17.refit.values, r17.hand.values, None)]:
        d, lo, hi, p = cluster_bootstrap_paired(y, a, b, cl) if cl is not None else delong_paired(y, a, b)
        rows.append({"dataset": name, "auroc_refit": roc_auc_score(y, a), "auroc_handset": roc_auc_score(y, b),
                     "delta_refit_minus_handset": d, "ci_lo": lo, "ci_hi": hi, "p": p,
                     "method": "subject cluster bootstrap" if cl is not None else "DeLong"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_SUM, index=False)
    print(out.round(3).to_string(index=False))
    print(f"[Saved] {OUT_SUM}")


if __name__ == "__main__":
    main()
