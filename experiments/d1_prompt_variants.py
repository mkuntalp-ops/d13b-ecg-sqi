"""
Prompt sensitivity of LiquidAI/d1-3B (sensitivity analysis; the main results keep the primary configuration).

Primary configuration (every main result): modality 'multimodal' = rendered strip + text summary of the SQIs
(format_sqi_for_prompt; 10 of the 13 features are written into the prompt) + the primary usability question.

Alternative configurations, run on the same windows:
- image_only: the primary question with the strip only (no SQI text).
- alt_wording: a reworded usability question with the same target (unusable = Class 3 on BUT QDB, unacceptable
  on CinC) and the same SQI text.

Validation first: for the first 10 BUT QDB windows the primary configuration is recomputed and compared with the
stored primary predictions (prob_LiquidAI_d1_3B_Neural). The run stops if any difference exceeds 0.05.

Outputs:
- results/d1_prompt_validation.csv: stored vs recomputed primary probabilities (10 BUT QDB windows)
- results/d1_prompt_variants_predictions.csv: every configuration on every window (primary, image_only, alt_wording)
- results/d1_prompt_variants_summary.csv: AUROC, AP, Brier, sensitivity and false reject at 0.5 (BUT QDB, CinC);
  mean P(usable) and paired change against the clean partner per SNR label (NSTDB)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.sqi_extractor import SQI_FEATURE_NAMES
from models.d1_real_neural_model import RealD1DecisionModel

RESULTS = PROJECT_ROOT / "results"
BUT_MANIFEST = RESULTS / "manifest_butqdb_1000_cohort.csv"
BUT_STORED = RESULTS / "cohort1000_consolidated_oof_predictions.csv"
CINC_STORED = RESULTS / "cinc2011_expanded_predictions.csv"
CINC_RECORDS = PROJECT_ROOT / "cache" / "datasets" / "cinc2011" / "cinc2011_random_draws_records.csv"
NSTDB_CSV = PROJECT_ROOT / "cache" / "datasets" / "nstdb" / "nstdb_paired_windows.csv"
NSTDB_STORED = RESULTS / "exp4_revised_nstdb_predictions.csv"
PRIMARY_COL = "prob_LiquidAI_d1_3B_Neural"
OUT_VAL = RESULTS / "d1_prompt_validation.csv"
OUT_PRED = RESULTS / "d1_prompt_variants_predictions.csv"
OUT_SUM = RESULTS / "d1_prompt_variants_summary.csv"
REUSE_BUT_FROM = RESULTS / "_backup_pre_render_fix_2026-10-13" / "d1_prompt_variants_predictions.csv"

ALT_QUESTION = {
    "usability": {
        "type": "choice",
        "instructions": (
            "Decide whether this 10-second single-lead ECG strip can be used for automated heart-rhythm analysis. "
            "It is unusable if noise, baseline drift, saturation or flatline prevents reliable QRS detection, "
            "or if QRS complexes cannot be seen; otherwise it is usable."
        ),
        "criteria": {
            "usable": "Individual QRS complexes can be detected reliably, so rhythm analysis is possible.",
            "unusable": "QRS detection is not reliable because of severe noise, drift, saturation, flatline or absent complexes.",
        },
    }
}


def score(d1: RealD1DecisionModel, df: pd.DataFrame, variant: str) -> np.ndarray:
    probs = []
    for _, r in df.iterrows():
        sqi_vec = np.nan_to_num(r[SQI_FEATURE_NAMES].values.astype(float), nan=0.0)
        sqi = dict(zip(SQI_FEATURE_NAMES, sqi_vec))
        with Image.open(r["image_path"]) as img:
            if variant == "primary":
                dec = d1.decide_single(img, sqi, modality="multimodal", segment_id=r["segment_id"])
            elif variant == "image_only":
                dec = d1.decide_single(img, sqi, modality="image_only", segment_id=r["segment_id"])
            elif variant == "alt_wording":
                dec = d1.decide_single(img, sqi, modality="multimodal", segment_id=r["segment_id"], questions=ALT_QUESTION)
            else:
                raise ValueError(variant)
        probs.append(dec["prob_unusable"])
    return np.array(probs, dtype=float)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p >= 0.5
    return {
        "n": len(y), "n_unusable": int(y.sum()),
        "auroc": roc_auc_score(y, p), "ap": average_precision_score(y, p), "brier": brier_score_loss(y, p),
        "sens_at_0.5": float(pred[y == 1].mean()), "fr_at_0.5": float(pred[y == 0].mean()),
    }


def main():
    but = pd.read_csv(BUT_MANIFEST)
    but_stored = pd.read_csv(BUT_STORED)[["segment_id", PRIMARY_COL]]
    cinc_stored = pd.read_csv(CINC_STORED)
    cinc_rec = pd.read_csv(CINC_RECORDS).drop_duplicates("segment_id").set_index("segment_id")
    cinc = cinc_rec.loc[cinc_stored["segment_id"]].reset_index()
    cinc["label"] = cinc_stored["label"].values
    nstdb = pd.read_csv(NSTDB_CSV)
    nstdb_stored = pd.read_csv(NSTDB_STORED)[["segment_id", "p_usable_d1_3B"]]
    d1 = RealD1DecisionModel()

    # 1. Validation: recompute the primary configuration on 10 BUT QDB windows and compare with the stored run
    val_df = but.head(10).merge(but_stored, on="segment_id", how="left", validate="one_to_one")
    assert val_df[PRIMARY_COL].notna().all(), "stored primary predictions missing for validation windows"
    val = pd.DataFrame({"segment_id": val_df["segment_id"], "stored": val_df[PRIMARY_COL].values,
                        "recomputed": score(d1, val_df, "primary")})
    val["abs_diff"] = (val["stored"] - val["recomputed"]).abs()
    val.to_csv(OUT_VAL, index=False)
    print(f"[validation] max |stored - recomputed| on 10 BUT windows = {val['abs_diff'].max():.4f}", flush=True)
    if val["abs_diff"].max() > 0.05:
        raise RuntimeError("Recomputed primary predictions do not match the stored run; stopping before the variants.")

    # 2. Variants on all windows; primary predictions come from the stored runs
    pred_rows = []
    but_p = but.merge(but_stored, on="segment_id", how="left", validate="one_to_one")
    pred_rows.append(pd.DataFrame({"dataset": "BUT_QDB", "segment_id": but_p["segment_id"], "label": but_p["label"].values,
                                   "variant": "primary", "prob_unusable": but_p[PRIMARY_COL].values}))
    pred_rows.append(pd.DataFrame({"dataset": "CinC_2011_SetA", "segment_id": cinc["segment_id"], "label": cinc["label"].values,
                                   "variant": "primary", "prob_unusable": cinc_stored[PRIMARY_COL].values}))
    nst_p = nstdb.merge(nstdb_stored, on="segment_id", how="left", validate="one_to_one")
    pred_rows.append(pd.DataFrame({"dataset": "NSTDB_paired", "segment_id": nst_p["segment_id"], "label": np.nan,
                                   "variant": "primary", "prob_unusable": 1.0 - nst_p["p_usable_d1_3B"].values}))

    # --reuse-but takes the BUT QDB variant predictions from the pre-fix run: BUT QDB images and SQIs were
    # unchanged by the 2026-10-13 renderer fix, so only CinC and NSTDB need recomputation
    reuse_but = None
    if "--reuse-but" in sys.argv:
        reuse_but = pd.read_csv(REUSE_BUT_FROM).query("dataset == 'BUT_QDB' and variant != 'primary'")
    for name, df, label in [("BUT_QDB", but, "label"), ("CinC_2011_SetA", cinc, "label"), ("NSTDB_paired", nstdb, None)]:
        src = {"BUT_QDB": but, "CinC_2011_SetA": cinc, "NSTDB_paired": nstdb}[name]
        for variant in ["image_only", "alt_wording"]:
            if name == "BUT_QDB" and reuse_but is not None:
                prev = reuse_but[reuse_but["variant"] == variant].set_index("segment_id")["prob_unusable"]
                p = prev.loc[src["segment_id"]].values.astype(float)
            else:
                p = score(d1, src, variant)
            pred_rows.append(pd.DataFrame({"dataset": name, "segment_id": src["segment_id"].values,
                                           "label": src[label].values if label else np.nan,
                                           "variant": variant, "prob_unusable": p}))
            pd.concat(pred_rows, ignore_index=True).to_csv(OUT_PRED, index=False)
            print(f"[{name}] {variant} done ({len(p)} windows)", flush=True)

    preds = pd.concat(pred_rows, ignore_index=True)
    preds.to_csv(OUT_PRED, index=False)

    # 3. Summary
    summary = []
    for (dataset, variant), g in preds[preds.dataset != "NSTDB_paired"].groupby(["dataset", "variant"]):
        summary.append({"dataset": dataset, "variant": variant, **metrics(g["label"].values.astype(int), g["prob_unusable"].values)})
    nst = preds[preds.dataset == "NSTDB_paired"].merge(nstdb[["segment_id", "condition", "snr_db", "partner_segment_id"]], on="segment_id")
    for variant, g in nst.groupby("variant"):
        clean = g[g.condition == "clean_reference"].set_index("segment_id")["prob_unusable"]
        noisy = g[g.condition == "noisy"].copy()
        noisy["paired_delta_usable"] = (1 - noisy["prob_unusable"]).values - (1 - clean.loc[noisy["partner_segment_id"]]).values
        summary.append({"dataset": "NSTDB_paired", "variant": variant, "snr_db": np.nan,
                        "mean_p_usable": float(1 - clean.mean()), "n": len(clean), "condition": "clean_reference"})
        for snr, s in noisy.groupby("snr_db"):
            summary.append({"dataset": "NSTDB_paired", "variant": variant, "snr_db": snr,
                            "mean_p_usable": float((1 - s["prob_unusable"]).mean()),
                            "paired_delta_usable": float(s["paired_delta_usable"].mean()),
                            "n": len(s), "condition": "noisy"})
    out = pd.DataFrame(summary)
    out.to_csv(OUT_SUM, index=False)
    print(out.round(3).to_string(index=False))
    print(f"[Saved] {OUT_PRED}\n[Saved] {OUT_SUM}")


if __name__ == "__main__":
    main()
