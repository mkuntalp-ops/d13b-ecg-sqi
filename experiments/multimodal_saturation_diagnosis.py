"""
Why the multimodal net gives one score on CinC 2011 (analysis C of ANALYSIS_PLAN_2026-11-09_additional_analyses.md).

Descriptive; no primary result is changed. The net is fitted on the 988 BUT QDB windows as in the main analysis, and CinC 2011
lead II is rescored with: the original inputs; each stream alone (the other one zeroed by the model's own ablation mode);
the flatline input replaced by the quantization-insensitive measure. The standardised inputs are also reported.

Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python experiments/multimodal_saturation_diagnosis.py
Output: $ECG_RESULTS_DIR/multimodal_saturation_{inputs,scores}.csv (never overwritten)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from cinc_random_draws import COHORT  # noqa: E402
from config import CINC2011_LEADII_FULL, RESULTS_DIR  # noqa: E402
from external_validation import WIN_OUT  # noqa: E402
from features.sqi_extractor import MODEL_INPUT_FEATURE_NAMES as F  # noqa: E402
from models.multimodal_fusion import SinglePassMultimodalClassifier  # noqa: E402

OUT_IN, OUT_SC = RESULTS_DIR / "multimodal_saturation_inputs.csv", RESULTS_DIR / "multimodal_saturation_scores.csv"


def main() -> None:
    for p in (OUT_IN, OUT_SC):
        if p.exists():
            raise FileExistsError(f"{p} exists")
    but, c11 = pd.read_csv(COHORT), pd.read_csv(CINC2011_LEADII_FULL)
    x_but, x_c11 = (np.nan_to_num(d[F].values.astype(float)) for d in (but, c11))
    net = SinglePassMultimodalClassifier(epochs=25, batch_size=32).fit(but.image_path.tolist(), x_but, but.label.values)
    z = (x_c11 - net.sqi_mean) / net.sqi_scale
    inputs = pd.DataFrame({"feature": F, "but_mean": net.sqi_mean, "but_sd": net.sqi_scale, "cinc2011_median_z": np.median(z, axis=0),
                           "cinc2011_max_abs_z": np.abs(z).max(axis=0), "frac_records_abs_z_gt_10": (np.abs(z) > 10).mean(axis=0)})
    robust = pd.read_csv(RESULTS_DIR / "flatline_robust_features.csv").set_index("segment_id").flatline_fraction_robust
    x_rob = x_c11.copy()
    x_rob[:, F.index("flatline_fraction")] = robust.loc[c11.segment_id].values
    imgs, y, rows = c11.image_path.tolist(), c11.label.values, []
    variants = {"original": (x_c11, None), "image_only": (x_c11, "image_only"), "sqi_only": (x_c11, "sqi_only"), "flatline_robust": (x_rob, None)}
    for name, (x, mode) in variants.items():
        p = net.predict_proba(imgs, x, override_mode=mode)
        rows.append({"variant": name, "n": len(p), "n_distinct": len(np.unique(p)), "min": float(p.min()), "max": float(p.max()), "auroc": roc_auc_score(y, p)})
    stored = pd.read_csv(WIN_OUT, low_memory=False)
    stored = stored[stored.dataset == "CinC2011"].Multimodal_Fusion
    print(f"[check] stored multimodal scores on CinC 2011: {stored.nunique()} distinct, min {stored.min()}", flush=True)
    inputs.to_csv(OUT_IN, index=False)
    pd.DataFrame(rows).to_csv(OUT_SC, index=False)
    print(inputs.round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
