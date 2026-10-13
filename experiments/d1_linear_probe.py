"""
Capacity vs adaptation for LiquidAI/d1-3B: a linear probe on frozen d1 representations.

The zero-shot d1-3B lags the task-specific CNNs. If a linear classifier trained on d1's frozen internal
representation (with the same subject-independent folds) reaches CNN-level discrimination, the information is
present and the gap is one of adaptation (prompting / readout); if it does not, the representation itself is
the limit.

One forward pass per window with the primary configuration (strip + SQI text + primary question), during which
two representations are captured with forward hooks:
- decision state: output of the final RMSNorm of the language model at the last position (the slot the
  answer is read from), 2048-d;
- image tokens: mean of the multimodal projector output over the image tokens, 2048-d.
The d1 probability from the same pass is stored; on BUT QDB it is compared with the stored primary run.

Windows: BUT QDB (1,047), CinC 2011 lead II (565 records of the transfer analyses), CinC 2017 subset (all noisy
recordings plus an equal number of random non-noisy recordings, seed 42; all of their windows), and the 12 leads
of the 150 reported CinC 2011 records (d1 probabilities for the 12-lead analysis).

Probe: StandardScaler + L2 logistic regression; C chosen by inner GroupKFold (subjects) from {1e-3 .. 1}.
BUT QDB: the five subject folds of the main analysis (OOF predictions, subject-level bootstrap CI).
Transfer: probe fit on all BUT QDB windows, applied frozen to CinC 2011 and CinC 2017.

Usage: python experiments/d1_linear_probe.py              (extract, then analyse)
       python experiments/d1_linear_probe.py --analysis-only
Outputs: cache/d1_probe/*.npy (features), results/d1_external_predictions.csv, results/d1_probe_oof_predictions.csv,
         results/d1_probe_summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_subject_level_bootstrap_ci
from features.sqi_extractor import SQI_FEATURE_NAMES
from models.d1_real_neural_model import RealD1DecisionModel

RESULTS = PROJECT_ROOT / "results"
FEAT_DIR = PROJECT_ROOT / "cache" / "d1_probe"
CINC11 = PROJECT_ROOT / "cache" / "datasets" / "cinc2011"
CINC17 = PROJECT_ROOT / "cache" / "datasets" / "cinc2017" / "cinc2017_windows.csv"
TWELVE = CINC11 / "twelve_lead" / "cinc2011_twelve_lead_windows.csv"
PRED_OUT = RESULTS / "d1_external_predictions.csv"
OOF_OUT = RESULTS / "d1_probe_oof_predictions.csv"
SUM_OUT = RESULTS / "d1_probe_summary.csv"
C_GRID = {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]}


def window_sets() -> dict:
    but = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv")
    c11 = pd.read_csv(CINC11 / "cinc2011_random_draws_records.csv").drop_duplicates("segment_id")
    mem = pd.read_csv(CINC11 / "cinc2011_random_draws_manifest.csv")
    lab = mem.drop_duplicates("record_name").set_index("record_name")["label"]
    c11["label"] = lab.loc[c11["record_name"].astype(int)].values
    c17 = pd.read_csv(CINC17)
    rec = c17.drop_duplicates("record_name")[["record_name", "label"]]
    noisy = rec[rec.label == 1]
    clean = rec[rec.label == 0].sample(len(noisy), random_state=42)
    c17s = c17[c17.record_name.isin(pd.concat([noisy, clean]).record_name)]
    tw = pd.read_csv(TWELVE)
    first = mem[mem.draw == "first_N_original"].record_name.astype(str)
    tw = tw[tw.record_name.astype(str).isin(first)]
    return {"BUT_QDB": but, "CinC2011_leadII": c11, "CinC2017_subset": c17s, "CinC2011_12lead_firstN": tw}


class Capture:
    """Forward hooks on the final norm and the multimodal projector."""

    def __init__(self, model):
        self.state, self.image = None, None
        norms = [m for n, m in model.named_modules() if n.endswith("embedding_norm")]
        projs = [m for n, m in model.named_modules() if n.endswith("multi_modal_projector")]
        if len(norms) != 1 or len(projs) != 1:
            raise RuntimeError(f"expected one final norm and one projector, found {len(norms)} and {len(projs)}")
        norms[0].register_forward_hook(self._norm)
        projs[0].register_forward_hook(self._proj)

    def _norm(self, _m, _i, out):
        o = out[0] if isinstance(out, tuple) else out
        self.state = o.reshape(-1, o.shape[-1])[-1].float().cpu().numpy()

    def _proj(self, _m, _i, out):
        o = out[0] if isinstance(out, tuple) else out
        self.image = o.reshape(-1, o.shape[-1]).float().mean(0).cpu().numpy()


def extract():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    d1 = RealD1DecisionModel(device="mps")
    cap = Capture(d1.model)
    preds = []
    for name, df in window_sets().items():
        st, im = [], []
        for i, (_, r) in enumerate(df.iterrows()):
            sqi = dict(zip(SQI_FEATURE_NAMES, np.nan_to_num(r[SQI_FEATURE_NAMES].values.astype(float), nan=0.0)))
            cap.state = cap.image = None
            with Image.open(r["image_path"]) as img:
                dec = d1.decide_single(img, sqi, modality="multimodal", segment_id=r["segment_id"])
            if cap.state is None or cap.image is None:
                raise RuntimeError(f"hooks did not fire for {r['segment_id']}")
            st.append(cap.state)
            im.append(cap.image)
            preds.append({"dataset": name, "segment_id": r["segment_id"], "record_name": r["record_name"],
                          "label": int(r["label"]), "prob_unusable": dec["prob_unusable"]})
            if i % 200 == 0:
                print(f"[{name}] {i}/{len(df)}", flush=True)
        np.save(FEAT_DIR / f"{name}_state.npy", np.stack(st).astype(np.float32))
        np.save(FEAT_DIR / f"{name}_image.npy", np.stack(im).astype(np.float32))
        pd.Series(df["segment_id"].values).to_csv(FEAT_DIR / f"{name}_ids.csv", index=False, header=["segment_id"])
        pd.DataFrame(preds).to_csv(PRED_OUT, index=False)
        print(f"[{name}] done ({len(df)} windows)", flush=True)


def probe(groups=None):
    return GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), C_GRID,
                        scoring="roc_auc", cv=GroupKFold(n_splits=4) if groups is not None else 4)


def analyse():
    sets = window_sets()
    but = sets["BUT_QDB"].reset_index(drop=True)
    preds = pd.read_csv(PRED_OUT)
    stored = pd.read_csv(RESULTS / "cohort1000_consolidated_oof_predictions.csv").set_index("segment_id")
    d1_but = preds[preds.dataset == "BUT_QDB"].set_index("segment_id")["prob_unusable"]
    diff = float(np.abs(d1_but.loc[but.segment_id].values - stored.loc[but.segment_id, "prob_LiquidAI_d1_3B_Neural"].values).max())
    print(f"[check] max |d1 recomputed - stored| on BUT QDB = {diff:.4f}")
    rows, oof_rows = [], []
    y = but["label"].values
    for feat in ["state", "image"]:
        X = np.load(FEAT_DIR / f"BUT_QDB_{feat}.npy")
        oof = np.zeros(len(y))
        for k in range(5):
            tr, te = but.fold_id.values != k, but.fold_id.values == k
            g = probe(groups=True).fit(X[tr], y[tr], groups=but.subject_id.values[tr])
            oof[te] = g.predict_proba(X[te])[:, 1]
        ci = compute_subject_level_bootstrap_ci(pd.DataFrame({"prob_unusable": oof, "label": y, "subject_id": but.subject_id}),
                                                n_bootstraps=1000, seed=42)["auroc"]
        rows.append({"features": feat, "dataset": "BUT_QDB (5-fold subject CV)", "n": len(y), "auroc": roc_auc_score(y, oof),
                     "auroc_ci_low": ci["ci_low"], "auroc_ci_high": ci["ci_high"], "ap": average_precision_score(y, oof)})
        oof_rows.append(pd.DataFrame({"segment_id": but.segment_id, "features": feat, "prob_unusable": oof, "label": y}))
        full = probe(groups=True).fit(X, y, groups=but.subject_id.values)
        # CinC 2011 lead II: reported subset and the five random subsets
        c11 = sets["CinC2011_leadII"].reset_index(drop=True)
        p11 = pd.Series(full.predict_proba(np.load(FEAT_DIR / f"CinC2011_leadII_{feat}.npy"))[:, 1], index=c11.record_name.astype(int).values)
        mem = pd.read_csv(CINC11 / "cinc2011_random_draws_manifest.csv")
        for draw, m in mem.groupby("draw"):
            rows.append({"features": feat, "dataset": f"CinC2011 {draw}", "n": len(m),
                         "auroc": roc_auc_score(m.label, p11.loc[m.record_name].values),
                         "ap": average_precision_score(m.label, p11.loc[m.record_name].values)})
        # CinC 2017 subset: record-level maximum over windows
        c17 = sets["CinC2017_subset"].reset_index(drop=True)
        p17 = pd.DataFrame({"record_name": c17.record_name, "label": c17.label,
                            "p": full.predict_proba(np.load(FEAT_DIR / f"CinC2017_subset_{feat}.npy"))[:, 1]})
        r17 = p17.groupby("record_name").agg(label=("label", "first"), p=("p", "max"))
        rows.append({"features": feat, "dataset": "CinC2017 subset (record max)", "n": len(r17),
                     "auroc": roc_auc_score(r17.label, r17.p), "ap": average_precision_score(r17.label, r17.p)})
    # zero-shot d1 on the same sets, for comparison
    ci = compute_subject_level_bootstrap_ci(pd.DataFrame({"prob_unusable": d1_but.loc[but.segment_id].values, "label": y,
                                                          "subject_id": but.subject_id}), n_bootstraps=1000, seed=42)["auroc"]
    rows.append({"features": "zero-shot d1 (primary)", "dataset": "BUT_QDB (5-fold subject CV)", "n": len(y),
                 "auroc": roc_auc_score(y, d1_but.loc[but.segment_id]), "auroc_ci_low": ci["ci_low"], "auroc_ci_high": ci["ci_high"],
                 "ap": average_precision_score(y, d1_but.loc[but.segment_id])})
    # zero-shot CinC 2011 scores from the same file as Table 2d: d1 scores are coarse (bf16) and tie-heavy, so
    # rounding differences of ~5e-7 between stored copies move the AUROC by up to 0.004
    z11 = pd.read_csv(RESULTS / "cinc_random_draws_predictions.csv").set_index("record_name")["LiquidAI_d1_3B_Neural"]
    for draw, m in pd.read_csv(CINC11 / "cinc2011_random_draws_manifest.csv").groupby("draw"):
        rows.append({"features": "zero-shot d1 (primary)", "dataset": f"CinC2011 {draw}", "n": len(m),
                     "auroc": roc_auc_score(m.label, z11.loc[m.record_name].values),
                     "ap": average_precision_score(m.label, z11.loc[m.record_name].values)})
    z17 = preds[preds.dataset == "CinC2017_subset"].groupby("record_name").agg(label=("label", "first"), p=("prob_unusable", "max"))
    rows.append({"features": "zero-shot d1 (primary)", "dataset": "CinC2017 subset (record max)", "n": len(z17),
                 "auroc": roc_auc_score(z17.label, z17.p), "ap": average_precision_score(z17.label, z17.p)})
    out = pd.DataFrame(rows)
    out.to_csv(SUM_OUT, index=False)
    pd.concat(oof_rows).to_csv(OOF_OUT, index=False)
    print(out.round(3).to_string(index=False))
    print(f"[Saved] {SUM_OUT}")


if __name__ == "__main__":
    if "--analysis-only" not in sys.argv:
        extract()
    analyse()
