"""
d1-3B image-token probe on all 12 leads of the 998 labelled CinC 2011 Set A records (v2, post hoc, secondary).

The probe is the one of the main analysis (experiments/d1_linear_probe.py): standardised L2 logistic regression, C chosen
by inner subject-grouped CV, fitted on all 988 BUT QDB windows. It is applied frozen to the image-token features cached
by experiments/d1_twelve_lead_full.py. No d1-3B forward pass and no CinC data are used for fitting.

Recording scores use the three rules of Table S5, which were chosen earlier for the other models: lead II, maximum over
the 12 leads, mean over the 12 leads.

Paired tests, fixed before this script was run (DeLong on 998 records, Holm over these four):
  - probe vs the best other model of Table S5, under the maximum and under the mean (same aggregation for both models);
  - probe with the maximum vs probe on lead II, and probe with the mean vs probe on lead II.

Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python experiments/d1_probe_twelve_lead.py
        ECG_RESULTS_DIR=results/v2_2026-11 python experiments/d1_probe_twelve_lead.py --constant-leads
Output: $ECG_RESULTS_DIR/d1_probe_twelve_lead_window_predictions.csv, d1_probe_twelve_lead_summary.csv,
        d1_probe_twelve_lead_comparisons.csv; with --constant-leads, d1_probe_twelve_lead_constant_leads.csv
        (descriptive, added after the first results were seen). Existing files are never overwritten.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from config import CINC2011_TWELVE_LEAD_FULL, RESULTS_DIR  # noqa: E402
from d1_linear_probe import FEAT_DIR, TRANSFER_OUT, probe, window_sets  # noqa: E402
from evaluation.paired_comparisons import delong_paired, holm  # noqa: E402
from external_validation import C12_OUT, D1, D12_PRED, MODELS, W12_OUT, rec_metrics  # noqa: E402
from feature_cache import load_features  # noqa: E402

FEATURE = "image"
PROBE = "d1_image_probe"
N_WINDOWS, N_RECORDS = 11976, 998
LEAD_II = 1
TOL = 1e-6
WIN_OUT = RESULTS_DIR / "d1_probe_twelve_lead_window_predictions.csv"
SUM_OUT = RESULTS_DIR / "d1_probe_twelve_lead_summary.csv"
CMP_OUT = RESULTS_DIR / "d1_probe_twelve_lead_comparisons.csv"
CONST_OUT = RESULTS_DIR / "d1_probe_twelve_lead_constant_leads.csv"


def score_windows() -> pd.DataFrame:
    """Fit the probe on all BUT QDB windows and score every CinC 2011 lead window."""
    but = window_sets()["BUT_QDB"].reset_index(drop=True)
    if len(but) != 988 or but.subject_id.nunique() != 15:
        raise ValueError(f"expected 988 BUT QDB windows from 15 subjects, got {len(but)} / {but.subject_id.nunique()}")
    fitted = probe(groups=True).fit(load_features(FEAT_DIR, FEATURE, but.segment_id), but.label.values,
                                    groups=but.subject_id.values)
    print(f"[probe] C = {fitted.best_params_['logisticregression__C']}", flush=True)
    df = pd.read_csv(CINC2011_TWELVE_LEAD_FULL).reset_index(drop=True)
    if df.segment_id.nunique() != N_WINDOWS or df.record_name.nunique() != N_RECORDS:
        raise ValueError(f"expected {N_WINDOWS} lead windows from {N_RECORDS} records")
    if set(df.groupby("record_name").size()) != {12} or df.groupby("record_name").label.nunique().max() != 1:
        raise ValueError("each record must have 12 lead windows with one label")
    p = fitted.predict_proba(load_features(FEAT_DIR, FEATURE, df.segment_id))[:, 1]
    if not np.isfinite(p).all():
        raise ValueError("non-finite probe scores")
    return df[["segment_id", "record_name", "lead_idx", "label"]].assign(prob_unusable=p)


def check_lead_ii(w: pd.DataFrame) -> None:
    """Lead II scores must equal the transfer scores of the main analysis."""
    stored = pd.read_csv(TRANSFER_OUT, dtype={"record_name": str})
    stored = stored[(stored.features == FEATURE) & (stored.dataset == "CinC2011")].set_index("record_name")
    l2 = w[w.lead_idx == LEAD_II].assign(record_name=lambda d: d.record_name.astype(str)).set_index("record_name")
    if len(stored) != N_RECORDS or set(stored.index) != set(l2.index):
        raise ValueError("lead II records differ from the stored transfer scores")
    worst = float(np.abs(l2.prob_unusable - stored.loc[l2.index, "prob_unusable"]).max())
    print(f"[check] max |lead II - main analysis| = {worst:.2e}", flush=True)
    if worst > TOL or not (l2.label == stored.loc[l2.index, "label"]).all():
        raise RuntimeError("the probe does not reproduce the main-analysis lead II scores")


def recording_scores(windows: pd.DataFrame, cols: list) -> dict:
    """Recording-level scores per aggregation rule; every frame is indexed and sorted by record name."""
    out = {"lead_II": windows[windows.lead_idx == LEAD_II].set_index("record_name")[cols + ["label"]].sort_index()}
    for agg in ["max", "mean"]:
        out[f"{agg}_12"] = windows.groupby("record_name").agg({**{c: agg for c in cols}, "label": "first"}).sort_index()
    for name, rec in out.items():
        if len(rec) != N_RECORDS or rec[cols].isna().any().any():
            raise ValueError(f"{name}: expected {N_RECORDS} complete records")
    return out


def compare(rec: dict) -> pd.DataFrame:
    summary = pd.read_csv(C12_OUT)
    summary = summary[summary.draw == "all_998"]
    pairs = []
    for agg in ["max_12", "mean_12"]:
        best = summary[summary.aggregation == agg].sort_values("auroc").iloc[-1].model
        pairs.append((f"probe {agg} vs best other ({best}, {agg})", rec[agg].label, rec[agg][PROBE], rec[agg][best]))
        pairs.append((f"probe {agg} vs probe lead_II", rec[agg].label, rec[agg][PROBE], rec["lead_II"].loc[rec[agg].index, PROBE]))
    rows = []
    for name, y, a, b in pairs:
        d, lo, hi, p = delong_paired(y.values, a.values, b.values)
        rows.append({"comparison": name, "auroc_probe": roc_auc_score(y, a), "auroc_other": roc_auc_score(y, b),
                     "delta": d, "ci_low": lo, "ci_high": hi, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.values)
    return out


def constant_lead_sensitivity() -> None:
    """Descriptive, no tests: how much of each 12-lead AUROC depends on records with at least one constant lead.

    A lead is constant when its signal SD is 0, the definition of experiments/cinc2011_constant_lead_check.py. The
    number of constant leads per record is also scored as a trivial one-variable reference.
    """
    if CONST_OUT.exists():
        raise FileExistsError(f"{CONST_OUT} exists; move it before re-running")
    manifest = pd.read_csv(CINC2011_TWELVE_LEAD_FULL)
    n_constant = manifest.assign(c=manifest.signal_std == 0).groupby("record_name").c.sum().sort_index()
    w = pd.read_csv(WIN_OUT)
    others = pd.read_csv(W12_OUT, low_memory=False).set_index("segment_id")
    zero_shot = pd.read_csv(D12_PRED).set_index("segment_id")["prob_unusable"]
    for m in MODELS:
        w[m] = w.segment_id.map(others[m])
    w[D1] = w.segment_id.map(zero_shot)
    w[PROBE] = w.prob_unusable
    cols = MODELS + [D1, PROBE]
    rec = recording_scores(w, cols)
    flat = n_constant.loc[rec["lead_II"].index]
    keep = flat.values == 0
    y = rec["lead_II"].label.values
    print(f"[constant] {int((~keep).sum())} of {N_RECORDS} records have a constant lead; "
          f"{int(y[~keep].sum())} of them unacceptable; remaining: {int(keep.sum())} records, {int(y[keep].sum())} unacceptable", flush=True)
    rows = [{"aggregation": "n_constant_leads", "model": "number of constant leads (trivial reference)", "n_all": N_RECORDS,
             "auroc_all": roc_auc_score(y, flat.values), "n_without_constant": int(keep.sum()),
             "n_pos_without_constant": int(y[keep].sum()), "auroc_without_constant": np.nan}]
    for agg, r in rec.items():
        for m in cols:
            rows.append({"aggregation": agg, "model": m, "n_all": len(r), "auroc_all": roc_auc_score(r.label, r[m]),
                         "n_without_constant": int(keep.sum()), "n_pos_without_constant": int(y[keep].sum()),
                         "auroc_without_constant": roc_auc_score(r.label[keep], r[m][keep])})
    out = pd.DataFrame(rows)
    out.to_csv(CONST_OUT, index=False)
    print(out[out.aggregation != "lead_II"].round(3).to_string(index=False))
    print(f"[Saved] {CONST_OUT}", flush=True)


def main() -> None:
    if "--constant-leads" in sys.argv:
        constant_lead_sensitivity()
        return
    for path in (WIN_OUT, SUM_OUT, CMP_OUT):
        if path.exists():
            raise FileExistsError(f"{path} exists; move it before re-running")
    w = score_windows()
    check_lead_ii(w)
    others = pd.read_csv(W12_OUT, low_memory=False).set_index("segment_id")
    zero_shot = pd.read_csv(D12_PRED).set_index("segment_id")["prob_unusable"]
    merged = w.copy()
    for m in MODELS:
        merged[m] = merged.segment_id.map(others[m])
    merged[D1] = merged.segment_id.map(zero_shot)
    merged[PROBE] = merged.prob_unusable
    rec = recording_scores(merged, MODELS + [D1, PROBE])
    summary = pd.DataFrame([{"draw": "all_998", "aggregation": a, "model": PROBE, **rec_metrics(r.label, r[PROBE])}
                            for a, r in rec.items()])
    comparisons = compare(rec)
    w.to_csv(WIN_OUT, index=False)
    summary.to_csv(SUM_OUT, index=False)
    comparisons.to_csv(CMP_OUT, index=False)
    print(summary[["aggregation", "n", "n_pos", "auroc", "auroc_ci_low", "auroc_ci_high", "ap", "sens_at_0.5", "fr_at_0.5"]].round(4).to_string(index=False))
    print(comparisons.round(4).to_string(index=False))
    print(f"[Saved] {WIN_OUT}\n[Saved] {SUM_OUT}\n[Saved] {CMP_OUT}", flush=True)


if __name__ == "__main__":
    main()
