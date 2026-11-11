"""
HuBERT-ECG probe on the 12 leads of the 998 labelled CinC 2011 Set A records (v2, post hoc, secondary).

The probe is the one of the main analysis (experiments/hubert_ecg_probe.py, authors' preprocessing): standardised L2
logistic regression, C chosen by inner subject-grouped CV, fitted on all 988 BUT QDB windows (single lead in the lead II
slot, other leads zero). No CinC data are used for fitting.

Two inputs:
- per_lead (same protocol as every other model of Table S5): each lead in turn goes in the lead II slot with the other
  leads zero; recording score = lead II, maximum or mean over the 12 leads.
- native_12 (secondary, descriptive, no tests): the record's real 12 leads in one input, one score per record, with the
  leads reordered by name to I, II, III, aVR, aVL, aVF, V1-V6 (the CinC 2011 files store aVF before aVL). The probe
  was fitted on zero-filled single-lead inputs, so this applies it to a kind of input it was not fitted on.

Paired tests, fixed before this script was run (DeLong on 998 records, Holm over these four, per_lead only):
  - HuBERT-ECG probe vs d1-3B image probe, under the maximum and under the mean;
  - HuBERT-ECG probe with the maximum, and with the mean, vs HuBERT-ECG probe on lead II.

Records with a constant lead (signal SD 0) are also excluded in a descriptive analysis, as in Table S35.

Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python experiments/hubert_probe_twelve_lead.py
Output: cache/hubert_ecg_authors/CinC2011_12lead_*.npy (embeddings; extraction resumes),
        $ECG_RESULTS_DIR/hubert_probe_twelve_lead_{window_predictions,native_predictions,summary,comparisons,constant_leads}.csv
        (existing result files are never overwritten)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from config import CINC2011_TWELVE_LEAD_FULL, RESULTS_DIR  # noqa: E402
from d1_probe_twelve_lead import LEAD_II, N_RECORDS, N_WINDOWS, PROBE as D1_PROBE, WIN_OUT as D1_WIN, recording_scores  # noqa: E402
from evaluation.paired_comparisons import delong_paired, holm  # noqa: E402
from external_validation import rec_metrics  # noqa: E402
from feature_cache import load_features, missing_ids, save_set  # noqa: E402
from hubert_ecg_probe import FEAT, FIR, HALF, PRED_OUT, embed, load_model, model_input_authors, probe  # noqa: E402
from lodo_multisource import load_sources  # noqa: E402

HUB = "HuBERT_ECG_probe"
SOURCE_LEADS = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]  # order in the CinC 2011 files
CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]  # order used for native_12
CHUNK = 1000
LEAD_II_TOL = 1e-3
OUT = {k: RESULTS_DIR / f"hubert_probe_twelve_lead_{k}.csv"
       for k in ["window_predictions", "native_predictions", "summary", "comparisons", "constant_leads"]}
NATIVE_SET = "CinC2011_12lead_native"


def manifest() -> pd.DataFrame:
    df = pd.read_csv(CINC2011_TWELVE_LEAD_FULL).sort_values(["record_name", "lead_idx"]).reset_index(drop=True)
    if df.segment_id.nunique() != N_WINDOWS or df.record_name.nunique() != N_RECORDS:
        raise ValueError(f"expected {N_WINDOWS} lead windows from {N_RECORDS} records")
    order = df.groupby("record_name").lead.agg(list)
    if not all(o == SOURCE_LEADS for o in order):
        raise ValueError(f"lead order differs from {SOURCE_LEADS}")
    return df


def native_input(twelve: np.ndarray) -> np.ndarray:
    """Authors' preprocessing on a real 12 x 5,000 array -> two flattened, decimated inputs of 6,000 samples."""
    if twelve.shape != (12, 2 * HALF):
        raise ValueError(f"expected (12, {2 * HALF}), got {twelve.shape}")
    x = signal.filtfilt(FIR, [1.0], np.asarray(twelve, dtype=np.float64), axis=1)
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    x = 2 * (x - lo) / (hi - lo + 1e-8) - 1
    return np.stack([signal.decimate(x[:, s:s + HALF].reshape(-1), 5) for s in (0, HALF)]).astype(np.float32)


def check_native_input(df: pd.DataFrame) -> None:
    """On a zero-filled array, native_input must equal the single-lead construction of the main analysis."""
    lead = np.load(df.signal_path.iloc[1])
    zero_filled = np.zeros((12, 2 * HALF))
    zero_filled[1] = lead[:2 * HALF]
    worst = float(np.abs(native_input(zero_filled) - model_input_authors(lead)).max())
    print(f"[check] max |native_input - single-lead input| on a zero-filled array = {worst:.2e}", flush=True)
    if worst > 1e-5:
        raise RuntimeError("native_input does not reproduce the main-analysis preprocessing")


@torch.no_grad()
def extract(df: pd.DataFrame) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(device)
    todo = df[df.segment_id.isin(set(missing_ids(FEAT, None, df.segment_id)))]
    print(f"[per_lead] {len(df) - len(todo)} windows cached, {len(todo)} to embed", flush=True)
    part = len(list(FEAT.glob("CinC2011_12lead_part*_ids.csv")))
    for start in range(0, len(todo), CHUNK):
        chunk = todo.iloc[start:start + CHUNK]
        save_set(FEAT, f"CinC2011_12lead_part{part:03d}", chunk.segment_id.tolist(),
                 {None: embed(model, chunk.signal_path.tolist(), device)})
        part += 1
        print(f"[per_lead] {start + len(chunk)}/{len(todo)}", flush=True)
    records = sorted(df.record_name.unique())
    native_ids = [f"cinc2011_{r}_native12" for r in records]
    if missing_ids(FEAT, None, native_ids):
        by_record = {r: g.set_index("lead").loc[CANONICAL_LEADS, "signal_path"].tolist() for r, g in df.groupby("record_name")}
        out = []
        for i in range(0, len(records), 32):
            batch = records[i:i + 32]
            xs = np.concatenate([native_input(np.stack([np.load(p)[:2 * HALF] for p in by_record[r]])) for r in batch])
            h = model(torch.from_numpy(xs).to(device)).last_hidden_state.mean(dim=1).float().cpu().numpy()
            out.append(h.reshape(-1, 2, h.shape[-1]).mean(axis=1))
        save_set(FEAT, NATIVE_SET, native_ids, {None: np.concatenate(out)})
        print(f"[native_12] {len(records)} records embedded", flush=True)


def fit_probe():
    but = load_sources()["BUT_QDB"]
    if len(but) != 988 or but.unit.nunique() != 15:
        raise ValueError(f"expected 988 BUT QDB windows from 15 subjects, got {len(but)} / {but.unit.nunique()}")
    fitted = probe().fit(load_features(FEAT, None, but.segment_id), but.label.values, groups=but.unit.values)
    print(f"[probe] C = {fitted.best_params_['logisticregression__C']}", flush=True)
    return fitted


def check_lead_ii(w: pd.DataFrame) -> None:
    """Lead II scores must equal the transfer scores of the main analysis."""
    stored = pd.read_csv(PRED_OUT, dtype={"record": str})
    stored = stored[(stored.setting == "but_only_external") & (stored.dataset == "CinC2011")].set_index("record")
    l2 = w[w.lead_idx == LEAD_II].assign(record=lambda d: d.record_name.astype(str)).set_index("record")
    if len(stored) != N_RECORDS or set(stored.index) != set(l2.index):
        raise ValueError("lead II records differ from the stored HuBERT-ECG transfer scores")
    worst = float(np.abs(l2.prob_unusable - stored.loc[l2.index, "prob_unusable"]).max())
    print(f"[check] max |lead II - main analysis| = {worst:.2e}; AUROC {roc_auc_score(l2.label, l2.prob_unusable):.4f}", flush=True)
    if worst > LEAD_II_TOL:
        raise RuntimeError("the probe does not reproduce the main-analysis lead II scores")


def main() -> None:
    for path in OUT.values():
        if path.exists():
            raise FileExistsError(f"{path} exists; move it before re-running")
    df = manifest()
    check_native_input(df)
    extract(df)
    fitted = fit_probe()
    p = fitted.predict_proba(load_features(FEAT, None, df.segment_id))[:, 1]
    if not np.isfinite(p).all():
        raise ValueError("non-finite probe scores")
    w = df[["segment_id", "record_name", "lead_idx", "label"]].assign(prob_unusable=p)
    check_lead_ii(w)

    records = df.drop_duplicates("record_name").sort_values("record_name")
    native_ids = [f"cinc2011_{r}_native12" for r in records.record_name]
    native = records[["record_name", "label"]].assign(
        prob_unusable=fitted.predict_proba(load_features(FEAT, None, native_ids))[:, 1]).set_index("record_name")

    merged = w.assign(**{HUB: p})
    merged[D1_PROBE] = merged.segment_id.map(pd.read_csv(D1_WIN).set_index("segment_id")["prob_unusable"])
    rec = recording_scores(merged, [HUB, D1_PROBE])

    summary = [{"draw": "all_998", "aggregation": a, "model": HUB, **rec_metrics(r.label, r[HUB])} for a, r in rec.items()]
    summary.append({"draw": "all_998", "aggregation": "native_12", "model": HUB, **rec_metrics(native.label, native.prob_unusable)})
    summary = pd.DataFrame(summary)

    rows = []
    for agg in ["max_12", "mean_12"]:
        r = rec[agg]
        for name, other in [(f"HuBERT {agg} vs d1 image probe {agg}", r[D1_PROBE]),
                            (f"HuBERT {agg} vs HuBERT lead_II", rec["lead_II"].loc[r.index, HUB])]:
            d, lo, hi, pv = delong_paired(r.label.values, r[HUB].values, other.values)
            rows.append({"comparison": name, "auroc_hubert": roc_auc_score(r.label, r[HUB]), "auroc_other": roc_auc_score(r.label, other),
                         "delta": d, "ci_low": lo, "ci_high": hi, "p": pv})
    comparisons = pd.DataFrame(rows)
    comparisons["p_holm"] = holm(comparisons.p.values)

    flat = df.assign(c=df.signal_std == 0).groupby("record_name").c.sum().loc[rec["lead_II"].index]
    keep = flat.values == 0
    constant = pd.DataFrame(
        [{"aggregation": a, "model": HUB, "n_all": N_RECORDS, "auroc_all": roc_auc_score(r.label, r[HUB]), "n_without_constant": int(keep.sum()),
          "n_pos_without_constant": int(r.label[keep].sum()), "auroc_without_constant": roc_auc_score(r.label[keep], r[HUB][keep])}
         for a, r in rec.items()]
        + [{"aggregation": "native_12", "model": HUB, "n_all": N_RECORDS, "auroc_all": roc_auc_score(native.label, native.prob_unusable),
            "n_without_constant": int(keep.sum()), "n_pos_without_constant": int(native.label[keep].sum()),
            "auroc_without_constant": roc_auc_score(native.label[keep], native.prob_unusable[keep])}])

    w.to_csv(OUT["window_predictions"], index=False)
    native.reset_index().to_csv(OUT["native_predictions"], index=False)
    summary.to_csv(OUT["summary"], index=False)
    comparisons.to_csv(OUT["comparisons"], index=False)
    constant.to_csv(OUT["constant_leads"], index=False)
    print(summary[["aggregation", "n", "n_pos", "auroc", "auroc_ci_low", "auroc_ci_high", "ap", "sens_at_0.5", "fr_at_0.5"]].round(4).to_string(index=False))
    print(comparisons.to_string(index=False))
    print(constant[["aggregation", "auroc_all", "n_without_constant", "auroc_without_constant"]].round(4).to_string(index=False))
    print("[Saved] " + ", ".join(str(p.name) for p in OUT.values()), flush=True)


if __name__ == "__main__":
    main()
