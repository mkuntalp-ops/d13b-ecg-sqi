"""
Sensitivity analysis for the BUT QDB quota of subject 105 (ANALYSIS_PLAN_2026-11-16_quota_sensitivity.md).

The primary cohort drew its quota over all annotated intervals and removed the windows outside the official intervals
afterwards, which left subject 105 with 1 unusable window instead of 54. Here the quota is drawn inside the official
intervals. Post hoc, added after review; the primary analysis is not changed.

Stages (run in this order; ECG_RESULTS_DIR must point to the sensitivity directory, never to the primary one):
    setup     cohort, checks, extraction of the added windows, manifest with the primary folds carried over
    d1        zero-shot d1-3B score and representations of the added windows (MPS)
    hubert    HuBERT-ECG embeddings of the added windows, cross-validated and BUT-only probe (no LODO)
    external  BUT-only task-specific models scored on CinC 2011 lead II and CinC 2017 (external_validation.score_windows)
    analyse   comparison with the primary analysis (reads both directories, writes only into the sensitivity one)
Stages taken unchanged from other scripts are listed in the plan and in experiments/quota_sensitivity_queue.sh.

Usage: ECG_RESULTS_DIR=results/v2_2026-11_quota_sensitivity python experiments/quota_sensitivity.py <stage>
       ... hubert --check-primary <scratch dir>   (reproduce the stored primary predictions; writes only to scratch)
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "evaluation"))

from config import RESULTS_DIR  # noqa: E402

PRIMARY = (ROOT / "results" / "v2_2026-11").resolve()
SET_NAME = "BUT_QDB_quota_sens_2026-11-16"
MANIFEST = "manifest_butqdb_1000_cohort.csv"
EXPECTED_ADDED = 53
KEY = ["record_name", "start_sample", "end_sample"]


# ---------- cohort rule (pure functions, covered by tests/test_v2.py) ----------
def sensitivity_cohort(candidates: pd.DataFrame, inside: np.ndarray, sampler) -> pd.DataFrame:
    """Quota sample drawn from the candidate windows inside the official intervals."""
    return sampler(candidates[np.asarray(inside, dtype=bool)].reset_index(drop=True))


def cohort_difference(primary: pd.DataFrame, cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(windows only in the sensitivity cohort, windows only in the primary cohort), matched by segment id."""
    added = cohort[~cohort.segment_id.isin(primary.segment_id)]
    removed = primary[~primary.segment_id.isin(cohort.segment_id)]
    return added, removed


def check_difference(added: pd.DataFrame, removed: pd.DataFrame) -> None:
    """The plan allows exactly one difference: 53 added Class 3 windows of subject 105. Anything else stops the run."""
    problems = []
    if len(removed):
        problems.append(f"{len(removed)} primary windows are not in the sensitivity cohort")
    if len(added) != EXPECTED_ADDED:
        problems.append(f"{len(added)} added windows, expected {EXPECTED_ADDED}")
    if len(added) and (set(added.subject_id.astype(str)) != {"105"} or set(added.label.astype(int)) != {1}):
        problems.append("added windows are not all unusable windows of subject 105")
    if problems:
        raise ValueError("sensitivity cohort differs from the plan: " + "; ".join(problems))


def carry_folds(cohort: pd.DataFrame, primary: pd.DataFrame) -> pd.Series:
    """Fold of each window = fold of its subject in the primary analysis (GroupKFold is not re-run)."""
    per_subject = primary.groupby(primary.subject_id.astype(str)).fold_id.agg(["nunique", "first"])
    if (per_subject["nunique"] != 1).any():
        raise ValueError("a subject has windows in more than one primary fold")
    folds = cohort.subject_id.astype(str).map(per_subject["first"])
    if folds.isna().any():
        raise ValueError("a subject of the sensitivity cohort has no primary fold")
    return folds.astype(int)


# ---------- helpers ----------
def guard() -> None:
    if "ECG_RESULTS_DIR" not in os.environ or RESULTS_DIR in (PRIMARY, (ROOT / "results").resolve()):
        raise SystemExit("Set ECG_RESULTS_DIR to the sensitivity directory; refusing to write into a primary directory.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def added_windows() -> pd.DataFrame:
    m = pd.read_csv(RESULTS_DIR / MANIFEST)
    prim = pd.read_csv(PRIMARY / MANIFEST)
    return m[~m.segment_id.isin(prim.segment_id)].reset_index(drop=True)


# ---------- stage: setup ----------
def setup():
    guard()
    from butqdb_official_split import official_mask
    from data.extract_butqdb_1000_cohort import build_candidate_windows, process_single_window, sample_stratified_cohort
    from features.sqi_extractor import SQI_FEATURE_NAMES

    primary = pd.read_csv(PRIMARY / MANIFEST)
    cand = build_candidate_windows()
    inside = official_mask(cand)
    cohort = sensitivity_cohort(cand, inside, sample_stratified_cohort)
    added, removed = cohort_difference(primary, cohort)
    print(f"[cohort] candidates {len(cand)} (inside official intervals {int(inside.sum())}); sensitivity cohort {len(cohort)} "
          f"windows, {int(cohort.label.sum())} unusable; added {len(added)}, removed {len(removed)}")
    added[["segment_id", "subject_id", "record_name", "start_sample", "end_sample", "original_quality_class", "label"]].to_csv(
        RESULTS_DIR / "quota_sensitivity_added_windows.csv", index=False)
    removed[["segment_id", "subject_id", "record_name", "start_sample", "end_sample", "label"]].to_csv(
        RESULTS_DIR / "quota_sensitivity_removed_windows.csv", index=False)
    check_difference(added, removed)

    # cache files of the primary windows must not change: hash before and after
    files = [Path(p) for p in list(primary.signal_path) + list(primary.image_path)]
    before = {str(f): sha(f) for f in files}

    # check of the code path: SQIs recomputed from the cached signals equal the primary manifest
    redo = pd.DataFrame([process_single_window(r) for r in primary[KEY + ["subject_id", "original_quality_class", "label",
                                                                         "mid_sample", "segment_id"]].to_dict("records")])
    diff = float(np.abs(redo[SQI_FEATURE_NAMES].values.astype(float) - primary[SQI_FEATURE_NAMES].values.astype(float)).max())
    print(f"[check] max |SQI recomputed - primary manifest| over the 988 primary windows = {diff:.3e}")
    if not diff < 1e-9:
        raise ValueError("the SQI code path does not reproduce the primary manifest")

    new_rows = pd.DataFrame([process_single_window(r) for r in added.to_dict("records")])
    after = {str(f): sha(f) for f in files}
    if before != after:
        raise RuntimeError("a cache file of the primary cohort changed")
    print(f"[check] {len(files)} cache files of the primary cohort unchanged (SHA-256)")

    new_rows["subject_id"] = new_rows["subject_id"].astype(primary.subject_id.dtype)
    out = pd.concat([primary, new_rows[[c for c in primary.columns if c != "fold_id"]]], ignore_index=True)
    out["fold_id"] = carry_folds(out, primary)
    out = out.sort_values(["subject_id", "record_name", "start_sample"]).reset_index(drop=True)[primary.columns]
    if out.segment_id.duplicated().any() or out[SQI_FEATURE_NAMES].isna().any().any():
        raise ValueError("duplicate segment ids or missing SQIs in the sensitivity manifest")
    pd.testing.assert_frame_equal(out.set_index("segment_id").loc[primary.segment_id], primary.set_index("segment_id"),
                                  check_exact=True)  # every primary row, fold included, is unchanged
    out.to_csv(RESULTS_DIR / MANIFEST, index=False)
    rep = out.groupby(["fold_id", "label"]).size().unstack(fill_value=0)
    print(rep.to_string())
    print(f"[Saved] {RESULTS_DIR / MANIFEST}: {len(out)} windows, {int(out.label.sum())} unusable, "
          f"{out.subject_id.nunique()} subjects; subject 105 unusable: {int(out[out.subject_id.astype(str) == '105'].label.sum())}")


# ---------- stage: d1 ----------
def d1():
    guard()
    from PIL import Image
    from d1_linear_probe import FEAT_DIR, Capture
    from feature_cache import missing_ids, save_set
    from features.sqi_extractor import SQI_FEATURE_NAMES
    from models.d1_real_neural_model import RealD1DecisionModel

    add = added_windows()
    todo = set(missing_ids(FEAT_DIR, "image", add.segment_id)) | set(missing_ids(FEAT_DIR, "state", add.segment_id))
    raw_out, ext_out = RESULTS_DIR / "raw_d1_neural_cohort1000_predictions.csv", RESULTS_DIR / "d1_external_predictions.csv"
    if raw_out.exists() or ext_out.exists():
        raise SystemExit(f"{raw_out.name} or {ext_out.name} exists; not overwriting")
    if len(todo) != len(add):
        raise SystemExit(f"{len(add) - len(todo)} of the added windows already have cached d1 features; inspect before rerunning")
    model = RealD1DecisionModel(device="mps")
    cap = Capture(model.model)
    rows, st, im = [], [], []
    for _, r in add.iterrows():
        sqi = dict(zip(SQI_FEATURE_NAMES, np.nan_to_num(r[SQI_FEATURE_NAMES].values.astype(float), nan=0.0)))
        cap.state = cap.image = None
        with Image.open(r["image_path"]) as img:
            dec = model.decide_single(img, sqi, modality="multimodal", segment_id=r["segment_id"])
        if cap.state is None or cap.image is None:
            raise RuntimeError(f"hooks did not fire for {r['segment_id']}")
        st.append(cap.state)
        im.append(cap.image)
        rows.append({"segment_id": r["segment_id"], "dataset": "BUT_QDB", "subject_id": r["subject_id"],
                     "record_name": r["record_name"], "label": int(r["label"]), "fold_id": int(r["fold_id"]),
                     "backend": dec["backend"], "model_name": dec["model_name"], "prob_unusable": dec["prob_unusable"],
                     "prob_usable": dec["prob_usable"], "choice": dec["choice"], "confidence": dec["confidence"],
                     "input_tokens": dec["input_tokens"], "output_tokens": dec["output_tokens"],
                     "latency_ms": dec["latency_ms"]})
    save_set(FEAT_DIR, SET_NAME, add.segment_id.tolist(), {"state": np.stack(st), "image": np.stack(im)})
    new = pd.DataFrame(rows)
    raw = pd.read_csv(PRIMARY / "raw_d1_neural_cohort1000_predictions.csv")
    pd.concat([raw, new[raw.columns]], ignore_index=True).to_csv(raw_out, index=False)
    ext = pd.read_csv(PRIMARY / "d1_external_predictions.csv")
    pd.concat([ext, new.assign(dataset="BUT_QDB")[ext.columns]], ignore_index=True).to_csv(ext_out, index=False)
    print(f"[d1] {len(new)} added windows scored; mean prob_unusable {new.prob_unusable.mean():.3f}. [Saved] {raw_out.name}, {ext_out.name}")


# ---------- stage: hubert ----------
def hubert_predictions(but: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Cross-validated and BUT-only external HuBERT-ECG probe, as in hubert_ecg_probe.analyse() without the LODO part."""
    from hubert_ecg_probe import FEAT, probe
    from feature_cache import load_features
    from lodo_multisource import load_sources
    src = load_sources()
    src["BUT_QDB"] = but
    F = {k: load_features(FEAT, None, src[k].segment_id) for k in src}
    oof = np.zeros(len(but))
    for k in range(5):
        tr, te = folds != k, folds == k
        oof[te] = probe().fit(F["BUT_QDB"][tr], but.label.values[tr], groups=but.unit.values[tr]).predict_proba(F["BUT_QDB"][te])[:, 1]
    cols = ["dataset", "segment_id", "unit", "record", "label"]
    preds = [but[cols].assign(setting="butqdb_cv", prob_unusable=oof)]
    full = probe().fit(F["BUT_QDB"], but.label.values, groups=but.unit.values)
    print(f"[hubert] BUT-only probe: C = {full.best_params_}")
    for k in ["CinC2011", "CinC2017"]:
        preds.append(src[k][cols].assign(setting="but_only_external", prob_unusable=full.predict_proba(F[k])[:, 1]))
    return pd.concat(preds, ignore_index=True)


def _but_source(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest.assign(dataset="BUT_QDB", unit=manifest["subject_id"].astype(str),
                           record=manifest["record_name"].astype(str)).reset_index(drop=True)


def hubert():
    if "--check-primary" in sys.argv:  # the new code must reproduce the stored primary predictions
        scratch = Path(sys.argv[sys.argv.index("--check-primary") + 1])
        prim = pd.read_csv(PRIMARY / MANIFEST)
        P = hubert_predictions(_but_source(prim), prim.fold_id.values)
        ref = pd.read_csv(PRIMARY / "hubert_probe_predictions.csv", low_memory=False)
        ref = ref[ref.setting.isin(["butqdb_cv", "but_only_external"])].set_index(["setting", "segment_id"]).prob_unusable
        d = float(np.abs(P.set_index(["setting", "segment_id"]).prob_unusable.loc[ref.index].values - ref.values).max())
        scratch.mkdir(parents=True, exist_ok=True)
        P.to_csv(scratch / "hubert_primary_recomputed.csv", index=False)
        print(f"[check-primary] HuBERT-ECG probe: {len(ref)} predictions, max |recomputed - stored| = {d:.3e}")
        sys.exit(0 if d < 1e-6 else 1)
    guard()
    import torch
    from hubert_ecg_probe import FEAT, embed, load_model
    from feature_cache import missing_ids, save_set
    add = added_windows()
    todo = missing_ids(FEAT, None, add.segment_id)
    if todo:
        if len(todo) != len(add):
            raise SystemExit("some of the added windows already have cached HuBERT-ECG embeddings; inspect before rerunning")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        E = embed(load_model(device), add["signal_path"].tolist(), device)
        save_set(FEAT, SET_NAME, add.segment_id.tolist(), {None: E})
        print(f"[hubert] embedded {E.shape}")
    out = RESULTS_DIR / "hubert_probe_predictions.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    m = pd.read_csv(RESULTS_DIR / MANIFEST)
    hubert_predictions(_but_source(m), m.fold_id.values).to_csv(out, index=False)
    print(f"[Saved] {out}")


# ---------- stage: external ----------
def external():
    guard()
    import external_validation as ev
    if ev.WIN_OUT.exists():
        raise SystemExit(f"{ev.WIN_OUT} exists; not overwriting")
    ev.score_windows()


# ---------- stage: analyse ----------
def _families(results_dir: Path) -> dict:
    import paired_comparisons as pc
    pc.RESULTS = results_dir
    return {k: v[1] for k, v in pc.families().items()}


def analyse():
    guard()
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from paired_comparisons import delong_paired
    sens, prim = _families(RESULTS_DIR), _families(PRIMARY)
    names = {"BUT_QDB (5-fold subject CV)": "BUT QDB", "CinC2011 lead II (998 records)": "CinC 2011 lead II",
             "CinC2017 (8,512 recordings)": "CinC 2017"}
    rows, auc = [], {}
    for fam, short in names.items():
        s, p = sens[fam], prim[fam]
        models = [c for c in p.columns if c not in ("unit", "y", "cluster")]
        if models != [c for c in s.columns if c not in ("unit", "y", "cluster")]:
            raise ValueError(f"{fam}: model columns differ between the two analyses")
        for m in models:
            a_s, a_p = roc_auc_score(s.y, s[m]), roc_auc_score(p.y, p[m])
            auc[(short, m)] = (a_p, a_s)
            row = {"set": short, "model": m, "n_primary": len(p), "n_sensitivity": len(s), "auroc_primary": a_p,
                   "auroc_sensitivity": a_s, "delta": a_s - a_p}
            if short != "BUT QDB":  # same records, two versions of the model: paired DeLong (descriptive, unadjusted)
                q = s.set_index("unit").loc[p.unit]
                if not (q.y.values == p.y.values).all():
                    raise ValueError(f"{fam}: labels differ")
                if np.array_equal(q[m].values, p[m].values):
                    row.update(delta_ci_lo=0.0, delta_ci_hi=0.0, p_delong=np.nan, identical_scores=True)
                else:
                    d, lo, hi, pv = delong_paired(p.y.values.astype(int), q[m].values, p[m].values)
                    row.update(delta_ci_lo=lo, delta_ci_hi=hi, p_delong=pv, identical_scores=False)
            rows.append(row)
    tab = pd.DataFrame(rows)
    for short in names.values():
        for ver in ("primary", "sensitivity"):
            sel = tab.set == short
            tab.loc[sel, f"rank_{ver}"] = tab.loc[sel, f"auroc_{ver}"].rank(ascending=False, method="min")
    tab.to_csv(RESULTS_DIR / "quota_sensitivity_auroc.csv", index=False)

    models = tab[tab.set == "BUT QDB"].model.tolist()
    rho = []
    for a, b in [("BUT QDB", "CinC 2011 lead II"), ("BUT QDB", "CinC 2017"), ("CinC 2011 lead II", "CinC 2017")]:
        for i, ver in enumerate(("primary", "sensitivity")):
            rho.append({"pair": f"{a} vs {b}", "analysis": ver, "n_models": len(models),
                        "spearman_rho": spearmanr([auc[(a, m)][i] for m in models], [auc[(b, m)][i] for m in models])[0]})
    pd.DataFrame(rho).to_csv(RESULTS_DIR / "quota_sensitivity_rank_agreement.csv", index=False)

    # subject 105 and the other subjects with unusable windows: sensitivity at tau = 0.5, both analyses
    sub = []
    for ver, fam in (("primary", prim), ("sensitivity", sens)):
        b = fam["BUT_QDB (5-fold subject CV)"]
        for subj, g in b[b.y == 1].groupby("cluster"):
            for m in models:
                sub.append({"analysis": ver, "subject": subj, "model": m, "n_unusable": len(g),
                            "sensitivity_at_0.5": float((g[m] >= 0.5).mean())})
    pd.DataFrame(sub).to_csv(RESULTS_DIR / "quota_sensitivity_subject_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string(index=False))
    print(pd.DataFrame(rho).round(3).to_string(index=False))
    print(f"[Saved] quota_sensitivity_auroc.csv, quota_sensitivity_rank_agreement.csv, quota_sensitivity_subject_sensitivity.csv in {RESULTS_DIR}")


if __name__ == "__main__":
    stages = {"setup": setup, "d1": d1, "hubert": hubert, "external": external, "analyse": analyse}
    if len(sys.argv) < 2 or sys.argv[1] not in stages:
        raise SystemExit(f"usage: quota_sensitivity.py {{{'|'.join(stages)}}}")
    stages[sys.argv[1]]()
