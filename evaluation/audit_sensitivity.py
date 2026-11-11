"""
Sensitivity analyses requested by the independent code audit of 2026-11-11 (findings CODE-001, -003, -004, -011).

All four are post hoc: they were asked for after every result of the v2 analysis was known, they use saved predictions
only (no model is fitted, no threshold is chosen) and they do not replace any primary result.

1. CinC 2011 without constant lead II (CODE-001). 73 of the 998 lead II records are constant and all of them are
   unacceptable; any flat-line check finds them. The pairwise DeLong tests of the CinC 2011 family are repeated on the
   remaining records, Holm-adjusted over all pairs (same family definition as evaluation/paired_comparisons.py) and,
   as in evaluation/paired_vs_best.py, over the comparisons with the highest-AUROC model.
   -> cinc2011_nonconstant_paired.csv
2. BUT QDB per-subject AUROC (CODE-003). Unusable windows occur in 7 of the 15 subjects. For every model: the AUROC
   inside each subject that has both classes, the pooled AUROC over those subjects only, and the windows per subject.
   Descriptive; no tests. A subject with one unusable window gives an AUROC that rests on that window.
   -> butqdb_per_subject_auroc.csv
3. CinC 2017 by number of windows (CODE-004). A recording contributes 1 to 3 windows, and noisy recordings are
   shorter, so the recording maximum (primary rule) is taken over fewer windows for them. AUROC per model for the
   maximum, mean and first-window rules, in all recordings and within recordings of 1, 2 and 3 windows, and the AUROC
   of the window count alone (fewer windows = higher score). Descriptive; no tests.
   -> cinc2017_length_stratified.csv
4. PTB-XL sample, patients (CODE-011). The stored sample lists record ids only. The patient of every sampled record is
   read from ptbxl_database.csv to confirm that no patient enters twice. Skipped when the sample or the local PTB-XL
   metadata is absent. Reads metadata only; no PTB-XL score is recomputed.
   -> ptbxl_frozen_sample_patient_check.csv

5. BUT QDB quota and the official intervals (CODE-002). The cohort quota (up to 60 unusable windows per subject) was
   drawn over all annotated intervals and the windows outside the official intervals were removed afterwards. Per
   subject: Class 3 candidate windows in all intervals and inside the official intervals, and unusable cohort windows.
   Skipped when the BUT QDB annotation files are not in the cache. Counts only.
   -> butqdb_quota_candidates_by_subject.csv

audit_sensitivity_provenance.json records the commit, whether the working tree was dirty, and package versions.

Usage: ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/audit_sensitivity.py
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CINC2011_LEADII_FULL, LOCAL_DATASETS_DIR  # noqa: E402
from paired_comparisons import RESULTS, cinc2017_windows, delong_paired, families, holm  # noqa: E402

META = ("unit", "y", "cluster")
C11_FAMILY = "CinC2011 lead II (998 records)"
BUT_FAMILY = "BUT_QDB (5-fold subject CV)"
OUT_C11 = RESULTS / "cinc2011_nonconstant_paired.csv"
OUT_SUBJ = RESULTS / "butqdb_per_subject_auroc.csv"
OUT_LEN = RESULTS / "cinc2017_length_stratified.csv"
OUT_PTB = RESULTS / "ptbxl_frozen_sample_patient_check.csv"
OUT_QUOTA = RESULTS / "butqdb_quota_candidates_by_subject.csv"
OUT_PROV = RESULTS / "audit_sensitivity_provenance.json"


def model_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META]


# ---------- 1. CinC 2011 without constant lead II ----------
def constant_lead_records(source: pd.DataFrame) -> set:
    """Record names whose lead II window has zero standard deviation (signal_std == 0)."""
    return set(source.record_name.astype(str)[source.signal_std == 0])


def paired_family(scores: pd.DataFrame) -> pd.DataFrame:
    """All pairwise DeLong tests on one score table; Holm over all pairs and over the pairs with the best model."""
    y = scores["y"].values.astype(int)
    if y.min() == y.max():
        raise ValueError("one class only")
    models = model_columns(scores)
    auc = {m: roc_auc_score(y, scores[m].values.astype(float)) for m in models}
    best = max(auc, key=auc.get)
    rows = []
    for a, b in combinations(models, 2):
        d, lo, hi, p = delong_paired(y, scores[a].values, scores[b].values)
        rows.append({"n_units": len(y), "n_positive": int(y.sum()), "model_a": a, "model_b": b, "auroc_a": auc[a],
                     "auroc_b": auc[b], "delta": d, "ci_lo": lo, "ci_hi": hi, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].values)
    with_best = ((out.model_a == best) | (out.model_b == best)).values
    out["best_model"] = best
    out["p_holm_vs_best"] = np.nan
    out.loc[with_best, "p_holm_vs_best"] = holm(out.loc[with_best, "p"].values)
    return out


def cinc2011_nonconstant() -> pd.DataFrame:
    scores = families()[C11_FAMILY][1]
    constant = constant_lead_records(pd.read_csv(CINC2011_LEADII_FULL))
    is_const = scores.unit.astype(str).isin(constant).values
    if is_const.sum() != len(constant):
        raise ValueError(f"{len(constant)} constant records in the source table, {is_const.sum()} found in the scores")
    n_unacc = int(scores.y.values[is_const].sum())
    kept = scores[~is_const].reset_index(drop=True)
    out = paired_family(kept)
    out.insert(0, "family", f"CinC2011 lead II without constant lead II ({len(kept)} records)")
    out["n_constant_removed"] = int(is_const.sum())
    out["n_constant_unacceptable"] = n_unacc
    out.to_csv(OUT_C11, index=False)
    best = out.best_model.iloc[0]
    v = out[(out.model_a == best) | (out.model_b == best)]
    print(f"[cinc2011] removed {is_const.sum()} constant records ({n_unacc} unacceptable); {len(kept)} kept, "
          f"{int(kept.y.sum())} unacceptable; best = {best}")
    print(v[["model_a", "model_b", "auroc_a", "auroc_b", "delta", "ci_lo", "ci_hi", "p", "p_holm", "p_holm_vs_best"]]
          .round(4).to_string(index=False))
    return out


# ---------- 2. BUT QDB per-subject AUROC ----------
def per_subject_auroc(scores: pd.DataFrame) -> pd.DataFrame:
    """Rows: one per model and mixed-class subject, plus 'pooled_mixed_subjects' and 'all_subjects' per model."""
    y, subj = scores["y"].values.astype(int), scores["cluster"].values
    mixed = [s for s in np.unique(subj) if 0 < y[subj == s].sum() < (subj == s).sum()]
    in_mixed = np.isin(subj, mixed)
    rows = []
    for m in model_columns(scores):
        p = scores[m].values.astype(float)
        for s in mixed:
            k = subj == s
            rows.append({"model": m, "subject": str(s), "n_windows": int(k.sum()), "n_unusable": int(y[k].sum()),
                         "auroc": roc_auc_score(y[k], p[k])})
        rows.append({"model": m, "subject": "pooled_mixed_subjects", "n_windows": int(in_mixed.sum()),
                     "n_unusable": int(y[in_mixed].sum()), "auroc": roc_auc_score(y[in_mixed], p[in_mixed])})
        rows.append({"model": m, "subject": "all_subjects", "n_windows": len(y), "n_unusable": int(y.sum()),
                     "auroc": roc_auc_score(y, p)})
    return pd.DataFrame(rows)


def butqdb_subjects() -> pd.DataFrame:
    out = per_subject_auroc(families()[BUT_FAMILY][1])
    out.to_csv(OUT_SUBJ, index=False)
    print("[butqdb] AUROC by subject (subjects with both classes), pooled over them, and over all 15 subjects")
    print(out.pivot(index="model", columns="subject", values="auroc").round(3).to_string())
    return out


# ---------- 3. CinC 2017 by number of windows ----------
def length_stratified(windows: pd.DataFrame, cols: list) -> pd.DataFrame:
    """AUROC per model, aggregation rule and stratum of windows per recording (strata with one class are skipped)."""
    g = windows.groupby("record_name")
    n_win, label = g.size(), g["label"].first().astype(int)
    first = windows[windows.unit_idx == 0].set_index("record_name")
    if len(first) != len(label) or first.index.duplicated().any():
        raise ValueError("every recording needs exactly one window with unit_idx == 0")
    agg = {"max": g[cols].max(), "mean": g[cols].mean(), "first_window": first.loc[label.index, cols]}
    strata = {"all": label.index}
    strata.update({f"{k}_window{'s' if k > 1 else ''}": n_win.index[n_win == k] for k in sorted(n_win.unique())})
    rows = []
    for sname, idx in strata.items():
        ys = label.loc[idx].values
        if ys.min() == ys.max():
            continue
        for rule, table in agg.items():
            for m in cols:
                rows.append({"stratum": sname, "aggregation": rule, "model": m, "n_recordings": len(ys),
                             "n_noisy": int(ys.sum()), "auroc": roc_auc_score(ys, table.loc[idx, m].values)})
    rows.append({"stratum": "all", "aggregation": "none", "model": "window_count_only (fewer windows = higher score)",
                 "n_recordings": len(label), "n_noisy": int(label.sum()),
                 "auroc": roc_auc_score(label.values, -n_win.loc[label.index].values)})
    return pd.DataFrame(rows)


def cinc2017_length() -> pd.DataFrame:
    w, cols = cinc2017_windows()
    out = length_stratified(w, cols)
    out.to_csv(OUT_LEN, index=False)
    t = out[out.aggregation.isin(["max", "mean"])]
    print("[cinc2017] AUROC by aggregation rule and windows per recording")
    print(t.pivot(index="model", columns=["aggregation", "stratum"], values="auroc").round(3).to_string())
    print(out[out.aggregation == "none"][["model", "n_recordings", "n_noisy", "auroc"]].round(3).to_string(index=False))
    print(out.drop_duplicates("stratum")[["stratum", "n_recordings", "n_noisy"]].to_string(index=False))
    return out


# ---------- 4. PTB-XL sample: one record per patient ----------
def patient_check(sample: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    pid = meta.set_index("ecg_id")["patient_id"]
    missing = sorted(set(sample.ecg_id) - set(pid.index))
    if missing:
        raise ValueError(f"{len(missing)} sampled records are not in the metadata (first: {missing[:3]})")
    s = sample.assign(patient_id=pid.loc[sample.ecg_id].values)
    rows = [{"stratum": k, "n_records": len(g), "n_patients": g.patient_id.nunique()} for k, g in s.groupby("stratum")]
    rows.append({"stratum": "all", "n_records": len(s), "n_patients": s.patient_id.nunique()})
    out = pd.DataFrame(rows)
    out["patients_with_more_than_one_record"] = out.n_records - out.n_patients
    return out


def ptbxl_patients():
    sample_path, meta_path = RESULTS / "ptbxl_frozen_sample.csv", LOCAL_DATASETS_DIR / "ptb-xl" / "ptbxl_database.csv"
    if not sample_path.exists() or not meta_path.exists():
        print(f"[ptbxl] skipped: {sample_path.name} or {meta_path} not found")
        return None
    out = patient_check(pd.read_csv(sample_path), pd.read_csv(meta_path, usecols=["ecg_id", "patient_id"]))
    out.to_csv(OUT_PTB, index=False)
    print("[ptbxl] sampled records and distinct patients")
    print(out.to_string(index=False))
    return out


# ---------- 5. BUT QDB quota versus official intervals ----------
def quota_table(candidates: pd.DataFrame, inside: np.ndarray, cohort: pd.DataFrame) -> pd.DataFrame:
    """Per subject: Class 3 candidate windows (all / inside the official intervals) and unusable cohort windows."""
    c = candidates.assign(inside=np.asarray(inside, dtype=bool), subject_id=candidates.subject_id.astype(str))
    c3 = c[c.label == 1]
    out = pd.DataFrame({"class3_candidates_all_intervals": c3.groupby("subject_id").size(),
                        "class3_candidates_inside_official": c3[c3.inside].groupby("subject_id").size(),
                        "cohort_unusable": cohort[cohort.label == 1].groupby(cohort.subject_id.astype(str)).size()})
    out = out.reindex(sorted(c.subject_id.unique())).fillna(0).astype(int)
    out["quota_if_drawn_inside_official"] = out.class3_candidates_inside_official.clip(upper=60)
    return out.rename_axis("subject_id").reset_index()


def butqdb_quota():
    sys.path.insert(0, str(ROOT / "experiments"))
    from data.extract_butqdb_1000_cohort import BUTQDB_DIR, build_candidate_windows
    if not list(BUTQDB_DIR.glob("*_ANN.csv")):
        print(f"[quota] skipped: no annotation files in {BUTQDB_DIR}")
        return None
    argv = list(sys.argv)
    from butqdb_official_split import official_mask  # this import edits sys.argv; restored below
    sys.argv = argv
    w = build_candidate_windows()
    out = quota_table(w, official_mask(w), pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv"))
    out.to_csv(OUT_QUOTA, index=False)
    print("[quota] Class 3 candidate windows and unusable cohort windows per subject")
    print(out.to_string(index=False))
    return out


# ---------- provenance ----------
def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def provenance() -> dict:
    import scipy
    import sklearn
    status = _git("status", "--porcelain")
    return {"script": "evaluation/audit_sensitivity.py", "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "results_dir": str(RESULTS.relative_to(ROOT)) if RESULTS.is_relative_to(ROOT) else str(RESULTS),
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(status),
            "git_changed_or_untracked_paths": len(status.splitlines()) if status else 0,
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}


def main():
    cinc2011_nonconstant()
    butqdb_subjects()
    cinc2017_length()
    ptbxl_patients()
    butqdb_quota()
    OUT_PROV.write_text(json.dumps(provenance(), indent=2) + "\n")
    print(f"[Saved] {OUT_C11.name}, {OUT_SUBJ.name}, {OUT_LEN.name}, {OUT_PTB.name}, {OUT_QUOTA.name}, {OUT_PROV.name} in {RESULTS}")


if __name__ == "__main__":
    main()
