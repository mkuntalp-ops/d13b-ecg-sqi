"""
BUT QDB official protocol: sensitivity analyses F1a and F1b (analysis plan, amendment of 2026-10-26).

Official split and evaluated intervals: Smital et al., Sci Data 2026 (Table 3) and choseDATA() of TESTrun_v01.m
(github.com/BioSig-BUT/BUT-QDB-supplement). Intervals are 1-based inclusive sample ranges at 1,000 Hz; a cohort window
[start_sample, end_sample) (0-based) is inside an interval if start_sample + 1 >= from and end_sample <= to. The only
cohort windows outside the intervals are 59 windows of 105001 in the excluded 13-h stretch of missing signal.

F1a: pooled out-of-fold AUROC of the stored BUT QDB predictions without those windows (evaluation-only exclusion;
     the models were trained with them), all 11 models, paired comparisons with the best model (subject-level
     cluster bootstrap, Holm over the 55 pairs).
F1b: train on the cohort windows of the official train records, test on those of the official test records (inside
     the intervals). Window-level binary task (Class 3 vs Classes 1-2), not the official sample-level 3-class metric.
     Trained with the fixed hyperparameters of the main analysis (seed 42): LR, LightGBM, MLP, 1D CNN, 2D CNN,
     multimodal net; untrained: rule heuristic, zero-shot d1-3B (stored scores); probes: d1-3B image tokens and
     HuBERT-ECG (authors' preprocessing) fitted on the train windows; TabPFN (v2) when its predictions exist.

v2 (revision of 2026-11-05): the primary cohort is the 988 windows inside the official intervals, so F1a has nothing to
exclude and is skipped (it is the primary analysis). F1b is the secondary analysis of plan item 3.1, with TabPFN.
Probe features are looked up by segment id (experiments/feature_cache.py), never by position.

Outputs: results/butqdb_official_split_assignment.csv (segment_id, split), results/butqdb_official_split_predictions.csv,
         results/butqdb_official_split_summary.csv
Usage (v2, TabPFN included):
    python experiments/butqdb_official_split.py --split-only                  # writes the assignment file
    /Volumes/KIOXIA/AI/tabpfn_env/bin/python experiments/tabpfn_runner.py --official-split
    python experiments/butqdb_official_split.py
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.argv = [a for a in sys.argv if not a.startswith("--preproc")]

from feature_cache import load_features  # noqa: E402
from hubert_ecg_probe import probe  # noqa: E402
from lodo_multisource import boot_ci, load_sources, train_and_score  # noqa: E402
from paired_comparisons import cluster_bootstrap_paired, families, holm  # noqa: E402

from config import RESULTS_DIR as _CFG_RESULTS  # noqa: E402  (ECG_RESULTS_DIR overrides)
RESULTS = _CFG_RESULTS
PRED_OUT = RESULTS / "butqdb_official_split_predictions.csv"
SUM_OUT = RESULTS / "butqdb_official_split_summary.csv"
SPLIT_OUT = RESULTS / "butqdb_official_split_assignment.csv"
TABPFN_PRED = RESULTS / "tabpfn_official_split_predictions.csv"  # tabpfn_runner.py --official-split
TRAIN_RECORDS = {100001, 100002, 103001, 103002, 103003, 104001, 111001, 113001, 114001, 115001, 126001}
TEST_RECORDS = {105001, 118001, 121001, 122001, 123001, 124001, 125001}
_TWO = [(28800001, 30000000), (57600001, 58800000)]
INTERVALS = {100001: [(1, 87087000)], 100002: _TWO, 103001: _TWO, 103002: _TWO, 103003: _TWO, 104001: _TWO,
             105001: [(46800001, 139147000)], 111001: [(1, 90645000)],
             113001: [(28800001, 30000000), (36120000, 37319999), (57600001, 58800000)],
             114001: [(11214751, 11334751), (11674751, 12874750), (28800001, 30000000), (57600001, 58800000)],
             115001: _TWO, 118001: _TWO, 121001: _TWO, 122001: _TWO, 123001: _TWO,
             124001: [(28800001, 30000000), (33700000, 34899999), (57600001, 58800000), (65100000, 66299999)],
             125001: _TWO, 126001: _TWO}


def official_mask(manifest: pd.DataFrame) -> np.ndarray:
    rec = manifest.record_name.str[:6].astype(int)
    return np.array([any(s + 1 >= a and e <= b for a, b in INTERVALS[r])
                     for r, s, e in zip(rec, manifest.start_sample, manifest.end_sample)])


def compare_with_best(df: pd.DataFrame, models: list, label: str) -> pd.DataFrame:
    y, cl = df["y"].values.astype(int), df["cluster"].values
    auc = {m: roc_auc_score(y, df[m]) for m in models}
    pairs = list(combinations(models, 2))
    res = [cluster_bootstrap_paired(y, df[a].values, df[b].values, cl) for a, b in pairs]
    p_holm = holm(np.array([r[3] for r in res]))
    best = max(auc, key=auc.get)
    rows = []
    for m in models:
        lo, hi = boot_ci(y, df[m].values, cl)
        rec = {"analysis": label, "model": m, "n_windows": len(y), "n_unusable": int(y.sum()),
               "n_subjects": len(np.unique(cl)), "auroc": auc[m], "auroc_lo": lo, "auroc_hi": hi, "best": m == best}
        if m != best:
            k = [i for i, (a, b) in enumerate(pairs) if {a, b} == {m, best}][0]
            d = res[k][0] if pairs[k][0] == m else -res[k][0]
            rec.update(delta_vs_best=d, p_holm_vs_best=p_holm[k])
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    man = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv")
    inside = official_mask(man)
    print(f"[official] cohort windows outside the official intervals: {(~inside).sum()} "
          f"(unusable {man.label[~inside].sum()})", flush=True)

    # F1b split (official train/test records, windows inside the intervals)
    rec = man.record_name.str[:6].astype(int)
    tr, te = inside & rec.isin(TRAIN_RECORDS).values, inside & rec.isin(TEST_RECORDS).values
    assert not set(man.subject_id[tr]) & set(man.subject_id[te]), "subject overlap between official train and test"
    assert not (tr & te).any() and (tr | te).sum() == inside.sum(), "every window inside the intervals in exactly one split"
    print(f"[official] train {tr.sum()} windows ({man.label[tr].sum()} unusable, {man.subject_id[tr].nunique()} subjects); "
          f"test {te.sum()} windows ({man.label[te].sum()} unusable, {man.subject_id[te].nunique()} subjects)", flush=True)
    split = np.where(tr, "train", np.where(te, "test", "excluded"))
    pd.DataFrame({"segment_id": man.segment_id, "subject_id": man.subject_id, "label": man.label,
                  "split": split}).to_csv(SPLIT_OUT, index=False)
    if "--split-only" in sys.argv:
        print(f"[Saved] {SPLIT_OUT}")
        return

    # F1a: evaluation-only exclusion on the stored out-of-fold predictions (v1 cohort only)
    but = families()["BUT_QDB (5-fold subject CV)"][1]
    models = [c for c in but.columns if c not in ("unit", "y", "cluster")]
    parts = []
    if (~inside).any():
        keep_ids = set(man.segment_id[inside])
        sub = but[but.unit.isin(keep_ids)].reset_index(drop=True)
        parts.append(compare_with_best(sub, models, "F1a: 5-fold CV, excluded stretch removed (evaluation only)"))
    else:
        print("[official] F1a skipped: every cohort window is inside the official intervals (primary analysis)", flush=True)

    # F1b: official train/test split
    src = load_sources()["BUT_QDB"]
    assert (src.segment_id.values == man.segment_id.values).all()
    pred = train_and_score(src[tr].reset_index(drop=True), src[te].reset_index(drop=True))
    stored = pd.read_csv(RESULTS / "cohort1000_consolidated_oof_predictions.csv").set_index("segment_id")
    for m in ["Rule_Based_SQI_Heuristic", "LiquidAI_d1_3B_Neural"]:  # untrained: stored scores are valid
        pred[m] = stored.loc[pred.segment_id, f"prob_{m}"].values
    groups = man.subject_id.values
    for name, cache_dir, feat in [("d1_image_probe", ROOT / "cache" / "d1_probe", "image"),
                                  ("HuBERT_ECG_probe", ROOT / "cache" / "hubert_ecg_authors", None)]:
        X = load_features(cache_dir, feat, man.segment_id)  # by segment id (the cached BUT_QDB set has 1,047 rows)
        pred[name] = probe().fit(X[tr], man.label.values[tr], groups=groups[tr]).predict_proba(X[te])[:, 1]
    if TABPFN_PRED.exists():
        t = pd.read_csv(TABPFN_PRED)
        if set(t.segment_id) != set(pred.segment_id):
            raise ValueError(f"{TABPFN_PRED}: segment ids differ from the official test windows; rerun --split-only "
                             "and tabpfn_runner.py --official-split")
        pred["TabPFN_SQI"] = t.set_index("segment_id").loc[pred.segment_id, "prob_unusable"].values
    else:
        print(f"[official] {TABPFN_PRED.name} not found: TabPFN not included", flush=True)
    pred.to_csv(PRED_OUT, index=False)
    test = pred.rename(columns={"label": "y"}).assign(cluster=man.subject_id.values[te])
    parts.append(compare_with_best(test, [m for m in models if m in test.columns], "F1b: official train/test split"))

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(SUM_OUT, index=False)
    print(out.drop(columns=["analysis"]).round(3).to_string(index=False))
    print(f"[Saved] {SUM_OUT}")


if __name__ == "__main__":
    main()
