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
     HuBERT-ECG (authors' preprocessing) fitted on the train windows. TabPFN is not included.

Outputs: results/butqdb_official_split_predictions.csv, results/butqdb_official_split_summary.csv
Usage: python experiments/butqdb_official_split.py
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

from hubert_ecg_probe import probe  # noqa: E402
from lodo_multisource import boot_ci, load_sources, train_and_score  # noqa: E402
from paired_comparisons import cluster_bootstrap_paired, families, holm  # noqa: E402

RESULTS = ROOT / "results"
PRED_OUT = RESULTS / "butqdb_official_split_predictions.csv"
SUM_OUT = RESULTS / "butqdb_official_split_summary.csv"
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

    # F1a: evaluation-only exclusion on the stored out-of-fold predictions
    but = families()["BUT_QDB (5-fold subject CV)"][1]
    keep_ids = set(man.segment_id[inside])
    sub = but[but.unit.isin(keep_ids)].reset_index(drop=True)
    models = [c for c in but.columns if c not in ("unit", "y", "cluster")]
    f1a = compare_with_best(sub, models, "F1a: 5-fold CV, excluded stretch removed (evaluation only)")

    # F1b: official train/test split
    src = load_sources()["BUT_QDB"]
    assert (src.segment_id.values == man.segment_id.values).all()
    rec = man.record_name.str[:6].astype(int)
    tr, te = inside & rec.isin(TRAIN_RECORDS).values, inside & rec.isin(TEST_RECORDS).values
    assert not set(man.subject_id[tr]) & set(man.subject_id[te]), "subject overlap between official train and test"
    print(f"[official] train {tr.sum()} windows ({man.label[tr].sum()} unusable, {man.subject_id[tr].nunique()} subjects); "
          f"test {te.sum()} windows ({man.label[te].sum()} unusable, {man.subject_id[te].nunique()} subjects)", flush=True)
    pred = train_and_score(src[tr].reset_index(drop=True), src[te].reset_index(drop=True))
    stored = pd.read_csv(RESULTS / "cohort1000_consolidated_oof_predictions.csv").set_index("segment_id")
    for m in ["Rule_Based_SQI_Heuristic", "LiquidAI_d1_3B_Neural"]:  # untrained: stored scores are valid
        pred[m] = stored.loc[pred.segment_id, f"prob_{m}"].values
    groups = man.subject_id.values
    for name, path in [("d1_image_probe", ROOT / "cache" / "d1_probe" / "BUT_QDB_image.npy"),
                       ("HuBERT_ECG_probe", ROOT / "cache" / "hubert_ecg_authors" / "BUT_QDB.npy")]:
        X = np.load(path)
        pred[name] = probe().fit(X[tr], man.label.values[tr], groups=groups[tr]).predict_proba(X[te])[:, 1]
    pred.to_csv(PRED_OUT, index=False)
    test = pred.rename(columns={"label": "y"}).assign(cluster=man.subject_id.values[te])
    f1b = compare_with_best(test, [m for m in models if m in test.columns], "F1b: official train/test split")

    out = pd.concat([f1a, f1b], ignore_index=True)
    out.to_csv(SUM_OUT, index=False)
    print(out.drop(columns=["analysis"]).round(3).to_string(index=False))
    print(f"[Saved] {SUM_OUT}")


if __name__ == "__main__":
    main()
