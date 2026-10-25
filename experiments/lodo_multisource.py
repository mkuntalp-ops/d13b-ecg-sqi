"""
Leave-one-dataset-out (LODO) multi-source training (Q1 revision, 2026-10-25).

Question: does training on more than one source improve transfer to an unseen dataset, compared with training
on BUT QDB alone?

Sources (all single-lead 10-s windows at 500 Hz, SQIs in mV, strips from the corrected renderer):
- BUT QDB: the 1,047-window cohort (label: Class 3 vs Classes 1-2); evaluation unit = subject.
- CinC 2011 Set A: lead II of the 565 records used in the transfer analyses (label: 12-lead unacceptable);
  evaluation unit = record.
- CinC 2017: windows of the 8,512 recordings (label: recording 'too noisy to classify', copied to each of its
  windows; a weak window label). When CinC 2017 is a training source, a fixed subsample is used so that it does
  not dominate the pool: every noisy recording plus 1,000 random non-noisy recordings (seed 42). When it is the
  test set, all windows are scored and the recording score is the maximum over windows.

Splits: train on two sources, test on the third. Models and hyperparameters are those of the main analysis,
fixed in advance (no tuning, no early stopping, seed 42); nothing is chosen on the test dataset.

The training-free rule heuristic and the BUT-only models are taken from the stored predictions of the main
analysis for comparison. The d1-3B linear probe (frozen image tokens) is run in the same LODO splits on the
windows for which d1 features exist (BUT QDB, CinC 2011 lead II, CinC 2017 balanced subset).

Usage: python experiments/lodo_multisource.py                 (train and score, then analyse)
       python experiments/lodo_multisource.py --analysis-only
       python experiments/lodo_multisource.py --probe-only      (re-run the d1-3B probe LODO, then analyse)
Outputs: results/lodo_predictions.csv, results/lodo_probe_predictions.csv, results/lodo_summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.sqi_extractor import SQI_FEATURE_NAMES
from models.cnn1d import ECG1DCNNClassifier
from models.cnn2d import ECG2DCNNClassifier
from models.multimodal_fusion import IndependentSQIClassifier, SinglePassMultimodalClassifier
from models.tabular_baselines import SQIGradientBoosting, SQILogisticRegression

RESULTS = PROJECT_ROOT / "results"
CINC11 = PROJECT_ROOT / "cache" / "datasets" / "cinc2011"
CINC17 = PROJECT_ROOT / "cache" / "datasets" / "cinc2017" / "cinc2017_windows.csv"
FEAT = PROJECT_ROOT / "cache" / "d1_probe"
PRED_OUT = RESULTS / "lodo_predictions.csv"
PROBE_OUT = RESULTS / "lodo_probe_predictions.csv"
SUM_OUT = RESULTS / "lodo_summary.csv"
TRAINED = ["Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG", "2D_CNN_Image", "Multimodal_Fusion"]
N_CLEAN_CINC17_TRAIN = 1000
SEED = 42


def load_sources() -> dict:
    but = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv")
    but = but.assign(dataset="BUT_QDB", unit=but["subject_id"].astype(str), record=but["record_name"].astype(str))
    c11 = pd.read_csv(CINC11 / "cinc2011_random_draws_records.csv").drop_duplicates("segment_id")
    lab = pd.read_csv(CINC11 / "cinc2011_random_draws_manifest.csv").drop_duplicates("record_name").set_index("record_name")["label"]
    c11 = c11.assign(dataset="CinC2011", label=lab.loc[c11["record_name"].astype(int)].values,
                     unit=c11["record_name"].astype(str), record=c11["record_name"].astype(str))
    c17 = pd.read_csv(CINC17)
    c17 = c17.assign(dataset="CinC2017", unit=c17["record_name"].astype(str), record=c17["record_name"].astype(str))
    cols = ["dataset", "segment_id", "unit", "record", "label", "signal_path", "image_path"] + SQI_FEATURE_NAMES
    return {k: v[cols].reset_index(drop=True) for k, v in [("BUT_QDB", but), ("CinC2011", c11), ("CinC2017", c17)]}


def cinc17_training_subset(c17: pd.DataFrame) -> pd.DataFrame:
    rec = c17.drop_duplicates("record")[["record", "label"]]
    keep = pd.concat([rec[rec.label == 1].record,
                      rec[rec.label == 0].sample(N_CLEAN_CINC17_TRAIN, random_state=SEED).record])
    return c17[c17.record.isin(keep)].reset_index(drop=True)


def train_and_score(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    Xtr = np.nan_to_num(train[SQI_FEATURE_NAMES].values.astype(float))
    Xte = np.nan_to_num(test[SQI_FEATURE_NAMES].values.astype(float))
    ytr = train["label"].values.astype(int)
    out = test[["dataset", "segment_id", "unit", "record", "label"]].copy()
    out["Logistic_Regression_SQI"] = SQILogisticRegression().fit(Xtr, ytr).predict_proba(Xte)
    out["LightGBM_SQI"] = SQIGradientBoosting().fit(Xtr, ytr).predict_proba(Xte)
    out["Independent_SQI_MLP"] = IndependentSQIClassifier(epochs=35, lr=1e-3).fit(Xtr, ytr).predict_proba(Xte)
    print("  tabular done", flush=True)
    sig_tr = [np.load(p) for p in train["signal_path"]]
    out["1D_CNN_RawECG"] = ECG1DCNNClassifier(epochs=25, batch_size=32).fit(sig_tr, ytr).predict_proba(
        [np.load(p) for p in test["signal_path"]])
    del sig_tr
    print("  1D CNN done", flush=True)
    out["2D_CNN_Image"] = ECG2DCNNClassifier(epochs=25, batch_size=32).fit(train["image_path"].tolist(), ytr).predict_proba(
        test["image_path"].tolist())
    print("  2D CNN done", flush=True)
    out["Multimodal_Fusion"] = SinglePassMultimodalClassifier(epochs=25, batch_size=32).fit(
        train["image_path"].tolist(), Xtr, ytr).predict_proba(test["image_path"].tolist(), Xte)
    print("  multimodal done", flush=True)
    return out


def run_lodo():
    src = load_sources()
    c17_train = cinc17_training_subset(src["CinC2017"])
    parts = []
    for test_name in ["CinC2017", "CinC2011", "BUT_QDB"]:
        train = pd.concat([c17_train if k == "CinC2017" else v for k, v in src.items() if k != test_name], ignore_index=True)
        print(f"[LODO] test={test_name}: train {len(train)} windows "
              f"({train.groupby('dataset').size().to_dict()}), unusable share {train.label.mean():.3f}", flush=True)
        pred = train_and_score(train, src[test_name])
        pred["train_sources"] = "+".join(k for k in src if k != test_name)
        parts.append(pred)
        pd.concat(parts, ignore_index=True).to_csv(PRED_OUT, index=False)
        print(f"[LODO] test={test_name} saved", flush=True)


def run_probe():
    """d1-3B image-token probe in the same LODO splits (windows with stored d1 features only)."""
    src = load_sources()
    # With the full CinC 2017 features (analysis plan A2) the probe uses the same CinC 2017 training subset as the
    # other models and is tested on all 8,512 recordings; otherwise the balanced subset is used for both.
    full17 = (FEAT / "CinC2017_full_image.npy").exists()
    feats = {}
    for name, key in [("BUT_QDB", "BUT_QDB"), ("CinC2011", "CinC2011_leadII"),
                      ("CinC2017", "CinC2017_full" if full17 else "CinC2017_subset")]:
        X = np.load(FEAT / f"{key}_image.npy")
        ids = pd.read_csv(FEAT / f"{key}_ids.csv")["segment_id"].values
        meta = src[name].set_index("segment_id").loc[ids].reset_index()
        feats[name] = (X, meta)
    train_feats = dict(feats)
    if full17:
        keep = feats["CinC2017"][1].segment_id.isin(cinc17_training_subset(src["CinC2017"]).segment_id).values
        train_feats["CinC2017"] = (feats["CinC2017"][0][keep], feats["CinC2017"][1][keep].reset_index(drop=True))
    rows = []
    for test_name in ["CinC2017", "CinC2011", "BUT_QDB"]:
        tr = [k for k in feats if k != test_name]
        X = np.vstack([train_feats[k][0] for k in tr])
        meta = pd.concat([train_feats[k][1] for k in tr], ignore_index=True)
        groups = (meta["dataset"] + "_" + meta["unit"]).values
        probe = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                             {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]}, scoring="roc_auc",
                             cv=GroupKFold(n_splits=4)).fit(X, meta["label"].values, groups=groups)
        Xt, mt = feats[test_name]
        out = mt[["dataset", "segment_id", "unit", "record", "label"]].copy()
        out["d1_image_probe"] = probe.predict_proba(Xt)[:, 1]
        out["train_sources"] = "+".join(tr)
        rows.append(out)
        print(f"[probe] test={test_name}: train {len(meta)} windows, C={probe.best_params_}", flush=True)
    pd.concat(rows, ignore_index=True).to_csv(PROBE_OUT, index=False)


def unit_scores(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Evaluation units: BUT QDB windows (clustered by subject); CinC records (maximum over windows)."""
    if df["dataset"].iloc[0] == "BUT_QDB":
        return pd.DataFrame({"y": df["label"].values, "p": df[col].values, "cluster": df["unit"].values})
    g = df.groupby("record").agg(y=("label", "first"), p=(col, "max"))
    return pd.DataFrame({"y": g["y"].values, "p": g["p"].values, "cluster": g.index.values})


def paired_boot(y, a, b, cluster, B=2000, seed=SEED):
    """AUROC difference a - b with a cluster bootstrap (cluster = subject or record); two-sided percentile p."""
    rng = np.random.default_rng(seed)
    ug = np.unique(cluster)
    idx = {c: np.where(cluster == c)[0] for c in ug}
    d = []
    for _ in range(B):
        i = np.concatenate([idx[c] for c in rng.choice(ug, len(ug))])
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return roc_auc_score(y, a) - roc_auc_score(y, b), lo, hi, min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def boot_ci(y, p, cluster, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    ug = np.unique(cluster)
    idx = {c: np.where(cluster == c)[0] for c in ug}
    v = []
    for _ in range(B):
        i = np.concatenate([idx[c] for c in rng.choice(ug, len(ug))])
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def single_source_predictions() -> dict:
    """Stored BUT-only predictions on each external set, keyed by segment_id (same units as the LODO tests)."""
    c11 = pd.read_csv(RESULTS / "cinc_random_draws_predictions.csv").drop_duplicates("segment_id").set_index("segment_id")
    w = pd.read_csv(RESULTS / "external_window_predictions.csv", low_memory=False)
    c17 = w[w.dataset == "CinC2017"].set_index("segment_id")
    oof = pd.read_csv(RESULTS / "cohort1000_consolidated_oof_predictions.csv").set_index("segment_id")
    oof = oof.rename(columns={c: c.replace("prob_", "") for c in oof.columns if c.startswith("prob_")})
    return {"CinC2011": c11, "CinC2017": c17, "BUT_QDB": oof}


def analyse():
    pred = pd.read_csv(PRED_OUT, low_memory=False)
    probe = pd.read_csv(PROBE_OUT) if PROBE_OUT.exists() else None
    single = single_source_predictions()
    rows = []
    for test_name, g in pred.groupby("dataset"):
        ref = single[test_name]
        g = g.copy()
        for m in TRAINED + ["Rule_Based_SQI_Heuristic"]:
            g[f"single_{m}"] = ref.loc[g["segment_id"], m].values
        for m in TRAINED:
            u_multi, u_single = unit_scores(g, m), unit_scores(g, f"single_{m}")
            y, cl = u_multi["y"].values.astype(int), u_multi["cluster"].values
            lo, hi = boot_ci(y, u_multi["p"].values, cl)
            d, dlo, dhi, p = paired_boot(y, u_multi["p"].values, u_single["p"].values, cl)
            rows.append({"test": test_name, "model": m, "train_sources": g["train_sources"].iloc[0], "n_units": len(y),
                         "prevalence": y.mean(), "auroc_multi": roc_auc_score(y, u_multi["p"].values), "auroc_multi_lo": lo,
                         "auroc_multi_hi": hi, "ap_multi": average_precision_score(y, u_multi["p"].values),
                         "auroc_single": roc_auc_score(y, u_single["p"].values), "delta": d, "delta_lo": dlo, "delta_hi": dhi,
                         "p_boot": p, "single_reference": "in-domain 5-fold CV" if test_name == "BUT_QDB" else "BUT QDB only"})
        u = unit_scores(g, "single_Rule_Based_SQI_Heuristic")
        rows.append({"test": test_name, "model": "Rule_Based_SQI_Heuristic", "train_sources": "none (fixed rule)",
                     "n_units": len(u), "prevalence": u["y"].mean(), "auroc_multi": np.nan,
                     "auroc_single": roc_auc_score(u["y"].values.astype(int), u["p"].values), "single_reference": "fixed rule"})
    # tuning-free and foundation-model comparators run in the same LODO splits by their own scripts
    extras = {}
    if (RESULTS / "tabpfn_predictions.csv").exists():
        tp = pd.read_csv(RESULTS / "tabpfn_predictions.csv", low_memory=False)
        extras["TabPFN_SQI"] = (tp[tp.setting.str.startswith("lodo_test_")],
                                tp[tp.setting.isin(["butqdb_cv", "but_only_external"])])
    if (RESULTS / "hubert_probe_predictions.csv").exists():
        hp = pd.read_csv(RESULTS / "hubert_probe_predictions.csv", low_memory=False)
        extras["HuBERT_ECG_probe"] = (hp[hp.setting.str.startswith("lodo_test_")],
                                      hp[hp.setting.isin(["butqdb_cv", "but_only_external"])])
    for name, (lodo_p, single_p) in extras.items():
        for test_name, g in pred.groupby("dataset"):
            ml = lodo_p[lodo_p.dataset == test_name].set_index("segment_id")["prob_unusable"]
            ms = single_p[single_p.dataset == test_name].set_index("segment_id")["prob_unusable"]
            h = g[["dataset", "segment_id", "unit", "record", "label"]].copy()
            h["multi"], h["single"] = ml.loc[h.segment_id].values, ms.loc[h.segment_id].values
            um, us = unit_scores(h, "multi"), unit_scores(h, "single")
            y, cl = um["y"].values.astype(int), um["cluster"].values
            lo, hi = boot_ci(y, um["p"].values, cl)
            d, dlo, dhi, pv = paired_boot(y, um["p"].values, us["p"].values, cl)
            rows.append({"test": test_name, "model": name, "train_sources": g["train_sources"].iloc[0], "n_units": len(y),
                         "prevalence": y.mean(), "auroc_multi": roc_auc_score(y, um["p"].values), "auroc_multi_lo": lo,
                         "auroc_multi_hi": hi, "ap_multi": average_precision_score(y, um["p"].values),
                         "auroc_single": roc_auc_score(y, us["p"].values), "delta": d, "delta_lo": dlo, "delta_hi": dhi,
                         "p_boot": pv, "single_reference": "in-domain 5-fold CV" if test_name == "BUT_QDB" else "BUT QDB only"})
    if probe is not None:
        oofp = pd.read_csv(RESULTS / "d1_probe_oof_predictions.csv")
        oofp = oofp[oofp.features == "image"].set_index("segment_id")["prob_unusable"]
        trp = pd.read_csv(RESULTS / "d1_probe_transfer_predictions.csv")
        trp = trp[trp.features == "image"].assign(record_name=lambda d: d.record_name.astype(str))
        for test_name, g in probe.groupby("dataset"):
            u = unit_scores(g, "d1_image_probe")
            if test_name == "BUT_QDB":
                single = oofp.loc[g.segment_id].values
            elif test_name == "CinC2017" and (RESULTS / "d1_cinc2017_full_predictions.csv").exists():
                f = pd.read_csv(RESULTS / "d1_cinc2017_full_predictions.csv").assign(record=lambda d: d.record_name.astype(str))
                single = f.groupby("record").prob_image_probe.max().loc[g.groupby("record").size().index].values
            else:
                s = trp[trp.dataset == test_name].set_index("record_name")["prob_unusable"]
                single = s.loc[g.groupby("record").size().index.astype(str)].values  # same record order as unit_scores
            y, cl = u["y"].values.astype(int), u["cluster"].values
            lo, hi = boot_ci(y, u["p"].values, cl)
            d, dlo, dhi, pv = paired_boot(y, u["p"].values, np.asarray(single, float), cl)
            rows.append({"test": test_name, "model": "d1_image_probe", "train_sources": g["train_sources"].iloc[0],
                         "n_units": len(y), "prevalence": y.mean(), "auroc_multi": roc_auc_score(y, u["p"].values),
                         "auroc_multi_lo": lo, "auroc_multi_hi": hi, "ap_multi": average_precision_score(y, u["p"].values),
                         "auroc_single": roc_auc_score(y, single), "delta": d, "delta_lo": dlo, "delta_hi": dhi, "p_boot": pv,
                         "single_reference": ("in-domain 5-fold CV" if test_name == "BUT_QDB" else "BUT QDB only")
                         + (" (CinC 2017 balanced subset)" if test_name == "CinC2017" and len(y) < 1000 else "")})
    out = pd.DataFrame(rows)
    out.to_csv(SUM_OUT, index=False)
    print(out[["test", "model", "n_units", "auroc_multi", "auroc_multi_lo", "auroc_multi_hi", "auroc_single", "delta", "delta_lo",
               "delta_hi", "p_boot"]].round(3).to_string(index=False))
    print(f"[Saved] {SUM_OUT}")


if __name__ == "__main__":
    if "--probe-only" in sys.argv:
        run_probe()
    elif "--analysis-only" not in sys.argv:
        run_lodo()
        run_probe()
    analyse()
