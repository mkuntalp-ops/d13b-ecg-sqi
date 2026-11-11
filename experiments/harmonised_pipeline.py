"""
Harmonised-pipeline comparison (reviewer finding PAPER-002). Post hoc and secondary; it does not replace the primary
unchanged-transfer results. The definition was written into REVISION_PLAN_2026-11-05_post_desk_rejection.md and
committed (999c419) before any harmonised score was computed.

Safeguards (none uses target labels or target performance):
  G1 constant-input rule: a window whose signal SD is 0 is scored 1 (unusable) before the model; all 11 models.
  G2 quantization-insensitive flatline input (definition of 2026-11-06) instead of flatline_fraction: logistic
     regression, LightGBM, SQI MLP, TabPFN, the SQI branch of the multimodal net, and the hand-set heuristic.
  G3 standardised inputs clipped to [-3, 3] with training statistics: the same models except the heuristic.
Everything else is as in the primary analysis (folds, hyperparameters, seed 42). The MLP and the multimodal net are
also refitted with seeds 1-4 for BUT-only transfer. The 1D CNN, 2D CNN, zero-shot d1-3B and the two probes have no
tabular input and are not refitted; their stored scores get G1 only.

Evaluations: BUT QDB 5-fold out-of-fold (988 windows), CinC 2011 lead II (998 records), CinC 2017 (8,512 recordings,
recording maximum). Reported per model and set: original and harmonised AUROC, paired difference with 95% CI
(subject cluster bootstrap on BUT QDB, DeLong externally; descriptive, no multiplicity adjustment), number of distinct
scores, seed range, share of standardised inputs beyond 3 SD, and the Spearman rank agreement between sets.

Usage: ECG_RESULTS_DIR=results/v2_2026-11 python experiments/harmonised_pipeline.py            (fit and score)
       <tabpfn env>/bin/python experiments/harmonised_tabpfn.py
       ECG_RESULTS_DIR=results/v2_2026-11 python experiments/harmonised_pipeline.py --analysis-only
Outputs: harmonised_predictions.csv, harmonised_summary.csv, harmonised_seed_range.csv,
         harmonised_input_diagnostics.csv, harmonised_rank_agreement.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "evaluation"))

from config import RESULTS_DIR  # noqa: E402
from harmonised_inputs import CLIP, HARMONISED, ORIGINAL, load_tables, zclip_apply, zclip_fit, zscore  # noqa: E402

PRED, TAB_PRED = RESULTS_DIR / "harmonised_predictions.csv", RESULTS_DIR / "harmonised_tabpfn_predictions.csv"
SUM, SEEDS_OUT = RESULTS_DIR / "harmonised_summary.csv", RESULTS_DIR / "harmonised_seed_range.csv"
DIAG, RANK = RESULTS_DIR / "harmonised_input_diagnostics.csv", RESULTS_DIR / "harmonised_rank_agreement.csv"
SEEDS = [42, 1, 2, 3, 4]
LR, GBM, MLP, MM, HEUR, TAB = ("Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "Multimodal_Fusion",
                               "Rule_Based_SQI_Heuristic", "TabPFN_SQI")
REFIT = [LR, GBM, MLP, MM, HEUR, TAB]
FAMILY = {"BUT_QDB": "BUT_QDB (5-fold subject CV)", "CinC2011": "CinC2011 lead II (998 records)", "CinC2017": "CinC2017 (8,512 recordings)"}


def apply_constant_rule(scores: np.ndarray, constant: np.ndarray) -> np.ndarray:
    """G1: score 1.0 for windows whose signal is constant; other scores unchanged."""
    out = np.asarray(scores, dtype=float).copy()
    out[np.asarray(constant, dtype=bool)] = 1.0
    return out


def set_seed(seed: int) -> None:
    import models.multimodal_fusion as mmf
    mmf.RANDOM_SEED = seed


def rows(dataset, df, model, seed, p):
    return pd.DataFrame({"dataset": dataset, "segment_id": df.segment_id.values, "model": model, "seed": seed, "prob_unusable": np.asarray(p, float)})


def fit_and_score():
    import models.multimodal_fusion as mmf
    from models.multimodal_fusion import IndependentSQIClassifier, SinglePassMultimodalClassifier
    from models.rule_heuristic_baseline import RuleBasedSQIHeuristic
    from models.tabular_baselines import SQIGradientBoosting, SQILogisticRegression
    from PIL import Image

    T = load_tables(ROOT, RESULTS_DIR)
    but, ext = T["BUT_QDB"], {k: T[k] for k in ("CinC2011", "CinC2017")}
    X, y, fold = but[HARMONISED].values, but.label.values.astype(int), but.fold_id.values
    imgs = but.image_path.tolist()
    parts = []

    def tab_models(seed):
        set_seed(seed)
        return {LR: SQILogisticRegression(), GBM: SQIGradientBoosting(), MLP: IndependentSQIClassifier(epochs=35, lr=1e-3)}

    # tabular models, seed 42: out-of-fold and BUT-only transfer; MLP seeds 1-4: transfer only
    oof = {m: np.zeros(len(but)) for m in (LR, GBM, MLP, MM)}
    for k in range(5):
        tr, te = fold != k, fold == k
        mean, sd = zclip_fit(X[tr])
        Ztr, Zte = zclip_apply(X[tr], mean, sd), zclip_apply(X[te], mean, sd)
        for name, model in tab_models(42).items():
            oof[name][te] = model.fit(Ztr, y[tr]).predict_proba(Zte)
        set_seed(42)
        mm = SinglePassMultimodalClassifier(epochs=25, lr=1e-3).fit([imgs[i] for i in np.where(tr)[0]], Ztr, y[tr])
        oof[MM][te] = mm.predict_proba([imgs[i] for i in np.where(te)[0]], Zte)
        print(f"[cv] fold {k} done", flush=True)
    parts += [rows("BUT_QDB", but, m, 42, p) for m, p in oof.items()]
    mean, sd = zclip_fit(X)
    Z = zclip_apply(X, mean, sd)
    Zext = {k: zclip_apply(d[HARMONISED].values, mean, sd) for k, d in ext.items()}
    for name, model in tab_models(42).items():
        model.fit(Z, y)
        parts += [rows(k, ext[k], name, 42, model.predict_proba(Zext[k])) for k in ext]
    for seed in SEEDS:
        if seed != 42:
            set_seed(seed)
            mlp = IndependentSQIClassifier(epochs=35, lr=1e-3).fit(Z, y)
            parts += [rows(k, ext[k], MLP, seed, mlp.predict_proba(Zext[k])) for k in ext]
        set_seed(seed)
        mm = SinglePassMultimodalClassifier(epochs=25, lr=1e-3).fit(imgs, Z, y)
        for k in ext:
            paths, p = ext[k].image_path.tolist(), []
            for i in range(0, len(paths), 2000):    # chunks keep the decoded-strip cache small
                p.append(mm.predict_proba(paths[i:i + 2000], Zext[k][i:i + 2000]))
                for path in paths[i:i + 2000]:
                    mmf._MM_IMAGE_CACHE.pop(path, None)
            parts.append(rows(k, ext[k], MM, seed, np.concatenate(p)))
        print(f"[transfer] seed {seed} done", flush=True)
        pd.concat(parts, ignore_index=True).to_csv(PRED, index=False)

    # hand-set heuristic with the robust flatline input (coefficients unchanged; no clipping)
    rule = RuleBasedSQIHeuristic()
    for name, df in T.items():
        p = []
        for r in df.itertuples():
            sqi = {f: float(getattr(r, f)) for f in ORIGINAL}
            sqi["flatline_fraction"] = float(r.flatline_fraction_robust)
            with Image.open(r.image_path) as img:
                p.append(rule.decide_single(img, sqi, modality="multimodal")["prob_unusable"])
        parts.append(rows(name, df, HEUR, 42, p))
        print(f"[heuristic] {name} done", flush=True)
    pd.concat(parts, ignore_index=True).to_csv(PRED, index=False)
    print(f"[Saved] {PRED}")


def record_table(df: pd.DataFrame, scores: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """One row per evaluation unit (window on BUT QDB and CinC 2011, recording maximum on CinC 2017), after G1."""
    w = scores.apply(lambda col: apply_constant_rule(col.values, df.constant.values))
    w.index = df.index
    if dataset == "CinC2017":
        r = w.groupby(df.record.values).max()
        r["y"] = df.groupby("record").label.first().astype(int).loc[r.index].values
        return r
    return w.assign(y=df.label.values.astype(int))


def analyse():
    from paired_comparisons import cinc2017_windows, cluster_bootstrap_paired, delong_paired, families
    from scipy.stats import spearmanr

    T = load_tables(ROOT, RESULTS_DIR)
    P = pd.concat([pd.read_csv(PRED), pd.read_csv(TAB_PRED)], ignore_index=True)
    fam = families()
    w17, cols17 = cinc2017_windows()
    summary, seed_rows, auc_tables = [], [], {}
    for ds, df in T.items():
        # original scores of all 11 models at the window level, in the order of df
        if ds == "CinC2017":
            orig = w17.set_index("segment_id").loc[df.segment_id, cols17].reset_index(drop=True)
        else:
            f = fam[FAMILY[ds]][1]
            key = df.segment_id if ds == "BUT_QDB" else df.record_name.astype(str)
            orig = f.set_index(f.unit.astype(str)).loc[key.astype(str), [c for c in f.columns if c not in ("unit", "y", "cluster")]].reset_index(drop=True)
        models = list(orig.columns)
        if len(models) != 11:
            raise ValueError(f"{ds}: expected 11 models, got {len(models)}")
        harm = orig.copy()
        for m in REFIT:
            g = P[(P.dataset == ds) & (P.model == m) & (P.seed == 42)].set_index("segment_id")
            if len(g) != len(df):
                raise ValueError(f"{ds} {m}: {len(g)} harmonised scores for {len(df)} windows")
            harm[m] = g.loc[df.segment_id, "prob_unusable"].values
        ro = record_table(df, orig, ds).rename(columns={"y": "y_o"})
        rh = record_table(df, harm, ds)
        # the 'original' column of the comparison is the stored primary analysis (no safeguard at all)
        if ds == "CinC2017":
            raw = orig.groupby(df.record.values).max().loc[rh.index]
        else:
            raw = orig
        y = rh.y.values
        cluster = df.record.values if ds == "BUT_QDB" else None
        stored = fam[FAMILY[ds]][1]
        for m in models:
            a, b = rh[m].values, raw[m].values
            if not np.isclose(roc_auc_score(y, b), roc_auc_score(stored.y.values.astype(int), stored[m].values.astype(float))):
                raise ValueError(f"{ds} {m}: original scores do not reproduce the stored AUROC")
            d, lo, hi, p = cluster_bootstrap_paired(y, a, b, cluster) if cluster is not None else delong_paired(y, a, b)
            if np.array_equal(a, b):
                d, lo, hi, p = 0.0, 0.0, 0.0, np.nan
            summary.append({"dataset": ds, "model": m, "refitted": m in REFIT, "n_units": len(y), "n_positive": int(y.sum()),
                            "n_constant_windows": int(df.constant.sum()), "auroc_original": roc_auc_score(y, b),
                            "auroc_harmonised": roc_auc_score(y, a), "delta": d, "ci_lo": lo, "ci_hi": hi, "p_unadjusted": p,
                            "n_distinct_original": len(np.unique(b)), "n_distinct_harmonised": len(np.unique(a)),
                            "auroc_original_with_constant_rule_only": roc_auc_score(y, ro[m].values)})
        auc_tables[ds] = pd.Series({m: roc_auc_score(y, rh[m].values) for m in models})
        for m in (MLP, MM):
            if ds == "BUT_QDB":
                continue
            for seed in SEEDS:
                g = P[(P.dataset == ds) & (P.model == m) & (P.seed == seed)].set_index("segment_id")
                s = record_table(df, pd.DataFrame({m: g.loc[df.segment_id, "prob_unusable"].values}), ds)
                seed_rows.append({"dataset": ds, "model": m, "seed": seed, "auroc_harmonised": roc_auc_score(s.y.values, s[m].values),
                                  "n_distinct": len(np.unique(s[m].values))})
    S = pd.DataFrame(summary)
    S.to_csv(SUM, index=False)
    pd.DataFrame(seed_rows).to_csv(SEEDS_OUT, index=False)

    # input diagnostics: share of standardised inputs beyond CLIP SDs, BUT QDB (all windows) statistics
    but = T["BUT_QDB"]
    diag = []
    for label, feats in [("original", ORIGINAL), ("harmonised (before clipping)", HARMONISED)]:
        mean, sd = zclip_fit(but[feats].values)
        for ds, df in T.items():
            z = np.abs(zscore(df[feats].values, mean, sd))
            for j, f in enumerate(feats):
                diag.append({"inputs": label, "dataset": ds, "feature": f, "n_windows": len(df), "share_beyond_clip": float((z[:, j] > CLIP).mean()),
                             "median_abs_z": float(np.median(z[:, j])), "max_abs_z": float(z[:, j].max())})
    pd.DataFrame(diag).to_csv(DIAG, index=False)

    # rank agreement with the harmonised AUROCs, next to the original (rank_agreement_spearman.csv)
    A = pd.DataFrame(auc_tables)
    O = S.pivot(index="model", columns="dataset", values="auroc_original")
    rk = []
    for a, b in [("BUT_QDB", "CinC2011"), ("BUT_QDB", "CinC2017"), ("CinC2011", "CinC2017")]:
        ro_, rh_ = spearmanr(O[a], O[b]), spearmanr(A[a], A[b])
        rk.append({"set_a": a, "set_b": b, "n_models": len(A), "spearman_original": float(ro_.statistic), "p_original": float(ro_.pvalue),
                   "spearman_harmonised": float(rh_.statistic), "p_harmonised": float(rh_.pvalue)})
    pd.DataFrame(rk).to_csv(RANK, index=False)
    print(S[["dataset", "model", "refitted", "auroc_original", "auroc_harmonised", "delta", "ci_lo", "ci_hi", "n_distinct_original",
             "n_distinct_harmonised"]].round(3).to_string(index=False))
    print(pd.DataFrame(seed_rows).round(3).to_string(index=False))
    print(pd.DataFrame(rk).round(3).to_string(index=False))
    print(f"[Saved] {SUM.name}, {SEEDS_OUT.name}, {DIAG.name}, {RANK.name}")


if __name__ == "__main__":
    if "--analysis-only" not in sys.argv:
        fit_and_score()
    else:
        analyse()
