"""
Frozen external test on PTB-XL (analysis A of ANALYSIS_PLAN_2026-11-09_additional_analyses.md).

Models are fitted on the 988 BUT QDB windows only and applied unchanged. The outcome (annotated noise) is a proxy for the
study target, not unusability. Run the steps in order; each can be repeated except --sample and --analyse, whose
outputs are never overwritten.

Usage (ECG_RESULTS_DIR=results/v2_2026-11):
  python experiments/ptbxl_frozen_test.py --sample        one record per patient, from the metadata only
  python experiments/ptbxl_frozen_test.py --extract       signals, images and SQIs of the 12 leads
  python experiments/ptbxl_frozen_test.py --score-d1      zero-shot d1-3B with feature capture (about 1 s per window; resumes)
  python experiments/ptbxl_frozen_test.py --score-others  BUT-only models and the HuBERT-ECG embeddings
  python experiments/ptbxl_frozen_test.py --analyse
Outputs: $ECG_RESULTS_DIR/ptbxl_frozen_{sample,d1_predictions,other_predictions,window_scores,summary,comparisons,constant_leads}.csv
"""

import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from config import LOCAL_DATASETS_DIR, RESULTS_DIR  # noqa: E402
from data.ecg_rendering import render_ecg_image  # noqa: E402
from data.units import to_millivolts  # noqa: E402
from features.sqi_extractor import extract_sqi_features  # noqa: E402

PTB_DIR = LOCAL_DATASETS_DIR / "ptb-xl"
PILOT_IMAGES = LOCAL_DATASETS_DIR / "ecg_images" / "ptbxl_pilot"
WORK = PROJECT_ROOT / "cache" / "datasets" / "ptbxl"
RAW, SIG_DIR, IMG_DIR = WORK / "raw", WORK / "signals", WORK / "images"
WINDOWS = WORK / "ptbxl_windows.csv"
OUT = {k: RESULTS_DIR / f"ptbxl_frozen_{k}.csv"
       for k in ["sample", "d1_predictions", "other_predictions", "window_scores", "summary", "comparisons", "constant_leads"]}
SEED, PER_STRATUM, N_LEADS, FS, N_SAMPLES = 42, 300, 12, 500, 5000
STRATA = ["global", "partial", "clean"]
PROBE, HUB, D1 = "d1_image_probe", "HuBERT_ECG_probe", "LiquidAI_d1_3B_Neural"
TASK_MODELS = ["Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG", "2D_CNN_Image",
               "Multimodal_Fusion", "Rule_Based_SQI_Heuristic"]
ALL_MODELS = TASK_MODELS + [D1, PROBE, HUB]
CNN2 = "2D_CNN_Image"
LEAD_II = 1


def assign_strata(meta: pd.DataFrame) -> pd.Series:
    """Stratum of each record from the four noise columns; empty string for baseline-drift-only records."""
    text = meta[["static_noise", "burst_noise"]].fillna("").agg(" ".join, axis=1).str.lower()
    is_global = text.str.contains("alles") | text.str.contains("noisy recording")
    any_noise = meta[["static_noise", "burst_noise", "electrodes_problems"]].notna().any(axis=1)
    no_note = meta[["static_noise", "burst_noise", "baseline_drift", "electrodes_problems"]].isna().all(axis=1)
    out = pd.Series("", index=meta.index, dtype=object)
    out[any_noise] = "partial"
    out[is_global] = "global"
    out[no_note] = "clean"
    return out


def pilot_ids() -> set:
    ids = {int(p.stem.split("_")[0]) for p in (PTB_DIR / "records500").rglob("*.hea")}
    ids |= {int(p.stem.split("_")[1]) for p in PILOT_IMAGES.glob("ecg_*.png")}
    return ids


def sample() -> None:
    if OUT["sample"].exists():
        raise FileExistsError(f"{OUT['sample']} exists; the sample is drawn once")
    meta = pd.read_csv(PTB_DIR / "ptbxl_database.csv")
    meta["stratum"] = assign_strata(meta)
    excluded = pilot_ids()
    print(f"[sample] stratum sizes before exclusions: {meta.stratum.value_counts().to_dict()}; pilot records: {len(excluded)}")
    pool = meta[(meta.stratum != "") & ~meta.ecg_id.isin(excluded)]
    rng, seen, rows = np.random.default_rng(SEED), set(), []
    for stratum in STRATA:
        group = pool[pool.stratum == stratum]
        taken = 0
        for i in rng.permutation(len(group)):
            r = group.iloc[int(i)]
            if r.patient_id in seen:
                continue
            seen.add(r.patient_id)
            rows.append({"ecg_id": int(r.ecg_id), "stratum": stratum, "filename_hr": r.filename_hr})
            taken += 1
            if taken == PER_STRATUM:
                break
        if taken != PER_STRATUM:
            raise ValueError(f"{stratum}: only {taken} records available")
    out = pd.DataFrame(rows)
    if out.ecg_id.duplicated().any():
        raise ValueError("a record was sampled twice")
    out.to_csv(OUT["sample"], index=False)
    print(f"[Saved] {OUT['sample']}: {out.stratum.value_counts().to_dict()}", flush=True)


def process_record(row: tuple) -> list:
    ecg_id, stratum, name = row
    r = wfdb.rdrecord(str(RAW / name))
    if int(r.fs) != FS or r.p_signal.shape != (N_SAMPLES, N_LEADS):
        raise ValueError(f"{name}: fs {r.fs}, shape {r.p_signal.shape}")
    rows = []
    for li, lead in enumerate(r.sig_name):
        x = np.nan_to_num(to_millivolts(r.p_signal[:, li], r.units[li]).astype(np.float32), nan=0.0)
        seg = f"ptbxl_{ecg_id}_lead{li}"
        sig_path, img_path = SIG_DIR / f"{seg}.npy", IMG_DIR / f"{seg}.png"
        np.save(sig_path, x)
        if not img_path.exists():
            render_ecg_image(x, FS, dpi=100).save(img_path)
        rows.append({"segment_id": seg, "record_name": ecg_id, "label": int(stratum != "clean"), "stratum": stratum, "lead_idx": li,
                     "lead": lead.upper(), "signal_unit": "mV", "signal_path": str(sig_path), "image_path": str(img_path),
                     **extract_sqi_features(x, FS)})
    return rows


def extract() -> None:
    s = pd.read_csv(OUT["sample"])
    for d in (RAW, SIG_DIR, IMG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PTB_DIR / "ptb-xl-data.zip") as z:
        names = set(z.namelist())
        for f in s.filename_hr:
            for ext in (".hea", ".dat"):
                member = f + ext
                if member not in names:
                    raise FileNotFoundError(member)
                target = RAW / member
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(member))
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for rs in ex.map(process_record, s[["ecg_id", "stratum", "filename_hr"]].itertuples(index=False, name=None), chunksize=8):
            rows.extend(rs)
    df = pd.DataFrame(rows)
    expected = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    if len(df) != len(s) * N_LEADS or not all(list(g.lead) == expected for _, g in df.groupby("record_name")):
        raise ValueError("unexpected window count or lead order")
    df.to_csv(WINDOWS, index=False)
    print(f"[Saved] {WINDOWS}: {len(df)} windows from {df.record_name.nunique()} records", flush=True)


def score_d1() -> None:
    from d1_linear_probe import FEAT_DIR, Capture
    from d1_twelve_lead_full import score_row
    from feature_cache import save_set
    from models.d1_real_neural_model import RealD1DecisionModel

    cols, path, chunk_size = ["segment_id", "record_name", "lead_idx", "label", "prob_unusable"], OUT["d1_predictions"], 500
    df = pd.read_csv(WINDOWS)
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)
    if done.empty:
        pd.DataFrame(columns=cols).to_csv(path, index=False)
    todo = df[~df.segment_id.isin(done.segment_id)].reset_index(drop=True)
    print(f"[d1] {len(done)} windows done, {len(todo)} to score", flush=True)
    d1, part = RealD1DecisionModel(device="mps"), len(list(FEAT_DIR.glob("PTBXL_part*_ids.csv")))
    cap = Capture(d1.model)
    for start in range(0, len(todo), chunk_size):
        chunk = todo.iloc[start:start + chunk_size]
        rows, state, image = [], [], []
        for _, r in chunk.iterrows():
            p, s, i = score_row(d1, cap, r)
            rows.append({c: r[c] for c in cols[:-1]} | {"prob_unusable": p})
            state.append(s)
            image.append(i)
        save_set(FEAT_DIR, f"PTBXL_part{part:03d}", chunk.segment_id.tolist(), {"state": np.stack(state), "image": np.stack(image)})
        pd.DataFrame(rows)[cols].to_csv(path, mode="a", header=False, index=False)
        part += 1
        print(f"[d1] {start + len(chunk)}/{len(todo)}", flush=True)
    final = pd.read_csv(path)
    if len(final) != len(df) or final.segment_id.nunique() != len(df):
        raise ValueError(f"{path}: incomplete")
    print(f"[Saved] {path}", flush=True)


def score_others() -> None:
    import torch

    from cinc_random_draws import fit_models
    from external_validation import WIN_OUT, score_frame
    from feature_cache import missing_ids, save_set
    from hubert_ecg_probe import FEAT, embed, load_model

    if OUT["other_predictions"].exists():
        raise FileExistsError(f"{OUT['other_predictions']} exists")
    df = pd.read_csv(WINDOWS)
    models = fit_models()
    primary = pd.read_csv(WIN_OUT, low_memory=False)
    primary = primary[primary.dataset == "CinC2011"].set_index("segment_id")
    from config import CINC2011_LEADII_FULL
    check = score_frame(models, pd.read_csv(CINC2011_LEADII_FULL).assign(lead_idx=1), "check").set_index("segment_id")  # lead II manifest has no lead_idx column
    worst = max(float(np.abs(check[m] - primary.loc[check.index, m]).max()) for m in TASK_MODELS)
    print(f"[check] max |refit - primary| on CinC 2011 lead II over the task models = {worst:.6f}", flush=True)
    if worst > 1e-3:
        raise RuntimeError("the BUT-only models do not reproduce the primary scores")
    score_frame(models, df, "PTBXL").to_csv(OUT["other_predictions"], index=False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    todo = df[df.segment_id.isin(set(missing_ids(FEAT, None, df.segment_id)))]
    if len(todo):
        save_set(FEAT, "PTBXL_12lead", todo.segment_id.tolist(), {None: embed(load_model(device), todo.signal_path.tolist(), device)})
    print(f"[Saved] {OUT['other_predictions']}; HuBERT-ECG embeddings: {len(todo)}", flush=True)


def analyse() -> None:
    from sklearn.metrics import roc_auc_score

    from d1_linear_probe import FEAT_DIR, probe as d1_probe, window_sets
    from evaluation.paired_comparisons import delong_paired, holm
    from external_validation import rec_metrics
    from feature_cache import load_features
    from hubert_ecg_probe import FEAT, probe as hub_probe
    from lodo_multisource import load_sources

    for k in ["window_scores", "summary", "comparisons", "constant_leads"]:
        if OUT[k].exists():
            raise FileExistsError(f"{OUT[k]} exists; the analysis is run once")
    df = pd.read_csv(WINDOWS).reset_index(drop=True)
    but = window_sets()["BUT_QDB"].reset_index(drop=True)
    fit = d1_probe(groups=True).fit(load_features(FEAT_DIR, "image", but.segment_id), but.label.values, groups=but.subject_id.values)
    src = load_sources()["BUT_QDB"]
    hub_fit = hub_probe().fit(load_features(FEAT, None, src.segment_id), src.label.values, groups=src.unit.values)
    print(f"[probes] d1 C = {fit.best_params_['logisticregression__C']}; HuBERT-ECG C = {hub_fit.best_params_['logisticregression__C']}", flush=True)

    other = pd.read_csv(OUT["other_predictions"]).set_index("segment_id")
    zero = pd.read_csv(OUT["d1_predictions"]).set_index("segment_id")["prob_unusable"]
    w = df[["segment_id", "record_name", "stratum", "lead_idx", "signal_std"]].copy()
    for m in TASK_MODELS:
        w[m] = w.segment_id.map(other[m])
    w[D1] = w.segment_id.map(zero)
    w[PROBE] = fit.predict_proba(load_features(FEAT_DIR, "image", df.segment_id))[:, 1]
    w[HUB] = hub_fit.predict_proba(load_features(FEAT, None, df.segment_id))[:, 1]
    if not np.isfinite(w[ALL_MODELS].values).all():
        raise ValueError("non-finite scores")

    strata = w.groupby("record_name").stratum.first()
    recs = {"lead_II": w[w.lead_idx == LEAD_II].set_index("record_name")[ALL_MODELS].sort_index()}
    for agg in ["max", "mean"]:
        recs[f"{agg}_12"] = w.groupby("record_name")[ALL_MODELS].agg(agg).sort_index()
    strata = strata.loc[recs["lead_II"].index]

    def subset(pos: str):
        mask = strata.isin([pos, "clean"]).values
        return mask, (strata[mask] == pos).astype(int).values

    summary = []
    for outcome, pos in [("global_vs_clean", "global"), ("partial_vs_clean", "partial")]:
        mask, y = subset(pos)
        for agg, r in recs.items():
            summary += [{"outcome": outcome, "aggregation": agg, "model": m, **rec_metrics(y, r[m].values[mask])} for m in ALL_MODELS]
    summary = pd.DataFrame(summary).drop(columns=["sens_at_0.5", "fr_at_0.5", "ppv_at_0.5", "npv_at_0.5"])

    mask, y = subset("global")
    def test(a: pd.Series, b: pd.Series, name: str, family: str) -> dict:
        d, lo, hi, p = delong_paired(y, a.values[mask], b.values[mask])
        return {"family": family, "comparison": name, "auroc_a": roc_auc_score(y, a.values[mask]), "auroc_b": roc_auc_score(y, b.values[mask]),
                "delta": d, "ci_low": lo, "ci_high": hi, "p": p}
    primary = pd.DataFrame([test(recs[a][PROBE], recs[a][CNN2], f"probe vs 2D CNN ({a})", "primary") for a in ["max_12", "mean_12"]])
    primary["p_holm"] = holm(primary.p.values)
    second = pd.DataFrame([test(recs[a][PROBE], recs[a][m], f"probe vs {m} ({a})", "secondary") for a in ["max_12", "mean_12"]
                           for m in ALL_MODELS if m != PROBE])
    second["p_holm"] = holm(second.p.values)
    comparisons = pd.concat([primary, second], ignore_index=True)

    flat = w.assign(c=w.signal_std == 0).groupby("record_name").c.sum().loc[strata.index]
    rows = []
    for agg in ["max_12", "mean_12"]:
        for m in ALL_MODELS:
            keep = mask & (flat.values == 0)
            y_keep = (strata[keep] == "global").astype(int).values
            rows.append({"aggregation": agg, "model": m, "n_records_constant": int((flat.values > 0).sum()), "n_global_constant": int(((flat.values > 0) & (strata.values == "global")).sum()),
                         "n_without": int(keep.sum()), "auroc_without_constant": roc_auc_score(y_keep, recs[agg][m].values[keep])})
    constant = pd.DataFrame(rows)

    w.drop(columns="signal_std").to_csv(OUT["window_scores"], index=False)
    summary.to_csv(OUT["summary"], index=False)
    comparisons.to_csv(OUT["comparisons"], index=False)
    constant.to_csv(OUT["constant_leads"], index=False)
    print(summary.pivot_table(index="model", columns=["outcome", "aggregation"], values="auroc").round(3).to_string())
    print(primary.to_string(index=False))
    print(second.round(4).to_string(index=False))
    print(constant.head(1).to_string(index=False))
    print("[Saved] analysis outputs", flush=True)


if __name__ == "__main__":
    steps = {"--sample": sample, "--extract": extract, "--score-d1": score_d1, "--score-others": score_others, "--analyse": analyse}
    chosen = [f for a, f in steps.items() if a in sys.argv]
    if len(chosen) != 1:
        raise SystemExit(f"choose one of {list(steps)}")
    chosen[0]()
