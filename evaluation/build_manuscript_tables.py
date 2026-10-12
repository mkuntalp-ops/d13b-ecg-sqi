"""
Build every numeric table of the manuscript directly from the result files.

Writes results/manuscript_tables_2026-10-12.md. Nothing is hand-typed: each table is computed from
the OOF / transfer / NSTDB prediction files and the scaling-sensitivity CSV.

Operating-point rules (Tables 1b and 2b), with deterministic tie-breaking:
- Target 1: maximise sensitivity subject to FPR <= limit; ties -> minimum FPR
- Target 2: minimise FPR subject to sensitivity >= 80%; ties -> maximum sensitivity
Limits: FPR <= 5% on BUT QDB (Table 1b), FPR <= 4% on CinC (Table 2b; 3 of 75 usable records).

Usage:
    python evaluation/build_manuscript_tables.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_subject_level_bootstrap_ci
from features.sqi_extractor import SQI_FEATURE_NAMES
from models.rule_heuristic_baseline import RuleBasedSQIHeuristic
from PIL import Image

RESULTS = PROJECT_ROOT / "results"
OUT = RESULTS / "manuscript_tables_2026-10-12.md"

MODELS = [
    # (display name, prob column in OOF/transfer files, short key)
    ("2D CNN (Rendered Image)", "prob_2D_CNN_Image"),
    ("1D CNN (Raw Signal)", "prob_1D_CNN_RawECG"),
    ("Rule-Based SQI Heuristic", "prob_Rule_Based_SQI_Heuristic"),
    ("SinglePass Multimodal Net", "prob_Multimodal_Fusion"),
    ("LightGBM (SQI)", "prob_LightGBM_SQI"),
    ("Independent SQI MLP", "prob_Independent_SQI_MLP"),
    ("Logistic Regression (SQI)", "prob_Logistic_Regression_SQI"),
    ("LiquidAI d1-3B (Zero-Shot)", "prob_LiquidAI_d1_3B_Neural"),
]


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def fmt_ci(mean_ci):
    return f"[{100 * mean_ci['ci_low']:.1f} – {100 * mean_ci['ci_high']:.1f}%]"


def confusion_at(y, p, tau=0.5):
    pred = (p >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return tp, fp, tn, fn


def op_cell(a, b, kind):
    """Format an operating point. kind='sens': a=sensitivity, b=FPR; kind='fpr': a=FPR, b=sensitivity."""
    if np.isnan(a):
        return "not attainable (scores saturate)"
    if kind == "sens":
        return f"{pct(a, 2)} (FPR {pct(b, 2)})"
    if a >= 0.999 and b >= 0.999:
        return "trivial (all predicted unusable)"
    return f"{pct(a, 2)} (Sens {pct(b, 2)})"


def op_thr(t):
    if np.isnan(t):
        return "—"
    return f"{t:.5f}" if t >= 1e-3 or t == 0 else f"{t:.2e}"


def operating_point(y, p, target: int, fpr_limit: float = None, sens_floor: float = 0.80):
    """Return (sensitivity, FPR, threshold) for Target 1 or Target 2 with deterministic tie-breaking."""
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    best = None
    for t in np.unique(p):
        tp, fp, _, _ = confusion_at(y, p, t)
        sens, fpr = tp / n_pos, fp / n_neg
        if target == 1:
            if fpr > fpr_limit + 1e-12:
                continue
            key = (sens, -fpr)  # maximise sensitivity, then minimise FPR
        else:
            if sens < sens_floor - 1e-12:
                continue
            key = (-fpr, sens)  # minimise FPR, then maximise sensitivity
        if best is None or key > best[0]:
            best = (key, sens, fpr, float(t))
    if best is None:
        return float("nan"), float("nan"), float("nan")
    return best[1], best[2], best[3]


def table_1(oof: pd.DataFrame) -> str:
    rows = []
    for name, col in MODELS:
        p = oof[col].values.astype(float)
        y = oof["label"].values.astype(int)
        auc, ap, br = roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)
        tp, fp, tn, fn = confusion_at(y, p)
        boot = pd.DataFrame({"subject_id": oof["subject_id"].values, "label": y, "prob": p})
        ci = compute_subject_level_bootstrap_ci(boot, pred_col="prob", target_col="label", subject_col="subject_id",
                                                n_bootstraps=1000, seed=42)
        rows.append({
            "name": name, "auc": auc, "auc_ci": ci["auroc"], "ap": ap, "brier": br,
            "cm": (tp, fp, tn, fn), "bal": (tp / (tp + fn) + tn / (tn + fp)) / 2,
            "bal_ci": ci["balanced_accuracy"], "sens": tp / (tp + fn), "sens_ci": ci["unusable_sensitivity"],
            "fr": fp / (tn + fp), "fr_ci": ci["usable_false_reject_rate"],
        })
    rows.sort(key=lambda r: -r["auc"])
    out = ["| Model Architecture | AUROC [95% CI] | AP | Brier | Fixed τ = 0.50 [TP / FP / TN / FN] | Balanced Acc. [95% CI] | Sensitivity [95% CI] | False Reject Rate [95% CI] |",
           "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for r in rows:
        out.append(f"| **{r['name']}** | {r['auc']:.3f} [{r['auc_ci']['ci_low']:.3f} – {r['auc_ci']['ci_high']:.3f}] | "
                   f"{r['ap']:.3f} | {r['brier']:.3f} | [{r['cm'][0]} / {r['cm'][1]} / {r['cm'][2]} / {r['cm'][3]}] | "
                   f"{pct(r['bal'])} {fmt_ci(r['bal_ci'])} | {pct(r['sens'])} {fmt_ci(r['sens_ci'])} | "
                   f"{pct(r['fr'])} {fmt_ci(r['fr_ci'])} |")
    return "\n".join(out)


def table_1b(oof: pd.DataFrame) -> str:
    y = oof["label"].values.astype(int)
    out = ["| Model Architecture | Target 1: Sens. (Actual FPR) | Threshold τ | Target 2: FPR (Actual Sens.) | Threshold τ |",
           "| :--- | :---: | :---: | :---: | :---: |"]
    for name, col in MODELS:
        p = oof[col].values.astype(float)
        s1, f1, t1 = operating_point(y, p, target=1, fpr_limit=0.05)
        s2, f2, t2 = operating_point(y, p, target=2)
        out.append(f"| **{name}** | {op_cell(s1, f1, 'sens')} | {op_thr(t1)} | {op_cell(f2, s2, 'fpr')} | {op_thr(t2)} |")
    return "\n".join(out)


def table_2(cinc: pd.DataFrame) -> str:
    rows = []
    y = cinc["label"].values.astype(int)
    for name, col in MODELS:
        p = cinc[col].values.astype(float)
        tp, fp, tn, fn = confusion_at(y, p)
        rows.append((roc_auc_score(y, p), name, average_precision_score(y, p), brier_score_loss(y, p),
                     (tp, fp, tn, fn), tp / (tp + fn), fp / (tn + fp), (tp / (tp + fn) + tn / (tn + fp)) / 2))
    rows.sort(key=lambda r: -r[0])
    out = ["| Model Architecture | Transfer AUROC | Transfer AP | Brier Score | Fixed τ = 0.50 [TP / FP / TN / FN] | Sensitivity | False Reject Rate | Balanced Accuracy |",
           "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for auc, name, ap, br, cm, sens, fr, bal in rows:
        out.append(f"| **{name}** | {auc:.3f} | {ap:.3f} | {br:.3f} | [{cm[0]} / {cm[1]} / {cm[2]} / {cm[3]}] | "
                   f"{pct(sens)} | {pct(fr)} | {pct(bal)} |")
    return "\n".join(out)


def table_2b(cinc: pd.DataFrame) -> str:
    y = cinc["label"].values.astype(int)
    out = ["| Model Architecture | Target 1: Sens. (Actual FPR) | Threshold τ | Target 2: FPR (Actual Sens.) | Threshold τ |",
           "| :--- | :---: | :---: | :---: | :---: |"]
    for name, col in MODELS:
        p = cinc[col].values.astype(float)
        s1, f1, t1 = operating_point(y, p, target=1, fpr_limit=3 / 75)
        s2, f2, t2 = operating_point(y, p, target=2)
        out.append(f"| **{name}** | {op_cell(s1, f1, 'sens')} | {op_thr(t1)} | {op_cell(f2, s2, 'fpr')} | {op_thr(t2)} |")
    return "\n".join(out)


def table_2c(scal: pd.DataFrame) -> str:
    variants = list(dict.fromkeys(scal["features"]))
    out = ["| Feature set | LR · StandardScaler AUROC / AP | LR · RobustScaler AUROC / AP | LR · Z-clip [-3, 3] AUROC / AP | MLP (internal scaler) AUROC / AP |",
           "| :--- | :---: | :---: | :---: | :---: |"]
    for v in variants:
        cells = []
        for model, scaler in [("LR", "StandardScaler"), ("LR", "RobustScaler"), ("LR", "Z-clip [-3,3]"),
                              ("Independent_SQI_MLP", "internal StandardScaler")]:
            r = scal[(scal["features"] == v) & (scal["model"] == model) & (scal["scaler"] == scaler)]
            cells.append(f"{r.transfer_auroc.iloc[0]:.3f} / {r.transfer_ap.iloc[0]:.3f}" if len(r) else "—")
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_3(nstdb: pd.DataFrame) -> str:
    # Rule heuristic is not produced by exp4; compute its usable probability on the same windows.
    rule = RuleBasedSQIHeuristic()
    rule_p = []
    for _, row in nstdb.iterrows():
        sqi = {k: float(row[k]) for k in SQI_FEATURE_NAMES}
        with Image.open(row["image_path"]) as img:
            rule_p.append(1.0 - rule.decide_single(img, sqi, modality="multimodal")["prob_unusable"])
    nstdb = nstdb.assign(p_usable_Rule=rule_p)
    cols = [("Logistic Regression (SQI)", "p_usable_LR"), ("LightGBM (SQI)", "p_usable_LGBM"),
            ("1D CNN (Raw Signal)", "p_usable_1D_CNN"), ("2D CNN (Image)", "p_usable_2D_CNN"),
            ("Multimodal Net", "p_usable_Multimodal"), ("Rule Heuristic Baseline", "p_usable_Rule")]

    def cell(sub, c):
        return f"{sub[c].mean():.3f} ± {sub[c].std(ddof=1):.3f}"

    out = ["| SNR Level | " + " | ".join(n for n, _ in cols) + " |",
           "| :--- | " + " | ".join([":---:"] * len(cols)) + " |"]
    clean = nstdb[nstdb["condition"] == "clean_reference"]
    out.append("| **Clean reference (MIT-BIH base)** | " + " | ".join(cell(clean, c) for _, c in cols) + " |")
    for snr in [24, 18, 12, 6, 0, -6]:
        sub = nstdb[(nstdb["condition"] == "noisy") & (nstdb["snr_db"] == snr)]
        label = {24: "+24 dB (Low Noise)", 0: "0 dB (Equal Power)", -6: "-6 dB (Severe Noise)"}.get(snr, f"{'+' if snr > 0 else ''}{snr} dB")
        out.append(f"| **{label}** | " + " | ".join(cell(sub, c) for _, c in cols) + " |")
    return "\n".join(out), len(clean), int(nstdb[nstdb["condition"] == "noisy"].shape[0])


def table_4() -> str:
    """Latency table from the per-sample measurements. Devices are the ones the profiling run actually used:
    classifiers without an explicit device default to MPS when available, so only the sklearn models run on CPU."""
    summary = pd.read_csv(RESULTS / "empirical_computational_profile_summary.csv").set_index("Model")
    raw = pd.read_csv(RESULTS / "empirical_latency_raw_measurements.csv")
    mapping = {  # raw model id -> (display name, summary row name, execution device)
        "Rule_Based_SQI_Heuristic": ("Rule-Based SQI Heuristic", "Rule-Based SQI Heuristic", "CPU"),
        "Logistic_Regression_SQI": ("Logistic Regression (SQI)", "Logistic Regression (SQI)", "CPU"),
        "LightGBM_SQI": ("LightGBM (SQI)", "LightGBM (SQI)", "CPU"),
        "Independent_SQI_MLP": ("Independent SQI MLP", "Independent SQI MLP", "Apple M2 (MPS)"),
        "1D_CNN_RawECG": ("1D CNN (Raw Signal)", "1D CNN (Raw Signal)", "Apple M2 (MPS)"),
        "2D_CNN_Image": ("2D CNN (Rendered Image)", "2D CNN (Rendered Image)", "Apple M2 (MPS)"),
        "Multimodal_Fusion": ("SinglePass Multimodal Net", "SinglePass Multimodal Net", "Apple M2 (MPS)"),
        "LiquidAI_d1_3B_Neural": ("LiquidAI d1-3B (Zero-Shot)", "Liquid AI d1-3B (Real Neural Model)", "Apple M2 (MPS)"),
    }
    out = ["| Model Architecture | Parameters | Disk Weights | Execution Device | Preproc Median (p95) | Forward Median (p95) | Total Median (p95) | Preproc Share (Median of Ratios) |",
           "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]

    def ms(x):
        return f"{x:.2f} ms"

    for raw_name, (label, summary_name, device) in mapping.items():
        r = raw[raw["model"] == raw_name]
        if r.empty:
            continue
        share = float(np.median(r["pre_ms"] / r["total_ms"]) * 100)
        srow = summary.loc[summary_name]
        params = f"{int(srow['Parameters']):,}"
        wmib = f"{float(srow['Weights_MiB']):.2f} MiB"
        out.append(f"| **{label}** | {params} | {wmib} | {device} | "
                   f"{ms(np.median(r['pre_ms']))} ({ms(np.percentile(r['pre_ms'], 95))}) | "
                   f"{ms(np.median(r['fwd_ms']))} ({ms(np.percentile(r['fwd_ms'], 95))}) | "
                   f"{ms(np.median(r['total_ms']))} ({ms(np.percentile(r['total_ms'], 95))}) | {share:.1f}% |")
    return "\n".join(out)

def table_5() -> str:
    """Seed stability: per-seed values (seed 42 = reported run, seeds 1-4 = experiments/seed_stability.py)."""
    seeds_csv = RESULTS / "seed_stability_2026-10-12.csv"
    if not seeds_csv.exists():
        return "(seed stability results not yet available)"
    st = pd.read_csv(seeds_csv)
    oof = pd.read_csv(RESULTS / "cohort1000_consolidated_oof_predictions.csv")
    cinc = pd.read_csv(RESULTS / "cinc2011_expanded_predictions.csv")
    y_b = oof["label"].values.astype(int)
    y_c = cinc["label"].values.astype(int)
    defs = [("1D CNN (Raw Signal)", "1D_CNN", "prob_1D_CNN_RawECG"),
            ("2D CNN (Rendered Image)", "2D_CNN", "prob_2D_CNN_Image"),
            ("SinglePass Multimodal Net", "Multimodal", "prob_Multimodal_Fusion")]
    out = ["| Model | Metric | Seed 42 (reported) | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± SD (seeds 1–4) |",
           "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for name, key, col in defs:
        b42 = roc_auc_score(y_b, oof[col].values)
        c42 = roc_auc_score(y_c, cinc[col].values)
        tn, fp, fn, tp = confusion_matrix(y_c, (cinc[col].values >= 0.5).astype(int), labels=[0, 1]).ravel()
        s42, f42 = tp / (tp + fn), fp / (tn + fp)
        b = st[(st.model == key) & (st.setting == "BUT_QDB_CV_pooled")].sort_values("seed")
        c = st[(st.model == key) & (st.setting == "CinC_transfer")].sort_values("seed")
        def row(label, v42, vals, scale=1.0, unit=""):
            cells = [f"{x * scale:.3f}{unit}" if unit == "" else f"{x * scale:.1f}{unit}" for x in vals]
            m, sd = np.mean(vals) * scale, np.std(vals, ddof=1) * scale
            tail = f"{m:.3f} ± {sd:.3f}" if unit == "" else f"{m:.1f} ± {sd:.1f}{unit}"
            head = f"{v42:.3f}" if unit == "" else f"{v42:.1f}{unit}"
            return f"| **{name}** | {label} | {head} | " + " | ".join(cells) + f" | {tail} |"
        out.append(row("BUT QDB pooled AUROC", b42, b["auroc"].values))
        out.append(row("CinC AUROC", c42, c["auroc"].values))
        out.append(row("CinC sensitivity at τ = 0.50", s42 * 100, c["sens_at_0.5"].values, 100.0, "%"))
        out.append(row("CinC false reject at τ = 0.50", f42 * 100, c["fr_at_0.5"].values, 100.0, "%"))
    return "\n".join(out)

MODEL_LABELS = {"Logistic_Regression_SQI": "Logistic Regression (SQI)", "LightGBM_SQI": "LightGBM (SQI)",
                "Independent_SQI_MLP": "Independent SQI MLP", "1D_CNN_RawECG": "1D CNN (Raw Signal)",
                "2D_CNN_Image": "2D CNN (Rendered Image)", "Multimodal_Fusion": "SinglePass Multimodal Net",
                "Rule_Based_SQI_Heuristic": "Rule-Based SQI Heuristic", "LiquidAI_d1_3B_Neural": "LiquidAI d1-3B (Zero-Shot)"}


def table_1d() -> str:
    """BUT QDB sensitivity analysis: class-2 windows relabelled as unusable (experiments/butqdb_class2_sensitivity.py)."""
    path = RESULTS / "butqdb_class2_sensitivity.csv"
    if not path.exists():
        return "(class-2 sensitivity results not yet available)"
    d = pd.read_csv(path)
    o = d[d.scheme == "original"].set_index("model")
    c = d[d.scheme == "class2_unusable"].set_index("model")
    bys = RESULTS / "butqdb_class2_by_seed.csv"
    seeds = pd.read_csv(bys) if bys.exists() else None
    out = ["| Model | AUROC, class 3 unusable (297) [95% CI] | AUROC, classes 2–3 unusable (490), seed 42 [95% CI] | AUROC, classes 2–3 unusable, seeds 1–4 (range) | Sens. / FR at τ = 0.50, class 3 unusable | Sens. / FR at τ = 0.50, classes 2–3 unusable |",
           "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for m in MODEL_LABELS:
        if m not in o.index:
            continue
        r1, r2 = o.loc[m], c.loc[m]
        if seeds is not None and m in set(seeds.model):
            a = seeds[(seeds.model == m) & (seeds.seed != 42)]["auroc"].values
            rng = f"{a.min():.3f}–{a.max():.3f}"
        else:
            rng = "—"
        out.append(f"| **{MODEL_LABELS[m]}** | {r1.auroc:.3f} [{r1.auroc_ci_low:.3f} – {r1.auroc_ci_high:.3f}] | "
                   f"{r2.auroc:.3f} [{r2.auroc_ci_low:.3f} – {r2.auroc_ci_high:.3f}] | {rng} | "
                   f"{pct(r1['sens_at_0.5'])} / {pct(r1['fr_at_0.5'])} | {pct(r2['sens_at_0.5'])} / {pct(r2['fr_at_0.5'])} |")
    return "\n".join(out)


def table_2d() -> str:
    """CinC transfer over random balanced draws (experiments/cinc_random_draws.py); models trained once on the cohort."""
    path = RESULTS / "cinc_random_draws_summary.csv"
    if not path.exists():
        return "(random-draw results not yet available)"
    d = pd.read_csv(path)
    rnd = d[d.draw.str.startswith("random_seed")]
    orig = d[d.draw == "first_N_original"].set_index("model")
    out = ["| Model | AUROC: first-N (reported) | AUROC: random draws, mean ± SD (min–max) | Sensitivity at τ = 0.50: first-N | Sensitivity at τ = 0.50: random draws, mean ± SD | False reject at τ = 0.50: random draws, mean ± SD |",
           "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for m in MODEL_LABELS:
        if m not in orig.index:
            continue
        r = rnd[rnd.model == m]
        a = r["auroc"].values
        sens = r["sens_at_0.5"].values * 100
        fr = r["fr_at_0.5"].values * 100
        out.append(f"| **{MODEL_LABELS[m]}** | {orig.loc[m, 'auroc']:.3f} | {a.mean():.3f} ± {a.std(ddof=1):.3f} ({a.min():.3f}–{a.max():.3f}) | "
                   f"{orig.loc[m, 'sens_at_0.5'] * 100:.1f}% | {sens.mean():.1f} ± {sens.std(ddof=1):.1f}% | {fr.mean():.1f} ± {fr.std(ddof=1):.1f}% |")
    return "\n".join(out)

def table_6() -> str:
    """Variance decomposition for the CinC subsets and training seeds (experiments/cinc_draws_by_seed.py)."""
    path = RESULTS / "cinc_draws_by_seed_summary.csv"
    if not path.exists():
        return "(subset-by-seed results not yet available)"
    d = pd.read_csv(path)
    rnd = d[d.draw.str.startswith("random_seed")]
    out = ["| Model | AUROC range over seeds × subsets | SD across training seeds (mean over subsets) | SD across subsets (mean over seeds) |",
           "| :--- | :---: | :---: | :---: |"]
    for m in MODEL_LABELS:
        r = rnd[rnd.model == m]
        if r.empty:
            continue
        piv = r.pivot_table(index="seed", columns="draw", values="auroc")
        across_seeds = piv.std(axis=0, ddof=1).mean()
        across_subsets = piv.std(axis=1, ddof=1).mean()
        out.append(f"| **{MODEL_LABELS[m]}** | {r.auroc.min():.3f}–{r.auroc.max():.3f} | {across_seeds:.3f} | {across_subsets:.3f} |")
    return "\n".join(out)

def main():
    oof = pd.read_csv(RESULTS / "cohort1000_consolidated_oof_predictions.csv")
    cinc = pd.read_csv(RESULTS / "cinc2011_expanded_predictions.csv")
    scal = pd.read_csv(RESULTS / "cinc2011_scaling_sensitivity.csv")
    nstdb = pd.read_csv(RESULTS / "exp4_revised_nstdb_predictions.csv")
    t3, n_clean, n_active = table_3(nstdb)

    sections = [
        ("Table 1", table_1(oof)),
        ("Table 1b", table_1b(oof)),
        ("Table 2", table_2(cinc)),
        ("Table 2b", table_2b(cinc)),
        ("Table 2c", table_2c(scal)),
        ("Table 3", t3),
        ("Table 4", table_4()),
        ("Table 5", table_5()),
        ("Table 1d", table_1d()),
        ("Table 2d", table_2d()),
        ("Table 6", table_6()),
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Manuscript tables regenerated 2026-10-12\n\n")
        f.write(f"NSTDB windows: clean control = {n_clean}, active noise = {n_active}\n\n")
        for title, body in sections:
            f.write(f"## {title}\n\n{body}\n\n")
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
