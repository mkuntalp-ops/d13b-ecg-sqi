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
            ("Multimodal Net", "p_usable_Multimodal"), ("Rule Heuristic Baseline", "p_usable_Rule"),
            ("d1-3B (Zero-Shot)", "p_usable_d1_3B")]

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


def table_3b(nstdb: pd.DataFrame) -> str:
    """Paired change in P(usable): each noisy window minus its clean base-record partner (same start time)."""
    cols = [("Logistic Regression (SQI)", "p_usable_LR"), ("LightGBM (SQI)", "p_usable_LGBM"),
            ("1D CNN (Raw Signal)", "p_usable_1D_CNN"), ("2D CNN (Image)", "p_usable_2D_CNN"),
            ("Multimodal Net", "p_usable_Multimodal"), ("d1-3B (zero-shot)", "p_usable_d1_3B")]
    clean = nstdb[nstdb["condition"] == "clean_reference"].set_index("segment_id")
    noisy = nstdb[nstdb["condition"] == "noisy"]
    out = ["| SNR Level | " + " | ".join(n for n, _ in cols) + " |",
           "| :--- | " + " | ".join([":---:"] * len(cols)) + " |"]
    for snr in [24, 18, 12, 6, 0, -6]:
        sub = noisy[noisy["snr_db"] == snr]
        label = {24: "+24 dB (Low Noise)", 0: "0 dB (Equal Power)", -6: "-6 dB (Severe Noise)"}.get(snr, f"{'+' if snr > 0 else ''}{snr} dB")
        cells = []
        for _, c in cols:
            d = sub[c].values - clean.loc[sub["partner_segment_id"], c].values
            cells.append(f"{d.mean():+.3f} ± {d.std(ddof=1):.3f}")
        out.append(f"| **{label}** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def _missing(path: Path) -> str:
    return f"({path.name} not yet available)"


def table_1c() -> str:
    """Nested threshold selection on BUT QDB (experiments/nested_thresholds.py)."""
    path = RESULTS / "nested_thresholds_summary.csv"
    if not path.exists():
        return _missing(path)
    s = pd.read_csv(path).set_index(["model", "rule"])
    out = ["| Model Architecture | Target 1 (FPR ≤ 5%): nested Sens. / FPR | Target 1: post-hoc Sens. / FPR | Target 2 (Sens. ≥ 80%): nested Sens. / FPR | Target 2: post-hoc Sens. / FPR | Youden: nested Sens. / FPR |",
           "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for m, label in MODEL_LABELS.items():
        if (m, "target1_fpr5") not in s.index:
            continue
        cell = lambda r, a, b: f"{100 * s.loc[(m, r), a]:.1f}% / {100 * s.loc[(m, r), b]:.1f}%"
        out.append(f"| **{label}** | {cell('target1_fpr5', 'sens', 'fpr')} | {cell('target1_fpr5', 'posthoc_sens', 'posthoc_fpr')} | "
                   f"{cell('target2_sens80', 'sens', 'fpr')} | {cell('target2_sens80', 'posthoc_sens', 'posthoc_fpr')} | "
                   f"{cell('youden', 'sens', 'fpr')} |")
    return "\n".join(out)


def table_1e() -> str:
    """BUT QDB metrics re-weighted to the natural prevalence (experiments/prevalence_subjects_features.py)."""
    path = RESULTS / "butqdb_natural_prevalence_metrics.csv"
    if not path.exists():
        return _missing(path)
    d = pd.read_csv(path)
    out = ["| Model Architecture | AP, quota (28.4%) | AP, natural (16.0%) | AP, natural excl. 105001 (2.8%) | PPV at τ = 0.50, 16.0% | PPV at τ = 0.50, 2.8% | NPV at τ = 0.50, 2.8% | False rejections per hour of usable signal |",
           "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for m, label in MODEL_LABELS.items():
        g = d[d.model == m].set_index("prevalence_scope")
        if g.empty:
            continue
        q, a, e = g.loc["quota cohort"], g.loc["natural, all records"], g.loc["natural, excl. 105001"]
        out.append(f"| **{label}** | {q.ap_reweighted:.3f} | {a.ap_reweighted:.3f} | {e.ap_reweighted:.3f} | {a['ppv_at_0.5']:.3f} | "
                   f"{e['ppv_at_0.5']:.3f} | {e['npv_at_0.5']:.3f} | {e.false_rejections_per_hour_usable:.1f} |")
    return "\n".join(out)


def table_1f() -> str:
    """Per-subject sensitivity and false reject rate at tau = 0.5 (pooled seed-42 OOF predictions)."""
    path = RESULTS / "butqdb_subject_heterogeneity.csv"
    if not path.exists():
        return _missing(path)
    d = pd.read_csv(path)
    models = ["1D_CNN_RawECG", "2D_CNN_Image", "Multimodal_Fusion", "LightGBM_SQI", "Rule_Based_SQI_Heuristic", "LiquidAI_d1_3B_Neural"]
    out = ["| Held-out subject (fold) | Unusable / usable windows | " + " | ".join(MODEL_LABELS[m] for m in models) + " |",
           "| :--- | :---: | " + " | ".join([":---:"] * len(models)) + " |"]
    for _, r in d[d.n_unusable > 0].iterrows():
        out.append(f"| **{int(r.subject_id)} ({int(r.fold_id)})** | {int(r.n_unusable)} / {int(r.n_usable)} | "
                   + " | ".join(f"{100 * r[f'{m}_sens']:.0f}%" for m in models) + " |")
    out.append("| **False reject, range over 15 subjects** | — / 750 | "
               + " | ".join(f"{100 * d[f'{m}_fr'].min():.0f}–{100 * d[f'{m}_fr'].max():.0f}%" for m in models) + " |")
    return "\n".join(out)


def table_2e() -> str:
    """CinC 2011 scored on all 12 leads (experiments/external_validation.py)."""
    path = RESULTS / "cinc2011_twelve_lead_summary.csv"
    if not path.exists():
        return _missing(path)
    t = pd.read_csv(path)
    first = t[t.draw == "first_N_original"].pivot_table(index="model", columns="aggregation", values="auroc")
    rnd = t[t.draw.str.startswith("random")].groupby(["model", "aggregation"]).auroc
    out = ["| Model Architecture | Reported subset: lead II | Reported subset: max over 12 leads | Reported subset: mean over 12 leads | Random subsets: lead II, mean (min–max) | Random subsets: max over 12, mean (min–max) | Random subsets: mean over 12, mean (min–max) |",
           "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for m, label in MODEL_LABELS.items():
        if m not in first.index:
            continue
        cells = [f"{first.loc[m, a]:.3f}" for a in ["lead_II", "max_12", "mean_12"]]
        for a in ["lead_II", "max_12", "mean_12"]:
            try:
                v = rnd.get_group((m, a))
                cells.append(f"{v.mean():.3f} ({v.min():.3f}–{v.max():.3f})")
            except KeyError:
                cells.append("—")
        out.append(f"| **{label}** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_2f() -> str:
    """Thresholds chosen on BUT QDB applied frozen to the external sets (experiments/nested_thresholds.py)."""
    path = RESULTS / "threshold_transfer_summary.csv"
    if not path.exists():
        return _missing(path)
    d = pd.read_csv(path)
    out = ["| Model Architecture | Rule | Threshold (from BUT QDB) | BUT QDB Sens. / FPR | CinC 2011 reported Sens. / FPR | CinC 2011 random subsets Sens. range | CinC 2017 Sens. / FPR |",
           "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"]
    for m, label in MODEL_LABELS.items():
        for rule, rname in [("target1_fpr5", "Target 1 (FPR ≤ 5%)"), ("target2_sens80", "Target 2 (Sens. ≥ 80%)")]:
            g = d[(d.model == m) & (d.rule == rule)]
            if g.empty:
                continue
            b = g[g.dataset.str.startswith("BUT")].iloc[0]
            c = g[g.dataset == "CinC2011 first-N (lead II)"].iloc[0]
            rr = g[g.dataset.str.contains("random")]
            s7 = g[g.dataset.str.startswith("CinC2017")].iloc[0]
            out.append(f"| **{label}** | {rname} | {b.threshold:.4g} | {100 * b.sens:.1f}% / {100 * b.fpr:.1f}% | "
                       f"{100 * c.sens:.1f}% / {100 * c.fpr:.1f}% | {100 * rr.sens.min():.1f}–{100 * rr.sens.max():.1f}% | "
                       f"{100 * s7.sens:.1f}% / {100 * s7.fpr:.1f}%{' †' if 'balanced' in s7.dataset else ''} |")
    return "\n".join(out)


def table_2g() -> str:
    """Univariate AUROC of each SQI across datasets (experiments/prevalence_subjects_features.py)."""
    path = RESULTS / "feature_shift_univariate_auroc.csv"
    if not path.exists():
        return _missing(path)
    d = pd.read_csv(path).dropna(subset=["BUT QDB (1,047 windows)"])
    cols = [c for c in d.columns if c != "feature"]
    out = ["| SQI feature | " + " | ".join(cols) + " |", "| :--- | " + " | ".join([":---:"] * len(cols)) + " |"]
    for _, r in d.iterrows():
        out.append(f"| `{r.feature}` | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    return "\n".join(out)


def table_4b() -> str:
    """Extended latency (experiments/latency_extended.py)."""
    path = RESULTS / "latency_extended_summary.csv"
    if not path.exists():
        return _missing(path)
    d = pd.read_csv(path).set_index("measurement")
    names = [("convert_ms", "Unit conversion (µV to mV)"), ("resample_ms", "Resampling 1,000 → 500 Hz"),
             ("sqi_ms", "SQI extraction (13 features)"), ("render_matplotlib_ms", "Rendering, Matplotlib (used by all image models)"),
             ("render_raster_ms", "Rendering, direct raster (PIL; timing only)")]
    for dev, dname in [("cpu", "CPU"), ("mps", "MPS")]:
        names += [(f"cnn1d_forward_{dev}_ms", f"1D CNN forward, {dname}"), (f"cnn1d_raw_to_decision_{dev}_ms", f"1D CNN raw-to-decision, {dname}"),
                  (f"cnn2d_forward_{dev}_ms", f"2D CNN forward, {dname}"), (f"mm_forward_{dev}_ms", f"Multimodal net forward, {dname}")]
    out = ["| Step | Median (three runs) | Range of run medians | p95 (median over runs) |", "| :--- | :---: | :---: | :---: |"]
    for k, n in names:
        if k in d.index:
            r = d.loc[k]
            out.append(f"| {n} | {r.median_over_runs_ms:.2f} ms | {r.min_run_ms:.2f}–{r.max_run_ms:.2f} ms | {r.p95_median_over_runs_ms:.2f} ms |")
    return "\n".join(out)


def table_7b() -> str:
    """Linear probe on frozen d1-3B representations (experiments/d1_linear_probe.py)."""
    path = RESULTS / "d1_probe_summary.csv"
    if not path.exists():
        return _missing(path)
    d = pd.read_csv(path)
    out = ["| d1-3B representation | BUT QDB AUROC [95% CI] (5-fold subject CV) | CinC 2011 reported subset AUROC | CinC 2011 random subsets AUROC, mean (min–max) | CinC 2017 balanced subset AUROC (recording max) |",
           "| :--- | :---: | :---: | :---: | :---: |"]
    for f, name in [("zero-shot d1 (primary)", "Zero-shot choice probability (primary configuration)"),
                    ("state", "Linear probe on the decision state (final norm, answer position)"),
                    ("image", "Linear probe on image tokens (projector output, mean)")]:
        g = d[d.features == f]
        b = g[g.dataset.str.startswith("BUT")].iloc[0]
        c = g[g.dataset == "CinC2011 first_N_original"].iloc[0]
        r = g[g.dataset.str.contains("random")].auroc
        s = g[g.dataset.str.startswith("CinC2017")].iloc[0]
        out.append(f"| **{name}** | {b.auroc:.3f} [{b.auroc_ci_low:.3f} – {b.auroc_ci_high:.3f}] | {c.auroc:.3f} | "
                   f"{r.mean():.3f} ({r.min():.3f}–{r.max():.3f}) | {s.auroc:.3f} |")
    return "\n".join(out)


def table_8() -> str:
    """CinC 2017 external validation at natural prevalence (experiments/external_validation.py)."""
    path = RESULTS / "external_cinc2017_summary.csv"
    if not path.exists():
        return _missing(path)
    d = pd.read_csv(path)
    a = d[(d.cohort == "all recordings")]
    mx, mn = a[a.aggregation == "max"].set_index("model"), a[a.aggregation == "mean"].set_index("model")
    out = ["| Model Architecture | AUROC [95% CI], recording max | AP [95% CI], recording max (prevalence 3.3%) | Sens. / FR at τ = 0.50 | PPV / NPV at τ = 0.50 | AUROC, recording mean (secondary) |",
           "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for m, label in MODEL_LABELS.items():
        if m not in mx.index:
            continue
        r = mx.loc[m]
        out.append(f"| **{label}** | {r.auroc:.3f} [{r.auroc_ci_low:.3f} – {r.auroc_ci_high:.3f}] | {r.ap:.3f} [{r.ap_ci_low:.3f} – {r.ap_ci_high:.3f}] | "
                   f"{100 * r['sens_at_0.5']:.1f}% / {100 * r['fr_at_0.5']:.1f}% | {r['ppv_at_0.5']:.3f} / {r['npv_at_0.5']:.3f} | {mn.loc[m, 'auroc']:.3f} |")
    return "\n".join(out)


def table_8b() -> str:
    """CinC 2017 balanced subset on which d1-3B was evaluated."""
    path = RESULTS / "external_cinc2017_summary.csv"
    if not path.exists():
        return _missing(path)
    d = pd.read_csv(path)
    b = d[(d.cohort == "balanced subset (d1)") & (d.aggregation == "max")].set_index("model")
    out = ["| Model Architecture | AUROC [95% CI], recording max (278 noisy + 278 non-noisy) | Sens. / FR at τ = 0.50 |",
           "| :--- | :---: | :---: |"]
    for m, label in MODEL_LABELS.items():
        if m in b.index:
            r = b.loc[m]
            out.append(f"| **{label}** | {r.auroc:.3f} [{r.auroc_ci_low:.3f} – {r.auroc_ci_high:.3f}] | {100 * r['sens_at_0.5']:.1f}% / {100 * r['fr_at_0.5']:.1f}% |")
    return "\n".join(out)


def table_9() -> str:
    """Paired comparisons with the best model of each dataset (evaluation/paired_comparisons.py)."""
    path = RESULTS / "paired_comparisons.csv"
    if not path.exists():
        return _missing(path)
    c = pd.read_csv(path)
    labels = {**MODEL_LABELS, "TabPFN_SQI": "TabPFN (SQI)", "HuBERT_ECG_probe": "HuBERT-ECG probe", "d1_image_probe": "d1-3B image-token probe"}
    fams = list(dict.fromkeys(c.family))
    auc = {}
    for f in fams:
        g = c[c.family == f]
        s = pd.concat([g[["model_a", "auroc_a"]].set_axis(["m", "a"], axis=1), g[["model_b", "auroc_b"]].set_axis(["m", "a"], axis=1)])
        auc[f] = s.drop_duplicates("m").set_index("m")["a"]
    out = ["| Model | " + " | ".join(fams) + " |", "| :--- | " + " | ".join([":---:"] * len(fams)) + " |"]
    for m in labels:
        cells = []
        for f in fams:
            if m not in auc[f].index:
                cells.append("—")
                continue
            best = auc[f].idxmax()
            if m == best:
                cells.append(f"**{auc[f][m]:.3f}** (best)")
                continue
            g = c[(c.family == f) & (((c.model_a == m) & (c.model_b == best)) | ((c.model_b == m) & (c.model_a == best)))].iloc[0]
            cells.append(f"{auc[f][m]:.3f} (p = {g.p_holm:.3f})" if g.p_holm >= 0.001 else f"{auc[f][m]:.3f} (p < 0.001)")
        out.append(f"| **{labels[m]}** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_10() -> str:
    """Leave-one-dataset-out multi-source training vs training on BUT QDB only (experiments/lodo_multisource.py)."""
    path = RESULTS / "lodo_summary.csv"
    if not path.exists():
        return _missing(path)
    s = pd.read_csv(path)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paired_comparisons import holm
    s["p_holm"] = np.nan
    for key, g in s[s.p_boot.notna()].groupby("test"):  # Holm within each held-out dataset
        s.loc[g.index, "p_holm"] = holm(g.p_boot.values)
    labels = {**MODEL_LABELS, "TabPFN_SQI": "TabPFN (SQI)", "HuBERT_ECG_probe": "HuBERT-ECG probe", "d1_image_probe": "d1-3B image-token probe"}
    tests = [("BUT_QDB", "BUT QDB (train: CinC 2011 + CinC 2017)"), ("CinC2011", "CinC 2011 (train: BUT QDB + CinC 2017)"),
             ("CinC2017", "CinC 2017 (train: BUT QDB + CinC 2011)")]
    out = ["| Model | " + " | ".join(n for _, n in tests) + " |", "| :--- | " + " | ".join([":---:"] * len(tests)) + " |"]
    for m, label in labels.items():
        cells = []
        for key, _ in tests:
            r = s[(s.test == key) & (s.model == m)]
            if r.empty:
                cells.append("—")
                continue
            r = r.iloc[0]
            if pd.isna(r.auroc_multi):
                cells.append(f"{r.auroc_single:.3f} (fixed rule)")
                continue
            sign = "+" if r.delta >= 0 else "−"
            p = "p_Holm < 0.001" if r.p_holm < 0.001 else f"p_Holm = {r.p_holm:.3f}"
            sub = " †" if key == "CinC2017" and r.n_units < 1000 else ""  # CinC 2017 balanced subset
            cells.append(f"{r.auroc_multi:.3f} [{r.auroc_multi_lo:.3f}–{r.auroc_multi_hi:.3f}]; Δ {sign}{abs(r.delta):.3f} ({p}){sub}")
        out.append(f"| **{label}** | " + " | ".join(cells) + " |")
    return "\n".join(out)


EXTRA_LABELS = {"TabPFN_SQI": "TabPFN (SQI)", "HuBERT_ECG_probe": "HuBERT-ECG probe", "d1_image_probe": "d1-3B image-token probe"}


def table_10b() -> str:
    """Seed repetition of the neural models' LODO effects (experiments/seed_repeats_transfer.py, plan A1)."""
    path = RESULTS / "seed_repeats_consistency.csv"
    if not path.exists():
        return _missing(path)
    s = pd.read_csv(path)
    s = s[s.seed_varied]
    out = ["| Model | Held-out dataset | Seeds | LODO AUROC, median (range) | BUT-only AUROC, median (range) | Δ, median (range) | Consistent |",
           "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"]
    full = pd.read_csv(RESULTS / "seed_repeats_summary.csv")
    for r in s.itertuples():
        lodo = f"{r.lodo_median:.3f} ({r.lodo_min:.3f}–{r.lodo_max:.3f})"
        if r.test == "BUT_QDB":  # no seed-matched BUT-only reference for the in-domain test (plan A1)
            out.append(f"| **{MODEL_LABELS[r.model]}** | {r.test} | {r.n_seeds} | {lodo} | — | — | — |")
            continue
        sg = full[(full.test == r.test) & (full.model == r.model)].auroc_single
        out.append(f"| **{MODEL_LABELS[r.model]}** | {r.test} | {r.n_seeds} | {lodo} | "
                   f"{r.single_median:.3f} ({sg.min():.3f}–{sg.max():.3f}) | "
                   f"{r.delta_median:+.3f} ({r.delta_min:+.3f} to {r.delta_max:+.3f}) | {'yes' if r.consistent else 'no'} |")
    return "\n".join(out)


def _recal_table(n_values) -> str:
    path = RESULTS / "local_recalibration_summary.csv"
    if not path.exists():
        return _missing(path)
    s = pd.read_csv(path)
    s = s[s.n.isin(n_values)]
    labels = {**MODEL_LABELS, **EXTRA_LABELS}
    out = ["| Dataset | Model | n | Transferred threshold: FPR / Sens. (median) | Local threshold: FPR median [95% range] | "
           "Local threshold: Sens. median [95% range] | Draws with FPR ≤ 5% / ≤ 7.5% (local) | Fallback draws | "
           "Brier: raw → Platt (median; Platt skipped) |",
           "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for _, r in s.iterrows():
        out.append(f"| {r['dataset']} | **{labels.get(r['model'], r['model'])}** | {r['n']} | "
                   f"{100 * r['fpr_transferred_median']:.1f}% / {100 * r['sens_transferred_median']:.1f}% | "
                   f"{100 * r['fpr_local_median']:.1f}% [{100 * r['fpr_local_p2.5']:.1f}–{100 * r['fpr_local_p97.5']:.1f}%] | "
                   f"{100 * r['sens_local_median']:.1f}% [{100 * r['sens_local_p2.5']:.1f}–{100 * r['sens_local_p97.5']:.1f}%] | "
                   f"{100 * r['share_fpr_le_5_local']:.0f}% / {100 * r['share_fpr_le_7.5_local']:.0f}% | {r['fallback_draws']} | "
                   f"{r['brier_raw_median']:.3f} → {r['brier_platt_median']:.3f} ({r['platt_skipped']}) |")
    return "\n".join(out)


def table_11() -> str:
    """Local recalibration with n = 100 labelled target recordings (experiments/local_recalibration.py, plan A3)."""
    return _recal_table([100])


def table_11b() -> str:
    """Local recalibration, all sample sizes."""
    return _recal_table([25, 50, 100, 200])


def table_12() -> str:
    """HuBERT-ECG: preprocessing sensitivity (plan A4) and partial fine-tuning (plan F2)."""
    variants = [("Probe, authors' preprocessing (primary)", RESULTS / "hubert_probe_summary.csv"),
                ("Probe, our first preprocessing (sensitivity)", RESULTS / "hubert_probe_summary_ourpreproc.csv"),
                ("Fine-tuned (last 4 layers + head), authors' preprocessing", RESULTS / "hubert_finetune_summary.csv")]
    cols = [("butqdb_cv", "BUT_QDB", "BUT QDB (5-fold subject CV)"), ("but_only_external", "CinC2011", "CinC 2011 (BUT-only)"),
            ("but_only_external", "CinC2017", "CinC 2017 (BUT-only)"), ("lodo_test_BUT_QDB", "BUT_QDB", "LODO: BUT QDB"),
            ("lodo_test_CinC2011", "CinC2011", "LODO: CinC 2011"), ("lodo_test_CinC2017", "CinC2017", "LODO: CinC 2017")]
    out = ["| HuBERT-ECG variant | " + " | ".join(c[2] for c in cols) + " |", "| :--- | " + " | ".join([":---:"] * len(cols)) + " |"]
    for name, path in variants:
        if not path.exists():
            out.append(f"| **{name}** | " + " | ".join(["not run"] * len(cols)) + " |")
            continue
        s = pd.read_csv(path)
        cells = []
        for setting, ds, _ in cols:
            r = s[(s.setting == setting) & (s.dataset == ds)]
            cells.append(f"{r.auroc.iloc[0]:.3f}" if len(r) else "—")
        out.append(f"| **{name}** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_13() -> str:
    """Official BUT QDB protocol: sensitivity analyses F1a and F1b (experiments/butqdb_official_split.py)."""
    path = RESULTS / "butqdb_official_split_summary.csv"
    if not path.exists():
        return _missing(path)
    s = pd.read_csv(path)
    labels = {**MODEL_LABELS, **EXTRA_LABELS}
    a, b = [g.set_index("model") for _, g in s.groupby("analysis", sort=False)]
    head = lambda g: f"{g.n_windows.iloc[0]:,} windows, {g.n_unusable.iloc[0]} unusable, {g.n_subjects.iloc[0]} subjects"
    out = [f"| Model | F1a: 5-fold CV without the excluded 105001 stretch ({head(a)}) | "
           f"F1b: official train/test split, test set ({head(b)}) |", "| :--- | :---: | :---: |"]

    def cell(g, m):
        if m not in g.index:
            return "not run"
        r = g.loc[m]
        txt = f"{r.auroc:.3f} [{r.auroc_lo:.3f}–{r.auroc_hi:.3f}]"
        if r.best:
            return f"**{txt}** (best)"
        return txt + (f"; p_Holm < 0.001" if r.p_holm_vs_best < 0.001 else f"; p_Holm = {r.p_holm_vs_best:.3f}")
    for m in labels:
        if m in a.index or m in b.index:
            out.append(f"| **{labels[m]}** | {cell(a, m)} | {cell(b, m)} |")
    return "\n".join(out)


def table_4c() -> str:
    """Latency of the foundation-model probes and TabPFN (experiments/latency_foundation_probes.py, latency_tabpfn.py)."""
    rows = []
    for path, names in [(RESULTS / "latency_probes_summary.csv",
                         {"hubert_preproc_ms": "HuBERT-ECG: preprocessing (authors' pipeline)",
                          "hubert_forward_cpu_ms": "HuBERT-ECG: forward pass, CPU",
                          "hubert_forward_mps_ms": "HuBERT-ECG: forward pass, MPS",
                          "hubert_linear_probe_ms": "HuBERT-ECG: linear probe",
                          "hubert_total_cpu_ms": "HuBERT-ECG probe: total, CPU",
                          "hubert_total_mps_ms": "HuBERT-ECG probe: total, MPS",
                          "d1_linear_probe_ms": "d1-3B image probe: linear probe (added to the d1-3B forward pass of main-text Table 5)"}),
                        (RESULTS / "latency_tabpfn_summary.csv",
                         {"fit_ms": "TabPFN: fit on 1,047 windows (once per model)",
                          "tabpfn_predict_cpu_ms": "TabPFN: prediction of one window, CPU"})]:
        if not path.exists():
            return _missing(path)
        s = pd.read_csv(path).set_index("measurement")
        for k, lab in names.items():
            r = s.loc[k]
            rows.append(f"| {lab} | {r.median_over_runs_ms:.2f} ms | {r.min_run_ms:.2f}–{r.max_run_ms:.2f} ms | "
                        f"{r.p95_median_over_runs_ms:.2f} ms |")
    return "\n".join(["| Step | Median (three runs) | Range of run medians | p95 (median over runs) |",
                      "| :--- | :---: | :---: | :---: |"] + rows)


def table_8c() -> str:
    """CinC 2017 (8,512 recordings, natural prevalence) for all 11 models trained on BUT QDB only: AUROC and AP with the
    pre-specified recording maximum, decisions at tau = 0.5, and AUROC with the post hoc recording mean. Window scores:
    results/external_window_predictions.csv (main models), tabpfn_predictions.csv, hubert_probe_predictions.csv
    (primary preprocessing) and d1_cinc2017_full_predictions.csv (zero-shot d1-3B and the d1-3B probe)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    w = pd.read_csv(RESULTS / "external_window_predictions.csv", low_memory=False)
    w = w[w.dataset == "CinC2017"].copy()
    extra = {"TabPFN_SQI": ("tabpfn_predictions.csv", "prob_unusable"), "HuBERT_ECG_probe": ("hubert_probe_predictions.csv", "prob_unusable")}
    for m, (f, col) in extra.items():
        e = pd.read_csv(RESULTS / f, low_memory=False)
        e = e[(e.setting == "but_only_external") & (e.dataset == "CinC2017")].set_index("segment_id")[col]
        w[m] = e.loc[w.segment_id].values
    d1 = pd.read_csv(RESULTS / "d1_cinc2017_full_predictions.csv").set_index("segment_id")
    w["LiquidAI_d1_3B_Neural"] = d1.loc[w.segment_id, "prob_unusable"].values
    w["d1_image_probe"] = d1.loc[w.segment_id, "prob_image_probe"].values
    labels = {**MODEL_LABELS, **EXTRA_LABELS}
    out = ["| Model | AUROC, recording max (pre-specified) | AP, recording max | Sens. / FR at τ = 0.50 | PPV / NPV at τ = 0.50 | "
           "AUROC, recording mean (post hoc) |", "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for m in labels:
        g = w.groupby("record_name").agg(y=("label", "first"), mx=(m, "max"), mn=(m, "mean"))
        y, mx = g.y.values.astype(int), g.mx.values
        pred = mx >= 0.5
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        ppv = tp / (tp + fp) if tp + fp else float("nan")
        npv = tn / (tn + fn) if tn + fn else float("nan")
        out.append(f"| **{labels[m]}** | {roc_auc_score(y, mx):.3f} | {average_precision_score(y, mx):.3f} | "
                   f"{100 * tp / (tp + fn):.1f}% / {100 * fp / (fp + tn):.1f}% | {ppv:.3f} / {npv:.3f} | "
                   f"{roc_auc_score(y, g.mn.values):.3f} |")
    return "\n".join(out)


def table_2h() -> str:
    """CinC 2011 records with a constant lead II (experiments/cinc2011_constant_lead_check.py; post hoc, descriptive)."""
    path = RESULTS / "cinc2011_constant_lead_check.csv"
    if not path.exists():
        return _missing(path)
    s = pd.read_csv(path).set_index("model")
    labels = {**MODEL_LABELS, **EXTRA_LABELS}
    out = ["| Model | Constant-lead records flagged at τ = 0.50 | AUROC, all 565 records | AUROC without the 65 constant-lead records |",
           "| :--- | :---: | :---: | :---: |"]
    for m in labels:
        if m in s.index:
            r = s.loc[m]
            out.append(f"| **{labels[m]}** | {int(r['constant_flagged_at_0.5'])} / {int(r.n_constant)} | {r.auroc_all:.3f} | "
                       f"{r.auroc_without_constant:.3f} |")
    return "\n".join(out)


def table_1g() -> str:
    """AUROC and AP with 95% bootstrap CIs for all models in each primary evaluation set (evaluation/model_cis.py)."""
    path = RESULTS / "model_cis.csv"
    if not path.exists():
        return _missing(path)
    c = pd.read_csv(path)
    labels = {**MODEL_LABELS, **EXTRA_LABELS}
    fams = list(dict.fromkeys(c.family))
    out = ["| Model | " + " | ".join(fams) + " |", "| :--- | " + " | ".join([":---:"] * len(fams)) + " |"]
    for m in labels:
        cells = []
        for f in fams:
            r = c[(c.family == f) & (c.model == m)]
            if r.empty:
                cells.append("—")
                continue
            r = r.iloc[0]
            cells.append(f"AUROC {r.auroc:.3f} [{r.auroc_lo:.3f}–{r.auroc_hi:.3f}]; AP {r.ap:.3f} [{r.ap_lo:.3f}–{r.ap_hi:.3f}]")
        out.append(f"| **{labels[m]}** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_2i() -> str:
    """SQI distributions by dataset and class: median [IQR] per window (TRIPOD+AI item 20c)."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    from lodo_multisource import load_sources
    from features.sqi_extractor import SQI_FEATURE_NAMES
    src = load_sources()
    groups = [("BUT_QDB", 0, "BUT QDB usable"), ("BUT_QDB", 1, "BUT QDB unusable"), ("CinC2011", 0, "CinC 2011 acceptable"),
              ("CinC2011", 1, "CinC 2011 unacceptable"), ("CinC2017", 0, "CinC 2017 not noisy"), ("CinC2017", 1, "CinC 2017 noisy")]
    head = [f"{lab} (n = {int((src[d].label == y).sum()):,})" for d, y, lab in groups]
    out = ["| SQI | " + " | ".join(head) + " |", "| :--- | " + " | ".join([":---:"] * len(groups)) + " |"]
    for f in SQI_FEATURE_NAMES[:11]:
        cells = []
        for d, y, _ in groups:
            v = src[d].loc[src[d].label == y, f]
            q1, q2, q3 = v.quantile([0.25, 0.5, 0.75])
            cells.append(f"{q2:.3g} [{q1:.3g}–{q3:.3g}]")
        out.append(f"| `{f}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_2j() -> str:
    """Univariate SQI AUROC with and without the CinC 2011 constant-lead records (cinc2011_constant_lead_check.py)."""
    path = RESULTS / "cinc2011_constant_lead_univariate.csv"
    if not path.exists():
        return _missing(path)
    s = pd.read_csv(path)
    out = ["| SQI feature | BUT QDB (1,047 windows) | CinC 2011 lead II, all 565 records | CinC 2011 lead II, without the 65 constant-lead records (500) |",
           "| :--- | :---: | :---: | :---: |"]
    for r in s.itertuples():
        out.append(f"| `{r.feature}` | {r.but_qdb:.3f} | {r.cinc2011_all:.3f} | {r.cinc2011_without_constant:.3f} |")
    return "\n".join(out)


def table_9b() -> str:
    """Holm over the ten comparisons with the best model of each set (evaluation/paired_vs_best.py)."""
    path = RESULTS / "paired_vs_best.csv"
    if not path.exists():
        return _missing(path)
    s = pd.read_csv(path)
    labels = {**MODEL_LABELS, **EXTRA_LABELS}
    fmt = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    out = ["| Evaluation set | Best model | Model | AUROC (best) | p, unadjusted | p_Holm, all 55 pairs | p_Holm, 10 comparisons with the best |",
           "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |"]
    for r in s.itertuples():
        out.append(f"| {r.family} | {labels.get(r.best, r.best)} | **{labels.get(r.model, r.model)}** | {r.auroc:.3f} ({r.auroc_best:.3f}) | "
                   f"{fmt(r.p_unadjusted)} | {fmt(r.p_holm_all_pairs)} | {fmt(r.p_holm_vs_best)} |")
    return "\n".join(out)


def table_7() -> str:
    """Prompt sensitivity of d1-3B (experiments/d1_prompt_variants.py). The AUROC CI on BUT QDB is the subject-level
    bootstrap used for Table 1 (seed 42); CinC has no subject identifiers and is reported as a point estimate."""
    summary_path = RESULTS / "d1_prompt_variants_summary.csv"
    if not summary_path.exists():
        return "(prompt variants not yet available)"
    s = pd.read_csv(summary_path)
    p = pd.read_csv(RESULTS / "d1_prompt_variants_predictions.csv")
    subj = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv")[["segment_id", "subject_id"]]
    names = {"primary": "Primary (SQI text + image)", "image_only": "Image only (no SQI text)",
             "alt_wording": "Reworded question (SQI text + image)"}
    out = ["| Configuration | BUT QDB AUROC [95% CI] | BUT QDB Sens. / FR at τ = 0.50 | CinC AUROC | CinC Sens. / FR at τ = 0.50 | NSTDB clean P(usable) | NSTDB paired change, +24 dB | NSTDB paired change, 0 dB |",
           "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for variant, name in names.items():
        b = p[(p["dataset"] == "BUT_QDB") & (p["variant"] == variant)].merge(subj, on="segment_id")
        ci = compute_subject_level_bootstrap_ci(b, pred_col="prob_unusable", target_col="label",
                                                subject_col="subject_id", n_bootstraps=1000, seed=42)["auroc"]
        bs = s[(s["dataset"] == "BUT_QDB") & (s["variant"] == variant)].iloc[0]
        cs = s[(s["dataset"] == "CinC_2011_SetA") & (s["variant"] == variant)].iloc[0]
        nst = s[(s["dataset"] == "NSTDB_paired") & (s["variant"] == variant)]
        clean = nst[nst["condition"] == "clean_reference"].iloc[0]
        d24 = nst[(nst["condition"] == "noisy") & (nst["snr_db"] == 24)].iloc[0]
        d0 = nst[(nst["condition"] == "noisy") & (nst["snr_db"] == 0)].iloc[0]
        out.append(f"| **{name}** | {bs['auroc']:.3f} [{ci['ci_low']:.3f} – {ci['ci_high']:.3f}] | "
                   f"{100 * bs['sens_at_0.5']:.1f}% / {100 * bs['fr_at_0.5']:.1f}% | {cs['auroc']:.3f} | "
                   f"{100 * cs['sens_at_0.5']:.1f}% / {100 * cs['fr_at_0.5']:.1f}% | {clean['mean_p_usable']:.3f} | "
                   f"{d24['paired_delta_usable']:+.3f} | {d0['paired_delta_usable']:+.3f} |")
    return "\n".join(out)


def table_4() -> str:
    """Latency table over the sequential profiling runs in results/latency_repeats (experiments/latency_repeats.sh).
    Each cell is the median over runs of the per-run median, with the median over runs of the per-run p95 in
    parentheses. The last column gives the range of the per-run total medians. Devices are the ones the profiling run
    actually used: classifiers without an explicit device default to MPS when available, so only the sklearn models run on CPU."""
    run_raw = sorted((RESULTS / "latency_repeats").glob("run*_raw.csv"))
    if not run_raw:
        return "(latency repeats not yet available)"
    raws = [pd.read_csv(p) for p in run_raw]
    summary = pd.read_csv(run_raw[0].with_name(run_raw[0].name.replace("_raw.csv", "_summary.csv"))).set_index("Model")
    mapping = {  # raw model id -> (display name, summary row name, execution device)
        "Rule_Based_SQI_Heuristic": ("Rule-Based SQI Heuristic", "Rule-Based SQI Heuristic", "CPU"),
        "Logistic_Regression_SQI": ("Logistic Regression (SQI)", "Logistic Regression (SQI)", "CPU"),
        "LightGBM_SQI": ("LightGBM (SQI)", "LightGBM (SQI)", "CPU"),
        "Independent_SQI_MLP": ("Independent SQI MLP", "Independent SQI MLP", "Apple M4 (MPS)"),
        "1D_CNN_RawECG": ("1D CNN (Raw Signal)", "1D CNN (Raw Signal)", "Apple M4 (MPS)"),
        "2D_CNN_Image": ("2D CNN (Rendered Image)", "2D CNN (Rendered Image)", "Apple M4 (MPS)"),
        "Multimodal_Fusion": ("SinglePass Multimodal Net", "SinglePass Multimodal Net", "Apple M4 (MPS)"),
        "LiquidAI_d1_3B_Neural": ("LiquidAI d1-3B (Zero-Shot)", "Liquid AI d1-3B (Real Neural Model)", "Apple M4 (MPS)"),
    }
    out = ["| Model Architecture | Parameters | Disk Weights | Execution Device | Preproc Median (p95) | Forward Median (p95) | Total Median (p95) | Total Median, Range over Runs | Preproc Share (Median of Ratios) |",
           "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]

    def ms(x):
        return f"{x:.2f} ms"

    for raw_name, (label, summary_name, device) in mapping.items():
        runs = [r[r["model"] == raw_name] for r in raws]
        if any(r.empty for r in runs):
            continue
        per_run = [{
            "pre": np.median(r["pre_ms"]), "pre95": np.percentile(r["pre_ms"], 95),
            "fwd": np.median(r["fwd_ms"]), "fwd95": np.percentile(r["fwd_ms"], 95),
            "tot": np.median(r["total_ms"]), "tot95": np.percentile(r["total_ms"], 95),
            "share": np.median(r["pre_ms"] / r["total_ms"]) * 100,
        } for r in runs]
        med = {k: float(np.median([p[k] for p in per_run])) for k in per_run[0]}
        tot_lo, tot_hi = min(p["tot"] for p in per_run), max(p["tot"] for p in per_run)
        srow = summary.loc[summary_name]
        params = f"{int(srow['Parameters']):,}"
        wmib = f"{float(srow['Weights_MiB']):.2f} MiB"
        out.append(f"| **{label}** | {params} | {wmib} | {device} | "
                   f"{ms(med['pre'])} ({ms(med['pre95'])}) | "
                   f"{ms(med['fwd'])} ({ms(med['fwd95'])}) | "
                   f"{ms(med['tot'])} ({ms(med['tot95'])}) | {tot_lo:.2f}–{tot_hi:.2f} ms | {med['share']:.1f}% |")
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
        ("Table 3b", table_3b(nstdb)),
        ("Table 4", table_4()),
        ("Table 5", table_5()),
        ("Table 1d", table_1d()),
        ("Table 2d", table_2d()),
        ("Table 6", table_6()),
        ("Table 7", table_7()),
        ("Table 1c", table_1c()),
        ("Table 1e", table_1e()),
        ("Table 1f", table_1f()),
        ("Table 2e", table_2e()),
        ("Table 2f", table_2f()),
        ("Table 2g", table_2g()),
        ("Table 4b", table_4b()),
        ("Table 7b", table_7b()),
        ("Table 8", table_8()),
        ("Table 8b", table_8b()),
        ("Table 9", table_9()),
        ("Table 10", table_10()),
        ("Table 10b", table_10b()),
        ("Table 11", table_11()),
        ("Table 11b", table_11b()),
        ("Table 12", table_12()),
        ("Table 13", table_13()),
        ("Table 4c", table_4c()),
        ("Table 8c", table_8c()),
        ("Table 2h", table_2h()),
        ("Table 1g", table_1g()),
        ("Table 2i", table_2i()),
        ("Table 2j", table_2j()),
        ("Table 9b", table_9b()),
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Manuscript tables regenerated 2026-10-12\n\n")
        f.write(f"NSTDB windows: clean control = {n_clean}, active noise = {n_active}\n\n")
        for title, body in sections:
            f.write(f"## {title}\n\n{body}\n\n")
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
