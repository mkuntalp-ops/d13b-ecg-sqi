"""
Check the numbers typed in the v2 manuscript prose against the saved outputs (results/v2_2026-11).

Each check renders a value computed from a CSV into a sentence fragment and requires that fragment to occur in
manuscript/v2/main_v2_template.md (or supplement_v2_template.md). `--fill` first replaces the placeholder tokens that
depend on results produced late in the queue (latency, fine-tuning, the DeLong/bootstrap implementation check); each token
must occur the expected number of times.

Usage: ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/verify_v2_prose.py [--fill] [--variant srep]

`--variant srep` runs the same checks on the Scientific Reports templates (manuscript/srep/). There, "Supplementary
Table Sn" is read as "Table Sn" before matching, and two extra checks apply: the variant's own wording checks, and a
number-provenance check that every number in the prose of the srep main text also occurs in the checked v2 manuscript
(main or supplement, assembled) or in the srep supplement, so that shortening cannot introduce an unchecked number.
Exit status 1 if any check fails or any placeholder remains.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
import config  # noqa: E402
import paired_comparisons as pc  # noqa: E402

R = config.RESULTS_DIR
VARIANT = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "v2"
if VARIANT not in ("v2", "srep"):
    raise SystemExit(f"unknown variant {VARIANT!r}")
MAIN = ROOT / "manuscript" / VARIANT / f"main_{VARIANT}_template.md"
SUPP = ROOT / "manuscript" / VARIANT / f"supplement_{VARIANT}_template.md"
TOKEN_RE = r"\b(?:FT_(?:BUT|C11|C17)|DL_CI|BOOT_CI|LAT_[A-Z0-9_]+)\b"

BUT, C11, C17, BAL = "BUT_QDB (5-fold subject CV)", "CinC2011 lead II (998 records)", "CinC2017 (8,512 recordings)", "CinC2017 balanced subset (556 recordings)"
LR, GBM, MLP, CNN1, CNN2, MM, HEUR, ZS, PROBE, TAB, HUB = ("Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG",
                                                          "2D_CNN_Image", "Multimodal_Fusion", "Rule_Based_SQI_Heuristic", "LiquidAI_d1_3B_Neural",
                                                          "d1_image_probe", "TabPFN_SQI", "HuBERT_ECG_probe")
pairs = pd.read_csv(R / "paired_comparisons.csv")
vs_best = pd.read_csv(R / "paired_vs_best.csv")


def auroc(family, model):
    x = pairs[(pairs.family == family) & (pairs.model_a == model)]
    return float(x.auroc_a.iloc[0]) if len(x) else float(pairs[(pairs.family == family) & (pairs.model_b == model)].auroc_b.iloc[0])


def pair(family, a, b):
    x = pairs[(pairs.family == family) & (((pairs.model_a == a) & (pairs.model_b == b)) | ((pairs.model_a == b) & (pairs.model_b == a)))]
    if len(x) != 1:
        raise ValueError((family, a, b))
    return x.iloc[0]


def holm_best(family, model):
    return float(vs_best[(vs_best.family == family) & (vs_best.model == model)].p_holm_all_pairs.iloc[0])


def f3(v):
    return f"{v:.3f}"


def latency_values() -> dict:
    runs = [pd.read_csv(R / "latency_repeats" / f"run{k}_summary.csv").set_index("Model") for k in (1, 2, 3)]

    def med(model, col):
        return float(np.median([r.loc[model, col] for r in runs]))

    ext = pd.read_csv(R / "latency_extended_summary.csv").set_index("measurement").median_over_runs_ms
    pro = pd.read_csv(R / "latency_probes_summary.csv").set_index("measurement").median_over_runs_ms
    tab = pd.read_csv(R / "latency_tabpfn_summary.csv").set_index("measurement").median_over_runs_ms
    cnn2 = "2D CNN (Rendered Image)"
    return {"LAT_RENDER": f"{med(cnn2, 'Preprocess_Median_ms'):.0f}",
            "LAT_RENDER_PCT": f"{100 * med(cnn2, 'Preprocess_Median_ms') / med(cnn2, 'Total_End_to_End_Median_ms'):.0f}",
            "LAT_2DFWD": f"{med(cnn2, 'Forward_Median_ms'):.1f}",
            "LAT_RASTER": f"{ext['render_raster_ms']:.2f}",
            "LAT_1D": f"{med('1D CNN (Raw Signal)', 'Total_End_to_End_Median_ms'):.2f}",
            "LAT_D1": f"{med('Liquid AI d1-3B (Real Neural Model)', 'Total_End_to_End_Median_ms'):.0f}",
            "LAT_D1FWD": f"{med('Liquid AI d1-3B (Real Neural Model)', 'Forward_Median_ms'):.0f}",
            "LAT_HUB": f"{pro['hubert_total_mps_ms']:.1f}",
            "LAT_HUBCPU": f"{pro['hubert_total_cpu_ms']:.1f}",
            "LAT_HUBFWD": f"{pro['hubert_forward_mps_ms']:.1f}",
            "LAT_D1PROBE": f"{pro['d1_linear_probe_ms']:.2f}",
            "LAT_TAB": f"{tab['tabpfn_predict_cpu_ms'] / 1000:.1f}",
            "LAT_T5_1D": f"{med('1D CNN (Raw Signal)', 'Forward_Median_ms'):.2f}",
            "LAT_S8_1D": f"{ext['cnn1d_forward_mps_ms']:.2f}",
            "LAT_T5_2D": f"{med(cnn2, 'Forward_Median_ms'):.2f}",
            "LAT_S8_2D": f"{ext['cnn2d_forward_mps_ms']:.2f}"}


def finetune_values() -> dict:
    s = pd.read_csv(R / "hubert_finetune_summary.csv")
    g = lambda st, ds: f3(float(s[(s.setting == st) & (s.dataset == ds)].auroc.iloc[0]))  # noqa: E731
    return {"FT_BUT": g("butqdb_cv", "BUT_QDB"), "FT_C11": g("but_only_external", "CinC2011"), "FT_C17": g("but_only_external", "CinC2017")}


def implementation_check() -> dict:
    """DeLong (paired_comparisons.csv) and an independent recording-bootstrap interval for one pair."""
    p = pair(C11, CNN2, PROBE)
    df = pc.families()[C11][1]
    rng = np.random.default_rng(42)
    from sklearn.metrics import roc_auc_score
    y = df.y.values
    a, b = df[PROBE].values, df[CNN2].values
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"DL_CI": f"{-p.ci_hi:.3f}–{-p.ci_lo:.3f}", "BOOT_CI": f"{lo:.3f}–{hi:.3f}"}


def fill() -> None:
    vals = {**latency_values(), **finetune_values(), **implementation_check()}
    for path in (MAIN, SUPP):
        t = path.read_text(encoding="utf-8")
        t = re.sub(TOKEN_RE, lambda m: vals[m.group(0)], t)
        path.write_text(t, encoding="utf-8")
    print("filled", vals)


def checks() -> list:
    c = []
    for fam, models in [(BUT, {CNN2: "The 2D CNN reached {}", CNN1: "the 1D CNN {}", MM: "the multimodal net {}", GBM: "LightGBM {}", TAB: "TabPFN {}", HUB: "HuBERT-ECG probe {}",
                               MLP: "The MLP ({};", LR: "logistic regression ({};", ZS: "Zero-shot d1-3B reached {}"}),
                        (C11, {PROBE: "ranked recordings best (AUROC {}", CNN2: "including the 2D CNN ({};", HUB: "close to chance ({})", MM: "same score ({})", LR: "was inverted ({})"}),
                        (C17, {HUB: "highest AUROC ({},", PROBE: "the d1-3B probe ({},", GBM: "LightGBM ({},", HEUR: "the rule heuristic ({},", MLP: "the MLP ({},", LR: "logistic regression ({},",
                               TAB: "TabPFN ({})", CNN2: "the 2D CNN ({})", MM: "the multimodal net ({})", CNN1: "the 1D CNN ({})", ZS: "zero-shot d1-3B ({})"}),
                        (BAL, {HUB: "the HuBERT-ECG probe ({}) was best", PROBE: "the d1-3B probe ({}), LightGBM", GBM: "LightGBM ({}), the heuristic", HEUR: "the heuristic ({}) and", MLP: "the MLP ({}) were not"})]:
        for m, frag in models.items():
            c.append((frag.format(f3(auroc(fam, m))), MAIN))
    c.append((f"(p_Holm = {holm_best(BUT, ZS):.3f}).", MAIN))
    p = pair(BUT, CNN2, ZS)
    c.append((f"({f3(p.delta)}; 95% CI {f3(p.ci_lo)}–{f3(p.ci_hi)})", MAIN))
    p = pair(BUT, ZS, PROBE)
    c.append((f"a gain of {f3(-p.delta)} (95% CI {f3(-p.ci_hi)}–{f3(-p.ci_lo)}; unadjusted p = {f3(p.p)}, p_Holm = {f3(p.p_holm)})", MAIN))
    p = pair(C11, CNN2, PROBE)
    c.append((f"difference {f3(-p.delta)}, 95% CI {f3(-p.ci_hi)}–{f3(-p.ci_lo)}; p_Holm = {f3(p.p_holm)}", MAIN))
    for m, frag in {HEUR: "the rule heuristic (0.887, p_Holm = {})", MLP: "the MLP (0.876, p_Holm = {})", LR: "logistic regression (0.870, p_Holm = {})"}.items():
        c.append((frag.format(f3(holm_best(C17, m))), MAIN))
    # Abstract (the Scientific Reports abstract is worded differently; its numbers are checked further down)
    zs = [auroc(f, ZS) for f in (BUT, C11, C17)]
    if VARIANT == "v2":
        c += [(f"Zero-shot d1-3B was weak ({f3(min(zs))}–{f3(max(zs))})", MAIN),
              (f"rose from eighth to first ({f3(auroc(C17, HUB))}) while near chance on CinC 2011 ({f3(auroc(C11, HUB))})", MAIN),
              (f"fell from second ({f3(auroc(BUT, CNN1))}) to tenth ({f3(auroc(C17, CNN1))})", MAIN)]
    # 12 leads
    t = pd.read_csv(R / "cinc2011_twelve_lead_summary.csv")
    tl = lambda a, m: f3(float(t[(t.aggregation == a) & (t.model == m)].auroc.iloc[0]))  # noqa: E731
    assert len(t[t.draw == "all_998"]) == len(t)
    c += [(f"Averaging the leads gave {tl('mean_12', CNN2)} for the 2D CNN, {tl('mean_12', GBM)} for LightGBM and {tl('mean_12', MLP)} for the MLP", MAIN),
          (f"the maximum gave {tl('max_12', HEUR)} for the heuristic, {tl('max_12', MLP)} for the MLP and {tl('max_12', ZS)} for zero-shot d1-3B", MAIN)]
    p12 = pd.read_csv(R / "d1_probe_twelve_lead_summary.csv").set_index("aggregation")
    pc = pd.read_csv(R / "d1_probe_twelve_lead_comparisons.csv")
    assert (pc.p_holm < 0.001).all() and len(pc) == 4
    pmax, pmean = pc[pc.comparison.str.contains("best other.*max_12")].iloc[0], pc[pc.comparison.str.contains("best other.*mean_12")].iloc[0]
    assert "Rule_Based" in pmax.comparison and "2D_CNN" in pmean.comparison
    kc = pd.read_csv(R / "d1_probe_twelve_lead_constant_leads.csv")
    wo = lambda a, m: f3(float(kc[(kc.aggregation == a) & (kc.model == m)].auroc_without_constant.iloc[0]))  # noqa: E731
    triv = kc[kc.aggregation == "n_constant_leads"].iloc[0]
    assert (triv.n_without_constant, triv.n_pos_without_constant) == (863, 96)
    c += [(f"reached {f3(p12.auroc['max_12'])} (95% CI {f3(p12.auroc_ci_low['max_12'])}–{f3(p12.auroc_ci_high['max_12'])}) with the maximum and {f3(p12.auroc['mean_12'])} (95% CI {f3(p12.auroc_ci_low['mean_12'])}–{f3(p12.auroc_ci_high['mean_12'])}) with the mean", MAIN),
          (f"(difference {f3(pmax.delta)}, 95% CI {f3(pmax.ci_low)}–{f3(pmax.ci_high)})", MAIN),
          (f"(difference {f3(pmean.delta)}, 95% CI {f3(pmean.ci_low)}–{f3(pmean.ci_high)}; both p_Holm < 0.001", MAIN),
          (f"the number of constant leads alone gave an AUROC of {f3(triv.auroc_all)}", MAIN),
          (f"the probe reached {wo('max_12', 'd1_image_probe')} with the maximum and {wo('mean_12', 'd1_image_probe')} with the mean, against {wo('max_12', CNN2)} and {wo('mean_12', CNN2)} for the 2D CNN", MAIN)]
    h12 = pd.read_csv(R / "hubert_probe_twelve_lead_summary.csv").set_index("aggregation")
    hc = pd.read_csv(R / "hubert_probe_twelve_lead_comparisons.csv")
    hk = pd.read_csv(R / "hubert_probe_twelve_lead_constant_leads.csv").set_index("aggregation")
    k_hub_73 = float(pd.read_csv(R / "cinc2011_constant_lead_check.csv").set_index("model").auroc_without_constant[HUB])
    assert (hc.p_holm < 0.001).all() and len(hc) == 4
    hd = lambda a: f3(abs(float(hc[hc.comparison == f"HuBERT {a} vs d1 image probe {a}"].delta.iloc[0])))  # noqa: E731
    assert f3(h12.auroc["lead_II"]) == "0.562"
    c += [(f"it reached {f3(h12.auroc['max_12'])} with the maximum and {f3(h12.auroc['mean_12'])} with the mean, {hd('max_12')} and {hd('mean_12')} below the d1-3B probe", MAIN),
          (f"Without the records with a constant lead its AUROC was {f3(hk.auroc_without_constant['max_12'])} and {f3(hk.auroc_without_constant['mean_12'])}", MAIN),
          (f"Given the real 12 leads in one input, it reached {f3(h12.auroc['native_12'])}", MAIN),
          ] + ([(f"its lead II AUROC was {f3(hk.auroc_without_constant['lead_II'])} (Table S35; without only the 73 records whose lead II is constant, {f3(k_hub_73)}, Table S26)", MAIN)]
               if VARIANT == "v2" else [])
    # Raster
    r = pd.read_csv(R / "raster_2dcnn_summary.csv").set_index("dataset")
    a = r.loc["BUT QDB (5-fold subject CV)"]
    b = r.loc["CinC 2011 lead II"]
    d = r.loc["CinC 2017 (recording max)"]
    c += [(f"BUT QDB {f3(a.auroc_raster)} versus {f3(a.auroc_matplotlib)}, Δ {a.delta_raster_minus_mpl:.3f}".replace("Δ -", "Δ −") + f", 95% CI {a.delta_ci_low:.3f} to {a.delta_ci_high:.3f}".replace("CI -", "CI −"), MAIN),
          (f"CinC 2017 {f3(d.auroc_raster)} versus {f3(d.auroc_matplotlib)}", MAIN),
          (f"{f3(b.auroc_raster)} versus {f3(b.auroc_matplotlib)}, Δ +{f3(b.delta_raster_minus_mpl)}, 95% CI {f3(b.delta_ci_low)}–{f3(b.delta_ci_high)}, p_Holm = {f3(b.p_holm)}", MAIN)]
    # Heuristic refit
    h = pd.read_csv(R / "heuristic_refit_summary.csv")
    c += [(f"{f3(h.auroc_handset[0])} versus {f3(h.auroc_refit[0])} (cluster-bootstrap p = {f3(h.p[0])})", MAIN), (f"{f3(h.auroc_handset[1])} versus {f3(h.auroc_refit[1])} (DeLong p < 0.001)", MAIN),
          (f"{f3(h.auroc_handset[2])} versus {f3(h.auroc_refit[2])} (DeLong p < 0.001)", MAIN), (f"fitted inside the folds, reached {f3(h.auroc_refit[0])}", MAIN)]
    assert h.p[1] < 0.001 and h.p[2] < 0.001
    # Constant lead
    k = pd.read_csv(R / "cinc2011_constant_lead_check.csv").set_index("model")
    assert (k.n_constant == 73).all()
    c += [(f"remaining highest ({f3(k.loc[PROBE, 'auroc_without_constant'])})", MAIN), (f"the 2D CNN ({f3(k.loc[CNN2, 'auroc_without_constant'])}) and the HuBERT-ECG probe ({f3(k.loc[HUB, 'auroc_without_constant'])})", MAIN),
          (f"rose from {f3(k.loc[LR, 'auroc_all'])} to {f3(k.loc[LR, 'auroc_without_constant'])} without them", MAIN)]
    flagged = k.constant_flagged_at_0_5 if "constant_flagged_at_0_5" in k else k["constant_flagged_at_0.5"]
    assert set(flagged[flagged == 73].index) == {MLP, CNN1, CNN2, MM, HEUR, ZS, PROBE} and set(flagged[flagged == 0].index) == {LR, GBM, TAB, HUB}
    # Univariate
    u = pd.read_csv(R / "feature_shift_univariate_auroc.csv").set_index("feature")
    for f, frag in {"high_frequency_noise_ratio": "the high-frequency ratio went from {} to {}", "powerline_noise_ratio": "the powerline ratio from {} to {}", "zero_crossing_rate": "the zero-crossing rate from {} to {}"}.items():
        c.append((frag.format(f3(u.loc[f].iloc[0]), f3(u.loc[f].iloc[1])), MAIN))
    v = pd.read_csv(R / "cinc2011_constant_lead_univariate.csv")
    col = [x for x in v.columns if "without" in x][0]
    v = v.set_index(v.columns[0])
    c.append((f"the three AUROCs were {f3(v.loc['high_frequency_noise_ratio', col])}, {f3(v.loc['powerline_noise_ratio', col])} and {f3(v.loc['zero_crossing_rate', col])}", MAIN))
    # Flatline
    fl = pd.read_csv(R / "flatline_robust_summary.csv")
    fl = fl[fl.auroc_original_flatline.notna()].set_index(["dataset", "model"])
    q = lambda m: (f3(fl.loc[("CinC2011", m), "auroc_original_flatline"]), f3(fl.loc[("CinC2011", m), "auroc_robust_flatline"]))  # noqa: E731
    c += [("CinC 2011 AUROC from {} to {} for the MLP".format(*q(MLP)), MAIN), ("from {} to {} for logistic regression".format(*q(LR)), MAIN), ("from {} to {} for LightGBM".format(*q(GBM)), MAIN)]
    dB = (fl.auroc_robust_flatline - fl.auroc_original_flatline).abs()
    c += [(f"at most {f3(dB.loc['BUT_QDB'].max())} on BUT QDB and at most {f3(dB.loc['CinC2017'].max())} on CinC 2017", MAIN)]
    sc = pd.read_csv(R / "cinc2011_scaling_sensitivity.csv")
    sv = lambda m, f, s: f3(float(sc[(sc.model == m) & (sc.features == f) & (sc.scaler == s)].transfer_auroc.iloc[0]))  # noqa: E731
    d3 = "Drop saturation, flatline, constants"
    c.append((f"logistic regression reached {sv('LR', d3, 'StandardScaler')} and the MLP {sv('Independent_SQI_MLP', d3, 'internal StandardScaler')}", MAIN))
    c.append((f"alone gave logistic regression {sv('LR', 'All 11 model features', 'Z-clip [-3,3]')}", MAIN))
    # Thresholds, local recalibration
    th = pd.read_csv(R / "threshold_transfer_summary.csv")
    g = th[(th.rule == "target1_fpr5") & th.dataset.str.startswith("CinC2017") & ~th.model.isin([HEUR, ZS])]
    c.append((f"of {100 * g.fpr.min():.1f}–{100 * g.fpr.max():.1f}% for the trained models", MAIN))
    l = pd.read_csv(R / "local_recalibration_summary.csv")  # noqa: E741
    x = l[(l.dataset == "CinC2011") & (l.n == 100)]
    c.append((f"at most 5% in {100 * x.share_fpr_le_5_local.min():.0f}–{100 * x.share_fpr_le_5_local.max():.0f}% and at most 7.5% in {100 * x['share_fpr_le_7.5_local'].min():.0f}–{100 * x['share_fpr_le_7.5_local'].max():.0f}% of draws", MAIN))
    x = l[(l.dataset == "CinC2017") & (l.n == 100)]
    c.append((f"at most 5% in {100 * x.share_fpr_le_5_local.min():.0f}–{100 * x.share_fpr_le_5_local.max():.0f}% and at most 7.5% in at least {100 * x['share_fpr_le_7.5_local'].min():.0f}% of draws", MAIN))
    c.append((f"median sensitivity was only {100 * x.sens_local_median.min():.0f}–{100 * x.sens_local_median.max():.0f}%", MAIN))
    c.append((f"(0.029–0.032 at n = 100)" if (f3(x.brier_platt_median.min())[:5] == "0.029" and f3(x.brier_platt_median.max())[:5] == "0.032") else "BRIER RANGE CHANGED", MAIN))
    y = l[(l.dataset == "CinC2017") & (l.n == 200)]
    c.append((f"the median sensitivity remained {100 * y.sens_local_median.min():.0f}–{100 * y.sens_local_median.max():.0f}%", MAIN))
    # Calibration
    dc = pd.read_csv(R / "decision_calibration.csv")
    x = dc[dc.family == C17].set_index("model")
    assert len(x) == 11 and (x.calibration_intercept < 0).all()
    c.append((f"negative for all 11 models ({x.calibration_intercept.max():.2f} to {x.calibration_intercept.min():.2f})".replace("(-", "(−").replace("to -", "to −"), MAIN))
    c.append((f"Slopes ranged from {x.calibration_slope.min():.2f} (1D CNN) to {x.calibration_slope.max():.2f} (heuristic)", MAIN))
    c.append((f"had the slope closest to 1 ({x.loc[HUB, 'calibration_slope']:.2f})", MAIN))
    assert x.calibration_slope.idxmin() == CNN1 and x.calibration_slope.idxmax() == HEUR and (x.calibration_slope - 1).abs().idxmin() == HUB
    # Counts stated in Methods
    m = pd.read_csv(R / "manifest_butqdb_1000_cohort.csv")
    assert (len(m), int(m.label.sum()), m.subject_id.nunique(), m.record_name.nunique(), m[m.label == 1].subject_id.nunique()) == (988, 238, 15, 17, 7)
    w = pd.read_csv(ROOT / "cache" / "datasets" / "cinc2017" / "cinc2017_windows.csv")
    clip = int(w.kurtosis_ksqi.isin([-5, 50]).sum() + w.skewness_ssqi.abs().eq(10).sum())
    c.append((f"and {clip} (CinC 2017) kurtosis or skewness values", MAIN))
    assert len(w) == 23985 and (w.flatline_fraction == 1).sum() == 0
    # Multimodal saturation diagnosis
    inp = pd.read_csv(R / "multimodal_saturation_inputs.csv").set_index("feature").loc["flatline_fraction"]
    v = pd.read_csv(R / "multimodal_saturation_scores.csv").set_index("variant")
    assert v.n_distinct["original"] == 1 and v.n_distinct["sqi_only"] == 1 and inp.frac_records_abs_z_gt_10 == 1.0
    c.append((f"near zero (SD {inp.but_sd:.5f})", MAIN))
    c.append((f"with a median of {inp.cinc2011_median_z:,.0f} SDs", MAIN))
    c.append((f"the same fitted net reached {v.auroc['image_only']:.3f}", MAIN))
    c.append((f"gave {v.n_distinct['flatline_robust']} distinct scores and {v.auroc['flatline_robust']:.3f}", MAIN))
    # Quality gating
    g = pd.read_csv(R / "cinc2017_gating_summary.csv")
    d = pd.read_csv(R / "cinc2017_gating_differences.csv").set_index(["fraction", "comparison"])
    at = lambda f: g[g.fraction == f].set_index("model")
    a = at(0.10)
    assert (a.n_all == 8234).all() and (a.n_kept == 7411).all()
    c.append((f"macro-F1 of {a.macro_f1_all.iloc[0]:.3f} on the 8,234", MAIN))
    x = a.change_minus_random
    c.append((f"by {x.min():.3f} to {x.max():.3f} across the eleven quality models", MAIN))
    cifrag = lambda r: f"{r.change_minus_random:.3f}, 95% CI {r.ci_low:.3f}–{r.ci_high:.3f}"
    c.append((f"d1-3B image probe ({cifrag(a.loc[PROBE])})", MAIN))
    c.append((f"the 1D CNN ({cifrag(a.loc[CNN1])})", MAIN))
    c.append((f"TabPFN ({cifrag(a.loc[TAB])})", MAIN))
    assert set(a.index[a.ci_low > 0]) == {PROBE, CNN1, TAB}
    for other, name in [(CNN2, "the 2D CNN"), (HUB, "the HuBERT-ECG probe")]:
        r = d.loc[(0.10, f"d1_image_probe minus {other}")]
        c.append((f"exceeded {name} by {r.difference:.3f} (95% CI {r.ci_low:.3f}–{r.ci_high:.3f})" if other == CNN2 else f"{name} by {r.difference:.3f} (95% CI {r.ci_low:.3f}–{r.ci_high:.3f})", MAIN))
    x5, x20 = at(0.05).change_minus_random, at(0.20).change_minus_random
    c.append((f"from {x5.min():.3f} to {x5.max():.3f} when 5% were rejected and from {x20.min():.3f} to {x20.max():.3f} when 20%".replace("-", "−"), MAIN))
    r1, r2 = d.loc[(0.20, f"d1_image_probe minus {CNN2}")], d.loc[(0.20, f"d1_image_probe minus {HUB}")]
    assert r1.ci_low < 0 < r1.ci_high and r2.ci_low < 0 < r2.ci_high
    p = a.loc[PROBE]
    c.append((f"removed {int(p.rejected_N)} normal, {int(p.rejected_A)} atrial fibrillation and {int(p.rejected_O)} other-rhythm", MAIN))
    c.append((f"({100 * p.rejected_A / 823:.1f}% of rejected recordings against {100 * 757 / 8234:.1f}% of all recordings)", MAIN))
    c.append((f"({100 * p.rejected_O / 823:.1f}% against {100 * 2411 / 8234:.1f}%)", MAIN))
    assert p.rejected_N + p.rejected_A + p.rejected_O == 823 and (a.change_minus_random < 0.01).all()
    # PTB-XL frozen test
    s = pd.read_csv(R / "ptbxl_frozen_summary.csv")
    px = pd.read_csv(R / "ptbxl_frozen_comparisons.csv")
    const = pd.read_csv(R / "ptbxl_frozen_constant_leads.csv")
    assert (const.n_records_constant == 0).all() and (s.n == 600).all() and (s.n_pos == 300).all()
    au = lambda m, a, o="global_vs_clean": s[(s.model == m) & (s.aggregation == a) & (s.outcome == o)].iloc[0]
    aucf = lambda m, a, o="global_vs_clean": f"{au(m, a, o).auroc:.3f}"
    M2, ME = "max_12", "mean_12"
    pr = lambda fam, name: px[(px.family == fam) & (px.comparison == name)].iloc[0]
    for a, tail in [(M2, "with the maximum"), (ME, "with the mean")]:
        r = au(PROBE, a)
        c.append((f"{r.auroc:.3f} (95% CI {r.auroc_ci_low:.3f}–{r.auroc_ci_high:.3f}) {tail}", MAIN))
    c.append((f"against {aucf(CNN2, M2)} and {aucf(CNN2, ME)} for the 2D CNN", MAIN))
    p1, p2 = pr("primary", "probe vs 2D CNN (max_12)"), pr("primary", "probe vs 2D CNN (mean_12)")
    assert p1.delta > 0 and p2.delta > 0 and p1.p_holm < 0.05 and p2.p_holm < 0.05
    c.append((f"(difference {p1.delta:.3f}, 95% CI {p1.ci_low:.3f}–{p1.ci_high:.3f}, p_Holm = {p1.p_holm:.3f}; difference {p2.delta:.3f}, 95% CI {p2.ci_low:.3f}–{p2.ci_high:.3f}, p_Holm = {p2.p_holm:.3f})", MAIN))
    sec = lambda m, a: pr("secondary", f"probe vs {m} ({a})")
    c.append((f"HuBERT-ECG probe ({aucf(HUB, M2)} with the maximum, {aucf(HUB, ME)} with the mean) and LightGBM ({aucf(GBM, M2)} and {aucf(GBM, ME)})", MAIN))
    assert all(sec(m, a).delta < 0 and sec(m, a).p_holm < 0.001 for m in [HUB, GBM] for a in [M2, ME])
    assert aucf(MLP, M2) == aucf(MLP, ME)
    c.append((f"The MLP ({aucf(MLP, M2)} under both rules; p_Holm = {sec(MLP, M2).p_holm:.3f} and {sec(MLP, ME).p_holm:.3f})", MAIN))
    c.append((f"the 1D CNN under the mean ({aucf(CNN1, ME)}; p_Holm = {sec(CNN1, ME).p_holm:.3f})", MAIN))
    assert sec(CNN1, ME).p_holm > 0.05 and sec(MLP, M2).p_holm > 0.05 and sec(MLP, ME).p_holm > 0.05
    assert all(sec(m, a).delta > 0 and sec(m, a).p_holm < 0.05 for m in [LR, MM, HEUR, ZS] for a in [M2, ME]) and sec(CNN1, M2).delta > 0 and sec(CNN1, M2).p_holm < 0.05
    c.append((f"under the maximum ({aucf(CNN1, M2)})", MAIN))
    c.append((f"Zero-shot d1-3B was inverted ({aucf(ZS, M2)} and {aucf(ZS, ME)}), as was the heuristic ({aucf(HEUR, M2)} and {aucf(HEUR, ME)}) and, under the mean, logistic regression ({aucf(LR, ME)})", MAIN))
    c.append((f"almost no discrimination ({aucf(MM, M2)} and {aucf(MM, ME)})", MAIN))
    P = "partial_vs_clean"
    c.append((f"the d1-3B probe reached {aucf(PROBE, M2, P)} and {aucf(PROBE, ME, P)}, with intervals that included 0.5", MAIN))
    assert all(au(PROBE, a, P).auroc_ci_low < 0.5 < au(PROBE, a, P).auroc_ci_high for a in [M2, ME])
    c.append((f"HuBERT-ECG probe ({aucf(HUB, M2, P)} and {aucf(HUB, ME, P)}) and LightGBM ({aucf(GBM, M2, P)} and {aucf(GBM, ME, P)})", MAIN))
    ptb = pd.read_csv(R / "ptbxl_frozen_summary.csv")
    ptb_max = float(ptb[(ptb.outcome == "global_vs_clean") & (ptb.aggregation == "max_12")].auroc.max())
    c += [(f"Under the maximum rule no model exceeded an AUROC of {f3(ptb_max)} on this outcome.", MAIN)]
    assert aucf(PROBE, M2) == "0.579"
    if VARIANT == "v2":  # conclusions of the full-length version
        c.append((f"(AUROC {aucf(PROBE, M2)}–{aucf(PROBE, ME)})", MAIN))
    # 12-lead value in the abstract
    tw = pd.read_csv(R / "d1_probe_twelve_lead_summary.csv").set_index("aggregation")
    assert f"{tw.auroc['max_12']:.3f}" == "0.951"  # quoted in section 3.3 (checked above with its interval)
    # ---- additions after the code audit of 2026-11-11 (evaluation/audit_sensitivity.py)
    # bootstrap p floor: MLP and logistic regression versus the best model on BUT QDB
    hb = [holm_best(BUT, m) for m in (MLP, LR)]
    assert hb[0] == hb[1] and pair(BUT, CNN2, MLP).p == pair(BUT, CNN2, LR).p == 1 / 2000
    c.append((f"(both p_Holm = {f3(hb[0])}): no resample reversed the sign, so the unadjusted p is the 1/n floor of 0.0005", MAIN))
    # CinC 2011 without constant lead II
    nc = pd.read_csv(R / "cinc2011_nonconstant_paired.csv")
    assert set(nc.best_model) == {PROBE} and set(nc.n_constant_removed) == {73} and set(nc.n_constant_unacceptable) == {73}
    vb = nc[(nc.model_a == PROBE) | (nc.model_b == PROBE)]
    x = vb[(vb.model_a == CNN2) | (vb.model_b == CNN2)].iloc[0]
    sign = -1 if x.model_b == PROBE else 1
    d, lo, hi = sign * x.delta, min(sign * x.ci_lo, sign * x.ci_hi), max(sign * x.ci_lo, sign * x.ci_hi)
    assert lo < 0 < hi
    c.append((f"on the remaining {int(x.n_units)} records ({int(x.n_positive)} unacceptable; Table S40; post hoc)", MAIN))
    c.append((f"(difference {f3(d)}, 95% CI −{f3(-lo)} to {f3(hi)}; unadjusted p = {f3(x.p)}, p_Holm = {f3(x.p_holm)})", MAIN))
    others = vb[(vb.model_a != CNN2) & (vb.model_b != CNN2)]
    assert len(others) == 9
    c.append((f"the other nine models stayed significantly below the probe (all p_Holm ≤ {f3(others.p_holm.max())})", MAIN))
    aw = lambda m: f3(float(x.auroc_a if x.model_a == m else x.auroc_b))  # noqa: E731
    assert (aw(PROBE), aw(CNN2)) == ("0.740", "0.716")
    if VARIANT == "v2":  # abstract of the full-length version
        c.append((f"(Holm-adjusted p = {f3(pair(C11, CNN2, PROBE).p_holm)} with them, {f3(x.p_holm)} without", MAIN))
    c.append((f"({int(x.n_units)} records, {int(x.n_positive)} unacceptable)", SUPP))
    # BUT QDB per-subject AUROC
    ps = pd.read_csv(R / "butqdb_per_subject_auroc.csv", dtype={"subject": str})
    s111 = ps[ps.subject == "111"].set_index("model").auroc
    cnn, low = s111[[CNN1, CNN2]], s111[[HEUR, GBM, LR, MLP, ZS, HUB]]
    assert s111.drop([CNN1, CNN2]).max() < cnn.min() and ps[ps.subject.str.isdigit()].subject.nunique() == 7
    c.append((f"there both CNNs kept an AUROC of {f3(cnn.min())}–{f3(cnn.max())}, whereas the heuristic, LightGBM, logistic regression, "
              f"the MLP, zero-shot d1-3B and the HuBERT-ECG probe had {f3(low.min())}–{f3(low.max())}", MAIN))
    s105 = ps[ps.subject == "105"]
    assert set(s105.n_unusable) == {1} and (s105.auroc == 1.0).all()
    # CinC 2017 by number of windows
    ls = pd.read_csv(R / "cinc2017_length_stratified.csv")
    nst = ls.drop_duplicates("stratum").set_index("stratum")
    one, three, allr = nst.loc["1_window"], nst.loc["3_windows"], nst.loc["all"]
    c.append((f"{int(one.n_noisy)} of the {int(allr.n_noisy)} noisy recordings but only {int(one.n_recordings - one.n_noisy)} of the "
              f"{int(allr.n_recordings - allr.n_noisy):,} others contributed a single window", MAIN))
    c.append((f"the number of windows alone gave an AUROC of {f3(float(ls[ls.aggregation == 'none'].auroc.iloc[0]))}", MAIN))
    m3 = ls[(ls.stratum == "3_windows") & (ls.aggregation == "max")].set_index("model").auroc.sort_values(ascending=False)
    assert list(m3.index[:5]) == [HUB, HEUR, LR, PROBE, GBM]
    c.append((f"Among the {int(three.n_recordings):,} recordings with three windows ({int(three.n_noisy)} noisy), the maximum gave {f3(m3[HUB])} for the "
              f"HuBERT-ECG probe, {f3(m3[HEUR])} for the heuristic, {f3(m3[LR])} for logistic regression, {f3(m3[PROBE])} for the d1-3B probe and "
              f"{f3(m3[GBM])} for LightGBM", MAIN))
    # BUT QDB quota versus official intervals
    q = pd.read_csv(R / "butqdb_quota_candidates_by_subject.csv").set_index("subject_id")
    c.append((f"subject 105 contributes {int(q.cohort_unusable[105])} unusable window, although {int(q.class3_candidates_inside_official[105])} "
              f"Class 3 windows of that subject lie inside the official intervals", MAIN))
    assert int(q.cohort_unusable.sum()) == 238 and int(q.class3_candidates_inside_official.sum()) == 605
    # ---- reframing of 2026-11-11: rank agreement, transferred thresholds
    rk = pd.read_csv(R / "rank_agreement.csv").set_index("model")
    rho = pd.read_csv(R / "rank_agreement_spearman.csv").set_index(["family_a", "family_b"]).spearman_rho
    assert set(pd.read_csv(R / "rank_agreement_spearman.csv").n_models) == {11}
    r11, r17, rx = rho[(BUT, C11)], rho[(BUT, C17)], rho[(C11, C17)]
    assert r17 < 0
    c.append((f"was {r11:.2f} on CinC 2011 and −{-r17:.2f} on CinC 2017, and between the two external datasets {rx:.2f}", MAIN))
    if VARIANT == "v2":
        c.append((f"(Spearman ρ = {r11:.2f}), with no relation detected on CinC 2017 (ρ = −{-r17:.2f})", MAIN))
    else:  # Scientific Reports abstract
        c.append((f"moderate on one dataset (Spearman ρ = {r11:.2f}) and not detected on the other (ρ = −{-r17:.2f})", MAIN))
        c.append(("a raw-signal network fell from second to tenth, and an ECG foundation-model probe rose from eighth to first", MAIN))
    rank = lambda m, f: int(rk.loc[m, f"rank | {f}"])  # noqa: E731
    assert [rank(CNN1, f) for f in (BUT, C11, C17)] == [2, 5, 10] and [rank(CNN2, f) for f in (BUT, C11, C17)] == [1, 2, 8]
    assert [rank(HUB, f) for f in (BUT, C11, C17)] == [8, 9, 1]
    c.append((f"The 1D CNN was second in-domain ({f3(auroc(BUT, CNN1))}), fifth on CinC 2011 and tenth on CinC 2017 ({f3(auroc(C17, CNN1))})", MAIN))
    c.append((f"the HuBERT-ECG probe, eighth in-domain ({f3(auroc(BUT, HUB))}), was ninth on CinC 2011 and first on CinC 2017", MAIN))
    tt = pd.read_csv(R / "threshold_transfer_summary.csv")
    tt = tt[tt.rule == "target1_fpr5"]
    # ---- round V10 (review PAPER-002/004/006): observed ranks of the d1-3B probe, thresholds meeting the target, label file
    assert [rank(PROBE, f) for f in (BUT, C11, C17)] == [4, 1, 2]
    top4 = [m for m in rk.index if all(rank(m, f) <= 4 for f in (BUT, C11, C17))]
    assert top4 == [PROBE], top4
    c.append(("By observed rank, the d1-3B image probe was fourth in-domain, first on CinC 2011 and second on CinC 2017, "
              "the only model in the top four of all three sets", MAIN))
    qs = pd.read_csv(R.parent / f"{R.name}_quota_sensitivity" / "quota_sensitivity_auroc.csv").set_index(["set", "model"])
    sets_qs = ["BUT QDB", "CinC 2011 lead II", "CinC 2017"]
    top4_qs = sorted(m for m in qs.loc["BUT QDB"].index if all(qs.loc[(s, m)].rank_sensitivity <= 4 for s in sets_qs))
    assert top4_qs == sorted([PROBE, HUB]), top4_qs
    assert [int(qs.loc[(s, HUB)].rank_sensitivity) for s in sets_qs] == [3, 4, 4]
    met = tt[(~tt.dataset.str.contains("BUT")) & (tt.fpr <= 0.05)]
    assert len(tt[~tt.dataset.str.contains("BUT")]) == 22 and len(met) == 2
    m11 = met[met.dataset.str.startswith("CinC2011")].iloc[0]
    m17 = met[met.dataset.str.startswith("CinC2017")].iloc[0]
    assert m11.model == "Independent_SQI_MLP" and m17.model == "LiquidAI_d1_3B_Neural"
    c.append((f"Two of these 22 thresholds met the target: that of the MLP on CinC 2011 (false reject rate {100 * m11.fpr:.1f}%, "
              f"sensitivity {100 * m11.sens:.1f}%) and that of zero-shot d1-3B on CinC 2017 ({100 * m17.fpr:.1f}%, {100 * m17.sens:.1f}%)", MAIN))
    lp = pd.read_csv(R / "cinc2017_label_provenance.csv").set_index("file")
    assert int(lp.loc["REFERENCE-v3.csv", "records_agreeing_with_used_file"]) == 8528
    c.append((f"which marks {int(lp.loc['REFERENCE-v3.csv', 'noisy_records'])} recordings as noisy (the original labels, REFERENCE-v0.csv, "
              f"mark {int(lp.loc['REFERENCE-v0.csv', 'noisy_records'])}, and REFERENCE-v2.csv, the file inside the training archive, "
              f"{int(lp.loc['REFERENCE-v2.csv', 'noisy_records'])})", MAIN))
    pa = pd.read_csv(R / "butqdb_bootstrap_p_audit.csv")
    assert (pa.n_valid == 2000).all()
    # round V18 (S28 caption, PAPER-003): 1D CNN scores of exactly 1.0 on CinC 2017 all belong to usable recordings
    _w = pd.read_csv(R / "external_window_predictions.csv", low_memory=False)
    _r = _w[_w.dataset == "CinC2017"].groupby("record_name").agg(s=("1D_CNN_RawECG", "max"), y=("label", "first"))
    assert len(_r) == 8512 and int(_r[_r.s >= 1.0].y.sum()) == 0
    if VARIANT == "srep":
        c.append((f"On CinC 2017, {int((_r.s >= 1.0).sum())} recordings have a 1D CNN score of exactly 1.0, all of them usable", SUPP))
    # round V18 (Methods, Inputs): acceptable records with a constant lead follow from the S35 caption counts
    _s = SUPP.read_text()
    assert "with and without the 135 records that have at least one constant lead (signal SD 0; 129 of them unacceptable)" in _s
    if VARIANT == "srep":
        c.append((f"135 records had at least one constant lead, {135 - 129} of them acceptable (Table S35)", MAIN))
    # round V19 (PAPER-005): 55 BUT QDB pairs with 20,000 resamples, plus-one convention without the 1/n floor
    bb = pd.read_csv(R / "butqdb_bootstrap_b20000_sensitivity.csv")
    assert len(bb) == 55 and (bb.n_valid == 20000).all()
    bb = bb.set_index(["model_a", "model_b"])
    hb = lambda a, b: f"{bb.loc[(a, b), 'p_plus1_holm']:.3f}"
    h_mlp, h_lr = hb("Independent_SQI_MLP", "2D_CNN_Image"), hb("Logistic_Regression_SQI", "2D_CNN_Image")
    floor_min = 2 * 55 / 20001
    assert abs(bb.p_plus1_holm.min() - floor_min) < 1e-12 and int((bb.p_plus1_holm < 0.05).sum()) == 4
    if VARIANT == "srep":
        assert float(h_mlp) < 0.05 and float(h_lr) < 0.05  # V23: the values themselves stay in the Table 1 caption
        c.append(("and both remained significant with 20,000 resamples (Table S54)", MAIN))
        c.append((f"gave {h_mlp} for the MLP and {h_lr} for logistic regression", MAIN))
        c.append((f"whose smallest attainable Holm-adjusted value is {floor_min:.4f} (Table S54)", MAIN))
        c.append((f"the MLP against the 1D CNN ({bb.loc[('Independent_SQI_MLP', '1D_CNN_RawECG'), 'p_plus1_holm']:.4f}), against the 2D CNN "
                  f"and against the rule heuristic (both {h_mlp}), and logistic regression against the 2D CNN ({h_lr})", SUPP))
        assert hb("Independent_SQI_MLP", "Rule_Based_SQI_Heuristic") == h_mlp
        lh = bb.loc[("Logistic_Regression_SQI", "Rule_Based_SQI_Heuristic")]
        mh = bb.loc[("Independent_SQI_MLP", "Rule_Based_SQI_Heuristic")]
        c.append((f"whereas logistic regression against the rule heuristic, {lh.p_holm_primary_2000:.3f} at the floor, gave "
                  f"{lh.p_plus1_holm:.3f}, and the MLP against the heuristic, {mh.p_holm_primary_2000:.3f} with 2,000 resamples, gave "
                  f"{mh.p_plus1_holm:.3f}. The next value was {bb.loc[('2D_CNN_Image', 'LiquidAI_d1_3B_Neural'), 'p_plus1_holm']:.3f}", SUPP))
        assert sorted(bb.p_plus1_holm)[4] == bb.loc[("2D_CNN_Image", "LiquidAI_d1_3B_Neural"), "p_plus1_holm"]
    # round V19 (PAPER-002): bSQI detector diagnostic
    bs = pd.read_csv(R / "bsqi_diagnostic_summary.csv")
    q = bs[bs.label >= 0].set_index(["dataset", "label", "quantity"])["median"]
    au = bs[bs.label == -1].set_index(["dataset", "quantity"])["median"]
    _c11, _bq = "CinC 2011 lead II", "BUT QDB"
    if VARIANT == "srep":
        c.append((f"detected a median of {q[(_c11, 0, 'n_pt')]:.0f} beats per record and agreed with the wfdb XQRS detector (median "
                  f"agreement {q[(_c11, 0, 'pt_vs_xqrs')]:.3f}), whereas the second-difference detector detected {q[(_c11, 0, 'n_diff')]:.0f}, "
                  f"about two peaks per beat, so the stored bSQI had a median of {q[(_c11, 0, 'bsqi_orig')]:.3f}", SUPP))
        c.append((f"the second-difference detector detected {q[(_c11, 0, 'n_diff_lp40')]:.0f} beats, and bSQI computed with it had a "
                  f"median of {q[(_c11, 0, 'bsqi_lp40')]:.3f} in these records. In usable BUT QDB windows the second-difference detector "
                  f"detected {q[(_bq, 0, 'n_diff')]:.0f} beats ({q[(_bq, 0, 'n_diff_lp40')]:.0f} after the filter)", SUPP))
        c.append((f"rose from {au[(_bq, 'AUROC bsqi_orig (all)')]:.3f} to {au[(_bq, 'AUROC bsqi_lp40 (all)')]:.3f} on BUT QDB and from "
                  f"{au[(_c11, 'AUROC bsqi_orig (all)')]:.3f} to {au[(_c11, 'AUROC bsqi_lp40 (all)')]:.3f} on CinC 2011 lead II (from "
                  f"{au[(_c11, 'AUROC bsqi_orig (without constant lead)')]:.3f} to {au[(_c11, 'AUROC bsqi_lp40 (without constant lead)')]:.3f} "
                  f"without the 73 constant-lead records)", SUPP))
        c.append((f"GQRS detected {q[(_bq, 0, 'n_gqrs')]:.0f} peaks against {q[(_bq, 0, 'n_xqrs')]:.0f} for XQRS, and the AUROC of their "
                  f"agreement there was {au[(_bq, 'AUROC xqrs_vs_gqrs_150 (all)')]:.3f}", SUPP))
        assert q[(_c11, 0, "bsqi_orig")] < 0.6 and abs(q[(_c11, 0, "n_diff")] / q[(_c11, 0, "n_pt")] - 2) < 0.25
    # round V19 (PAPER-004): SQI inputs on PTB-XL
    px = pd.read_csv(R / "ptbxl_sqi_diagnostic.csv").set_index(["unit", "sqi"])
    l2 = lambda k: px.loc[("lead II", k)]
    _oof = pd.read_csv(R / "cohort1000_consolidated_oof_predictions.csv")
    z = l2("flatline_fraction, z relative to BUT QDB")
    if VARIANT == "srep":
        c.append((f"(median {l2('flatline_fraction').median_clean:.3f} against {l2('flatline_fraction').median_global:.3f}; univariate "
                  f"AUROC for global noise {l2('flatline_fraction').auroc_global_vs_clean:.3f})", SUPP))
        c.append((f"whose flatline fraction is at most {_oof.flatline_fraction.max():.4f}, these medians lie {z.median_clean:,.0f} and "
                  f"{z.median_global:,.0f} SDs above the BUT QDB mean", SUPP))
        c.append((f"the expected direction (AUROC {l2('bsqi_detector_agreement').auroc_global_vs_clean:.3f}), and the high-frequency ratio "
                  f"and the zero-crossing rate were higher ({l2('high_frequency_noise_ratio').auroc_global_vs_clean:.3f} and "
                  f"{l2('zero_crossing_rate').auroc_global_vs_clean:.3f})", SUPP))
    # round V20 (PAPER-010): PTB-XL folds of the sampled records (HuBERT-ECG pre-training exposure, Table S51)
    fc = pd.read_csv(R / "ptbxl_strat_fold_counts.csv")
    prim = fc[fc.stratum.isin(["global", "clean"])]
    n_prim, n_f10, n_f9 = int(prim.records.sum()), int(prim[prim.strat_fold == 10].records.sum()), int(prim[prim.strat_fold == 9].records.sum())
    assert n_prim == 600 and int(fc.records.sum()) == 900
    if VARIANT == "srep":
        c.append((f"probably except the test fold that holds {n_f10} of the {n_prim} primary PTB-XL records", MAIN))
        f10 = prim[prim.strat_fold == 10].set_index("stratum").records
        c.append((f"{n_f10} are in fold 10 ({f10['global']} global noise, {f10['clean']} clean; AUROCs in Table S58) and "
                  f"{n_prim - n_f10} in folds 1–9, {n_f9} of them in fold 9", SUPP))
    # round V21 (PAPER-003): subject influence, evaluation only
    _vsi = pd.read_csv(R / "butqdb_subject_influence.csv")
    _vlo = _vsi[_vsi.left_out != "none"]
    assert len(_vlo) == 15 and (_vsi.delta_LR_minus_2D_CNN < 0).all() and (_vsi.delta_MLP_minus_2D_CNN < 0).all()
    if VARIANT == "srep":
        c.append((f"the 2D CNN kept the highest AUROC in {int((_vlo.best_model == '2D_CNN_Image').sum())} of 15 cases", SUPP))
        c.append((f"ranged from {_vlo.delta_LR_minus_2D_CNN.min():.3f} to {_vlo.delta_LR_minus_2D_CNN.max():.3f} and those of the MLP from "
                  f"{_vlo.delta_MLP_minus_2D_CNN.min():.3f} to {_vlo.delta_MLP_minus_2D_CNN.max():.3f}".replace("-", "−"), SUPP))
        c.append((f"with those of the full cohort was at least {np.floor(_vlo.spearman_vs_full.min() * 100) / 100:.2f}", SUPP))
        assert set(_vlo[_vlo.best_model != "2D_CNN_Image"].left_out.astype(int)) == {100, 125}
    # round V21 (PAPER-004): PTB-XL AUROC on the test fold
    _vf10 = pd.read_csv(R / "ptbxl_fold10_subset.csv").set_index(["aggregation", "subset", "model"])
    _vhf = _vf10.loc[("max", "fold 10", "HuBERT_ECG_probe")]
    if VARIANT == "srep":
        c.append((f"on which its AUROC was {_vhf.auroc:.3f} (95% CI {_vhf.ci_lo:.3f}–{_vhf.ci_hi:.3f}; Table S58)", MAIN))
        c.append((f"reached an AUROC of {_vhf.auroc:.3f} (95% CI {_vhf.ci_lo:.3f}–{_vhf.ci_hi:.3f}) with the maximum, against "
                  f"{_vf10.loc[('max', 'folds 1-9', 'HuBERT_ECG_probe')].auroc:.3f} on the {int(_vf10.loc[('max', 'folds 1-9', 'HuBERT_ECG_probe')].n)} "
                  f"records of folds 1–9, and LightGBM {_vf10.loc[('max', 'fold 10', 'LightGBM_SQI')].auroc:.3f}", SUPP))
    # round V21 (PAPER-006): decomposition of the CinC 2011 margin by flat records
    _vfd = pd.read_csv(R / "cinc2011_flat_lead_decomposition.csv").set_index("model")
    _vpr, _vcn = _vfd.loc["d1_image_probe"], _vfd.loc["2D_CNN_Image"]
    _vdm, _vdfl, _vdnf = _vpr.auroc_all - _vcn.auroc_all, _vpr.contribution_flat - _vcn.contribution_flat, _vpr.contribution_nonflat - _vcn.contribution_nonflat
    _vdsub = _vpr.auroc_nonflat_vs_negatives - _vcn.auroc_nonflat_vs_negatives
    if VARIANT == "srep":
        c.append((f"Of the probe's margin of {_vdm:.3f}, {_vdfl:.3f} came from the flat records and {_vdnf:.3f} from the others (Table S59)", MAIN))
        c.append((f"the leaders' AUROC difference fell from {_vdm:.3f} to {_vdsub:.3f}, losing significance", MAIN))
        c.append((f"against which the probe reached {_vpr.auroc_flat_vs_negatives:.3f} and the 2D CNN {_vcn.auroc_flat_vs_negatives:.3f}, and "
                  f"{_vdnf:.3f} from the non-flat records ({_vpr.auroc_nonflat_vs_negatives:.3f} against {_vcn.auroc_nonflat_vs_negatives:.3f})", SUPP))
        c.append((f"yet ranked them above the acceptable records with an AUROC of {_vfd.loc['LightGBM_SQI'].auroc_flat_vs_negatives:.3f}", SUPP))
    # round V21 (PAPER-002): Target 1 transfer with the 1,041-window models
    tq = pd.read_csv(R / "threshold_transfer_quota_sensitivity.csv")
    tq = tq[(tq.rule == "target1_fpr5") & ~tq.dataset.str.startswith("BUT")]
    _qmet = {cq: set(zip(g[g.fpr <= 0.05].model, g[g.fpr <= 0.05].dataset)) for cq, g in tq.groupby("cohort")}
    assert _qmet["primary"] == _qmet["sensitivity"] and len(_qmet["primary"]) == 2
    _qs = tq[tq.cohort == "sensitivity"].set_index(["model", "dataset"])
    _qm11 = _qs.loc[("Independent_SQI_MLP", "CinC2011 all 998 records (lead II)")]
    _qr17 = tq[(tq.cohort == "sensitivity") & tq.dataset.str.startswith("CinC2017")].fpr
    _qr11 = tq[(tq.cohort == "sensitivity") & tq.dataset.str.startswith("CinC2011")].fpr
    if VARIANT == "srep":
        c.append((f"that of the MLP on CinC 2011 (false reject rate {100 * _qm11.fpr:.1f}%, sensitivity {100 * _qm11.sens:.1f}%)", SUPP))
        c.append((f"False reject rates ranged from {100 * _qr17.min():.1f}% to {100 * _qr17.max():.1f}% on CinC 2017 and from "
                  f"{100 * _qr11.min():.1f}% to {100 * _qr11.max():.1f}% on CinC 2011", SUPP))
    # round V17 (S50 caption): the "-0.000" upper limit of the multimodal net is a negative value that rounds to zero
    mm = pairs[pairs.family.str.startswith("BUT") & (pairs.model_a == "2D_CNN_Image") & (pairs.model_b == "Multimodal_Fusion")]
    assert len(mm) == 1 and -0.0005 < -mm.ci_lo.iloc[0] < 0
    c.append((f"the upper limit for the multimodal net is {-mm.ci_lo.iloc[0]:.4f}, so its interval excludes zero", SUPP))
    # ---- round V11 (review PAPER-001/009): all 55 pairs, floor-dependent significance
    ap = pd.read_csv(R / "butqdb_bootstrap_p_audit_all_pairs.csv")
    assert len(ap) == 55 and (ap.n_valid == 2000).all()
    assert int((ap.p_holm < 0.05).sum()) == 4 and int((ap.p_plus1_holm < 0.05).sum()) == 0
    sig = ap[ap.p_holm < 0.05]
    assert (sig[["n_delta_le0", "n_delta_ge0"]].min(axis=1) == 0).all()  # every significant pair is at the floor
    assert round(55 * 0.0005, 4) == 0.0275 and round(55 * 2 / 2001, 3) == 0.055
    vb = ap[(ap.model_b == "2D_CNN_Image") & ap.model_a.isin([LR, MLP])]
    assert len(vb) == 2 and np.allclose(vb.p_holm, 0.0275)
    # round V12: Table 1 / Fig. 2 (two significant comparisons with the best, both at the floor) and the quota cohort
    assert int((ap[(ap.model_a == "2D_CNN_Image") | (ap.model_b == "2D_CNN_Image")].p_holm < 0.05).sum()) == 2
    qp = pd.read_csv(R.parent / f"{R.name}_quota_sensitivity" / "paired_comparisons.csv")
    qp = qp[qp.family.str.startswith("BUT")]
    assert len(qp) == 55 and (qp[qp.p_holm < 0.05].p == 0.0005).all()
    c.append(("On BUT QDB the two open markers rest on the 1/n floor of the bootstrap p-value", MAIN))
    if VARIANT == "v2":  # V15: the srep Discussion no longer repeats this (its Results and Fig. 2 legend keep it)
        c.append(("in both cohorts every such in-domain decision rests on the 1/n floor of the bootstrap p-value", MAIN))
    # round V13 (statistics review): AUROC differences from the in-domain best with their cluster-bootstrap intervals
    fam_b = "BUT_QDB (5-fold subject CV)"

    def vs_cnn2(m):  # (Δ, lo, hi) oriented as model − 2D CNN
        r = pair(fam_b, m, CNN2)
        return (r.delta, r.ci_lo, r.ci_hi) if r.model_a == m else (-r.delta, -r.ci_hi, -r.ci_lo)

    seven = [CNN1, HEUR, PROBE, MM, GBM, TAB, HUB]
    assert all(holm_best(fam_b, m) >= 0.64 for m in seven)
    sd = [vs_cnn2(m) for m in seven]
    def n3(v):
        return f"{v:.3f}".replace("-", "−")

    c.append((f"The AUROC differences of these seven from the 2D CNN ranged from {n3(min(x[0] for x in sd))} to "
              f"{n3(max(x[0] for x in sd))}, and the lower limits of their unadjusted 95% cluster-bootstrap intervals ranged "
              f"from {n3(min(x[1] for x in sd))} to {n3(max(x[1] for x in sd))}", MAIN))
    dm, dl = vs_cnn2(MLP), vs_cnn2(LR)
    assert dm[2] < 0 and dl[2] < 0  # both intervals exclude zero
    c.append((f"The MLP (0.904; difference {n3(dm[0])}, 95% CI {n3(dm[1])} to {n3(dm[2])}) and logistic regression (0.899; "
              f"{n3(dl[0])}, 95% CI {n3(dl[1])} to {n3(dl[2])})", MAIN))
    c.append(("Their 95% intervals, which do not depend on this convention, exclude zero but are wide.", MAIN))
    c.append(("as throughout CinC 2017, the intervals and tests treat recordings as independent", MAIN))
    c.append(("contain only one class discarded (none were, in any of the 55 pairs)", MAIN))
    c.append(("the smallest attainable Holm-adjusted value over 55 pairs is 0.0275 under this floor and 0.055 if one is added", MAIN))
    c.append(("no BUT QDB pair could reach p_Holm < 0.05 with 2,000 resamples (smallest attainable value 0.055", MAIN))
    f17 = 100 * tt[tt.dataset == "CinC2017 all (record max)"].fpr
    f11 = 100 * tt[tt.dataset == "CinC2011 all 998 records (lead II)"].fpr
    assert len(f17) == len(f11) == 11
    c.append((f"{f17.min():.1f}–{f17.max():.1f}% across the 11 models on CinC 2017 and {f11.min():.1f}–{f11.max():.1f}% on CinC 2011 (Table S4)", MAIN))
    if VARIANT == "v2":
        c.append((f"gave {f17.min():.0f}–{f17.max():.0f}% on CinC 2017", MAIN))
        c.append((f"the number of windows alone gave AUROC {f3(float(ls[ls.aggregation == 'none'].auroc.iloc[0]))}", MAIN))
    else:
        c.append((f"recording length alone gave an AUROC of {f3(float(ls[ls.aggregation == 'none'].auroc.iloc[0]))}", MAIN))
    # ---- latency prose (session repeated 2026-11-11 with a common input boundary; PAPER-009)
    lv = latency_values()
    runs = [pd.read_csv(R / "latency_repeats" / f"run{k}_summary.csv").set_index("Model") for k in (1, 2, 3)]
    lmed = lambda m, col: float(np.median([r.loc[m, col] for r in runs]))  # noqa: E731
    HN = "Rule-Based SQI Heuristic"
    sqi_tot = [lmed(m, "Total_End_to_End_Median_ms") for m in ("Logistic Regression (SQI)", "LightGBM (SQI)", "Independent SQI MLP")]
    c += [(f"about {lv['LAT_RENDER']} ms ({lv['LAT_RENDER_PCT']}% of the total) against {lv['LAT_2DFWD']} ms for the forward pass", SUPP),
          (f"it took {lmed(HN, 'Total_End_to_End_Median_ms'):.2f} ms, of which {lmed(HN, 'Preprocess_Median_ms'):.2f} ms was preprocessing; only the three SQI models "
           f"({min(sqi_tot):.2f}–{max(sqi_tot):.2f} ms) and the 1D CNN avoid rendering", SUPP),
          (f"a direct raster renderer took {lv['LAT_RASTER']} ms (Table S8)", SUPP),
          (f"took {lv['LAT_1D']} ms on the GPU from the stored 500-Hz window (Table S45). d1-3B took {lv['LAT_D1']} ms per window, dominated by its {lv['LAT_D1FWD']}-ms forward pass.", SUPP),
          (f"the HuBERT-ECG probe took {lv['LAT_HUB']} ms per window on MPS ({lv['LAT_HUBCPU']} ms on the CPU), of which {lv['LAT_HUBFWD']} ms was the forward pass; "
           f"the linear d1-3B probe adds {lv['LAT_D1PROBE']} ms to the d1-3B forward pass", SUPP),
          (f"took {lv['LAT_TAB']} s per window on the CPU", SUPP),
          (f"(1D CNN {lv['LAT_T5_1D']} vs {lv['LAT_S8_1D']} ms; 2D CNN {lv['LAT_T5_2D']} vs {lv['LAT_S8_2D']} ms)", SUPP),
          (f"namely rendering of about {lv['LAT_RENDER']} ms versus the forward pass, and {lv['LAT_D1FWD']} ms for d1-3B", SUPP)]
    # ---- harmonised pipeline (PAPER-002; experiments/harmonised_pipeline.py)
    hs = pd.read_csv(R / "harmonised_summary.csv").set_index(["dataset", "model"])
    hseed = pd.read_csv(R / "harmonised_seed_range.csv")
    hr = pd.read_csv(R / "harmonised_rank_agreement.csv").set_index(["set_a", "set_b"])
    ho = lambda ds, m: f3(hs.loc[(ds, m), "auroc_original"])  # noqa: E731
    hh = lambda ds, m: f3(hs.loc[(ds, m), "auroc_harmonised"])  # noqa: E731
    def seeds(ds, m):
        g = hseed[(hseed.dataset == ds) & (hseed.model == m)].auroc_harmonised
        assert len(g) == 5
        return f"{f3(g.min())}–{f3(g.max())}"
    D = "CinC2011"
    c.append((f"logistic regression rose from {ho(D, LR)} to {hh(D, LR)}, the multimodal net from {ho(D, MM)} to {hh(D, MM)} (seeds: {seeds(D, MM)}) "
              f"and the MLP from {ho(D, MLP)} to {hh(D, MLP)} (seeds: {seeds(D, MLP)})", MAIN))
    assert not hs.loc[(D, HUB), "refitted"] and not hs.loc[(D, CNN1), "refitted"]
    c.append((f"raised the HuBERT-ECG probe from {ho(D, HUB)} to {hh(D, HUB)} and the 1D CNN from {ho(D, CNN1)} to {hh(D, CNN1)}", MAIN))
    c11h = hs.loc[D]
    others = c11h.drop(ZS)
    rng_h, rng_o = f"{f3(others.auroc_harmonised.min())}–{f3(others.auroc_harmonised.max())}", f"{f3(others.auroc_original.min())}–{f3(others.auroc_original.max())}"
    c.append((f"then spanned {rng_h} instead of {rng_o}", MAIN))
    if VARIANT == "v2":
        c.append((f"from {rng_o} to {rng_h}", MAIN))                       # abstract
        c.append((f"({rng_h} for the ten models other than zero-shot d1-3B, against {rng_o})", MAIN))
    top2 = c11h.auroc_harmonised.sort_values(ascending=False).index[:2]
    assert list(top2) == [PROBE, CNN2] and hh(D, ZS) == ho(D, ZS)
    c.append((f"the d1-3B probe ({hh(D, PROBE)}) and the 2D CNN ({hh(D, CNN2)}) remained the two highest, and zero-shot d1-3B was unchanged ({hh(D, ZS)})", MAIN))
    dmax = lambda ds: hs.loc[ds].delta.abs().max()  # noqa: E731
    mm17 = hs.loc[("CinC2017", MM)]
    assert hs.loc["CinC2017"].delta.abs().idxmax() == MM
    c.append((f"In-domain AUROCs changed by at most {f3(dmax('BUT_QDB'))} and CinC 2017 AUROCs by at most {f3(dmax('CinC2017'))} "
              f"(multimodal net, {f3(mm17.auroc_original)} to {f3(mm17.auroc_harmonised)})", MAIN))
    a, b_, x_ = hr.loc[("BUT_QDB", "CinC2011")], hr.loc[("BUT_QDB", "CinC2017")], hr.loc[("CinC2011", "CinC2017")]
    assert b_.spearman_harmonised < 0 and b_.spearman_original < 0
    c.append((f"was {a.spearman_harmonised:.2f} on CinC 2011 and −{-b_.spearman_harmonised:.2f} on CinC 2017, against {a.spearman_original:.2f} and "
              f"−{-b_.spearman_original:.2f} with the original pipeline, while the agreement between the two external datasets rose from "
              f"{x_.spearman_original:.2f} to {x_.spearman_harmonised:.2f}", MAIN))
    c.append((f"lead II AUROC was {hh(D, HUB)} (Table S46)" if VARIANT == "v2" else
              f"raised its CinC 2011 lead II AUROC to {hh(D, HUB)} (Table S46)", MAIN))
    # ---- third review (2026-11-13): PAPER-001/002 rank uncertainty, PAPER-003 calibration, PAPER-004 demographics,
    # PAPER-006 zero-rejection thresholds, PAPER-012 pairwise comparisons among the eight in-domain leaders
    u = pd.read_csv(R / "rank_uncertainty.csv").set_index(["model_set", "set_a", "set_b", "statistic"])
    ua = lambda a, b, st, ms="all 11 models": u.loc[(ms, a, b, st)]  # noqa: E731
    r1, r2, r3 = ua("BUT QDB", "CinC 2011", "spearman_rho"), ua("BUT QDB", "CinC 2017", "spearman_rho"), ua("CinC 2011", "CinC 2017", "spearman_rho")
    assert r1.ci_lo > 0 and r2.ci_lo < 0 < r2.ci_hi and r3.ci_lo > 0
    assert abs(r1.estimate - r11) < 1e-12 and abs(r2.estimate - r17) < 1e-12 and abs(r3.estimate - rx) < 1e-12
    c.append((f"gave 95% intervals of {r1.ci_lo:.2f} to {r1.ci_hi:.2f} for CinC 2011, −{-r2.ci_lo:.2f} to {r2.ci_hi:.2f} for CinC 2017 and "
              f"{r3.ci_lo:.2f} to {r3.ci_hi:.2f} between the external datasets", MAIN))
    l1, l2 = ua("BUT QDB", "CinC 2011", "same_leader"), ua("BUT QDB", "CinC 2017", "same_leader")
    g1, g2 = ua("BUT QDB", "CinC 2011", "regret_auroc"), ua("BUT QDB", "CinC 2017", "regret_auroc")
    assert l1.estimate == 0 and l2.estimate == 0
    assert l2.prop_positive == 0 and int(l2.resamples) == 2000  # V3: "none of the 2,000"
    # V6 (review PAPER-013): ties for the highest AUROC; primary rule = first in the fixed model order
    n_pos = lambda a, b, st: int(ua(a, b, st).n_resamples_positive)  # noqa: E731
    assert round(l1.prop_positive * 2000) == 11
    assert (n_pos("BUT QDB", "CinC 2011", "ties_selection"), n_pos("BUT QDB", "CinC 2017", "ties_target")) == (7, 1)
    assert n_pos("BUT QDB", "CinC 2011", "ties_target") == 0
    assert all(ua(a, b, st).estimate == 0 for a, b in (("BUT QDB", "CinC 2011"), ("BUT QDB", "CinC 2017"), ("CinC 2011", "CinC 2017"))
               for st in ("ties_selection", "ties_target"))  # no tie in the full-data ranking
    any1, any2 = n_pos("BUT QDB", "CinC 2011", "same_leader_any"), n_pos("BUT QDB", "CinC 2017", "same_leader_any")
    for a_, b_ in (("BUT QDB", "CinC 2011"), ("BUT QDB", "CinC 2017"), ("CinC 2011", "CinC 2017")):
        lo_, hi_ = ua(a_, b_, "regret_best_case"), ua(a_, b_, "regret_worst_case")
        assert max(abs(hi_.ci_lo - lo_.ci_lo), abs(hi_.ci_hi - lo_.ci_hi)) <= 0.002 and lo_.estimate == hi_.estimate
    if VARIANT == "srep":  # V15: tie counts left to the S44 caption and table
        # V18: percentage dropped from the text (S44 prints it to one decimal); the count is derived
        c.append((f"The in-domain leader also led CinC 2011 in {round(2000 * l1.prop_positive)} of 2,000 resamples and CinC 2017 in none of "
                  f"the 2,000 (not observed, which does not make the probability zero; tie handling in Table S44). Selecting the in-domain "
                  f"leader cost {g1.estimate:.3f} AUROC on CinC 2011 and {g2.estimate:.3f} on CinC 2017", MAIN))
    else:
      c.append((f"The in-domain leader also led CinC 2011 in 11 of 2,000 resamples ({100 * l1.prop_positive:.2f}%) and CinC 2017 in none of the 2,000 "
              f"(not observed, which does not make the probability zero). Models tied for the highest AUROC in 7 in-domain resamples and in 1 CinC 2017 "
              f"resample; ties were broken by a fixed model order, and breaking every tie in favour of agreement would give {any1} and {any2} of 2,000. "
              f"Selecting the in-domain leader cost {g1.estimate:.3f} AUROC on CinC 2011 and {g2.estimate:.3f} on CinC 2017", MAIN))
    h1, h2 = ua("BUT QDB", "CinC 2011", "spearman_rho", "heuristic excluded"), ua("BUT QDB", "CinC 2017", "spearman_rho", "heuristic excluded")
    assert h2.estimate < 0
    c.append((f"Without the rule heuristic the correlations were {h1.estimate:.2f} and −{-h2.estimate:.2f}", MAIN))
    c.append((f"CinC 2017 was wide (−{-r2.ci_lo:.2f} to {r2.ci_hi:.2f})", MAIN))
    pcomp = pd.read_csv(R / "paired_comparisons.csv")
    pb = pcomp[pcomp.family == BUT]
    au = pd.concat([pb.set_index("model_a").auroc_a, pb.set_index("model_b").auroc_b])
    au = au[~au.index.duplicated()]
    top8 = au[au >= 0.9375].index
    assert len(top8) == 8 and len(pb) == 55
    t8 = pb[pb.model_a.isin(top8) & pb.model_b.isin(top8)]
    assert len(t8) == 28
    c.append((f"None of the 28 pairwise comparisons among these eight models was significant after Holm adjustment over all 55 pairs "
              f"(all p_Holm ≥ {np.floor(1000 * t8.p_holm.min()) / 1000:.3f})", MAIN))
    tr = pd.read_csv(R / "local_threshold_triviality.csv")
    t17, t11 = tr[(tr.dataset == "CinC2017") & (tr.n == 100)], tr[(tr.dataset == "CinC2011") & (tr.n == 100)].set_index("model")
    assert len(t17) == 11 and len(t11) == 11
    assert t11.loc[LR, "zero_rejection"] == 1 and t11.loc[MM, "zero_rejection"] == 1 and (t11.zero_rejection == 1).sum() == 2
    c.append((f"flagged no held-out recording in {100 * t17.zero_rejection.min():.0f}–{100 * t17.zero_rejection.max():.0f}% of these draws, depending on the model, "
              f"so a false reject rate of at most 5% together with a sensitivity above zero was reached in only {100 * t17.useful_control.min():.0f}–"
              f"{100 * t17.useful_control.max():.0f}% of them", MAIN))
    if VARIANT == "v2":  # V15: per-model detail left to Table S22 in the srep version
        c.append((f"and in {100 * t11.loc[CNN1, 'zero_rejection']:.0f}% of draws for the 1D CNN", MAIN))
    dc = pd.read_csv(R / "decision_calibration.csv")
    d11 = dc[dc.family == C11].set_index("model")
    assert (dc.calibration_intercept_status == "estimated").all() and d11.calibration_intercept.idxmin() == MM and d11.calibration_intercept.idxmax() == MLP
    assert d11.loc[MM, "calibration_slope_status"] == "not identifiable (constant score)"
    assert set(d11.index[d11.calibration_slope_status == "not estimable (quasi-complete separation)"]) == {LR, MLP}
    assert (dc.calibration_slope_status != "estimated").sum() == 3
    assert not dc.calibration_slope_status.str.contains("not converged").any()
    # V4: the separation certificate from the data. LR: every record below the upper clipping bound unusable, both
    # classes at it (slope -> -inf); MLP: every record above the lower bound unusable, both classes at it (slope -> +inf)
    from decision_calibration import logit as _lg, separation as _sep
    from paired_comparisons import families as _families
    f11 = _families()[C11][1]
    y_ = f11.y.values.astype(int)
    for m_, bound, off_side, n_off, n_at, n_at_pos in ((LR, "max", "below", 76, 922, 149), (MLP, "min", "above", 75, 923, 150)):
        z_ = _lg(f11[m_].values)
        assert _sep(y_, z_) == "quasi-complete"
        v = z_.max() if bound == "max" else z_.min()
        off = z_ < v if off_side == "below" else z_ > v
        assert off.sum() == n_off and y_[off].min() == 1, m_
        assert (z_ == v).sum() == n_at and y_[z_ == v].sum() == n_at_pos and 0 < n_at_pos < n_at, m_
        assert not (z_ > v).any() if off_side == "below" else not (z_ < v).any()
    c.append(("logistic regression: all 76 records below the upper bound unusable, 149 of the 922 at it; MLP: all 75 records "
              "above the lower bound unusable, 150 of the 923 at it), a quasi-complete separation", MAIN))
    diag = pd.read_csv(R / "calibration_separation_diagnostic.csv")
    boot = pd.read_csv(R / "calibration_separation_bootstrap.csv")
    est = diag[diag.status == "estimated"]
    assert len(est) == 41 and est.within_tolerance.all()
    assert (boot.separation_not_confirmed_by_brute_force == 0).all()
    assert not any("not converged" in col for col in boot.columns if (boot[col] != 0).any())
    c.append((f"On CinC 2011 the intercept ranged from −{-d11.loc[MM, 'calibration_intercept']:.2f} (multimodal net, which gave every record the same score) "
              f"to {d11.loc[MLP, 'calibration_intercept']:.2f} (MLP)", MAIN))
    from config import LOCAL_DATASETS_DIR
    si = pd.read_csv(next(Path(LOCAL_DATASETS_DIR, "butqdb").glob("*/subject-info.csv")), sep=";")
    si["subject"] = si.ID // 1000
    subj = si.drop_duplicates("subject")
    assert len(subj) == 15 and (si.groupby("subject").Gender.nunique() == 1).all()
    c.append((f"The 15 subjects ({(subj.Gender == 'F').sum()} women and {(subj.Gender == 'M').sum()} men) were aged "
              f"{subj.Age.min()}–{subj.Age.max()} years (median {subj.Age.median():.0f})", MAIN))
    # ---- quota sensitivity analysis (round V9; experiments/quota_sensitivity.py, ANALYSIS_PLAN_2026-11-16_quota_sensitivity.md)
    Q = R.parent / f"{R.name}_quota_sensitivity"
    qa = pd.read_csv(Q / "quota_sensitivity_auroc.csv").set_index(["set", "model"])
    qr = pd.read_csv(Q / "quota_sensitivity_rank_agreement.csv").set_index(["pair", "analysis"]).spearman_rho
    qb = pd.read_csv(Q / "paired_vs_best.csv").set_index(["family", "model"])
    qp = pd.read_csv(Q / "paired_comparisons.csv")
    qm = pd.read_csv(Q / "manifest_butqdb_1000_cohort.csv")
    added = pd.read_csv(Q / "quota_sensitivity_added_windows.csv")
    removed = pd.read_csv(Q / "quota_sensitivity_removed_windows.csv")
    assert len(removed) == 0 and (added.subject_id == 105).all() and (added.label == 1).all()
    prim = pd.read_csv(R / "manifest_butqdb_1000_cohort.csv")
    assert len(qm) == len(prim) + len(added) and set(qm.segment_id) == set(prim.segment_id) | set(added.segment_id)
    c.append((f"a sensitivity cohort of {len(qm):,} windows ({int(qm.label.sum())} unusable) that differs from the primary cohort only by "
              f"{len(added)} added unusable windows of subject 105", MAIN))
    QB, Q11, Q17 = "BUT QDB", "CinC 2011 lead II", "CinC 2017"
    sa = lambda s, m: f3(qa.loc[(s, m), "auroc_sensitivity"])  # noqa: E731
    pa = lambda s, m: f3(qa.loc[(s, m), "auroc_primary"])  # noqa: E731
    for s, fam in [(QB, BUT), (Q11, C11), (Q17, C17)]:  # the primary column of the sensitivity table is the primary analysis
        for m in qa.loc[s].index:
            assert abs(qa.loc[(s, m), "auroc_primary"] - auroc(fam, m)) < 1e-12, (s, m)
    d_in = qa.loc[QB].delta
    assert (d_in > 0).all() and qa.loc[QB].auroc_sensitivity.idxmax() == CNN2 and (qb.loc[BUT].best == CNN2).all()
    c.append((f"In-domain AUROC rose for every model, by {f3(d_in.min())}–{f3(d_in.max())}, and the 2D CNN remained first ({sa(QB, CNN2)})", MAIN))
    hq = lambda fam, m: f3(qb.loc[(fam, m), "p_holm_all_pairs"])  # noqa: E731
    sig = qb.loc[BUT].p_holm_all_pairs < 0.05
    assert list(sig[sig].index) == [ZS] and holm_best(BUT, LR) < 0.05 and holm_best(BUT, MLP) < 0.05 and holm_best(BUT, ZS) >= 0.05
    v2c = []
    v2c.append((f"Logistic regression ({sa(QB, LR)}) and the MLP ({sa(QB, MLP)}) were no longer significantly below it (p_Holm = {hq(BUT, LR)} and "
              f"{hq(BUT, MLP)}), whereas zero-shot d1-3B was, again at the 1/n floor ({sa(QB, ZS)}; p_Holm = {hq(BUT, ZS)})", MAIN))
    assert qa.loc[Q11].auroc_sensitivity.idxmax() == PROBE and qa.loc[Q17].auroc_sensitivity.idxmax() == PROBE
    x = qp[(qp.family == C11) & (qp.model_a == CNN2) & (qp.model_b == PROBE)]
    assert len(x) == 1
    x = x.iloc[0]
    v2c.append((f"remained first ({sa(Q11, PROBE)}) and significantly above the 2D CNN ({sa(Q11, CNN2)}; difference {f3(-x.delta)}, 95% CI "
              f"{f3(-x.ci_hi)}–{f3(-x.ci_lo)}; p_Holm = {f3(x.p_holm)})", MAIN))
    h11, h17 = qa.loc[(Q11, HUB)], qa.loc[(Q17, HUB)]
    sig17 = qb.loc[C17].p_holm_all_pairs
    assert set(sig17[sig17 >= 0.05].index) == {GBM, HEUR} and h11.rank_sensitivity == 4 and h17.rank_sensitivity == 4 and h17.rank_primary == 1
    v2c.append((f"its AUROC rose from {pa(Q11, HUB)} to {sa(Q11, HUB)} on CinC 2011 (difference {f3(h11.delta)}, 95% CI {f3(h11.delta_ci_lo)}–"
              f"{f3(h11.delta_ci_hi)}) and fell from {pa(Q17, HUB)} to {sa(Q17, HUB)} on CinC 2017, where the d1-3B probe became first "
              f"({sa(Q17, PROBE)}) and the HuBERT-ECG probe was significantly below it (p_Holm = {hq(C17, HUB)}); LightGBM and the rule "
              f"heuristic were again not significantly different from the leader (p_Holm = {hq(C17, GBM)} and {hq(C17, HEUR)})", MAIN))
    others = qa.loc[[Q11, Q17]].drop(index=HUB, level="model").delta.abs()
    v2c.append((f"No other external AUROC changed by more than {f3(others.max())}.", MAIN))
    rq = lambda pair_, an: qr.loc[(pair_, an)]  # noqa: E731
    p1, p2, p3 = f"{QB} vs {Q11}", f"{QB} vs {Q17}", f"{Q11} vs {Q17}"
    assert abs(rq(p1, "primary") - r11) < 1e-12 and abs(rq(p2, "primary") - r17) < 1e-12 and abs(rq(p3, "primary") - rx) < 1e-12
    assert rq(p2, "primary") < 0 < rq(p2, "sensitivity")
    v2c.append((f"were {rq(p1, 'sensitivity'):.2f} on CinC 2011 and {rq(p2, 'sensitivity'):.2f} on CinC 2017, and between the two external datasets "
              f"{rq(p3, 'sensitivity'):.2f}, against {rq(p1, 'primary'):.2f}, −{-rq(p2, 'primary'):.2f} and {rq(p3, 'primary'):.2f} in the primary analysis", MAIN))
    v2c.append((f"the same probe reached {sa(Q11, HUB)} on CinC 2011 and was fourth on CinC 2017", MAIN))
    v2c.append((f"the correlation with CinC 2017 remained low ({rq(p2, 'sensitivity'):.2f})", MAIN))
    if VARIANT == "srep":  # V15: shortened paragraph; the AUROCs and intervals dropped here stay in Table S49
        c.append((f"logistic regression and the MLP were no longer significantly below it (p_Holm = {hq(BUT, LR)} and {hq(BUT, MLP)}), "
                  f"whereas zero-shot d1-3B was, again at the 1/n floor (p_Holm = {hq(BUT, ZS)})", MAIN))
        # V21: this sentence was dropped for length; the value is in the generated Table 1 (sensitivity columns)
        c.append((f"its AUROC rose from {pa(Q11, HUB)} to {sa(Q11, HUB)} on CinC 2011 and fell from {pa(Q17, HUB)} to {sa(Q17, HUB)} on CinC "
                  f"2017, where the d1-3B probe became first ({sa(Q17, PROBE)}) and the HuBERT-ECG probe was significantly below it "
                  f"(p_Holm = {hq(C17, HUB)})", MAIN))
        c.append((f"were {rq(p1, 'sensitivity'):.2f} on CinC 2011 and {rq(p2, 'sensitivity'):.2f} on CinC 2017, and between the two "
                  f"external datasets {rq(p3, 'sensitivity'):.2f}.", MAIN))
        c.append((f"the correlation with CinC 2017 remained low ({rq(p2, 'sensitivity'):.2f}; Table S49)", MAIN))
    # "No other external AUROC" is checked for both variants
    c.extend(v2c if VARIANT == "v2" else [x_ for x_ in v2c if x_[0].startswith("No other external AUROC")])
    if VARIANT == "srep":
        # V15: p-values added where 'significant' had none nearby (Scientific Reports guidance)
        hb = lambda fam, m: f3(holm_best(fam, m))  # noqa: E731
        assert (vs_best[vs_best.family == C17].set_index("model").best == HUB).all()
        rest17 = [m for m in vs_best[vs_best.family == C17].model if m not in (HUB, PROBE, GBM, HEUR)]
        assert len(rest17) == 7 and all(holm_best(C17, m) < 0.05 for m in rest17)
        assert all(holm_best(C17, m) >= 0.05 for m in (PROBE, GBM, HEUR))
        c.append((f"were not significantly different from it (p_Holm = {hb(C17, PROBE)}, {hb(C17, GBM)} and {hb(C17, HEUR)}), and all "
                  f"others were significantly worse (all p_Holm ≤ {f3(max(holm_best(C17, m) for m in rest17))})", MAIN))
        c.append((f"not significantly different from the best in-domain or on CinC 2017 (p_Holm = {hb(BUT, PROBE)} and {hb(C17, PROBE)})", MAIN))
        assert (vs_best[vs_best.family == C11].set_index("model").best == PROBE).all()
        c.append((f"the one set where the probe was significantly best (p_Holm = {hb(C11, CNN2)})", MAIN))
        rest11 = [m for m in vs_best[vs_best.family == C11].model if m != PROBE]
        assert len(rest11) == 10
        c.append((f"significantly better than all ten other models (all p_Holm ≤ {f3(max(holm_best(C11, m) for m in rest11))})", MAIN))
        # V23: the Discussion gives the CinC 2017 ranks of the two highest in-domain models instead of the test
        ra = pd.read_csv(R / "rank_agreement.csv").set_index("model")
        top2 = ra.sort_values("rank | BUT_QDB (5-fold subject CV)").index[:2].tolist()
        assert top2 == [CNN2, "1D_CNN_RawECG"]
        _r17 = ra["rank | CinC2017 (8,512 recordings)"]
        _ord = {8: "eighth", 10: "tenth"}
        c.append((f"the two highest in-domain models, the 2D and 1D CNNs, were {_ord[int(_r17[CNN2])]} and "
                  f"{_ord[int(_r17['1D_CNN_RawECG'])]} on CinC 2017", MAIN))
        ncx = pd.read_csv(R / "cinc2011_nonconstant_paired.csv")
        ncx = ncx[((ncx.model_a == PROBE) & (ncx.model_b == CNN2)) | ((ncx.model_a == CNN2) & (ncx.model_b == PROBE))]
        assert len(ncx) == 1
        c.append((f"a constant lead II (p_Holm = {hb(C11, CNN2)}; without them {f3(ncx.iloc[0].p_holm)})", MAIN))
        pxx = pd.read_csv(R / "ptbxl_frozen_comparisons.csv").set_index("comparison").p_holm
        sec = [pxx[f"probe vs {m} ({a})"] for m in ("LightGBM_SQI", "HuBERT_ECG_probe") for a in ("max_12", "mean_12")]
        assert max(sec) < 0.001
        c.append(("and LightGBM (0.689; both p_Holm < 0.001; Table S39)", MAIN))
        c.append(("significantly behind two models on the PTB-XL proxy outcome (both p_Holm < 0.001)", MAIN))
        c.append((f"after multiplicity adjustment (all p_Holm ≥ {np.floor(1000 * t8.p_holm.min()) / 1000:.3f}), yet", MAIN))
        # round V23 (PAPER-004): null-centred bootstrap p-values and the 1/n floor for Table 3
        _vn = pd.read_csv(R / "bootstrap_null_centred.csv")
        _vp = _vn[_vn.analysis == "BUT QDB, primary cohort (988 windows)"].set_index(["model_a", "model_b"])
        _vs = _vn[_vn.analysis == "BUT QDB, sensitivity cohort (1,041 windows)"].set_index(["model_a", "model_b"])
        _vl = _vn[_vn.analysis == "Table 3 (LODO)"].set_index(["family", "model_a"])
        assert len(_vp) == 55 and len(_vs) == 55 and len(_vl) == 27
        _hl = [f"{_vp.loc[(m, CNN2), 'p_centred_holm']:.3f}" for m in ("Independent_SQI_MLP", "Logistic_Regression_SQI")]
        assert _hl[0] == _hl[1]
        c.append((f"With the post hoc null-centred p-value neither was (both p_Holm = {_hl[0]})", MAIN))  # V24 wording
        assert (_vp.p_centred_holm >= 0.05).all() and (_vs.p_centred_holm >= 0.05).all()
        c.append(("and no in-domain pair was significant in either cohort (Table S62)", MAIN))
        c.append(("with the post hoc null-centred bootstrap p-value no BUT QDB comparison was significant in either cohort, Table S62, so "
                  "BUT QDB values below 0.05 are not regarded as established)", MAIN))
        c.append(("but not with the post hoc null-centred p-value, with which no BUT QDB comparison was significant (Table S62), so these "
                  "two differences are not regarded as established", MAIN))
        assert _vs.loc[(CNN2, "LiquidAI_d1_3B_Neural"), "p_percentile_holm"] < 0.05
        c.append((f"though not with the null-centred p-value (p_Holm = {_vs.loc[(CNN2, 'LiquidAI_d1_3B_Neural'), 'p_centred_holm']:.3f}; "
                  f"Table S62)", MAIN))
        _p4 = lambda x: f"{x:.4f}" if x < 0.01 else f"{x:.3f}"  # the Table 3 format
        c.append((f"from 0.562 to 0.802 (p_Holm = {_p4(_vl.loc[('CinC2011', HUB), 'p_floor_holm'])})", MAIN))
        c.append((f"(0.922, Δ +0.028, p_Holm = {_p4(_vl.loc[('CinC2017', PROBE), 'p_floor_holm'])})", MAIN))
        assert ((_vl.p_floor_holm < 0.05) == (_vl.p_percentile_holm < 0.05)).all()  # the floor changed no decision
        _min = _vl.p_floor_holm.min()
        assert abs(_min - 9 / 2000) < 1e-12
        c.append((f"so the smallest attainable Holm-adjusted value is {_min:.4f}", MAIN))
        _chg = _vl[(_vl.p_floor_holm < 0.05) != (_vl.p_centred_holm < 0.05)]
        assert sorted(_chg.index) == [("BUT_QDB", "Independent_SQI_MLP"), ("CinC2011", CNN2)]
        assert _chg.loc[("BUT_QDB", "Independent_SQI_MLP"), "p_centred_holm"] >= 0.05 and _chg.loc[("CinC2011", CNN2), "p_centred_holm"] < 0.05
        c.append(("except for the MLP with BUT QDB held out (not significant) and the 2D CNN with CinC 2011 held out (significant; Table S62)", MAIN))
        # round V23 (PAPER-006): rank agreement without the d1-3B image-token probe
        _rq = pd.read_csv(R / "rank_sensitivity_probe.csv")
        _rq = _rq[_rq.model_set == "d1-3B image-token probe excluded"].set_index(["set_a", "set_b", "statistic"])
        _f2 = lambda x: f"{x:.2f}".replace("-", "−")
        _rho = [_rq.loc[(a, b, "spearman_rho"), "estimate"] for a, b in
                (("BUT QDB", "CinC 2011"), ("BUT QDB", "CinC 2017"), ("CinC 2011", "CinC 2017"))]
        c.append((f"the three correlations were {_f2(_rho[0])}, {_f2(_rho[1])} and {_f2(_rho[2])}", MAIN))
        assert _rq.loc[("BUT QDB", "CinC 2011", "spearman_rho"), "leader_a"] == CNN2 == _rq.loc[("BUT QDB", "CinC 2011", "spearman_rho"), "leader_b"]
        _ru = pd.read_csv(R / "rank_uncertainty.csv")
        _ru = _ru[_ru.model_set == "all 11 models"].set_index(["set_a", "set_b", "statistic"])
        assert _ru.loc[("BUT QDB", "CinC 2011", "regret_auroc"), "estimate"] > 0 == _rq.loc[("BUT QDB", "CinC 2011", "regret_auroc"), "estimate"]
        assert abs(_ru.loc[("BUT QDB", "CinC 2017", "regret_auroc"), "estimate"] - _rq.loc[("BUT QDB", "CinC 2017", "regret_auroc"), "estimate"]) < 1e-12
        c.append(("so the CinC 2011 selection loss depended on the probe, unlike that on CinC 2017 (Table S44)", MAIN))
        assert _ru.loc[("BUT QDB", "CinC 2017", "same_leader"), "prop_positive"] == 0 and _ru.loc[("BUT QDB", "CinC 2011", "same_leader"), "prop_positive"] < 0.01
        assert 0.4 < _rq.loc[("BUT QDB", "CinC 2011", "same_leader"), "prop_positive"] < 0.6
        c.append(("and CinC 2011 in almost none, though in about half once the post hoc d1-3B probe was excluded (Table S44)", MAIN))
        # round V24 (review of v23, PAPER-004): the percentile p-value is primary, the null-centred one a sensitivity analysis,
        # and in-domain differences from the best that are significant with only one of them are not regarded as established
        _hp = [_vp.loc[(m, CNN2), "p_percentile_holm"] for m in ("Independent_SQI_MLP", "Logistic_Regression_SQI")]
        assert max(_hp) < 0.05 and f"{_hp[0]:.3f}" == f"{_hp[1]:.3f}"
        c.append((f"With the primary percentile p-value they were significant (both p_Holm = {_hp[0]:.3f})", MAIN))
        c.append(("so we do not regard them as established.", MAIN))
        _best = _vp[(_vp.index.get_level_values(0) == CNN2) | (_vp.index.get_level_values(1) == CNN2)]
        _only = _best[(_best.p_percentile_holm < 0.05) != (_best.p_centred_holm < 0.05)]
        assert sorted(_only.index) == sorted([("Independent_SQI_MLP", CNN2), ("Logistic_Regression_SQI", CNN2)])
        c.append(("with 15 clusters neither is known to control the type I error rate, so in-domain differences that are significant "
                  "with only one of them are not regarded as established", MAIN))
        c.append(("both hold the fitted models fixed and do not include variability from refitting, fold assignment or training seeds", MAIN))
        c.append(("the main text does not regard BUT QDB differences that are significant with only one of them as established", SUPP))
        # PAPER-012: Fig. 3 markers (evaluation/generate_q1_figures.py: Holm over the unfloored p_boot) give the same decisions as
        # the floored Table 3 values, and the legend names the two markers that the null-centred p-value would change
        _ls = pd.read_csv(R / "lodo_summary.csv")
        _ls = _ls[_ls.p_boot.notna()].copy()
        _ls["p_holm"] = np.nan
        from paired_comparisons import holm as _holm  # the name pc is a local table inside checks()
        for _, _g in _ls.groupby("test"):
            _ls.loc[_g.index, "p_holm"] = _holm(_g.p_boot.values)
        _ls = _ls.set_index(["test", "model"])
        assert len(_ls) == 27 and all((_ls.loc[k, "p_holm"] < 0.05) == (_vl.loc[k, "p_floor_holm"] < 0.05) for k in _vl.index)
        assert _ls.loc[("BUT_QDB", "Independent_SQI_MLP"), "p_holm"] < 0.05 <= _ls.loc[("CinC2011", CNN2), "p_holm"]
        c.append(("with the post hoc null-centred p-value the MLP marker for BUT QDB would be open and the 2D CNN marker for CinC 2011 "
                  "filled (Table S62)", MAIN))
        # PAPER-009: the HuBERT-ECG developers' preprint is now a numbered reference (entry checked in main(), assembled file)
        c.append(("which the HuBERT-ECG developers' preprint [@hubertpreprint] implies was not used in its pre-training", MAIN))
        c.append(("(19) After a further review, no analysis was added;", MAIN))
        # PAPER-013: every analysis script and saved output is pinned to the commit of round V23
        c.append((f"Every analysis script and saved analysis output reported here is as in commit {PIN_COMMIT} of that branch; later "
                  "commits changed only the manuscript files, the checks of their wording and the file manifest.", MAIN))
    return c


# round V24 (PAPER-013): the commit named in Code availability. Every tracked file changed since then must be a manuscript file,
# this checker or the file manifest, and no untracked file may sit among the analysis code and outputs.
PIN_COMMIT = "b75120df79d23e50cfaec037b9079deb62ba6e3a"
PIN_ALLOWED = {"evaluation/verify_v2_prose.py", "results/v2_2026-11/submission_manifest.csv"}


def commit_pin() -> int:
    import subprocess
    if not (ROOT / ".git").exists():
        print(f"commit pin check NOT RUN: no Git repository (compare the files with commit {PIN_COMMIT} by hand)")
        return 0
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    changed = [p for p in run("diff", "--name-only", PIN_COMMIT) if p]
    untracked = [p for p in run("ls-files", "--others", "--exclude-standard", "--", "evaluation", "experiments", "features",
                                "models", "results") if p]
    bad = [p for p in changed if not (p.startswith("manuscript/") or p in PIN_ALLOWED)] + untracked
    for p in bad:
        print(f"COMMIT PIN: {p} differs from commit {PIN_COMMIT[:7]} or is untracked")
    print(f"commit pin: {len(changed)} files changed since {PIN_COMMIT[:7]}, {len(bad)} outside the manuscript files, "
          f"this checker and the manifest")
    return len(bad)


def checklist_consistency(assembled_main: str) -> int:
    """Round V25 (review of v24, PAPER-014): the TRIPOD+AI checklist gives the same commits as Code availability, the
    abstract word count of the assembled main text and the last item of Deviations from the analysis plan."""
    tri = (ROOT / "manuscript" / "srep" / "tripod_ai_checklist_srep.md").read_text(encoding="utf-8")
    fails = 0
    analysis = "a21e70b89d2448565624f15be6d6ec98caebdaca"
    for h in (analysis, PIN_COMMIT):
        if h not in tri or h not in assembled_main:
            print(f"CHECKLIST: commit {h[:7]} missing from the checklist or from Code availability")
            fails += 1
    abstract = assembled_main[assembled_main.index("## Abstract") + len("## Abstract"):assembled_main.index("**Keywords")]
    n = len(abstract.split())
    if f"Unstructured abstract ({n} words;" not in tri:
        print(f"CHECKLIST: item 2 does not give the abstract length of {n} words")
        fails += 1
    dev = assembled_main[assembled_main.index("### Deviations from the analysis plan"):]
    dev = dev[:dev.index("\n### ")]
    last = max(int(x) for x in re.findall(r"\((\d{1,2})\) [A-Z]", dev))
    if f"items 11–{last} there" not in tri:
        print(f"CHECKLIST: item 18c does not end at Deviations item {last}")
        fails += 1
    print(f"checklist: commits, abstract length ({n} words) and Deviations items (1–{last}) checked, {fails} failed")
    return fails


# round V28: the Zenodo archive of release srep-submission-2026-11-21, named with the same commit and DOI in the code
# availability statement and in TRIPOD+AI item 18f, and once in the reference list; no DOI placeholder is left
ARCHIVE_COMMIT = "20aa28226867e83c32cdfe1aafb7e1ab4f6703e5"
ARCHIVE_DOI = "10.5281/zenodo.24415747"


def archive_doi() -> int:
    d = ROOT / "manuscript" / VARIANT
    asm = (d / f"main_{VARIANT}.md").read_text(encoding="utf-8")
    tri = (d / f"tripod_ai_checklist_{VARIANT}.md").read_text(encoding="utf-8")
    head = "## Code availability" if VARIANT == "srep" else "## Data and code availability"
    stmt = asm[asm.index(head):]
    stmt = stmt[:stmt.index("\n## ")]
    refs = asm[asm.index("## References"):]
    refs = refs[:refs.index("\n## ")] if "\n## " in refs else refs
    item = [ln for ln in tri.splitlines() if ln.startswith("| 18f |")]
    sentence = f"The repository at commit {ARCHIVE_COMMIT} is archived at https://doi.org/{ARCHIVE_DOI} ["
    entry = ("Kuntalp, M. mkuntalp-ops/d13b-ecg-sqi: Scientific Reports submission, 21 November 2026. *Zenodo* "
             f"https://doi.org/{ARCHIVE_DOI} (2026)." if VARIANT == "srep" else
             f"Kuntalp M. mkuntalp-ops/d13b-ecg-sqi: Scientific Reports submission, 21 November 2026. Zenodo; 2026. doi:{ARCHIVE_DOI}")
    problems = [(stmt.count(sentence) == 1, "the availability statement does not name the archived commit and DOI once"),
                (refs.count(entry) == 1 and refs.count(ARCHIVE_DOI) == 1, "the reference list does not give the archive once"),
                # V29: the srep Data availability statement also names the archive
                (asm.count(ARCHIVE_DOI) == (3 if VARIANT == "srep" else 2),
                 "the archive DOI occurs outside the availability statements and its reference"),
                (len(item) == 1 and ARCHIVE_COMMIT in item[0] and f"https://doi.org/{ARCHIVE_DOI}" in item[0],
                 "TRIPOD+AI item 18f does not give the archived commit and DOI"),
                ("TO BE ADDED" not in asm and "TO BE ADDED" not in tri, "a placeholder is left")]
    fails = 0
    for ok, msg in problems:
        if not ok:
            print("ARCHIVE:", msg)
            fails += 1
    print(f"archive: commit {ARCHIVE_COMMIT[:7]} and DOI {ARCHIVE_DOI} checked in the availability statement, the "
          f"reference list and TRIPOD+AI item 18f, {fails} failed")
    return fails


# round V29 (editorial check of 23 November 2026): the Data availability statement of the manuscript must be the one entered
# in the submission system, word for word
DATA_STATEMENT = (
    "All datasets analysed in this study are publicly available on PhysioNet: the PhysioNet/Computing in Cardiology Challenge "
    "2011 (https://physionet.org/content/challenge-2011/1.0.0/), the Brno University of Technology ECG Quality Database (BUT "
    "QDB, https://doi.org/10.13026/kah4-0w24), the PhysioNet/Computing in Cardiology Challenge 2017 "
    "(https://physionet.org/content/challenge-2017/1.0.0/), the MIT-BIH Noise Stress Test Database "
    "(https://physionet.org/content/nstdb/1.0.0/) and PTB-XL version 1.0.3 (https://physionet.org/content/ptb-xl/1.0.3/). The "
    "saved predictions and label files used in the analyses are available with the code at "
    "https://github.com/mkuntalp-ops/d13b-ecg-sqi and archived at https://doi.org/10.5281/zenodo.24415747.")


def data_statement() -> int:
    asm = (ROOT / "manuscript" / "srep" / "main_srep.md").read_text(encoding="utf-8")
    sec = asm[asm.index("## Data availability") + len("## Data availability"):]
    sec = sec[:sec.index("\n## ")].strip()
    ok = sec == DATA_STATEMENT
    if not ok:
        print("DATA AVAILABILITY: the manuscript statement differs from the one entered in the submission system")
    print(f"data availability: statement compared word for word with the submission-system text, {0 if ok else 1} failed")
    return 0 if ok else 1


def SREP_REQUIRED(main_t: str, supp_t: str) -> list:
    """Statements the Scientific Reports version must keep (it is shortened, so the v2 sentences differ)."""
    return [("results-informed sensitivity analysis and not a pre-specified or independent validation", main_t),
            ("designed with knowledge of the earlier external results", main_t),
            ("receives the eight SQIs as text in its prompt", main_t),
            ("so only G1 applies to it", main_t),
            ("does not establish equivalent performance", main_t),
            ("reported as conditional projections", main_t),
            ("All tests were two-sided", main_t),
            ("a results-informed sensitivity analysis whose definition was committed", supp_t)]


def number_provenance(main_t: str, supp_t: str) -> int:
    """Every number in the srep main prose must occur in the checked v2 manuscript or in the srep supplement (template text,
    and from V23 also the assembled supplement, whose generated tables are written from saved outputs only)."""
    v2 = "".join((ROOT / "manuscript" / "v2" / f).read_text(encoding="utf-8") for f in ("main_v2.md", "supplement_v2.md"))
    rendered = (ROOT / "manuscript" / "srep" / "supplement_srep.md").read_text(encoding="utf-8")
    pool = set(re.findall(r"\d[\d,]*(?:\.\d+)?", v2 + supp_t + rendered))
    main_t = main_t[main_t.index("## Abstract"):]  # the title page (address, ORCID) carries no results
    prose = "\n".join(ln for ln in main_t.splitlines()
                      if not ln.startswith(("|", "![", "<!--", "{{", "#")) and "@" not in ln[:1])
    prose = re.sub(r"\[@[^\]]+\]", "", prose)
    prose = re.sub(r"\b[0-9a-f]{40}\b", "", prose)  # V5: a commit id is an identifier, not a result; its digits
    # would otherwise be "matched" against the same id in the v2 text
    prose = re.sub(r"https://\S+", "", prose)  # V28: a DOI is an identifier as well; V29: so are the dataset web addresses
    bad = sorted({n for n in re.findall(r"\d[\d,]*(?:\.\d+)?", prose) if n.rstrip(",") not in pool})
    for n in bad:
        print("NUMBER NOT IN CHECKED SOURCES:", n)
    print(f"number provenance: {len(set(re.findall(r'\d[\d,]*(?:\.\d+)?', prose)))} distinct numbers, {len(bad)} unmatched")
    return len(bad)


def main() -> int:
    if "--fill" in sys.argv:
        fill()
    main_t, supp_t = MAIN.read_text(encoding="utf-8"), SUPP.read_text(encoding="utf-8")
    if VARIANT == "srep":
        def norm(s):
            return s.replace("Supplementary Tables S", "Tables S").replace("Supplementary Table S", "Table S")
        main_t, supp_t = norm(main_t), norm(supp_t)
    left = re.findall(TOKEN_RE, main_t) + re.findall(TOKEN_RE, supp_t)
    fails = moved = 0
    items = checks()
    for frag, path in items:
        here, other = (main_t, supp_t) if path == MAIN else (supp_t, main_t)
        if frag not in here:
            # Reframing of 2026-11-11: several Results paragraphs moved unchanged to the Supplementary Results, so a
            # fragment is also accepted in the other file; the move is reported, the number is still checked.
            if frag in other:
                moved += 1
                continue
            print("MISSING:", frag)
            fails += 1
    # Wording checks (review of 2026-11-12, PAPER-003/013/014/015): phrases that must not come back, and statements that
    # must stay. They check wording only; the numbers are checked above.
    both = main_t + supp_t
    forbidden = ["pre-stated", "Pre-stated", "indistinguishable", "reported re-weighted", "AP, natural",
                 "none of its three safeguards uses target labels or target performance",
                 "zero-shot d1-3B and the two probes have no tabular input",
                 # third review (2026-11-13)
                 "removes the tuning question", "from what input, matters more", "could not be estimated for the models with saturated",
                 "Not estimable: scores saturated", "constant-lead check and flatline analysis are descriptive and were not tested",
                 "which the public datasets could not support", "brought false rejections close to the target",
                 # V3 (2026-11-13)
                 "mattered more than the model class", "Rankings are reported only where", "not converged (separation)",
                 "did not converge for logistic regression", "CinC 2017 in 0.0%", "arXiv:2605", "Preprint at",
                 "the 74 records at one clipping bound", "have no peer-reviewed version", "an implementation following Pan and Tompkins",
                 "[COMMIT HASH AND ARCHIVE DOI", "increases without bound",
                 # V28: the archive DOI is filled in
                 "TO BE ADDED AT SUBMISSION"]
    required = SREP_REQUIRED(main_t, supp_t) if VARIANT == "srep" else [
                ("results-informed sensitivity analysis and not a pre-specified or independent validation", main_t),
                ("designed with knowledge of the earlier external results", main_t),
                ("receives the eight SQIs as text in its prompt", main_t),
                ("so only G1 applies to it", main_t),
                ("no significant difference was detected among eight models", main_t),
                ("no significant difference was detected among eight approaches in-domain, which does not establish their equivalence", main_t),
                ("reported as conditional projections", main_t),
                ("Post hoc input safeguards", main_t),
                ("a results-informed sensitivity analysis whose definition was committed", supp_t),
                ("are therefore conditional on those resamples", supp_t),
                ("approaches a finite supremum that no finite slope attains", supp_t),
                ("changed by at most 0.002 at either interval bound", supp_t),
                ("correspond to commit a21e70b89d2448565624f15be6d6ec98caebdaca on the branch", main_t),
                ("The constant-score check precedes the separation check", supp_t),
                ("searches on 13 November 2026 found no peer-reviewed version", main_t),
                ("a simplified re-implementation of the Pan–Tompkins detector", main_t),
                ("a simplified re-implementation of the Pan–Tompkins detector (its band-pass filtering", supp_t)]
    for w in forbidden:
        if w in both:
            print("FORBIDDEN WORDING:", w)
            fails += 1
    # V23 (review PAPER-009): the name of the preprint server stays out of the main text; the HuBERT-ECG preprint is cited
    # in full, as a non-peer-reviewed source, in Supplementary Table S51 (from V24 also as a numbered reference, see below).
    if VARIANT == "srep" and ("medRxiv" in main_t or supp_t.count("medRxiv") != 1):
        print("FORBIDDEN WORDING: medRxiv outside the one citation in Supplementary Table S51")
        fails += 1
    # V24 (review of v23, PAPER-009): the HuBERT-ECG preprint is the one preprint in the assembled reference list, in the
    # journal's "Preprint at" format; the TabPFN-3 preprint stays out ("arXiv:2605" above)
    if VARIANT == "srep":
        _asm = (ROOT / "manuscript" / "srep" / "main_srep.md").read_text(encoding="utf-8")
        _entry = ("Coppola, E. et al. HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications. "
                  "Preprint at https://doi.org/10.1101/2024.11.14.24317328 (2026).")
        if _asm.count("Preprint at") != 1 or _asm.count(_entry) != 1 or "medRxiv" in _asm[_asm.index("## Abstract"):]:
            print("REFERENCE LIST: expected exactly one preprint, the HuBERT-ECG preprint, in the assembled main text")
            fails += 1
        fails += commit_pin()
        fails += checklist_consistency(_asm)
        fails += data_statement()
    fails += archive_doi()
    for w, text in required:
        if w not in text:
            print("MISSING WORDING:", w)
            fails += 1
    n_word = len(forbidden) + len(required)
    if VARIANT == "srep":
        fails += number_provenance(main_t, supp_t)
    print(f"{len(items)} prose checks and {n_word} wording checks, {fails} failed ({moved} found in the other file); "
          f"placeholders left: {sorted(set(left))}")
    return 1 if fails or left else 0


if __name__ == "__main__":
    sys.exit(main())
