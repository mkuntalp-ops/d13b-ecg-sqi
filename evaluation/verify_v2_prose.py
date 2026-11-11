"""
Check the numbers typed in the v2 manuscript prose against the saved outputs (results/v2_2026-11).

Each check renders a value computed from a CSV into a sentence fragment and requires that fragment to occur in
manuscript/v2/main_v2_template.md (or supplement_v2_template.md). `--fill` first replaces the placeholder tokens that
depend on results produced late in the queue (latency, fine-tuning, the DeLong/bootstrap implementation check); each token
must occur the expected number of times.

Usage: ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/verify_v2_prose.py [--fill]
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
MAIN = ROOT / "manuscript" / "v2" / "main_v2_template.md"
SUPP = ROOT / "manuscript" / "v2" / "supplement_v2_template.md"
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
                               MLP: "The MLP ({})", LR: "logistic regression ({})", ZS: "Zero-shot d1-3B reached {}"}),
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
    # Abstract
    c += [(f"(0.828 in-domain; {f3(auroc(C11, ZS))} and {f3(auroc(C17, ZS))} externally)", MAIN), (f"highest on CinC 2017 ({f3(auroc(C17, HUB))})", MAIN)]
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
    assert (hc.p_holm < 0.001).all() and len(hc) == 4
    hd = lambda a: f3(abs(float(hc[hc.comparison == f"HuBERT {a} vs d1 image probe {a}"].delta.iloc[0])))  # noqa: E731
    assert f3(h12.auroc["lead_II"]) == "0.562"
    c += [(f"it reached {f3(h12.auroc['max_12'])} with the maximum and {f3(h12.auroc['mean_12'])} with the mean, {hd('max_12')} and {hd('mean_12')} below the d1-3B probe", MAIN),
          (f"Without the records with a constant lead its AUROC was {f3(hk.auroc_without_constant['max_12'])} and {f3(hk.auroc_without_constant['mean_12'])}", MAIN),
          (f"Given the real 12 leads in one input, it reached {f3(h12.auroc['native_12'])}", MAIN),
          (f"({f3(hk.auroc_without_constant['lead_II'])} on lead II without them)", MAIN)]
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
    c.append((f"(0.579 versus {aucf(CNN2, M2)}; adjusted p = {p1.p_holm:.3f}) but was weak and significantly below HuBERT-ECG ({aucf(HUB, M2)})", MAIN))
    assert aucf(PROBE, M2) == "0.579"
    c.append((f"(AUROC {aucf(PROBE, M2)}–{aucf(PROBE, ME)})", MAIN))
    # 12-lead value in the abstract
    tw = pd.read_csv(R / "d1_probe_twelve_lead_summary.csv").set_index("aggregation")
    c.append((f"; {tw.auroc['max_12']:.3f} post hoc with 12 leads)", MAIN))
    # ---- additions after the code audit of 2026-11-11 (evaluation/audit_sensitivity.py)
    # bootstrap p floor: MLP and logistic regression versus the best model on BUT QDB
    hb = [holm_best(BUT, m) for m in (MLP, LR)]
    assert hb[0] == hb[1] and pair(BUT, CNN2, MLP).p == pair(BUT, CNN2, LR).p == 1 / 2000
    c.append((f"(both p_Holm = {f3(hb[0])}; no resample reversed the sign, so the unadjusted p is at its floor of 0.0005)", MAIN))
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
    c.append((f"but not significantly without 73 flat-lead records ({aw(PROBE)} versus {aw(CNN2)}; p = {f3(x.p)})", MAIN))
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
    return c


def main() -> int:
    if "--fill" in sys.argv:
        fill()
    main_t, supp_t = MAIN.read_text(encoding="utf-8"), SUPP.read_text(encoding="utf-8")
    left = re.findall(TOKEN_RE, main_t) + re.findall(TOKEN_RE, supp_t)
    fails = 0
    items = checks()
    for frag, path in items:
        if frag not in (main_t if path == MAIN else supp_t):
            print("MISSING:", frag)
            fails += 1
    print(f"{len(items)} prose checks, {fails} failed; placeholders left: {sorted(set(left))}")
    return 1 if fails or left else 0


if __name__ == "__main__":
    sys.exit(main())
