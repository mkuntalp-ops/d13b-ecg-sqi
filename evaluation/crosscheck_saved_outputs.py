"""
Read-only cross-check of reported numbers against the saved outputs (review of 2026-11-12, PAPER-001/002/009).

It imports no project module: it reads the saved predictions and raw timings and recomputes
  - the 33 AUROCs of Table 1 (11 models on BUT QDB, CinC 2011 lead II and CinC 2017 with the recording maximum),
  - the original and harmonised AUROCs of Table S46 (applying the constant-input rule G1 itself), the Spearman
    correlations and seed ranges of Table S47,
  - the medians and 95th percentiles of Table S45 from the three raw latency runs,
and compares them with the stored summaries and with the text of the assembled manuscript. It fits no model and
measures no latency, so it shows that the tables follow from the saved outputs, not that the outputs can be regenerated.
A second calculation by the same author is a cross-check, not an independent review.

Run:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/crosscheck_saved_outputs.py
Exit code 0 if every comparison holds; an AssertionError otherwise. Writes nothing.
"""
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent.parent
R = Path(os.environ.get("ECG_RESULTS_DIR", ROOT / "results" / "v2_2026-11"))
R = R if R.is_absolute() else ROOT / R
SUPP = (ROOT / "manuscript/v2/supplement_v2.md").read_text(); MAIN = (ROOT / "manuscript/v2/main_v2.md").read_text()
M7 = ["Logistic_Regression_SQI", "LightGBM_SQI", "Independent_SQI_MLP", "1D_CNN_RawECG", "2D_CNN_Image", "Multimodal_Fusion", "Rule_Based_SQI_Heuristic"]
ZS, PR, TP, HB = "LiquidAI_d1_3B_Neural", "d1_image_probe", "TabPFN_SQI", "HuBERT_ECG_probe"

# ---------- original scores, one frame per set: index = unit id, columns = label + 11 models
coh = pd.read_csv(R / "cohort1000_consolidated_oof_predictions.csv").set_index("segment_id")
ext = pd.read_csv(R / "external_window_predictions.csv")
tab = pd.read_csv(R / "tabpfn_predictions.csv"); hub = pd.read_csv(R / "hubert_probe_predictions.csv")
def by_seg(df): assert not df.segment_id.duplicated().any(); return df.set_index("segment_id").prob_unusable
but = pd.DataFrame({"label": coh.label})
for m in M7 + [ZS]: but[m] = coh["prob_" + m]
but[TP] = by_seg(tab[tab.setting == "butqdb_cv"]).reindex(but.index)
but[HB] = by_seg(hub[hub.setting == "butqdb_cv"]).reindex(but.index)
po = pd.read_csv(R / "d1_probe_oof_predictions.csv"); but[PR] = by_seg(po[po.features == "image"]).reindex(but.index)

e11 = ext[ext.dataset == "CinC2011"].set_index("segment_id")
c11 = pd.DataFrame({"label": e11.label, "record": e11.record_name.astype(str)})
for m in M7: c11[m] = e11[m]
c11[TP] = by_seg(tab[(tab.setting == "but_only_external") & (tab.dataset == "CinC2011")]).reindex(c11.index)
c11[HB] = by_seg(hub[(hub.setting == "but_only_external") & (hub.dataset == "CinC2011")]).reindex(c11.index)
d1e = pd.read_csv(R / "d1_external_predictions.csv"); z = d1e[d1e.dataset == "CinC2011_leadII"]
c11[ZS] = z.assign(r=z.record_name.astype(str)).set_index("r").prob_unusable.reindex(c11.record).values
pt = pd.read_csv(R / "d1_probe_transfer_predictions.csv"); p = pt[(pt.features == "image") & (pt.dataset == "CinC2011")]
c11[PR] = p.assign(r=p.record_name.astype(str)).set_index("r").prob_unusable.reindex(c11.record).values

e17 = ext[ext.dataset == "CinC2017"].set_index("segment_id")
w17 = pd.DataFrame({"label": e17.label, "record": e17.record_name.astype(str)})
for m in M7: w17[m] = e17[m]
w17[TP] = by_seg(tab[(tab.setting == "but_only_external") & (tab.dataset == "CinC2017")]).reindex(w17.index)
w17[HB] = by_seg(hub[(hub.setting == "but_only_external") & (hub.dataset == "CinC2017")]).reindex(w17.index)
d17 = pd.read_csv(R / "d1_cinc2017_full_predictions.csv").set_index("segment_id")
w17[ZS] = d17.prob_unusable.reindex(w17.index); w17[PR] = d17.prob_image_probe.reindex(w17.index)
ALL = M7 + [ZS, TP, HB, PR]
for name, df in (("BUT", but), ("C11", c11), ("C17w", w17)):
    assert not df[ALL + ["label"]].isna().any().any(), name
recmax = lambda w: w.groupby("record").agg({**{m: "max" for m in ALL}, "label": "first"})
assert (w17.groupby("record").label.nunique() == 1).all()
c17 = recmax(w17)
print("units/positives:", len(but), int(but.label.sum()), len(c11), int(c11.label.sum()), len(c17), int(c17.label.sum()))
auc = lambda df: {m: roc_auc_score(df.label.astype(int), df[m]) for m in ALL}
orig = {"BUT_QDB": auc(but), "CinC2011": auc(c11), "CinC2017": auc(c17)}

# ---------- PAPER-001: Table 1 / model_cis.csv
ci = pd.read_csv(R / "model_cis.csv")
fam = {"BUT_QDB": "BUT_QDB (5-fold", "CinC2011": "CinC2011 lead II", "CinC2017": "CinC2017 (8"}
worst = 0
for ds, pref in fam.items():
    g = ci[ci.family.str.startswith(pref)].set_index("model"); assert len(g) == 11
    for m in ALL:
        worst = max(worst, abs(g.auroc[m] - orig[ds][m]))
        assert f"{orig[ds][m]:.3f}" in MAIN, (ds, m)
print(f"PAPER-001  33 AUROCs recomputed vs model_cis.csv: max |diff| = {worst:.2e}; all 33 three-decimal values occur in main_v2.md")

# ---------- PAPER-002: harmonised
hp = pd.concat([pd.read_csv(R / "harmonised_predictions.csv"), pd.read_csv(R / "harmonised_tabpfn_predictions.csv")])
h42 = hp[hp.seed == 42]
# Constant windows: the extractor gives a constant window a flatline fraction of exactly 1 (main text, section 2.2).
_ff = pd.read_csv(R / "flatline_robust_features.csv"); _ff = _ff[_ff.dataset == "CinC2011"].set_index("segment_id")
const11 = _ff.flatline_fraction.reindex(c11.index) == 1
_cache = ROOT / "cache/datasets/cinc2011/cinc2011_setA_leadII_full.csv"  # not in the repository; used only if present
if _cache.exists():
    assert ((pd.read_csv(_cache).set_index("segment_id").signal_std.reindex(c11.index) == 0) == const11).all()
    print("constant flags agree with signal SD = 0 in the local feature table")
print("constant CinC 2011 lead II windows:", int(const11.sum()), "all unacceptable:", bool((c11.label[const11] == 1).all()))
REF = sorted(h42.model.unique()); print("refitted models:", REF)
harm = {}
for ds, base in (("BUT_QDB", but), ("CinC2011", c11), ("CinC2017", w17)):
    df = base.copy()
    for m in REF:
        s = h42[(h42.dataset == ds) & (h42.model == m)]; assert not s.segment_id.duplicated().any() and len(s) == len(df)
        df[m] = s.set_index("segment_id").prob_unusable.reindex(df.index).values
    assert not df[ALL].isna().any().any()
    if ds == "CinC2011":
        for m in ALL: df.loc[const11.values, m] = 1.0          # G1
    if ds == "CinC2017": df = recmax(df)
    harm[ds] = auc(df)
hs = pd.read_csv(R / "harmonised_summary.csv").set_index(["dataset", "model"])
wo = max(abs(hs.auroc_original[(d, m)] - orig[d][m]) for d in orig for m in ALL)
wh = max(abs(hs.auroc_harmonised[(d, m)] - harm[d][m]) for d in harm for m in ALL)
print(f"PAPER-002  harmonised_summary.csv: original max |diff| = {wo:.2e}; harmonised max |diff| = {wh:.2e}")
ra = pd.read_csv(R / "harmonised_rank_agreement.csv")
for r in ra.itertuples():
    so = spearmanr([orig[r.set_a][m] for m in ALL], [orig[r.set_b][m] for m in ALL])[0]
    sh = spearmanr([harm[r.set_a][m] for m in ALL], [harm[r.set_b][m] for m in ALL])[0]
    print(f"   Spearman {r.set_a}-{r.set_b}: original {so:+.4f} (stored {r.spearman_original:+.4f}); harmonised {sh:+.4f} (stored {r.spearman_harmonised:+.4f})")
    assert abs(so - r.spearman_original) < 1e-9 and abs(sh - r.spearman_harmonised) < 1e-9
    assert f"{so:+.2f}" in SUPP and f"{sh:+.2f}" in SUPP
ten = [m for m in ALL if m != ZS]
print("   CinC 2011 range, ten models: original %.3f-%.3f, harmonised %.3f-%.3f" % (min(orig["CinC2011"][m] for m in ten), max(orig["CinC2011"][m] for m in ten), min(harm["CinC2011"][m] for m in ten), max(harm["CinC2011"][m] for m in ten)))
print("   max |harmonised - original|: BUT %.3f, CinC2017 %.3f" % (max(abs(harm["BUT_QDB"][m] - orig["BUT_QDB"][m]) for m in ALL), max(abs(harm["CinC2017"][m] - orig["CinC2017"][m]) for m in ALL)))
# seed ranges
for m in ("Independent_SQI_MLP", "Multimodal_Fusion"):
    vals = []
    for sd in (42, 1, 2, 3, 4):
        s = hp[(hp.dataset == "CinC2011") & (hp.model == m) & (hp.seed == sd)].set_index("segment_id").prob_unusable.reindex(c11.index).copy()
        s[const11.values] = 1.0; vals.append(roc_auc_score(c11.label, s))
    print(f"   {m} CinC 2011 seeds: {min(vals):.3f}-{max(vals):.3f}")

# ---------- PAPER-009: latency from raw run files
runs = [pd.read_csv(p) for p in sorted((R / "latency_repeats").glob("run*_raw.csv"))]; assert len(runs) == 3
tbl = SUPP[SUPP.index("## Table S45"):SUPP.index("## Table S46")]
bad = 0
for m in runs[0].model.unique():
    assert all(len(r[r.model == m]) == 25 for r in runs)
    st = {c: (np.median([np.median(r[r.model == m][c]) for r in runs]), np.median([np.percentile(r[r.model == m][c], 95) for r in runs])) for c in ("pre_ms", "fwd_ms", "total_ms")}
    cells = [f"{st[c][0]:.2f} ms ({st[c][1]:.2f} ms)" for c in ("pre_ms", "fwd_ms", "total_ms")]
    ok = all(c in tbl for c in cells); bad += not ok
    print(f"   {m:28s} pre/fwd/total median (p95): {' | '.join(cells)}  in Table S45: {ok}")
    assert all(np.allclose(r[r.model == m].pre_ms + r[r.model == m].fwd_ms, r[r.model == m].total_ms) for r in runs)
print("PAPER-009  rows not matching Table S45:", bad)
assert bad == 0 and worst < 1e-9 and wo < 1e-9 and wh < 1e-9
print("ALL CROSS-CHECKS PASSED")
