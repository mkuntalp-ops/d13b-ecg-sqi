"""
Figures of the Q1 manuscript (2026-10-26), generated from saved outputs only.

Figure 1: AUROC of every model in each evaluation set, marked by the Holm-adjusted paired comparison with the best
          model of that set (results/paired_comparisons.csv).
Figure 2: Leave-one-dataset-out effect (multi-source minus BUT-only AUROC) on the two external datasets: seed-42
          paired difference (results/lodo_summary.csv; Holm within each held-out dataset) and, for the
          seed-dependent networks, the range over five seeds (results/seed_repeats_consistency.csv).
Figure 3: Local recalibration: median false reject rate and sensitivity of thresholds re-set on n labelled target
          recordings, every model (thin lines) and the median over models (results/local_recalibration_summary.csv).

Colour carries one job per figure (the comparison outcome); model identity is carried by row labels or, in Figure 3,
is not encoded (per-model values are in Tables S23-S24). Palette validated with the dataviz validator (2 slots).
Outputs: figures/q1_figure{1,2,3}.png (300 dpi) and .pdf
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_comparisons import holm  # noqa: E402

RESULTS, FIG = ROOT / "results", ROOT / "figures"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
ORDER = [("Logistic_Regression_SQI", "Logistic regression (SQI)"), ("LightGBM_SQI", "LightGBM (SQI)"),
         ("Independent_SQI_MLP", "MLP (SQI)"), ("TabPFN_SQI", "TabPFN (SQI)"), ("1D_CNN_RawECG", "1D CNN"),
         ("2D_CNN_Image", "2D CNN"), ("Multimodal_Fusion", "Multimodal net"), ("Rule_Based_SQI_Heuristic", "Rule heuristic"),
         ("LiquidAI_d1_3B_Neural", "d1-3B zero-shot"), ("d1_image_probe", "d1-3B image probe"),
         ("HuBERT_ECG_probe", "HuBERT-ECG probe")]
FAMILY_BREAKS = [4, 7, 8]  # separators after the tabular, network and heuristic groups

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "axes.titlesize": 9, "axes.titlecolor": INK,
                     "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight"})


def _rows(ax, n, breaks=FAMILY_BREAKS):
    ax.set_ylim(n - 0.5, -0.5)
    for b in breaks:
        ax.axhline(b - 0.5, color=GRID, lw=0.8, zorder=0)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"[Saved] figures/{name}.png/.pdf")


def figure1():
    c = pd.read_csv(RESULTS / "paired_comparisons.csv")
    fams = [("BUT_QDB (5-fold subject CV)", "BUT QDB, subject CV\n(1,047 windows)"),
            ("CinC2011 lead II (565 records)", "CinC 2011, lead II\n(565 records)"),
            ("CinC2017 (8,512 recordings)", "CinC 2017\n(8,512 recordings)"),
            ("CinC2017 balanced subset (556 recordings)", "CinC 2017 subset\n(556 recordings)")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.6), sharey=True)
    for ax, (fam, title) in zip(axes, fams):
        g = c[c.family == fam]
        auc = pd.concat([g[["model_a", "auroc_a"]].set_axis(["m", "a"], axis=1),
                         g[["model_b", "auroc_b"]].set_axis(["m", "a"], axis=1)]).drop_duplicates("m").set_index("m").a
        best = auc.idxmax()
        for i, (m, _) in enumerate(ORDER):
            if m not in auc.index:
                continue
            if m == best:
                ax.scatter(auc[m], i, s=46, marker="D", color=BLUE, edgecolor="white", lw=1.2, zorder=3)
                continue
            r = g[((g.model_a == m) & (g.model_b == best)) | ((g.model_b == m) & (g.model_a == best))].iloc[0]
            ns = r.p_holm >= 0.05
            ax.scatter(auc[m], i, s=34, marker="o", color=BLUE if ns else "white", edgecolor=BLUE if ns else MUTED,
                       lw=1.2, zorder=3)
        _rows(ax, len(ORDER))
        ax.set_xlim(0.28, 1.02)
        ax.set_xticks([0.3, 0.5, 0.7, 0.9])
        ax.axvline(0.5, color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("AUROC")
    axes[0].set_yticks(range(len(ORDER)), [lab for _, lab in ORDER])
    handles = [Line2D([], [], marker="D", ls="", color=BLUE, markeredgecolor="white", markersize=7, label="Best in the set"),
               Line2D([], [], marker="o", ls="", color=BLUE, markersize=6, label="Not significantly different from best"),
               Line2D([], [], marker="o", ls="", color="white", markeredgecolor=MUTED, markersize=6,
                      label="Significantly worse than best (Holm p < 0.05)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.12), labelcolor=INK2)
    save(fig, "q1_figure1")


def figure2():
    s = pd.read_csv(RESULTS / "lodo_summary.csv")
    s["p_holm"] = np.nan
    for _, g in s[s.p_boot.notna()].groupby("test"):
        s.loc[g.index, "p_holm"] = holm(g.p_boot.values)
    seeds = pd.read_csv(RESULTS / "seed_repeats_consistency.csv")
    seeds = seeds[seeds.seed_varied]
    order = [(m, lab) for m, lab in ORDER if m in set(s.model) and m not in ("Rule_Based_SQI_Heuristic",)]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), sharey=True)
    for ax, (test, title) in zip(axes, [("CinC2011", "Held out: CinC 2011\n(train: BUT QDB + CinC 2017)"),
                                        ("CinC2017", "Held out: CinC 2017\n(train: BUT QDB + CinC 2011)")]):
        for i, (m, _) in enumerate(order):
            r = s[(s.test == test) & (s.model == m)]
            if r.empty or pd.isna(r.delta.iloc[0]):
                continue
            r = r.iloc[0]
            sr = seeds[(seeds.test == test) & (seeds.model == m)]
            if len(sr):
                ax.plot([sr.delta_min.iloc[0], sr.delta_max.iloc[0]], [i, i], color=BLUE, lw=2, alpha=0.35,
                        solid_capstyle="round", zorder=2)
            sig = r.p_holm < 0.05
            ax.scatter(r.delta, i, s=34, color=BLUE if sig else "white", edgecolor=BLUE if sig else MUTED, lw=1.2, zorder=3)
        _rows(ax, len(order), breaks=[4, 7])  # tabular | networks | probes
        ax.axvline(0, color=INK2, lw=0.9, zorder=1)
        ax.set_xlim(-0.40, 0.30)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Δ AUROC, multi-source − BUT-only")
    axes[0].set_yticks(range(len(order)), [lab for _, lab in order])
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, markersize=6, label="Seed 42: Holm p < 0.05"),
               Line2D([], [], marker="o", ls="", color="white", markeredgecolor=MUTED, markersize=6, label="Seed 42: not significant"),
               Line2D([], [], color=BLUE, lw=2, alpha=0.35, label="Range over five seeds (networks)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.13), labelcolor=INK2)
    save(fig, "q1_figure2")


def figure3():
    s = pd.read_csv(RESULTS / "local_recalibration_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True)
    for row, (ds, name) in enumerate([("CinC2011", "CinC 2011 (34.9% unacceptable)"), ("CinC2017", "CinC 2017 (3.3% noisy)")]):
        g = s[s.dataset == ds]
        for col, (k, lab, top) in enumerate([("fpr_local_median", "False reject rate (median)", 0.12),
                                             ("sens_local_median", "Sensitivity (median)", 1.0)]):
            ax = axes[row, col]
            for _, gm in g.groupby("model"):
                ax.plot(gm.n, gm[k], color=MUTED, lw=0.8, alpha=0.6, zorder=2)
            med = g.groupby("n")[k].median()
            ax.plot(med.index, med.values, color=BLUE, lw=2, marker="o", markersize=5, zorder=3)
            if k.startswith("fpr"):
                ax.axhline(0.05, color=ORANGE, lw=1, ls=(0, (3, 2)), zorder=1)
                ax.text(205, 0.05, "target 5%", color=INK2, va="center", fontsize=7.5)
            ax.set_ylim(0, top)
            ax.set_xscale("log")
            ax.set_xticks([25, 50, 100, 200], ["25", "50", "100", "200"])
            ax.minorticks_off()
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
            ax.grid(color=GRID, lw=0.6, zorder=0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_title(f"{name}: {lab.split(' (')[0].lower()}", loc="left")
            if row == 1:
                ax.set_xlabel("Labelled target recordings used to set the threshold (n)")
            ax.set_ylabel(lab)
    handles = [Line2D([], [], color=BLUE, lw=2, marker="o", markersize=5, label="Median over the 11 models"),
               Line2D([], [], color=MUTED, lw=0.8, label="Individual models"),
               Line2D([], [], color=ORANGE, lw=1, ls=(0, (3, 2)), label="Target false reject rate")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05), labelcolor=INK2)
    fig.tight_layout()
    save(fig, "q1_figure3")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
