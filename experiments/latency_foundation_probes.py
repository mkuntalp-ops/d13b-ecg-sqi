"""
Latency of the foundation-model probes (Q1 revision, 2026-10-26). Run alone (no other MPS or CPU-heavy job).

Same 25 BUT QDB windows and three runs as experiments/latency_extended.py (first window of each run is a warm-up and
excluded). Measured per window:
- HuBERT-ECG probe: preprocessing (authors' pipeline, experiments/hubert_ecg_probe.model_input), forward pass of the
  two 5-s halves on CPU and on MPS (synchronised), and the linear probe (scaler + logistic regression, fitted on all
  BUT QDB HuBERT features);
- d1-3B image-token probe: only the linear probe step on a stored 2048-d feature. Its forward pass is the zero-shot
  d1-3B forward pass of Table 4, which already captures the image tokens, so the probe adds only this step.
TabPFN is timed by experiments/latency_tabpfn.py in its own environment.

Output: results/latency_probes_raw.csv, results/latency_probes_summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.argv = [a for a in sys.argv if not a.startswith("--preproc")]

import hubert_ecg_probe as hp  # noqa: E402
from latency_extended import N_RUNS, select_windows, timed  # noqa: E402

RESULTS = ROOT / "results"


def main():
    assert hp.PREPROC == "authors"
    sel = select_windows()
    man = pd.read_csv(RESULTS / "manifest_butqdb_1000_cohort.csv")
    pos = pd.Series(np.arange(len(man)), index=man.segment_id).loc[sel.segment_id].values
    Hb, Db = np.load(ROOT / "cache" / "hubert_ecg_authors" / "BUT_QDB.npy"), np.load(ROOT / "cache" / "d1_probe" / "BUT_QDB_image.npy")
    y, g = man.label.values, man.subject_id.values
    hub_probe = hp.probe().fit(Hb, y, groups=g).best_estimator_
    d1_probe = hp.probe().fit(Db, y, groups=g).best_estimator_
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    models = {dev: hp.load_model(dev) for dev in devices}
    signals = [np.load(p) for p in sel.signal_path]
    rows = []
    for run in range(1, N_RUNS + 1):
        for i, (x, k) in enumerate(zip(signals, pos)):
            t_pre, inp = timed(lambda: hp.model_input(x))
            rec = {"run": run, "window": i, "warmup": i == 0, "hubert_preproc_ms": t_pre}
            for dev, m in models.items():
                with torch.no_grad():
                    t_fwd, h = timed(lambda: m(torch.from_numpy(inp).to(dev)).last_hidden_state.mean(dim=1).mean(dim=0)
                                     .float().cpu().numpy(), dev)
                rec[f"hubert_forward_{dev}_ms"] = t_fwd
            rec["hubert_linear_probe_ms"] = timed(lambda: hub_probe.predict_proba(h.reshape(1, -1)))[0]
            rec["d1_linear_probe_ms"] = timed(lambda: d1_probe.predict_proba(Db[k:k + 1]))[0]
            for dev in devices:
                rec[f"hubert_total_{dev}_ms"] = (rec["hubert_preproc_ms"] + rec[f"hubert_forward_{dev}_ms"]
                                                 + rec["hubert_linear_probe_ms"])
            rows.append(rec)
        print(f"[latency_probes] run {run} done", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "latency_probes_raw.csv", index=False)
    meas = [c for c in raw.columns if c.endswith("_ms")]
    per_run = raw[~raw.warmup].groupby("run")[meas].median()
    summ = pd.DataFrame({"median_over_runs_ms": per_run.median(), "min_run_ms": per_run.min(), "max_run_ms": per_run.max(),
                         "p95_median_over_runs_ms": raw[~raw.warmup].groupby("run")[meas].quantile(0.95).median()})
    summ.index.name = "measurement"
    summ.to_csv(RESULTS / "latency_probes_summary.csv")
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
