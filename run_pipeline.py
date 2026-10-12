"""
Pipeline orchestrator for "Zero-shot and task-specific models for ECG usability screening".

Runs the current stages in the order they depend on each other. Superseded v1 and pilot-era
scripts are in experiments/_superseded/ and are not used here.

Usage:
    python run_pipeline.py --stage all          # every stage, in dependency order (long: hours)
    python run_pipeline.py --stage sqi          # recompute SQI features (units, bSQI matching)
    python run_pipeline.py --stage supervised   # LR, LightGBM, MLP, 1D/2D CNN, multimodal (5-fold CV)
    python run_pipeline.py --stage d1           # zero-shot LiquidAI/d1-3B on BUT QDB and CinC (needs weights)
    python run_pipeline.py --stage consolidate  # merge OOF predictions and compute BUT QDB metrics
    python run_pipeline.py --stage cinc         # transfer to CinC 2011 Set A (needs d1 CinC predictions)
    python run_pipeline.py --stage nstdb        # NSTDB noise stress (needs d1 weights)
    python run_pipeline.py --stage scaling      # Table 2c scaler / feature-subset sensitivity
    python run_pipeline.py --stage seeds        # seed stability, seeds 1-4 (long)
    python run_pipeline.py --stage latency      # latency and parameter profiling (Table 4)
    python run_pipeline.py --stage figures      # Figures 1-4 from the result files
    python run_pipeline.py --stage tables       # all numeric manuscript tables -> results/manuscript_tables_*.md

Data preparation (cohort and segment caches) is done by data/extract_butqdb_1000_cohort.py and the
CinC/NSTDB loaders; this orchestrator assumes those caches exist.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STAGES = [
    ("sqi", "experiments/recompute_sqi_harmonized.py"),
    ("supervised", "experiments/run_supervised_cohort_cv.py"),
    ("d1", "experiments/run_d1_neural_inference.py"),
    ("consolidate", "experiments/compute_consolidated_metrics.py"),
    ("cinc", "experiments/run_cinc_transfer_benchmark.py"),
    ("nstdb", "experiments/exp4_revised_nstdb.py"),
    ("scaling", "experiments/scaling_sensitivity_cinc.py"),
    ("seeds", "experiments/seed_stability.py"),
    ("latency", "experiments/empirical_latency_benchmark.py"),
    ("figures", "evaluation/generate_publication_figures.py"),
    ("tables", "evaluation/build_manuscript_tables.py"),
]


def run_stage(name: str, script: str) -> None:
    print(f"\n[{name}] {script}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, "-u", str(PROJECT_ROOT / script)], cwd=PROJECT_ROOT, check=True)
    print(f"[{name}] finished in {time.time() - t0:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ECG usability assessment pipeline orchestrator")
    parser.add_argument("--stage", required=True, choices=[s for s, _ in STAGES] + ["all"],
                        help="Stage to run, or 'all' for every stage in dependency order")
    args = parser.parse_args()
    t_start = time.time()
    for name, script in STAGES:
        if args.stage in (name, "all"):
            run_stage(name, script)
    print(f"\nDone in {time.time() - t_start:.0f} s. Results: {PROJECT_ROOT / 'results'}")


if __name__ == "__main__":
    main()
