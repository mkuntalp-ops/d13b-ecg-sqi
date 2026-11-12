# Cross-Dataset ECG Usability Assessment with a Single-Pass Multimodal Decision Model

### Integrating Waveform Images and Interpretable Signal-Quality Indicators

---

## 1. Research Question & Study Rationale

Rather than treating ECG quality control as a naive binary noise detector, this study investigates:

> **"Can a lightweight single-pass multimodal decision model reliably classify ECG recording usability across heterogeneous clinical and wearable databases by fusing standardized visual waveform images with domain-specific signal quality indicators (SQIs)?"**

### Core Clinical Output:
- **Usable ($\mathcal{C}_0$)**: Suitable for automated QRS detection and rhythm interpretation.
- **Unusable ($\mathcal{C}_1$, Target Class)**: Unacceptable due to severe motion artifacts, baseline drift, saturation, or absent complexes; requires automated re-recording prompt or expert clinical review.

---

## 2. Experimental Databases (PhysioNet Open Access)

| Database | Modality & Size | Target Usability Role | Access / License |
|---|---|---|---|
| **BUT QDB** | 18 recordings (15 subjects, 1000 Hz) with 3 quality classes by consensus | **Primary Development & Leakage-Free Subject Cross-Validation** (GroupKFold) | CC BY 4.0 (Open access) |
| **CinC 2011 Set A** | 1,000 10-s 12-lead ECGs (500 Hz) with reference acceptability labels | **Independent Cross-Dataset Transfer Benchmark** (Distribution Shift) | ODC Attribution 1.0 |
| **MIT-BIH NSTDB** | Calibrated noise stress ECGs at 24, 18, 12, 6, 0, and -6 dB SNR | **Controlled Noise Degradation & Inflection Analysis** | ODC Attribution 1.0 |

---

## 3. Multimodal Decision Design & Modality Ablations

Every ECG segment (10-second window) is simultaneously evaluated across three distinct input modes:
1. **Image Only**: 10-s ECG strip (513 x 257 px, clinical-style grid, vertical axis scaled per strip; not a fixed 10 mm/mV gain).
2. **Numerical SQIs Only**: Structured domain-specific signal quality indicators:
   - $b\text{SQI}$: Inter-detector agreement between Pan-Tompkins and Difference filter R-peaks.
   - Baseline wander spectral power ratio ($< 0.5$ Hz vs $0.5 - 40$ Hz).
   - High-frequency EMG noise ratio ($> 40$ Hz).
   - Saturation and Flatline fractions (ADC clipping and disconnection detection).
   - QRS band energy concentration ($5 - 15$ Hz).
   - Higher-order statistical moments: Kurtosis ($k\text{SQI}$) and Skewness ($s\text{SQI}$).
3. **Multimodal (Image + SQIs)**: Gated fusion of visual representations and numerical SQIs.

---

## 4. Evaluation Metrics & Standards

- **Positive Class**: Unusable ($\mathcal{C}_1$).
- **Primary Clinical Metric**: Unusable Sensitivity (recall of bad recordings) and Usable False Reject Rate ($1 - \text{Specificity}$).
- **Statistical Discrimination**: Balanced Accuracy, Macro F1, AUROC, AUPRC.
- **Calibration**: Brier Score and Expected Calibration Error (ECE).
- **Uncertainty Quantification**: 95% Confidence Intervals computed via Subject-Level Non-Parametric Bootstrap (clustering by patient to prevent within-subject optimism).

---

## 5. Execution Guide

Stages run in dependency order through `run_pipeline.py`. Run one stage, or `all` (several hours; the CNN and seed stages dominate).

```bash
# Activate virtual environment
source .venv/bin/activate

# Recompute SQI features on harmonized units (mV) with one-to-one bSQI matching
python run_pipeline.py --stage sqi

# Patient-independent 5-fold CV on BUT QDB (LR, LightGBM, MLP, 1D/2D CNN, multimodal)
python run_pipeline.py --stage supervised

# Zero-shot LiquidAI/d1-3B on BUT QDB and CinC (needs the local weights; see models/d1_real_neural_model.py)
python run_pipeline.py --stage d1

# Merge OOF predictions and compute the BUT QDB metrics
python run_pipeline.py --stage consolidate

# Transfer to CinC 2011 Set A, NSTDB stress, scaler sensitivity, seed stability, latency
python run_pipeline.py --stage cinc
python run_pipeline.py --stage nstdb
python run_pipeline.py --stage scaling
python run_pipeline.py --stage seeds
python run_pipeline.py --stage latency

# Figures 1-4 and all numeric manuscript tables
python run_pipeline.py --stage figures
python run_pipeline.py --stage tables
```

Outputs:
- `results/manuscript_tables_2026-10-12.md`: every numeric table in the manuscript, generated from the result files.
- `figures/`: Figures 1–4.
- `manuscript/final_manuscript.md`: the manuscript; `.docx` and `.pdf` are exported from it with pandoc and headless Chrome.

**v2 revision (2026-11, not yet committed at the time of writing).** Outputs: `results/v2_2026-11/` (figures in its `figures/` folder); the v1 submission stays in `results/` and commit 55095ed. Set `ECG_RESULTS_DIR=results/v2_2026-11` and `ECG_FIGURES_DIR=results/v2_2026-11/figures`, then run `experiments/v2_setup.py`, `experiments/v2_queue.sh` and `experiments/v2_queue_part2.sh` to `part5.sh` in order. Cohort: BUT QDB 988 windows, CinC 2011 998 records, 11 tabular inputs. Latency has a single entry point, `experiments/latency_all.sh`, and must run alone. Manuscript: edit `manuscript/v2/main_v2_template.md`, `manuscript/v2/supplement_v2_template.md`, `manuscript/v2/references_v2.md`; run `python evaluation/build_manuscript_tables.py`, `python evaluation/assemble_q1_manuscript.py`, then `python evaluation/verify_v2_prose.py` (checks prose numbers against the saved outputs). Plan and progress: `REVISION_PLAN_2026-11-05_post_desk_rejection.md`.

Superseded material (v1 and 124-window pilot experiments, earlier audit outputs, the stale manuscript builders) is in `experiments/_superseded/`, `evaluation/_superseded/` and `results/_superseded_2026-10-12/`. Do not cite it.

Additional analyses (run directly; see REVIEW_2026-10-12.md section 7):
- `python experiments/nstdb_paired_extraction.py` then `python experiments/exp4_revised_nstdb.py`: paired NSTDB design (clean reference from MIT-BIH 118/119)
- `python experiments/butqdb_class2_sensitivity.py`: class-2 relabelling (Table 1d)
- `python experiments/cinc_random_draws.py`: five random CinC subsets (Table 2d)
- `python experiments/cinc_draws_by_seed.py` and `python experiments/butqdb_class2_by_seed.py`: training-seed variation (Table 6 and Table 1d)
- `python experiments/d1_prompt_variants.py`: d1-3B prompt sensitivity (image only; reworded question), checked against the stored primary run on 10 BUT QDB windows; Section 4.6
- Latency repeats: three sequential runs of `python experiments/empirical_latency_benchmark.py`, each copied to `results/latency_repeats/` before the next run (Table 4 reports the median and range over the runs)
- Second review (2026-10-13; see REVIEW_2026-10-13_reviewer2.md and manuscript/response_to_reviewer_2.md):
  - `python experiments/rerender_images.py` then `bash experiments/rerun_after_render_fix.sh`: renderer fix and recomputation
  - `python experiments/cinc2017_extraction.py`, `python experiments/cinc2011_twelve_lead_extraction.py`: external windows (local copies of the PhysioNet datasets under /Volumes/KIOXIA/Datasets)
  - `python experiments/d1_linear_probe.py`: d1-3B probes and external d1 predictions (Table 7b)
  - `python experiments/external_validation.py`: CinC 2017 and CinC 2011 12-lead (Tables 8, 8b, 2e)
  - `ECG_RESULTS_DIR=results/v2_2026-11 python experiments/d1_probe_twelve_lead.py [--constant-leads]`: d1-3B image-token probe on 12 leads of CinC 2011 (v2; Tables S5 and S35)
  - `ECG_RESULTS_DIR=results/v2_2026-11 python experiments/hubert_probe_twelve_lead.py`: HuBERT-ECG probe on 12 leads of CinC 2011 (v2; Tables S5 and S35)
  - `ECG_RESULTS_DIR=results/v2_2026-11 python experiments/multimodal_saturation_diagnosis.py`: why the multimodal net gives one score on CinC 2011 (v2; Table S36)
  - `ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/crosscheck_saved_outputs.py`: read-only cross-check that imports no project module. It recomputes the 33 AUROCs of Table 1, the original and harmonised AUROCs, Spearman correlations and seed ranges of Tables S46 and S47, and the latency medians and 95th percentiles of Table S45 from the saved predictions and raw timings, and compares them with the stored summaries and the assembled manuscript. It fits no model and writes nothing; it is a second calculation, not an independent review
  - `ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/audit_sensitivity.py`: post hoc sensitivity analyses asked for by the code audit of 2026-11-11 (`REVIEW_2026-11-11_code_audit_v2.md`), from saved predictions only: CinC 2011 paired tests without the 73 constant lead II records, BUT QDB per-subject AUROC, CinC 2017 AUROC by number of windows per recording, the patient check of the PTB-XL sample, and the BUT QDB quota versus the official intervals (Tables S40–S43). Run it before `evaluation/build_manuscript_tables.py`
  - `ECG_RESULTS_DIR=results/v2_2026-11 python experiments/harmonised_pipeline.py`, then `<tabpfn env>/bin/python experiments/harmonised_tabpfn.py`, then `python experiments/harmonised_pipeline.py --analysis-only`: harmonised-pipeline comparison (constant-input rule, quantization-insensitive flatline, z-clipping; post hoc, pre-stated in the revision plan; Tables S46 and S47). Needs `flatline_robust_features.csv` from `experiments/flatline_robust_exploratory.py`
  - `ECG_RESULTS_DIR=results/v2_2026-11 python experiments/cinc2017_quality_gating.py [--analysis-only]`: effect of quality gating on rhythm classification, CinC 2017 (v2; Table S37)
  - `ECG_RESULTS_DIR=results/v2_2026-11 python experiments/ptbxl_frozen_test.py --sample`, then `--extract`, `--score-d1`, `--score-others`, `--analyse` (once): frozen test on PTB-XL (v2; plan in `ANALYSIS_PLAN_2026-11-09_additional_analyses.md`)
  - `python experiments/nested_thresholds.py`: nested and transferred thresholds (Tables 1c, 2f)
  - `python experiments/prevalence_subjects_features.py`: natural prevalence, subject heterogeneity, feature shift (Tables 1e, 1f, 2g)
  - `python experiments/latency_extended.py`: extended latency (Table 4b; run alone)
  - `bash experiments/review2_queue.sh` runs the above in dependency order
  - `python evaluation/build_manuscript_tables.py` then `python evaluation/sync_manuscript_tables.py`: regenerate all tables and copy them into the manuscript

- Q1 revision (2026-10-25; run after the analyses above):
  - Environment: `pip install -r requirements.txt` (pinned direct dependencies); `requirements-lock.txt` lists every installed package. Machine paths are set by environment variables: `ECG_DATASETS_ROOT` (local PhysioNet copies), `D1_MODEL_PATH` (d1-3B weights), `HUBERT_ECG_PATH` (HuBERT-ECG base weights, CC BY-NC 4.0), `TABPFN_CKPT` (TabPFN v3 checkpoint)
  - `python experiments/lodo_multisource.py`: leave-one-dataset-out training over BUT QDB, CinC 2011 and CinC 2017, including the d1-3B probe (Table 10); `--analysis-only` re-runs the summary from saved predictions
  - `<tabpfn env>/bin/python experiments/tabpfn_runner.py`: TabPFN v3 on the 13 SQIs (CV, transfer, LODO); needs a separate environment with `tabpfn` 9.0.0
  - `python experiments/hubert_ecg_probe.py`: HuBERT-ECG frozen-feature probe (CV, transfer, LODO)
  - `python experiments/heuristic_refit.py`: rule heuristic with refitted coefficients (Table S22)
  - `python evaluation/paired_comparisons.py`: all pairwise AUROC tests per dataset, Holm-adjusted (Table 9); `--check` compares DeLong with a bootstrap
  - `python evaluation/build_manuscript_tables.py` then `python evaluation/assemble_q1_manuscript.py`: fills `manuscript/main_q1_template.md` and `manuscript/supplement_q1_template.md` with the generated tables, writing `manuscript/main_q1.md` and `manuscript/supplement_q1.md` (edit the templates, not the outputs)
  - Pre-specified additions (see `ANALYSIS_PLAN_2026-10-25_q1_additions.md`, committed before they were run):
    - `python experiments/seed_repeats_transfer.py`: seeds 1–4 of BUT-only and LODO training for the networks (Table 10b / S23)
    - `python experiments/d1_cinc2017_full.py`: zero-shot d1-3B and its image-token features on all CinC 2017 windows (resumable; about 6 h on an M4), then `python experiments/lodo_multisource.py --probe-only`
    - `python experiments/hubert_ecg_probe.py` uses the authors' preprocessing (primary); `--preproc=ours` reproduces the first version (outputs `*_ourpreproc.csv`)
    - `python experiments/hubert_ecg_finetune.py [--time-fold0]`: HuBERT-ECG partial fine-tuning (Table 12 / S25)
    - `python experiments/local_recalibration.py`: thresholds and Platt scaling re-set on small labelled target samples (Tables 11, 11b)
    - Run `python evaluation/paired_comparisons.py` after these, then the table builder and the assembler
  - `python -m unittest discover -s tests`: unit tests (metrics vs scikit-learn, DeLong and Holm, fold disjointness, feature definitions, renderer)

## 6. Review Status

See [REVIEW_2026-10-12.md](REVIEW_2026-10-12.md) for the first audit and [REVIEW_2026-10-13_reviewer2.md](REVIEW_2026-10-13_reviewer2.md) for the re-audit and the second review.
