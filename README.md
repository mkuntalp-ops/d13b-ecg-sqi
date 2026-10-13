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
  - `python experiments/nested_thresholds.py`: nested and transferred thresholds (Tables 1c, 2f)
  - `python experiments/prevalence_subjects_features.py`: natural prevalence, subject heterogeneity, feature shift (Tables 1e, 1f, 2g)
  - `python experiments/latency_extended.py`: extended latency (Table 4b; run alone)
  - `bash experiments/review2_queue.sh` runs the above in dependency order
  - `python evaluation/build_manuscript_tables.py` then `python evaluation/sync_manuscript_tables.py`: regenerate all tables and copy them into the manuscript

## 6. Review Status

See [REVIEW_2026-10-12.md](REVIEW_2026-10-12.md) for the first audit and [REVIEW_2026-10-13_reviewer2.md](REVIEW_2026-10-13_reviewer2.md) for the re-audit and the second review.
