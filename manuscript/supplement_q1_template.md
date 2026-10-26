# Supplementary Material

<!-- Template filled by evaluation/assemble_q1_manuscript.py (writes manuscript/supplement_q1.md). -->

Every table is generated from saved outputs by evaluation/build_manuscript_tables.py (results/manuscript_tables_2026-10-12.md), except Table S22, which is read from results/heuristic_refit_summary.csv, and Supplementary Methods, which are text. Positive class: unusable.

## Supplementary Methods

All definitions below were checked against the code in the repository (`features/sqi_extractor.py`, `features/qrs_detectors.py`, `models/*.py`).

### S-M1. Signal quality indices

Each 10-s window is at 500 Hz, in mV, with n = 5,000 samples. Spectral quantities use Welch's method with 2-s segments.

**The 13 tabular inputs.**
1. **bSQI.** F1 agreement between two QRS detectors:
   - an implementation following Pan and Tompkins;
   - a second-difference detector (0.5-Hz high-pass, |x[n+2] − 2x[n] + x[n−2]|, 80-ms moving average);
   - beats are matched one to one within 50 ms;
   - bSQI is 0 if either detector finds no beat.
2. **Baseline-wander ratio.** Power below 0.5 Hz divided by power in 0.5–40 Hz; clipped to [0, 100].
3. **High-frequency ratio.** Power above 40 Hz, up to min(0.49·fs, 100 Hz), divided by power in 0.5–40 Hz; clipped to [0, 100].
4. **Saturation fraction.** Share of samples within 10^-5 of the window minimum or maximum. Its floor is 2/5,000 = 4 × 10^-4.
5. **Flatline fraction.** Share of consecutive-sample differences with |Δx| < 10^-6 mV. It also counts repeats caused by amplitude quantization.
6. **QRS-band ratio.** Power in 5–15 Hz divided by power in 0.5–40 Hz; clipped to [0, 1].
7. **Kurtosis.** Fisher definition; clipped to [−5, 50].
8. **Skewness.** Clipped to [−10, 10].
9. **Zero-crossing rate.** Sign changes of the median-centred signal divided by n.
10. **Powerline ratio.** Power in 49–51 Hz and 59–61 Hz divided by power in 0.5–40 Hz; clipped to [0, 100].
11. **Amplitude SD.** In mV.
12. **Window duration.** In s.
13. **Sampling rate.** In Hz.

**Special values.**
- A window whose range is below 10^-6 receives flatline and saturation fractions of 1.
- Non-finite values are set to 0.

### S-M2. Network architectures

**Common training settings.**
- AdamW (learning rate 10^-3, weight decay 10^-3).
- Binary cross-entropy with the positive class weighted by the usable/unusable ratio of the training data.
- No early stopping; seed 42 for the reported run.

**SQI MLP (3,201 parameters).**
- Layers: Linear(13→64), BatchNorm, ReLU, Dropout 0.2, Linear(64→32), BatchNorm, ReLU, Linear(32→1).
- Inputs standardised with training-set mean and SD.
- 35 epochs, batch size 64.

**1D CNN (167,489 parameters).**
- Four blocks of Conv1d, BatchNorm, LeakyReLU (slope 0.1), MaxPool (×4) and Dropout 0.15.
  - Channels 32, 64, 128, 128.
  - Kernel sizes 15, 11, 7, 5.
- Global average pooling, then Linear(128→32), ReLU, Dropout 0.25, Linear(32→1).
- Input: the 5,000-sample window, z-scored per window (SD below 10^-6 replaced by 1).
- 25 epochs, batch size 32, cosine learning-rate schedule.

**2D CNN (245,697 parameters).**
- Input: the rendered strip resized to 3 × 128 × 256 and normalised with mean 0.95 and SD 0.15 per channel.
- Four blocks of 3 × 3 Conv2d, BatchNorm and ReLU, with 2 × 2 max-pooling after the first three.
  - Channels 32, 64, 128, 128.
- Global average pooling, then Linear(128→32), ReLU, Dropout 0.3, Linear(32→1).
- 25 epochs, batch size 32, cosine schedule.

**Multimodal net (131,969 parameters).**
- *Image encoder:* three blocks of Conv2d, BatchNorm and ReLU (channels 32, 64, 128), global average pooling, Linear(128→64), ReLU.
- *SQI encoder:* Linear(13→64), BatchNorm, ReLU, Dropout 0.2, Linear(64→64), ReLU.
- *Gated fusion:* the concatenated 128-d vector is multiplied element-wise by a sigmoid gate, Linear(128→128).
- *Head:* Linear(128→64), ReLU, Dropout 0.3, Linear(64→1).
- 25 epochs, batch size 32, cosine schedule.

### S-M3. Rule heuristic (`models/rule_heuristic_baseline.py`)

**SQI component.** With the inputs of S-M1:

logit_SQI = −3.0·(bSQI − 0.5) + 1.2·min(baseline-wander ratio, 3) + 1.5·min(high-frequency ratio, 3) + 5.0·saturation + 5.0·flatline − 0.5

p_SQI = 1 / (1 + exp(−logit_SQI))

**Image component.**
- The rendered strip is converted to 8-bit grey scale.
- *ink* is the share of pixels with value below 200.
- *spread* is the SD over image columns of the per-column ink share.

logit_vis = 4.0·(ink − 0.08) − 2.0·(spread − 0.02)

p_vis = 1 / (1 + exp(−logit_vis))

**Output.** The score is p = 0.6·p_SQI + 0.4·p_vis, clipped to [0.01, 0.99]. All coefficients are hand-set; their development history is not documented.

### S-M4. Zero-shot d1-3B input (`models/d1_real_neural_model.py`)

**Image.** The rendered strip.

**State text.** A summary of 10 inputs (S-M1, items 1–8, 12 and 13), in the format:

"ECG Signal Quality Indicators (SQIs): QRS Detector Agreement (bSQI): x (1.0 = perfect consensus, 0.0 = total disagreement); Baseline Wander Ratio; High-Frequency Noise Ratio; Saturation Fraction (%); Flatline Fraction (%); QRS Band Energy Ratio; Kurtosis and Skewness; Window Duration and Sampling Rate."

**Question.** A choice question with:
- *Instructions:* "Determine whether this electrocardiogram (ECG) recording segment is usable for reliable QRS-based rhythm analysis, or unusable due to severe artifacts, baseline wander, high-frequency noise, or saturation/flatline."
- *Criterion for "usable":* "QRS complexes are distinguishable and suitable for rhythm analysis. Artifacts are manageable."
- *Criterion for "unusable":* "ECG is severely corrupted by noise, baseline drift, saturation, or absent complexes; requires human review or re-recording."

**Output.** The probability of "unusable" from a single forward pass with zero output tokens. Model revision: 051bcc464b01b9f92942b364d9586b0ef5912432.

## Table S1. Patient-independent 5-fold cross-validation on BUT QDB (1,047 windows, 15 subjects), models of the main analysis. 95% CIs: subject-level cluster bootstrap.

<!-- source: Table 1 -->
{{Table 1}}

## Table S2. Nested versus post-hoc threshold selection on BUT QDB.

<!-- source: Table 1c -->
{{Table 1c}}

## Table S3. BUT QDB precision metrics re-weighted to the natural prevalence of Class 3 (16.0%; 2.8% without record 105001).

<!-- source: Table 1e -->
{{Table 1e}}

## Table S4. Thresholds chosen on pooled BUT QDB out-of-fold scores and applied unchanged to the external datasets. CinC 2017: all 8,512 recordings for every model, including zero-shot d1-3B (recording maximum). CinC 2011: the 150-record reported subset and the range over the five random subsets.

<!-- source: Table 2f -->
{{Table 2f}}

## Table S5. CinC 2011 scored on all 12 leads (aggregation rules chosen after the main analysis).

<!-- source: Table 2e -->
{{Table 2e}}

## Table S6. Univariate AUROC of each SQI across datasets (values below 0.5: higher feature value associated with usable).

<!-- source: Table 2g -->
{{Table 2g}}

## Table S7. NSTDB: paired change in usable probability relative to the clean partner window (mean ± SD).

<!-- source: Table 3b -->
{{Table 3b}}

## Table S8. Extended latency profile (Apple M4; `experiments/latency_extended.py`). This is a separate script from main-text Table 5 (`experiments/empirical_latency_benchmark.py`): CPU and MPS passes are interleaved per window, and the first window of each run is excluded. MPS forward times of a few milliseconds therefore differ from Table 5 (main text, Limitation 8).

<!-- source: Table 4b -->
{{Table 4b}}

## Table S9. Seed stability of the 1D CNN, 2D CNN and multimodal net.

<!-- source: Table 5 -->
{{Table 5}}

## Table S10. Training-seed and CinC-subset variation of the supervised models.

<!-- source: Table 6 -->
{{Table 6}}

## Table S11. Post-hoc ROC operating points on BUT QDB (exploratory; thresholds chosen on the evaluated data).

<!-- source: Table 1b -->
{{Table 1b}}

## Table S12. Sensitivity analysis with Class 2 relabelled as unusable (BUT QDB).

<!-- source: Table 1d -->
{{Table 1d}}

## Table S13. Subject-level sensitivity on BUT QDB.

<!-- source: Table 1f -->
{{Table 1f}}

## Table S14. CinC 2011 Set A, reported 150-record subset (first 75 acceptable and 75 unacceptable records).

<!-- source: Table 2 -->
{{Table 2}}

## Table S15. Post-hoc ROC operating points on the CinC 2011 reported subset (exploratory).

<!-- source: Table 2b -->
{{Table 2b}}

## Table S16. Feature-set and scaler sensitivity of the tabular models under transfer.

<!-- source: Table 2c -->
{{Table 2c}}

## Table S17. Transfer to five random balanced CinC 2011 subsets.

<!-- source: Table 2d -->
{{Table 2d}}

## Table S18. NSTDB: usable probability by nominal SNR.

<!-- source: Table 3 -->
{{Table 3}}

## Table S19. Prompt sensitivity of zero-shot d1-3B.

<!-- source: Table 7 -->
{{Table 7}}

## Table S20. Linear probes on frozen d1-3B representations.

<!-- source: Table 7b -->
{{Table 7b}}

## Table S21. CinC 2017 balanced subset (278 noisy + 278 non-noisy recordings) including zero-shot d1-3B.

<!-- source: Table 8b -->
{{Table 8b}}

## Table S22. Hand-set rule heuristic versus a logistic-regression replacement using the same seven input features (standardised L2 logistic regression fitted inside the BUT QDB folds and transferred like the other models). The replacement changes the functional form as well as the coefficients (Supplementary Methods S-M3). p: BUT QDB, subject-level cluster bootstrap; external sets, DeLong; unadjusted.

{{CSV:heuristic_refit}}

## Table S23. Seed repetition (seeds 42 and 1–4) of BUT-only training and leave-one-dataset-out training for the seed-dependent networks. Consistent: same sign of Δ in all five seeds and |median Δ| ≥ 0.02 (pre-specified).

{{Table 10b}}

## Table S24. Local recalibration on n = 25–200 randomly drawn labelled target recordings (500 draws each), evaluated on the remaining recordings. Prevalence-only Brier score: CinC 2011 0.227, CinC 2017 0.032.

{{Table 11b}}

## Table S25. HuBERT-ECG: preprocessing sensitivity and partial fine-tuning (AUROC). Fine-tuning was not run in LODO.

{{Table 12}}

## Table S26. BUT QDB official protocol (Smital et al. 2026): sensitivity analyses. F1a: pooled out-of-fold AUROC without the 59 windows from the excluded stretch of record 105001 (evaluation only). F1b: retrained on the official training records, tested on the official test records (window-level binary task). 95% CI: subject-level cluster bootstrap; p_Holm: paired comparison with the best model.

{{Table 13}}

## Table S27. Latency of the foundation-model probes and TabPFN (Apple M4; same 25 windows and three runs as main-text Table 5, harness of Table S8).

{{Table 4c}}

## Table S28. CinC 2011 records with a constant lead II (65 of 197 unacceptable records; post hoc, descriptive). Number flagged at τ = 0.50 and AUROC with and without these records; models trained on BUT QDB only.

{{Table 2h}}

## Table S29. CinC 2017 at natural prevalence with bootstrap confidence intervals, seven main models (recording maximum; mean in the last column).

{{Table 8}}

## Table S30. Discrimination of every model in each primary evaluation set: AUROC and average precision with 95% bootstrap confidence intervals (2,000 resamples, seed 42; resampling unit subject on BUT QDB and recording externally). Models trained on BUT QDB only; external sets scored with the pre-specified recording maximum. Intervals for the seven main models differ slightly from Tables S1 and S29, which used other resampling runs.

{{Table 1g}}

## Table S31. Distribution of each SQI by dataset and class: median [interquartile range] per window. CinC 2017 windows carry the label of their recording.

{{Table 2i}}
