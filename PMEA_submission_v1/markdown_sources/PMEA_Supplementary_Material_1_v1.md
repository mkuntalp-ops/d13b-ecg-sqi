# Supplementary Material 1

**In-domain rankings did not identify the best-transferring model in ECG usability screening: a patient-independent, multi-dataset benchmark of task-specific, tabular and foundation models**

Mehmet Kuntalp


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

**Output.** The score is p = 0.6·p_SQI + 0.4·p_vis, clipped to [0.01, 0.99]. All coefficients are hand-set. They were set during the initial prototyping of the analysis code, and their independence from BUT QDB cannot be guaranteed; the heuristic's BUT QDB results are therefore for reference only.

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

| Model Architecture | AUROC [95% CI] | AP | Brier | Fixed τ = 0.50 [TP / FP / TN / FN] | Balanced Acc. [95% CI] | Sensitivity [95% CI] | False Reject Rate [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D CNN (Raw Signal)** | 0.989 [0.964 – 0.999] | 0.975 | 0.058 | [228 / 5 / 745 / 69] | 88.1% [71.7 – 98.4%] | 76.8% [43.7 – 97.5%] | 0.7% [0.0 – 1.6%] |
| **2D CNN (Rendered Image)** | 0.989 [0.967 – 0.999] | 0.971 | 0.052 | [234 / 7 / 743 / 63] | 88.9% [75.5 – 99.5%] | 78.8% [51.8 – 100.0%] | 0.9% [0.3 – 1.7%] |
| **Rule-Based SQI Heuristic** | 0.977 [0.926 – 0.998] | 0.948 | 0.105 | [196 / 9 / 741 / 101] | 82.4% [57.2 – 96.1%] | 66.0% [15.5 – 92.9%] | 1.2% [0.3 – 2.4%] |
| **SinglePass Multimodal Net** | 0.967 [0.926 – 0.999] | 0.928 | 0.094 | [189 / 14 / 736 / 108] | 80.9% [63.4 – 98.5%] | 63.6% [28.1 – 98.9%] | 1.9% [0.7 – 3.3%] |
| **LightGBM (SQI)** | 0.964 [0.893 – 0.997] | 0.928 | 0.074 | [218 / 31 / 719 / 79] | 84.6% [64.8 – 96.1%] | 73.4% [33.9 – 94.6%] | 4.1% [1.6 – 7.3%] |
| **Independent SQI MLP** | 0.939 [0.834 – 0.990] | 0.880 | 0.088 | [231 / 65 / 685 / 66] | 84.6% [68.2 – 95.6%] | 77.8% [48.8 – 96.5%] | 8.7% [2.4 – 18.7%] |
| **Logistic Regression (SQI)** | 0.938 [0.838 – 0.994] | 0.888 | 0.084 | [227 / 47 / 703 / 70] | 85.1% [67.6 – 96.4%] | 76.4% [44.0 – 96.8%] | 6.3% [1.7 – 13.5%] |
| **LiquidAI d1-3B (Zero-Shot)** | 0.852 [0.717 – 0.944] | 0.740 | 0.154 | [158 / 30 / 720 / 139] | 74.6% [59.0 – 87.2%] | 53.2% [21.3 – 77.4%] | 4.0% [1.5 – 6.9%] |

## Table S2. Nested versus post-hoc threshold selection on BUT QDB.

| Model Architecture | Target 1 (FPR ≤ 5%): nested Sens. / FPR | Target 1: post-hoc Sens. / FPR | Target 2 (Sens. ≥ 80%): nested Sens. / FPR | Target 2: post-hoc Sens. / FPR | Youden: nested Sens. / FPR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 77.4% / 6.7% | 74.1% / 4.8% | 87.5% / 13.1% | 80.1% / 8.3% | 73.1% / 8.7% |
| **LightGBM (SQI)** | 70.4% / 3.7% | 80.5% / 4.8% | 92.9% / 9.5% | 80.5% / 4.8% | 62.6% / 5.7% |
| **Independent SQI MLP** | 65.3% / 4.8% | 70.0% / 4.9% | 88.6% / 12.3% | 80.5% / 9.3% | 66.3% / 9.1% |
| **1D CNN (Raw Signal)** | 97.3% / 3.5% | 96.0% / 3.5% | 81.5% / 1.2% | 80.1% / 1.3% | 97.6% / 3.5% |
| **2D CNN (Rendered Image)** | 94.6% / 4.7% | 94.3% / 4.7% | 76.8% / 0.9% | 80.1% / 0.9% | 91.9% / 5.2% |
| **SinglePass Multimodal Net** | 70.0% / 5.3% | 69.0% / 4.7% | 78.1% / 11.1% | 80.1% / 6.5% | 85.5% / 7.9% |
| **Rule-Based SQI Heuristic** | 84.8% / 5.3% | 84.5% / 4.9% | 69.0% / 3.2% | 80.5% / 4.1% | 90.2% / 8.5% |
| **LiquidAI d1-3B (Zero-Shot)** | 53.2% / 4.0% | 53.2% / 4.0% | 76.1% / 34.3% | 80.1% / 32.7% | 63.0% / 14.7% |

## Table S3. BUT QDB precision metrics re-weighted to the natural prevalence of Class 3 (16.0%; 2.8% without record 105001).

| Model Architecture | AP, quota (28.4%) | AP, natural (16.0%) | AP, natural excl. 105001 (2.8%) | PPV at τ = 0.50, 16.0% | PPV at τ = 0.50, 2.8% | NPV at τ = 0.50, 2.8% | False rejections per hour of usable signal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.888 | 0.823 | 0.587 | 0.700 | 0.260 | 0.993 | 22.6 |
| **LightGBM (SQI)** | 0.928 | 0.882 | 0.728 | 0.772 | 0.339 | 0.992 | 14.9 |
| **Independent SQI MLP** | 0.880 | 0.806 | 0.551 | 0.632 | 0.205 | 0.993 | 31.2 |
| **1D CNN (Raw Signal)** | 0.975 | 0.954 | 0.831 | 0.957 | 0.768 | 0.993 | 2.4 |
| **2D CNN (Rendered Image)** | 0.971 | 0.947 | 0.809 | 0.942 | 0.709 | 0.994 | 3.4 |
| **SinglePass Multimodal Net** | 0.928 | 0.878 | 0.700 | 0.867 | 0.496 | 0.989 | 6.7 |
| **Rule-Based SQI Heuristic** | 0.948 | 0.910 | 0.765 | 0.913 | 0.613 | 0.990 | 4.3 |
| **LiquidAI d1-3B (Zero-Shot)** | 0.740 | 0.630 | 0.351 | 0.718 | 0.277 | 0.986 | 14.4 |

## Table S4. Thresholds chosen on pooled BUT QDB out-of-fold scores and applied unchanged to the external datasets. CinC 2017: all 8,512 recordings for every model, including zero-shot d1-3B (recording maximum). CinC 2011: the 150-record reported subset and the range over the five random subsets.

| Model Architecture | Rule | Threshold (from BUT QDB) | BUT QDB Sens. / FPR | CinC 2011 reported Sens. / FPR | CinC 2011 random subsets Sens. range | CinC 2017 Sens. / FPR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | Target 1 (FPR ≤ 5%) | 0.5644 | 74.1% / 4.8% | 74.7% / 100.0% | 61.3–77.3% | 74.5% / 15.7% |
| **Logistic Regression (SQI)** | Target 2 (Sens. ≥ 80%) | 0.3929 | 80.1% / 8.3% | 74.7% / 100.0% | 61.3–77.3% | 83.5% / 25.5% |
| **LightGBM (SQI)** | Target 1 (FPR ≤ 5%) | 0.3602 | 80.5% / 4.8% | 41.3% / 1.3% | 42.7–58.7% | 79.5% / 19.2% |
| **LightGBM (SQI)** | Target 2 (Sens. ≥ 80%) | 0.3602 | 80.5% / 4.8% | 41.3% / 1.3% | 42.7–58.7% | 79.5% / 19.2% |
| **Independent SQI MLP** | Target 1 (FPR ≤ 5%) | 0.725 | 70.0% / 4.9% | 0.0% / 0.0% | 0.0–0.0% | 74.5% / 16.5% |
| **Independent SQI MLP** | Target 2 (Sens. ≥ 80%) | 0.4337 | 80.5% / 9.3% | 0.0% / 0.0% | 0.0–0.0% | 88.1% / 26.2% |
| **1D CNN (Raw Signal)** | Target 1 (FPR ≤ 5%) | 0.007939 | 96.0% / 3.5% | 70.7% / 28.0% | 70.7–86.7% | 99.6% / 54.5% |
| **1D CNN (Raw Signal)** | Target 2 (Sens. ≥ 80%) | 0.3318 | 80.1% / 1.3% | 62.7% / 16.0% | 62.7–78.7% | 93.5% / 36.9% |
| **2D CNN (Rendered Image)** | Target 1 (FPR ≤ 5%) | 0.05569 | 94.3% / 4.7% | 52.0% / 10.7% | 54.7–70.7% | 96.4% / 50.4% |
| **2D CNN (Rendered Image)** | Target 2 (Sens. ≥ 80%) | 0.4892 | 80.1% / 0.9% | 17.3% / 4.0% | 17.3–28.0% | 89.9% / 34.0% |
| **SinglePass Multimodal Net** | Target 1 (FPR ≤ 5%) | 0.1746 | 69.0% / 4.7% | 25.3% / 0.0% | 24.0–40.0% | 92.8% / 43.1% |
| **SinglePass Multimodal Net** | Target 2 (Sens. ≥ 80%) | 0.1054 | 80.1% / 6.5% | 25.3% / 0.0% | 24.0–40.0% | 94.6% / 46.2% |
| **Rule-Based SQI Heuristic** | Target 1 (FPR ≤ 5%) | 0.3762 | 84.5% / 4.9% | 100.0% / 100.0% | 100.0–100.0% | 52.9% / 6.2% |
| **Rule-Based SQI Heuristic** | Target 2 (Sens. ≥ 80%) | 0.407 | 80.5% / 4.1% | 98.7% / 98.7% | 97.3–98.7% | 43.5% / 5.2% |
| **LiquidAI d1-3B (Zero-Shot)** | Target 1 (FPR ≤ 5%) | 0.5 | 53.2% / 4.0% | 100.0% / 100.0% | 98.7–100.0% | 19.8% / 3.1% |
| **LiquidAI d1-3B (Zero-Shot)** | Target 2 (Sens. ≥ 80%) | 0.3775 | 80.1% / 32.7% | 100.0% / 100.0% | 100.0–100.0% | 40.6% / 10.3% |

## Table S5. CinC 2011 scored on all 12 leads (aggregation rules chosen after the main analysis).

| Model Architecture | Reported subset: lead II | Reported subset: max over 12 leads | Reported subset: mean over 12 leads | Random subsets: lead II, mean (min–max) | Random subsets: max over 12, mean (min–max) | Random subsets: mean over 12, mean (min–max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.383 | 0.393 | 0.226 | 0.323 (0.291–0.372) | 0.352 (0.320–0.407) | 0.157 (0.121–0.222) |
| **LightGBM (SQI)** | 0.731 | 0.729 | 0.858 | 0.750 (0.700–0.821) | 0.667 (0.624–0.726) | 0.877 (0.845–0.932) |
| **Independent SQI MLP** | 0.617 | 0.735 | 0.735 | 0.670 (0.628–0.703) | 0.807 (0.763–0.832) | 0.809 (0.766–0.834) |
| **1D CNN (Raw Signal)** | 0.726 | 0.330 | 0.803 | 0.707 (0.660–0.816) | 0.214 (0.153–0.251) | 0.801 (0.753–0.879) |
| **2D CNN (Rendered Image)** | 0.746 | 0.761 | 0.870 | 0.781 (0.713–0.884) | 0.692 (0.675–0.713) | 0.852 (0.822–0.910) |
| **SinglePass Multimodal Net** | 0.627 | 0.787 | 0.787 | 0.668 (0.620–0.707) | 0.837 (0.775–0.868) | 0.841 (0.783–0.872) |
| **Rule-Based SQI Heuristic** | 0.700 | 0.835 | 0.715 | 0.683 (0.639–0.737) | 0.860 (0.821–0.900) | 0.743 (0.701–0.778) |
| **LiquidAI d1-3B (Zero-Shot)** | 0.636 | 0.800 | 0.630 | — | — | — |

## Table S6. Univariate AUROC of each SQI across datasets (values below 0.5: higher feature value associated with usable).

| SQI feature | BUT QDB (1,047 windows) | CinC 2011 lead II (565 records) | CinC 2011 lead II (reported 150) | CinC 2017 (first window, all recordings) |
| :--- | :---: | :---: | :---: | :---: |
| `bsqi_detector_agreement` | 0.030 | 0.308 | 0.305 | 0.124 |
| `baseline_wander_ratio` | 0.421 | 0.437 | 0.435 | 0.595 |
| `high_frequency_noise_ratio` | 0.777 | 0.307 | 0.329 | 0.414 |
| `saturation_fraction` | 0.501 | 0.634 | 0.593 | 0.505 |
| `flatline_fraction` | 0.419 | 0.645 | 0.632 | 0.281 |
| `qrs_energy_ratio` | 0.132 | 0.226 | 0.263 | 0.231 |
| `kurtosis_ksqi` | 0.164 | 0.269 | 0.289 | 0.401 |
| `skewness_ssqi` | 0.093 | 0.338 | 0.358 | 0.342 |
| `zero_crossing_rate` | 0.727 | 0.248 | 0.311 | 0.470 |
| `powerline_noise_ratio` | 0.773 | 0.312 | 0.324 | 0.426 |
| `signal_std` | 0.385 | 0.402 | 0.385 | 0.786 |

## Table S7. NSTDB: paired change in usable probability relative to the clean partner window (mean ± SD).

| SNR Level | Logistic Regression (SQI) | LightGBM (SQI) | 1D CNN (Raw Signal) | 2D CNN (Image) | Multimodal Net | d1-3B (zero-shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+24 dB (Low Noise)** | -0.067 ± 0.117 | -0.107 ± 0.240 | -0.002 ± 0.073 | -0.110 ± 0.252 | -0.094 ± 0.183 | -0.018 ± 0.058 |
| **+18 dB** | -0.155 ± 0.141 | -0.436 ± 0.344 | -0.046 ± 0.159 | -0.318 ± 0.336 | -0.466 ± 0.402 | -0.037 ± 0.059 |
| **+12 dB** | -0.467 ± 0.214 | -0.811 ± 0.187 | -0.267 ± 0.364 | -0.602 ± 0.354 | -0.776 ± 0.307 | -0.072 ± 0.092 |
| **+6 dB** | -0.791 ± 0.139 | -0.901 ± 0.093 | -0.557 ± 0.417 | -0.666 ± 0.366 | -0.840 ± 0.276 | -0.092 ± 0.103 |
| **0 dB (Equal Power)** | -0.930 ± 0.055 | -0.902 ± 0.087 | -0.683 ± 0.373 | -0.668 ± 0.365 | -0.852 ± 0.271 | -0.092 ± 0.119 |
| **-6 dB (Severe Noise)** | -0.956 ± 0.032 | -0.893 ± 0.098 | -0.713 ± 0.356 | -0.673 ± 0.363 | -0.851 ± 0.273 | -0.078 ± 0.113 |

## Table S8. Extended latency profile (Apple M4; `experiments/latency_extended.py`). This is a separate script from main-text Table 5 (`experiments/empirical_latency_benchmark.py`): CPU and MPS passes are interleaved per window, and the first window of each run is excluded. Decoded image tensors are cached after first use, so image decoding and resizing are timed only in the first run; the reported median over runs reflects cached inputs. MPS forward times of a few milliseconds therefore differ from Table 5 (main text, Limitation 8).

| Step | Median (three runs) | Range of run medians | p95 (median over runs) |
| :--- | :---: | :---: | :---: |
| Unit conversion (µV to mV) | 0.01 ms | 0.01–0.01 ms | 0.02 ms |
| Resampling 1,000 → 500 Hz | 0.10 ms | 0.09–0.10 ms | 0.12 ms |
| SQI extraction (13 features) | 1.72 ms | 1.70–1.76 ms | 1.95 ms |
| Rendering, Matplotlib (used by all image models) | 58.41 ms | 58.18–59.91 ms | 121.16 ms |
| Rendering, direct raster (PIL; timing only) | 0.63 ms | 0.62–0.68 ms | 0.72 ms |
| 1D CNN forward, CPU | 1.30 ms | 1.27–1.35 ms | 1.49 ms |
| 1D CNN raw-to-decision, CPU | 1.41 ms | 1.39–1.46 ms | 1.61 ms |
| 2D CNN forward, CPU | 5.11 ms | 5.10–7.13 ms | 5.61 ms |
| Multimodal net forward, CPU | 4.39 ms | 4.32–5.58 ms | 5.10 ms |
| 1D CNN forward, MPS | 2.80 ms | 2.75–2.80 ms | 3.24 ms |
| 1D CNN raw-to-decision, MPS | 2.90 ms | 2.85–2.94 ms | 3.36 ms |
| 2D CNN forward, MPS | 2.92 ms | 2.90–3.75 ms | 3.87 ms |
| Multimodal net forward, MPS | 3.17 ms | 3.15–3.98 ms | 4.48 ms |

## Table S9. Seed stability of the 1D CNN, 2D CNN and multimodal net.

| Model | Metric | Seed 42 (reported) | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± SD (seeds 1–4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D CNN (Raw Signal)** | BUT QDB pooled AUROC | 0.989 | 0.971 | 0.979 | 0.979 | 0.977 | 0.976 ± 0.004 |
| **1D CNN (Raw Signal)** | CinC AUROC | 0.726 | 0.715 | 0.713 | 0.719 | 0.710 | 0.714 ± 0.004 |
| **1D CNN (Raw Signal)** | CinC sensitivity at τ = 0.50 | 61.3% | 62.7% | 61.3% | 61.3% | 62.7% | 62.0 ± 0.8% |
| **1D CNN (Raw Signal)** | CinC false reject at τ = 0.50 | 14.7% | 20.0% | 20.0% | 16.0% | 21.3% | 19.3 ± 2.3% |
| **2D CNN (Rendered Image)** | BUT QDB pooled AUROC | 0.989 | 0.984 | 0.983 | 0.984 | 0.993 | 0.986 ± 0.005 |
| **2D CNN (Rendered Image)** | CinC AUROC | 0.746 | 0.763 | 0.763 | 0.767 | 0.748 | 0.760 ± 0.008 |
| **2D CNN (Rendered Image)** | CinC sensitivity at τ = 0.50 | 17.3% | 41.3% | 42.7% | 20.0% | 42.7% | 36.7 ± 11.1% |
| **2D CNN (Rendered Image)** | CinC false reject at τ = 0.50 | 4.0% | 4.0% | 2.7% | 2.7% | 4.0% | 3.3 ± 0.8% |
| **SinglePass Multimodal Net** | BUT QDB pooled AUROC | 0.967 | 0.967 | 0.975 | 0.967 | 0.967 | 0.969 ± 0.004 |
| **SinglePass Multimodal Net** | CinC AUROC | 0.627 | 0.500 | 0.500 | 0.500 | 0.627 | 0.532 ± 0.063 |
| **SinglePass Multimodal Net** | CinC sensitivity at τ = 0.50 | 25.3% | 100.0% | 100.0% | 100.0% | 25.3% | 81.3 ± 37.3% |
| **SinglePass Multimodal Net** | CinC false reject at τ = 0.50 | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | 75.0 ± 50.0% |

## Table S10. Training-seed and CinC-subset variation of the supervised models.

| Model | AUROC range over seeds × subsets | SD across training seeds (mean over subsets) | SD across subsets (mean over seeds) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.291–0.372 | 0.000 | 0.032 |
| **LightGBM (SQI)** | 0.700–0.821 | 0.000 | 0.049 |
| **Independent SQI MLP** | 0.613–0.707 | 0.007 | 0.033 |
| **1D CNN (Raw Signal)** | 0.643–0.825 | 0.009 | 0.066 |
| **2D CNN (Rendered Image)** | 0.713–0.904 | 0.007 | 0.064 |
| **SinglePass Multimodal Net** | 0.500–0.707 | 0.091 | 0.015 |

## Table S11. Post-hoc ROC operating points on BUT QDB (exploratory; thresholds chosen on the evaluated data).

| Model Architecture | Target 1: Sens. (Actual FPR) | Threshold τ | Target 2: FPR (Actual Sens.) | Threshold τ |
| :--- | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | 94.28% (FPR 4.67%) | 0.05569 | 0.93% (Sens 80.13%) | 0.48918 |
| **1D CNN (Raw Signal)** | 95.96% (FPR 3.47%) | 0.00794 | 1.33% (Sens 80.13%) | 0.33178 |
| **Rule-Based SQI Heuristic** | 84.51% (FPR 4.93%) | 0.37622 | 4.13% (Sens 80.47%) | 0.40701 |
| **SinglePass Multimodal Net** | 69.02% (FPR 4.67%) | 0.17462 | 6.53% (Sens 80.13%) | 0.10541 |
| **LightGBM (SQI)** | 80.47% (FPR 4.80%) | 0.36019 | 4.80% (Sens 80.47%) | 0.36019 |
| **Independent SQI MLP** | 70.03% (FPR 4.93%) | 0.72503 | 9.33% (Sens 80.47%) | 0.43368 |
| **Logistic Regression (SQI)** | 74.07% (FPR 4.80%) | 0.56444 | 8.27% (Sens 80.13%) | 0.39292 |
| **LiquidAI d1-3B (Zero-Shot)** | 53.20% (FPR 4.00%) | 0.50000 | 32.67% (Sens 80.13%) | 0.37754 |

## Table S12. Sensitivity analysis with Class 2 relabelled as unusable (BUT QDB).

| Model | AUROC, class 3 unusable (297) [95% CI] | AUROC, classes 2–3 unusable (490), seed 42 [95% CI] | AUROC, classes 2–3 unusable, seeds 1–4 (range) | Sens. / FR at τ = 0.50, class 3 unusable | Sens. / FR at τ = 0.50, classes 2–3 unusable |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.938 [0.838 – 0.994] | 0.955 [0.920 – 0.976] | 0.955–0.955 | 76.4% / 6.3% | 85.3% / 8.6% |
| **LightGBM (SQI)** | 0.964 [0.893 – 0.997] | 0.951 [0.918 – 0.972] | 0.951–0.951 | 73.4% / 4.1% | 89.0% / 13.1% |
| **Independent SQI MLP** | 0.939 [0.834 – 0.990] | 0.837 [0.594 – 0.966] | 0.825–0.942 | 77.8% / 8.7% | 75.9% / 14.7% |
| **1D CNN (Raw Signal)** | 0.989 [0.964 – 0.999] | 0.961 [0.902 – 0.994] | 0.959–0.964 | 76.8% / 0.7% | 78.2% / 6.6% |
| **2D CNN (Rendered Image)** | 0.989 [0.967 – 0.999] | 0.984 [0.960 – 0.998] | 0.984–0.986 | 78.8% / 0.9% | 90.6% / 5.6% |
| **SinglePass Multimodal Net** | 0.967 [0.926 – 0.999] | 0.865 [0.611 – 0.993] | 0.867–0.984 | 63.6% / 1.9% | 78.6% / 6.5% |
| **Rule-Based SQI Heuristic** | 0.977 [0.926 – 0.998] | 0.973 [0.948 – 0.987] | — | 66.0% / 1.2% | 41.8% / 0.0% |
| **LiquidAI d1-3B (Zero-Shot)** | 0.852 [0.717 – 0.944] | 0.785 [0.655 – 0.875] | — | 53.2% / 4.0% | 35.3% / 2.7% |

## Table S13. Subject-level sensitivity on BUT QDB.

| Held-out subject (fold) | Unusable / usable windows | 1D CNN (Raw Signal) | 2D CNN (Rendered Image) | SinglePass Multimodal Net | LightGBM (SQI) | Rule-Based SQI Heuristic | LiquidAI d1-3B (Zero-Shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100 (4)** | 22 / 50 | 14% | 86% | 64% | 5% | 9% | 5% |
| **105 (3)** | 60 / 50 | 100% | 100% | 100% | 100% | 100% | 65% |
| **111 (2)** | 60 / 50 | 32% | 27% | 35% | 37% | 12% | 53% |
| **113 (0)** | 4 / 50 | 100% | 100% | 75% | 100% | 50% | 100% |
| **114 (1)** | 60 / 50 | 98% | 100% | 100% | 100% | 98% | 100% |
| **122 (0)** | 60 / 50 | 100% | 73% | 0% | 100% | 100% | 33% |
| **124 (4)** | 31 / 50 | 74% | 100% | 100% | 35% | 19% | 6% |
| **False reject, range over 15 subjects** | — / 750 | 0–6% | 0–4% | 0–10% | 0–22% | 0–8% | 0–16% |

## Table S14. CinC 2011 Set A, reported 150-record subset (first 75 acceptable and 75 unacceptable records).

| Model Architecture | Transfer AUROC | Transfer AP | Brier Score | Fixed τ = 0.50 [TP / FP / TN / FN] | Sensitivity | False Reject Rate | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | 0.746 | 0.748 | 0.356 | [13 / 3 / 72 / 62] | 17.3% | 4.0% | 56.7% |
| **LightGBM (SQI)** | 0.731 | 0.792 | 0.320 | [7 / 1 / 74 / 68] | 9.3% | 1.3% | 54.0% |
| **1D CNN (Raw Signal)** | 0.726 | 0.683 | 0.266 | [46 / 11 / 64 / 29] | 61.3% | 14.7% | 73.3% |
| **Rule-Based SQI Heuristic** | 0.700 | 0.759 | 0.239 | [72 / 63 / 12 / 3] | 96.0% | 84.0% | 56.0% |
| **LiquidAI d1-3B (Zero-Shot)** | 0.636 | 0.706 | 0.388 | [75 / 75 / 0 / 0] | 100.0% | 100.0% | 50.0% |
| **SinglePass Multimodal Net** | 0.627 | 0.627 | 0.373 | [19 / 0 / 75 / 56] | 25.3% | 0.0% | 62.7% |
| **Independent SQI MLP** | 0.617 | 0.627 | 0.500 | [0 / 0 / 75 / 75] | 0.0% | 0.0% | 50.0% |
| **Logistic Regression (SQI)** | 0.383 | 0.451 | 0.627 | [56 / 75 / 0 / 19] | 74.7% | 100.0% | 37.3% |

## Table S15. Post-hoc ROC operating points on the CinC 2011 reported subset (exploratory).

| Model Architecture | Target 1: Sens. (Actual FPR) | Threshold τ | Target 2: FPR (Actual Sens.) | Threshold τ |
| :--- | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | 17.33% (FPR 4.00%) | 0.57338 | 46.67% (Sens 81.33%) | 1.74e-04 |
| **1D CNN (Raw Signal)** | 9.33% (FPR 4.00%) | 0.99977 | 37.33% (Sens 80.00%) | 4.44e-04 |
| **Rule-Based SQI Heuristic** | 32.00% (FPR 4.00%) | 0.71544 | 68.00% (Sens 81.33%) | 0.53974 |
| **SinglePass Multimodal Net** | 25.33% (FPR 0.00%) | 1.00000 | trivial (all predicted unusable) | 0.00000 |
| **LightGBM (SQI)** | 41.33% (FPR 1.33%) | 0.36021 | 70.67% (Sens 84.00%) | 0.01955 |
| **Independent SQI MLP** | 25.33% (FPR 0.00%) | 2.98e-34 | trivial (all predicted unusable) | 0.00000 |
| **Logistic Regression (SQI)** | not attainable (scores saturate) | — | trivial (all predicted unusable) | 0.00000 |
| **LiquidAI d1-3B (Zero-Shot)** | 30.67% (FPR 4.00%) | 0.95257 | 73.33% (Sens 81.33%) | 0.84389 |

## Table S16. Feature-set and scaler sensitivity of the tabular models under transfer.

| Feature set | LR · StandardScaler AUROC / AP | LR · RobustScaler AUROC / AP | LR · Z-clip [-3, 3] AUROC / AP | MLP (internal scaler) AUROC / AP |
| :--- | :---: | :---: | :---: | :---: |
| All 13 features | 0.383 / 0.451 | 0.377 / 0.442 | 0.749 / 0.791 | 0.617 / 0.627 |
| Drop signal_std | 0.431 / 0.529 | 0.370 / 0.441 | 0.723 / 0.775 | 0.627 / 0.627 |
| Drop saturation, flatline, constants | 0.746 / 0.796 | 0.746 / 0.805 | 0.751 / 0.792 | 0.750 / 0.749 |
| Drop signal_std, saturation, flatline, constants | 0.713 / 0.770 | 0.721 / 0.781 | 0.723 / 0.774 | 0.729 / 0.762 |

## Table S17. Transfer to five random balanced CinC 2011 subsets.

| Model | AUROC: first-N (reported) | AUROC: random draws, mean ± SD (min–max) | Sensitivity at τ = 0.50: first-N | Sensitivity at τ = 0.50: random draws, mean ± SD | False reject at τ = 0.50: random draws, mean ± SD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.383 | 0.323 ± 0.032 (0.291–0.372) | 74.7% | 67.5 ± 6.8% | 100.0 ± 0.0% |
| **LightGBM (SQI)** | 0.731 | 0.750 ± 0.049 (0.700–0.821) | 9.3% | 16.0 ± 4.7% | 4.3 ± 1.7% |
| **Independent SQI MLP** | 0.617 | 0.670 ± 0.030 (0.628–0.703) | 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% |
| **1D CNN (Raw Signal)** | 0.726 | 0.707 ± 0.062 (0.660–0.816) | 61.3% | 66.7 ± 6.3% | 22.7 ± 3.7% |
| **2D CNN (Rendered Image)** | 0.746 | 0.781 ± 0.064 (0.713–0.884) | 17.3% | 21.6 ± 4.8% | 6.7 ± 3.0% |
| **SinglePass Multimodal Net** | 0.627 | 0.668 ± 0.038 (0.620–0.707) | 25.3% | 33.3 ± 6.9% | 0.3 ± 0.6% |
| **Rule-Based SQI Heuristic** | 0.700 | 0.683 ± 0.046 (0.639–0.737) | 96.0% | 92.5 ± 2.9% | 88.0 ± 2.5% |
| **LiquidAI d1-3B (Zero-Shot)** | 0.636 | 0.650 ± 0.059 (0.583–0.723) | 100.0% | 99.7 ± 0.6% | 100.0 ± 0.0% |

## Table S18. NSTDB: usable probability by nominal SNR.

| SNR Level | Logistic Regression (SQI) | LightGBM (SQI) | 1D CNN (Raw Signal) | 2D CNN (Image) | Multimodal Net | Rule Heuristic Baseline | d1-3B (Zero-Shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean reference (MIT-BIH base)** | 0.967 ± 0.028 | 0.975 ± 0.082 | 0.788 ± 0.326 | 0.685 ± 0.358 | 0.857 ± 0.271 | 0.740 ± 0.011 | 0.777 ± 0.104 |
| **+24 dB (Low Noise)** | 0.900 ± 0.136 | 0.867 ± 0.217 | 0.787 ± 0.314 | 0.575 ± 0.390 | 0.764 ± 0.310 | 0.717 ± 0.026 | 0.759 ± 0.100 |
| **+18 dB** | 0.812 ± 0.159 | 0.538 ± 0.334 | 0.742 ± 0.311 | 0.367 ± 0.390 | 0.391 ± 0.409 | 0.695 ± 0.025 | 0.740 ± 0.074 |
| **+12 dB** | 0.501 ± 0.225 | 0.163 ± 0.169 | 0.521 ± 0.349 | 0.083 ± 0.181 | 0.082 ± 0.187 | 0.653 ± 0.038 | 0.704 ± 0.052 |
| **+6 dB** | 0.176 ± 0.143 | 0.074 ± 0.051 | 0.231 ± 0.307 | 0.019 ± 0.078 | 0.017 ± 0.057 | 0.606 ± 0.044 | 0.685 ± 0.049 |
| **0 dB (Equal Power)** | 0.038 ± 0.046 | 0.072 ± 0.036 | 0.105 ± 0.188 | 0.017 ± 0.064 | 0.006 ± 0.018 | 0.550 ± 0.063 | 0.685 ± 0.051 |
| **-6 dB (Severe Noise)** | 0.011 ± 0.013 | 0.081 ± 0.060 | 0.075 ± 0.139 | 0.012 ± 0.041 | 0.006 ± 0.020 | 0.555 ± 0.070 | 0.699 ± 0.059 |

## Table S19. Prompt sensitivity of zero-shot d1-3B.

| Configuration | BUT QDB AUROC [95% CI] | BUT QDB Sens. / FR at τ = 0.50 | CinC AUROC | CinC Sens. / FR at τ = 0.50 | NSTDB clean P(usable) | NSTDB paired change, +24 dB | NSTDB paired change, 0 dB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary (SQI text + image)** | 0.852 [0.717 – 0.944] | 53.2% / 4.0% | 0.636 | 100.0% / 100.0% | 0.777 | -0.018 | -0.092 |
| **Image only (no SQI text)** | 0.913 [0.855 – 0.972] | 99.7% / 81.3% | 0.680 | 81.3% / 78.7% | 0.459 | -0.037 | -0.039 |
| **Reworded question (SQI text + image)** | 0.869 [0.732 – 0.955] | 17.5% / 0.5% | 0.627 | 100.0% / 98.7% | 0.870 | -0.017 | -0.073 |

## Table S20. Linear probes on frozen d1-3B representations.

| d1-3B representation | BUT QDB AUROC [95% CI] (5-fold subject CV) | CinC 2011 reported subset AUROC | CinC 2011 random subsets AUROC, mean (min–max) | CinC 2017 balanced subset AUROC (recording max) |
| :--- | :---: | :---: | :---: | :---: |
| **Zero-shot choice probability (primary configuration)** | 0.852 [0.717 – 0.944] | 0.636 | 0.650 (0.583–0.723) | 0.739 |
| **Linear probe on the decision state (final norm, answer position)** | 0.974 [0.930 – 0.997] | 0.475 | 0.443 (0.411–0.482) | 0.883 |
| **Linear probe on image tokens (projector output, mean)** | 0.979 [0.941 – 0.996] | 0.777 | 0.825 (0.756–0.907) | 0.910 |

## Table S21. CinC 2017 balanced subset (278 noisy + 278 non-noisy recordings) including zero-shot d1-3B.

| Model Architecture | AUROC [95% CI], recording max (278 noisy + 278 non-noisy) | Sens. / FR at τ = 0.50 |
| :--- | :---: | :---: |
| **Logistic Regression (SQI)** | 0.860 [0.826 – 0.890] | 80.2% / 18.3% |
| **LightGBM (SQI)** | 0.880 [0.849 – 0.905] | 67.6% / 15.5% |
| **Independent SQI MLP** | 0.866 [0.835 – 0.896] | 86.3% / 23.0% |
| **1D CNN (Raw Signal)** | 0.767 [0.720 – 0.814] | 92.4% / 35.3% |
| **2D CNN (Rendered Image)** | 0.859 [0.825 – 0.889] | 89.9% / 32.4% |
| **SinglePass Multimodal Net** | 0.848 [0.812 – 0.880] | 88.8% / 32.0% |
| **Rule-Based SQI Heuristic** | 0.876 [0.843 – 0.905] | 18.3% / 3.2% |
| **LiquidAI d1-3B (Zero-Shot)** | 0.739 [0.697 – 0.778] | 19.8% / 4.0% |

## Table S22. Hand-set rule heuristic versus a logistic-regression replacement using the same seven input features (standardised L2 logistic regression fitted inside the BUT QDB folds and transferred like the other models). The replacement changes the functional form as well as the coefficients (Supplementary Methods S-M3). p: BUT QDB, subject-level cluster bootstrap; external sets, DeLong; unadjusted.

| Evaluation set | Hand-set heuristic AUROC | LR replacement AUROC | Δ (replacement − hand-set) [95% CI] | p | Test |
| :--- | :---: | :---: | :---: | :---: | :--- |
| BUT_QDB (5-fold subject CV) | 0.977 | 0.925 | -0.052 [-0.143 to -0.001] | 0.001 | subject cluster bootstrap |
| CinC2011 lead II (565) | 0.685 | 0.325 | -0.360 [-0.436 to -0.284] | < 0.001 | DeLong |
| CinC2017 (8,512, recording max) | 0.887 | 0.864 | -0.023 [-0.037 to -0.009] | < 0.001 | DeLong |

## Table S23. Seed repetition (seeds 42 and 1–4) of BUT-only training and leave-one-dataset-out training for the seed-dependent networks. Consistent: same sign of Δ in all five seeds and |median Δ| ≥ 0.02 (pre-specified).

| Model | Held-out dataset | Seeds | LODO AUROC, median (range) | BUT-only AUROC, median (range) | Δ, median (range) | Consistent |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Independent SQI MLP** | CinC2011 | 5 | 0.326 (0.315–0.328) | 0.673 (0.672–0.679) | -0.353 (-0.358 to -0.345) | yes |
| **1D CNN (Raw Signal)** | CinC2011 | 5 | 0.780 (0.777–0.785) | 0.713 (0.704–0.722) | +0.066 (+0.060 to +0.076) | yes |
| **2D CNN (Rendered Image)** | CinC2011 | 5 | 0.768 (0.758–0.783) | 0.795 (0.787–0.799) | -0.023 (-0.035 to -0.015) | yes |
| **SinglePass Multimodal Net** | CinC2011 | 5 | 0.518 (0.503–0.769) | 0.500 (0.500–0.674) | +0.003 (-0.156 to +0.269) | no |
| **Independent SQI MLP** | CinC2017 | 5 | 0.873 (0.859–0.876) | 0.865 (0.860–0.868) | +0.010 (-0.006 to +0.016) | no |
| **1D CNN (Raw Signal)** | CinC2017 | 5 | 0.875 (0.854–0.885) | 0.770 (0.765–0.773) | +0.102 (+0.089 to +0.115) | yes |
| **2D CNN (Rendered Image)** | CinC2017 | 5 | 0.776 (0.768–0.802) | 0.848 (0.834–0.850) | -0.069 (-0.082 to -0.048) | yes |
| **SinglePass Multimodal Net** | CinC2017 | 5 | 0.860 (0.844–0.866) | 0.791 (0.764–0.835) | +0.053 (+0.022 to +0.098) | yes |
| **Independent SQI MLP** | BUT_QDB | 5 | 0.980 (0.979–0.983) | — | — | — |
| **1D CNN (Raw Signal)** | BUT_QDB | 5 | 0.989 (0.985–0.996) | — | — | — |
| **2D CNN (Rendered Image)** | BUT_QDB | 5 | 0.962 (0.705–0.989) | — | — | — |
| **SinglePass Multimodal Net** | BUT_QDB | 5 | 0.813 (0.741–0.992) | — | — | — |

## Table S24. Local recalibration on n = 25–200 randomly drawn labelled target recordings (500 draws each), evaluated on the remaining recordings. Prevalence-only Brier score: CinC 2011 0.227, CinC 2017 0.032.

| Dataset | Model | n | Transferred threshold: FPR / Sens. (median) | Local threshold: FPR median [95% range] | Local threshold: Sens. median [95% range] | Draws with FPR ≤ 5% / ≤ 7.5% (local) | Fallback draws | Brier: raw → Platt (median; Platt skipped) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CinC2011 | **Logistic Regression (SQI)** | 25 | 100.0% / 66.5% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 0 | 0.769 → 0.175 (0) |
| CinC2011 | **Logistic Regression (SQI)** | 50 | 100.0% / 66.5% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 0 | 0.767 → 0.172 (0) |
| CinC2011 | **Logistic Regression (SQI)** | 100 | 100.0% / 66.5% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 0 | 0.768 → 0.171 (0) |
| CinC2011 | **Logistic Regression (SQI)** | 200 | 100.0% / 66.2% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 0 | 0.770 → 0.169 (0) |
| CinC2011 | **LightGBM (SQI)** | 25 | 4.6% / 51.3% | 3.3% [0.0–14.6%] | 14.7% [0.0–59.5%] | 85% / 89% | 0 | 0.219 → 0.188 (0) |
| CinC2011 | **LightGBM (SQI)** | 50 | 4.7% / 51.4% | 4.7% [0.0–11.0%] | 49.0% [0.0–57.6%] | 73% / 86% | 0 | 0.219 → 0.183 (0) |
| CinC2011 | **LightGBM (SQI)** | 100 | 4.7% / 51.2% | 4.7% [0.3–11.7%] | 50.0% [6.3–57.7%] | 65% / 82% | 0 | 0.219 → 0.180 (0) |
| CinC2011 | **LightGBM (SQI)** | 200 | 4.6% / 51.2% | 4.6% [0.8–10.6%] | 48.5% [10.6–56.3%] | 66% / 86% | 0 | 0.218 → 0.178 (0) |
| CinC2011 | **Independent SQI MLP** | 25 | 0.0% / 0.0% | 0.9% [0.0–0.9%] | 36.0% [0.0–37.4%] | 100% / 100% | 0 | 0.348 → 0.233 (0) |
| CinC2011 | **Independent SQI MLP** | 50 | 0.0% / 0.0% | 0.9% [0.0–0.9%] | 36.0% [1.7–38.1%] | 100% / 100% | 0 | 0.349 → 0.229 (0) |
| CinC2011 | **Independent SQI MLP** | 100 | 0.0% / 0.0% | 1.0% [0.3–1.0%] | 36.2% [32.7–38.9%] | 100% / 100% | 0 | 0.348 → 0.228 (0) |
| CinC2011 | **Independent SQI MLP** | 200 | 0.0% / 0.0% | 0.8% [0.0–1.3%] | 36.3% [31.5–41.2%] | 100% / 100% | 0 | 0.348 → 0.227 (0) |
| CinC2011 | **1D CNN (Raw Signal)** | 25 | 30.5% / 76.1% | 0.0% [0.0–17.1%] | 0.0% [0.0–23.7%] | 77% / 80% | 0 | 0.242 → 0.210 (0) |
| CinC2011 | **1D CNN (Raw Signal)** | 50 | 30.5% / 76.1% | 0.0% [0.0–14.2%] | 0.0% [0.0–19.6%] | 75% / 83% | 0 | 0.242 → 0.205 (0) |
| CinC2011 | **1D CNN (Raw Signal)** | 100 | 30.5% / 76.1% | 2.6% [0.0–13.0%] | 1.2% [0.0–16.0%] | 67% / 81% | 0 | 0.243 → 0.202 (0) |
| CinC2011 | **1D CNN (Raw Signal)** | 200 | 30.5% / 76.0% | 4.0% [0.0–9.9%] | 1.6% [0.0–8.7%] | 68% / 84% | 0 | 0.242 → 0.200 (0) |
| CinC2011 | **2D CNN (Rendered Image)** | 25 | 10.6% / 60.3% | 2.3% [0.0–15.6%] | 9.1% [0.0–63.5%] | 68% / 89% | 0 | 0.244 → 0.183 (0) |
| CinC2011 | **2D CNN (Rendered Image)** | 50 | 10.6% / 60.3% | 3.6% [0.0–11.0%] | 13.9% [0.0–60.7%] | 68% / 81% | 0 | 0.245 → 0.178 (0) |
| CinC2011 | **2D CNN (Rendered Image)** | 100 | 10.7% / 60.6% | 4.6% [0.7–11.3%] | 18.3% [1.9–59.9%] | 61% / 74% | 0 | 0.244 → 0.174 (0) |
| CinC2011 | **2D CNN (Rendered Image)** | 200 | 10.5% / 60.3% | 4.6% [1.3–9.9%] | 17.8% [8.3–56.3%] | 63% / 88% | 0 | 0.245 → 0.173 (0) |
| CinC2011 | **SinglePass Multimodal Net** | 25 | 0.3% / 34.0% | 0.3% [0.0–0.3%] | 34.0% [0.0–35.5%] | 100% / 100% | 0 | 0.231 → 0.177 (0) |
| CinC2011 | **SinglePass Multimodal Net** | 50 | 0.3% / 34.1% | 0.3% [0.0–0.3%] | 34.1% [31.6–36.1%] | 100% / 100% | 0 | 0.233 → 0.174 (0) |
| CinC2011 | **SinglePass Multimodal Net** | 100 | 0.3% / 34.1% | 0.3% [0.0–0.3%] | 34.2% [31.0–37.2%] | 100% / 100% | 0 | 0.232 → 0.173 (0) |
| CinC2011 | **SinglePass Multimodal Net** | 200 | 0.4% / 34.3% | 0.4% [0.0–0.4%] | 34.4% [29.4–39.2%] | 100% / 100% | 0 | 0.230 → 0.172 (0) |
| CinC2011 | **Rule-Based SQI Heuristic** | 25 | 100.0% / 100.0% | 0.0% [0.0–15.2%] | 34.0% [31.7–50.9%] | 83% / 88% | 0 | 0.278 → 0.202 (0) |
| CinC2011 | **Rule-Based SQI Heuristic** | 50 | 100.0% / 100.0% | 2.4% [0.0–13.6%] | 36.2% [31.9–49.7%] | 76% / 86% | 0 | 0.277 → 0.198 (0) |
| CinC2011 | **Rule-Based SQI Heuristic** | 100 | 100.0% / 100.0% | 3.7% [0.0–12.5%] | 40.3% [32.0–47.8%] | 66% / 83% | 0 | 0.278 → 0.195 (0) |
| CinC2011 | **Rule-Based SQI Heuristic** | 200 | 100.0% / 100.0% | 4.6% [1.3–10.3%] | 41.4% [34.0–47.7%] | 60% / 91% | 0 | 0.278 → 0.194 (0) |
| CinC2011 | **LiquidAI d1-3B (Zero-Shot)** | 25 | 100.0% / 99.5% | 0.0% [0.0–11.4%] | 33.3% [22.1–43.1%] | 92% / 96% | 0 | 0.511 → 0.207 (0) |
| CinC2011 | **LiquidAI d1-3B (Zero-Shot)** | 50 | 100.0% / 99.4% | 0.0% [0.0–11.6%] | 36.3% [23.2–42.5%] | 90% / 96% | 0 | 0.510 → 0.200 (0) |
| CinC2011 | **LiquidAI d1-3B (Zero-Shot)** | 100 | 100.0% / 99.4% | 2.7% [0.0–7.9%] | 38.5% [31.4–43.4%] | 82% / 93% | 0 | 0.510 → 0.199 (0) |
| CinC2011 | **LiquidAI d1-3B (Zero-Shot)** | 200 | 100.0% / 99.2% | 3.7% [0.0–8.8%] | 39.4% [34.0–45.0%] | 82% / 87% | 0 | 0.511 → 0.197 (0) |
| CinC2011 | **d1-3B image-token probe** | 25 | 11.7% / 64.9% | 0.6% [0.0–13.6%] | 46.8% [32.7–66.8%] | 82% / 88% | 0 | 0.161 → 0.148 (0) |
| CinC2011 | **d1-3B image-token probe** | 50 | 11.7% / 65.0% | 2.1% [0.0–12.5%] | 51.1% [33.7–64.4%] | 77% / 87% | 0 | 0.161 → 0.144 (0) |
| CinC2011 | **d1-3B image-token probe** | 100 | 11.7% / 65.0% | 3.6% [0.2–11.7%] | 53.5% [44.5–63.5%] | 70% / 85% | 0 | 0.161 → 0.142 (0) |
| CinC2011 | **d1-3B image-token probe** | 200 | 11.7% / 65.0% | 3.9% [0.4–9.2%] | 54.7% [47.4–61.5%] | 70% / 90% | 0 | 0.160 → 0.141 (0) |
| CinC2011 | **TabPFN (SQI)** | 25 | 24.2% / 61.4% | 3.1% [0.0–13.2%] | 42.8% [0.0–56.4%] | 78% / 88% | 0 | 0.213 → 0.194 (0) |
| CinC2011 | **TabPFN (SQI)** | 50 | 24.2% / 61.4% | 3.0% [0.0–12.6%] | 43.1% [0.0–55.3%] | 74% / 89% | 0 | 0.214 → 0.188 (0) |
| CinC2011 | **TabPFN (SQI)** | 100 | 24.3% / 61.4% | 3.7% [0.2–11.1%] | 44.4% [3.1–53.2%] | 61% / 90% | 0 | 0.214 → 0.186 (0) |
| CinC2011 | **TabPFN (SQI)** | 200 | 24.2% / 61.3% | 4.1% [1.2–8.4%] | 44.8% [7.8–51.6%] | 60% / 88% | 0 | 0.213 → 0.184 (0) |
| CinC2011 | **HuBERT-ECG probe** | 25 | 40.0% / 48.7% | 0.6% [0.0–20.8%] | 0.0% [0.0–30.3%] | 77% / 82% | 0 | 0.319 → 0.241 (0) |
| CinC2011 | **HuBERT-ECG probe** | 50 | 39.9% / 48.9% | 0.9% [0.0–14.7%] | 2.8% [0.0–22.9%] | 71% / 83% | 0 | 0.319 → 0.233 (0) |
| CinC2011 | **HuBERT-ECG probe** | 100 | 40.0% / 48.8% | 3.6% [0.0–13.2%] | 4.9% [0.0–18.9%] | 64% / 78% | 0 | 0.319 → 0.228 (0) |
| CinC2011 | **HuBERT-ECG probe** | 200 | 39.9% / 48.4% | 3.8% [0.4–10.3%] | 5.5% [1.5–12.2%] | 73% / 86% | 0 | 0.319 → 0.226 (0) |
| CinC2017 | **Logistic Regression (SQI)** | 25 | 15.7% / 74.5% | 8.1% [0.0–15.8%] | 55.7% [0.0–74.5%] | 43% / 49% | 229 | 0.145 → 0.036 (229) |
| CinC2017 | **Logistic Regression (SQI)** | 50 | 15.7% / 74.4% | 2.8% [0.0–15.8%] | 17.7% [0.0–74.5%] | 69% / 79% | 82 | 0.145 → 0.032 (82) |
| CinC2017 | **Logistic Regression (SQI)** | 100 | 15.7% / 74.5% | 2.5% [0.0–15.7%] | 15.6% [0.0–74.5%] | 83% / 94% | 21 | 0.145 → 0.031 (21) |
| CinC2017 | **Logistic Regression (SQI)** | 200 | 15.7% / 74.5% | 3.4% [0.0–7.1%] | 25.0% [0.0–52.6%] | 83% / 98% | 0 | 0.145 → 0.030 (0) |
| CinC2017 | **LightGBM (SQI)** | 25 | 19.2% / 79.5% | 8.4% [0.0–19.2%] | 49.2% [0.0–79.5%] | 46% / 49% | 229 | 0.105 → 0.034 (229) |
| CinC2017 | **LightGBM (SQI)** | 50 | 19.2% / 79.4% | 1.4% [0.0–19.2%] | 17.9% [0.0–79.5%] | 76% / 80% | 82 | 0.105 → 0.030 (82) |
| CinC2017 | **LightGBM (SQI)** | 100 | 19.2% / 79.5% | 1.7% [0.0–19.1%] | 20.9% [0.0–79.5%] | 88% / 93% | 21 | 0.105 → 0.029 (21) |
| CinC2017 | **LightGBM (SQI)** | 200 | 19.2% / 79.5% | 2.6% [0.0–6.8%] | 28.4% [0.0–44.2%] | 93% / 99% | 0 | 0.105 → 0.029 (0) |
| CinC2017 | **Independent SQI MLP** | 25 | 16.5% / 74.5% | 9.6% [0.0–16.6%] | 50.0% [0.0–74.5%] | 43% / 47% | 229 | 0.175 → 0.034 (229) |
| CinC2017 | **Independent SQI MLP** | 50 | 16.5% / 74.5% | 1.6% [0.0–16.6%] | 6.3% [0.0–74.5%] | 71% / 78% | 82 | 0.175 → 0.031 (82) |
| CinC2017 | **Independent SQI MLP** | 100 | 16.5% / 74.5% | 1.8% [0.0–16.5%] | 7.3% [0.0–74.5%] | 83% / 93% | 21 | 0.175 → 0.031 (21) |
| CinC2017 | **Independent SQI MLP** | 200 | 16.5% / 74.5% | 3.2% [0.0–7.4%] | 18.1% [0.0–40.0%] | 84% / 97% | 0 | 0.175 → 0.030 (0) |
| CinC2017 | **1D CNN (Raw Signal)** | 25 | 54.5% / 99.6% | 3.5% [0.0–54.6%] | 0.2% [0.0–99.6%] | 50% / 50% | 229 | 0.308 → 0.033 (229) |
| CinC2017 | **1D CNN (Raw Signal)** | 50 | 54.5% / 99.6% | 0.0% [0.0–54.6%] | 0.0% [0.0–99.6%] | 83% / 83% | 82 | 0.308 → 0.032 (82) |
| CinC2017 | **1D CNN (Raw Signal)** | 100 | 54.5% / 99.6% | 0.0% [0.0–54.5%] | 0.0% [0.0–99.6%] | 95% / 96% | 21 | 0.308 → 0.032 (21) |
| CinC2017 | **1D CNN (Raw Signal)** | 200 | 54.5% / 99.6% | 0.0% [0.0–1.9%] | 0.0% [0.0–0.0%] | 99% / 100% | 0 | 0.308 → 0.032 (0) |
| CinC2017 | **2D CNN (Rendered Image)** | 25 | 50.4% / 96.4% | 8.8% [0.0–50.4%] | 41.4% [0.0–96.4%] | 44% / 48% | 229 | 0.284 → 0.035 (229) |
| CinC2017 | **2D CNN (Rendered Image)** | 50 | 50.4% / 96.4% | 0.5% [0.0–50.4%] | 4.3% [0.0–96.4%] | 72% / 78% | 82 | 0.284 → 0.032 (82) |
| CinC2017 | **2D CNN (Rendered Image)** | 100 | 50.4% / 96.4% | 0.4% [0.0–50.3%] | 3.3% [0.0–96.4%] | 86% / 93% | 21 | 0.284 → 0.031 (21) |
| CinC2017 | **2D CNN (Rendered Image)** | 200 | 50.4% / 96.3% | 2.1% [0.0–7.1%] | 14.6% [0.0–34.3%] | 86% / 99% | 0 | 0.284 → 0.030 (0) |
| CinC2017 | **SinglePass Multimodal Net** | 25 | 43.1% / 92.8% | 8.7% [0.0–43.1%] | 33.2% [0.0–92.8%] | 45% / 49% | 229 | 0.303 → 0.034 (229) |
| CinC2017 | **SinglePass Multimodal Net** | 50 | 43.1% / 92.8% | 0.0% [0.0–43.1%] | 0.0% [0.0–92.8%] | 74% / 81% | 82 | 0.303 → 0.032 (82) |
| CinC2017 | **SinglePass Multimodal Net** | 100 | 43.1% / 92.8% | 0.0% [0.0–43.0%] | 0.0% [0.0–92.8%] | 87% / 94% | 21 | 0.303 → 0.031 (21) |
| CinC2017 | **SinglePass Multimodal Net** | 200 | 43.1% / 92.7% | 2.6% [0.0–7.0%] | 9.9% [0.0–28.9%] | 86% / 98% | 0 | 0.303 → 0.031 (0) |
| CinC2017 | **Rule-Based SQI Heuristic** | 25 | 6.2% / 52.9% | 6.2% [0.0–8.7%] | 52.9% [0.0–70.1%] | 41% / 95% | 229 | 0.095 → 0.037 (229) |
| CinC2017 | **Rule-Based SQI Heuristic** | 50 | 6.2% / 52.9% | 2.9% [0.0–8.6%] | 20.6% [0.0–69.5%] | 65% / 94% | 82 | 0.095 → 0.033 (82) |
| CinC2017 | **Rule-Based SQI Heuristic** | 100 | 6.2% / 52.9% | 2.2% [0.0–7.5%] | 13.5% [0.0–62.4%] | 77% / 97% | 21 | 0.095 → 0.031 (21) |
| CinC2017 | **Rule-Based SQI Heuristic** | 200 | 6.2% / 52.9% | 3.6% [0.0–7.6%] | 25.0% [0.0–62.4%] | 76% / 97% | 0 | 0.095 → 0.031 (0) |
| CinC2017 | **LiquidAI d1-3B (Zero-Shot)** | 25 | 3.1% / 19.8% | 3.1% [0.0–7.0%] | 19.8% [0.0–33.2%] | 96% / 98% | 229 | 0.091 → 0.034 (229) |
| CinC2017 | **LiquidAI d1-3B (Zero-Shot)** | 50 | 3.1% / 19.9% | 0.0% [0.0–7.0%] | 0.0% [0.0–33.3%] | 94% / 99% | 82 | 0.091 → 0.032 (82) |
| CinC2017 | **LiquidAI d1-3B (Zero-Shot)** | 100 | 3.1% / 19.8% | 0.5% [0.0–7.0%] | 5.5% [0.0–33.2%] | 95% / 100% | 21 | 0.091 → 0.031 (21) |
| CinC2017 | **LiquidAI d1-3B (Zero-Shot)** | 200 | 3.1% / 19.8% | 2.1% [0.0–7.0%] | 16.3% [0.0–33.3%] | 95% / 100% | 0 | 0.091 → 0.031 (0) |
| CinC2017 | **TabPFN (SQI)** | 25 | 29.2% / 87.8% | 9.6% [0.0–29.2%] | 54.1% [0.0–87.8%] | 45% / 48% | 229 | 0.109 → 0.035 (229) |
| CinC2017 | **TabPFN (SQI)** | 50 | 29.2% / 87.7% | 1.1% [0.0–29.2%] | 13.3% [0.0–87.8%] | 71% / 79% | 82 | 0.109 → 0.031 (82) |
| CinC2017 | **TabPFN (SQI)** | 100 | 29.2% / 87.7% | 1.1% [0.0–29.2%] | 13.1% [0.0–87.8%] | 87% / 94% | 21 | 0.109 → 0.030 (21) |
| CinC2017 | **TabPFN (SQI)** | 200 | 29.2% / 87.7% | 2.8% [0.0–6.8%] | 23.8% [0.0–42.8%] | 86% / 99% | 0 | 0.109 → 0.029 (0) |
| CinC2017 | **HuBERT-ECG probe** | 25 | 51.6% / 99.6% | 5.6% [0.0–51.7%] | 53.7% [0.0–99.6%] | 49% / 51% | 229 | 0.175 → 0.033 (229) |
| CinC2017 | **HuBERT-ECG probe** | 50 | 51.6% / 99.6% | 1.9% [0.0–51.7%] | 25.7% [0.0–99.6%] | 72% / 81% | 82 | 0.175 → 0.029 (82) |
| CinC2017 | **HuBERT-ECG probe** | 100 | 51.6% / 99.6% | 2.2% [0.0–51.6%] | 28.5% [0.0–99.6%] | 87% / 95% | 21 | 0.175 → 0.028 (21) |
| CinC2017 | **HuBERT-ECG probe** | 200 | 51.6% / 99.6% | 3.2% [0.0–6.8%] | 36.4% [0.0–59.1%] | 88% / 99% | 0 | 0.175 → 0.028 (0) |
| CinC2017 | **d1-3B image-token probe** | 25 | 51.1% / 98.9% | 8.3% [0.0–51.2%] | 65.3% [0.0–98.9%] | 43% / 49% | 229 | 0.162 → 0.034 (229) |
| CinC2017 | **d1-3B image-token probe** | 50 | 51.1% / 98.9% | 1.8% [0.0–51.2%] | 27.8% [0.0–98.9%] | 71% / 80% | 82 | 0.162 → 0.030 (82) |
| CinC2017 | **d1-3B image-token probe** | 100 | 51.1% / 98.9% | 1.7% [0.0–51.1%] | 25.3% [0.0–98.9%] | 83% / 94% | 21 | 0.162 → 0.028 (21) |
| CinC2017 | **d1-3B image-token probe** | 200 | 51.1% / 98.9% | 3.1% [0.0–7.2%] | 36.9% [0.0–59.9%] | 81% / 99% | 0 | 0.162 → 0.028 (0) |

## Table S25. HuBERT-ECG: preprocessing sensitivity and partial fine-tuning (AUROC). Fine-tuning was not run in LODO.

| HuBERT-ECG variant | BUT QDB (5-fold subject CV) | CinC 2011 (BUT-only) | CinC 2017 (BUT-only) | LODO: BUT QDB | LODO: CinC 2011 | LODO: CinC 2017 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Probe, authors' preprocessing (primary)** | 0.972 | 0.562 | 0.907 | 0.982 | 0.804 | 0.871 |
| **Probe, our first preprocessing (sensitivity)** | 0.987 | 0.732 | 0.867 | 0.995 | 0.786 | 0.868 |
| **Fine-tuned (last 4 layers + head), authors' preprocessing** | 0.969 | 0.706 | 0.912 | — | — | — |

## Table S26. BUT QDB official protocol (Smital et al. 2026): sensitivity analyses. F1a: pooled out-of-fold AUROC without the 59 windows from the excluded stretch of record 105001 (evaluation only). F1b: retrained on the official training records, tested on the official test records (window-level binary task). 95% CI: subject-level cluster bootstrap; p_Holm: paired comparison with the best model.

| Model | F1a: 5-fold CV without the excluded 105001 stretch (988 windows, 238 unusable, 15 subjects) | F1b: official train/test split, test set (442 windows, 92 unusable, 7 subjects) |
| :--- | :---: | :---: |
| **Logistic Regression (SQI)** | 0.923 [0.798–0.991]; p_Holm = 0.343 | 0.980 [0.957–1.000]; p_Holm = 1.000 |
| **LightGBM (SQI)** | 0.955 [0.873–0.999]; p_Holm = 1.000 | 0.992 [0.940–1.000]; p_Holm = 1.000 |
| **Independent SQI MLP** | 0.924 [0.805–0.985]; p_Holm = 0.204 | 0.981 [0.960–1.000]; p_Holm = 1.000 |
| **1D CNN (Raw Signal)** | **0.987 [0.956–0.999]** (best) | **1.000 [1.000–1.000]** (best) |
| **2D CNN (Rendered Image)** | 0.986 [0.958–0.999]; p_Holm = 1.000 | 0.961 [0.933–1.000]; p_Holm = 1.000 |
| **SinglePass Multimodal Net** | 0.959 [0.903–0.998]; p_Holm = 1.000 | 0.878 [0.784–1.000]; p_Holm = 1.000 |
| **Rule-Based SQI Heuristic** | 0.971 [0.908–1.000]; p_Holm = 1.000 | 0.993 [0.971–1.000]; p_Holm = 1.000 |
| **LiquidAI d1-3B (Zero-Shot)** | 0.828 [0.685–0.939]; p_Holm = 0.368 | 0.723 [0.576–1.000]; p_Holm = 1.000 |
| **TabPFN (SQI)** | 0.953 [0.885–0.996]; p_Holm = 1.000 | not run |
| **HuBERT-ECG probe** | 0.965 [0.887–0.998]; p_Holm = 1.000 | 1.000 [1.000–1.000]; p_Holm = 1.000 |
| **d1-3B image-token probe** | 0.976 [0.936–0.996]; p_Holm = 1.000 | 0.991 [0.984–1.000]; p_Holm = 1.000 |

## Table S27. Latency of the foundation-model probes and TabPFN (Apple M4; same 25 windows and three runs as main-text Table 5, harness of Table S8).

| Step | Median (three runs) | Range of run medians | p95 (median over runs) |
| :--- | :---: | :---: | :---: |
| HuBERT-ECG: preprocessing (authors' pipeline) | 4.42 ms | 4.37–4.43 ms | 4.54 ms |
| HuBERT-ECG: forward pass, CPU | 59.44 ms | 59.25–60.75 ms | 64.32 ms |
| HuBERT-ECG: forward pass, MPS | 23.39 ms | 23.26–23.58 ms | 24.09 ms |
| HuBERT-ECG: linear probe | 0.30 ms | 0.29–0.31 ms | 0.35 ms |
| HuBERT-ECG probe: total, CPU | 64.10 ms | 63.92–65.74 ms | 69.61 ms |
| HuBERT-ECG probe: total, MPS | 28.02 ms | 27.92–28.42 ms | 28.97 ms |
| d1-3B image probe: linear probe (added to the d1-3B forward pass of main-text Table 5) | 0.13 ms | 0.12–0.13 ms | 0.17 ms |
| TabPFN: fit on 1,047 windows (once per model) | 186.73 ms | 135.93–490.23 ms | 186.73 ms |
| TabPFN: prediction of one window, CPU | 2503.49 ms | 2364.70–2507.85 ms | 2578.09 ms |

## Table S28. CinC 2011 records with a constant lead II (65 of 197 unacceptable records; post hoc, descriptive). Number flagged at τ = 0.50 and AUROC with and without these records; models trained on BUT QDB only.

| Model | Constant-lead records flagged at τ = 0.50 | AUROC, all 565 records | AUROC without the 65 constant-lead records |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0 / 65 | 0.320 | 0.478 |
| **LightGBM (SQI)** | 0 / 65 | 0.754 | 0.655 |
| **Independent SQI MLP** | 0 / 65 | 0.672 | 0.515 |
| **1D CNN (Raw Signal)** | 64 / 65 | 0.719 | 0.681 |
| **2D CNN (Rendered Image)** | 1 / 65 | 0.787 | 0.716 |
| **SinglePass Multimodal Net** | 65 / 65 | 0.674 | 0.514 |
| **Rule-Based SQI Heuristic** | 65 / 65 | 0.685 | 0.530 |
| **LiquidAI d1-3B (Zero-Shot)** | 65 / 65 | 0.650 | 0.477 |
| **TabPFN (SQI)** | 0 / 65 | 0.749 | 0.640 |
| **HuBERT-ECG probe** | 0 / 65 | 0.562 | 0.692 |
| **d1-3B image-token probe** | 65 / 65 | 0.833 | 0.751 |

## Table S29. CinC 2017 at natural prevalence with bootstrap confidence intervals, seven main models (recording maximum; mean in the last column).

| Model Architecture | AUROC [95% CI], recording max | AP [95% CI], recording max (prevalence 3.3%) | Sens. / FR at τ = 0.50 | PPV / NPV at τ = 0.50 | AUROC, recording mean (secondary) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.868 [0.848 – 0.886] | 0.173 [0.146 – 0.207] | 80.2% / 18.9% | 0.126 / 0.992 | 0.916 |
| **LightGBM (SQI)** | 0.880 [0.863 – 0.895] | 0.234 [0.187 – 0.285] | 67.6% / 14.6% | 0.135 / 0.987 | 0.930 |
| **Independent SQI MLP** | 0.868 [0.852 – 0.883] | 0.140 [0.120 – 0.171] | 86.3% / 24.0% | 0.108 / 0.994 | 0.934 |
| **1D CNN (Raw Signal)** | 0.773 [0.760 – 0.786] | 0.064 [0.057 – 0.073] | 92.4% / 34.5% | 0.083 / 0.996 | 0.801 |
| **2D CNN (Rendered Image)** | 0.850 [0.832 – 0.867] | 0.137 [0.115 – 0.172] | 89.9% / 33.8% | 0.082 / 0.995 | 0.879 |
| **SinglePass Multimodal Net** | 0.835 [0.818 – 0.852] | 0.107 [0.092 – 0.128] | 88.8% / 35.7% | 0.078 / 0.994 | 0.888 |
| **Rule-Based SQI Heuristic** | 0.887 [0.867 – 0.905] | 0.182 [0.155 – 0.217] | 18.3% / 2.5% | 0.199 / 0.973 | 0.933 |

## Table S30. Discrimination of every model in each primary evaluation set: AUROC and average precision with 95% bootstrap confidence intervals (2,000 resamples, seed 42; resampling unit subject on BUT QDB and recording externally). Models trained on BUT QDB only; external sets scored with the pre-specified recording maximum. Intervals for the seven main models differ slightly from Tables S1 and S29, which used other resampling runs.

| Model | BUT_QDB (5-fold subject CV) | CinC2011 lead II (565 records) | CinC2017 (8,512 recordings) | CinC2017 balanced subset (556 recordings) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | AUROC 0.938 [0.828–0.994]; AP 0.888 [0.598–0.984] | AUROC 0.320 [0.283–0.354]; AP 0.287 [0.251–0.321] | AUROC 0.868 [0.848–0.886]; AP 0.173 [0.145–0.209] | AUROC 0.860 [0.826–0.891]; AP 0.818 [0.762–0.871] |
| **LightGBM (SQI)** | AUROC 0.964 [0.887–1.000]; AP 0.928 [0.662–0.998] | AUROC 0.754 [0.708–0.801]; AP 0.680 [0.610–0.751] | AUROC 0.880 [0.864–0.895]; AP 0.234 [0.189–0.287] | AUROC 0.880 [0.849–0.906]; AP 0.856 [0.807–0.896] |
| **Independent SQI MLP** | AUROC 0.939 [0.838–0.990]; AP 0.880 [0.542–0.980] | AUROC 0.672 [0.639–0.709]; AP 0.568 [0.515–0.624] | AUROC 0.868 [0.852–0.884]; AP 0.140 [0.120–0.170] | AUROC 0.866 [0.834–0.897]; AP 0.807 [0.748–0.863] |
| **1D CNN (Raw Signal)** | AUROC 0.989 [0.967–0.999]; AP 0.975 [0.898–0.997] | AUROC 0.719 [0.674–0.763]; AP 0.508 [0.450–0.579] | AUROC 0.773 [0.760–0.786]; AP 0.064 [0.057–0.072] | AUROC 0.767 [0.719–0.812]; AP 0.624 [0.571–0.685] |
| **2D CNN (Rendered Image)** | AUROC 0.989 [0.966–0.999]; AP 0.971 [0.904–0.997] | AUROC 0.787 [0.744–0.828]; AP 0.655 [0.588–0.732] | AUROC 0.850 [0.832–0.867]; AP 0.137 [0.115–0.170] | AUROC 0.859 [0.825–0.889]; AP 0.816 [0.762–0.868] |
| **SinglePass Multimodal Net** | AUROC 0.967 [0.916–0.999]; AP 0.928 [0.754–0.996] | AUROC 0.674 [0.642–0.707]; AP 0.572 [0.518–0.623] | AUROC 0.835 [0.818–0.852]; AP 0.107 [0.092–0.128] | AUROC 0.848 [0.813–0.879]; AP 0.799 [0.741–0.857] |
| **Rule-Based SQI Heuristic** | AUROC 0.977 [0.917–1.000]; AP 0.948 [0.741–0.999] | AUROC 0.685 [0.631–0.735]; AP 0.683 [0.624–0.737] | AUROC 0.887 [0.868–0.905]; AP 0.182 [0.155–0.218] | AUROC 0.876 [0.842–0.905]; AP 0.839 [0.785–0.889] |
| **LiquidAI d1-3B (Zero-Shot)** | AUROC 0.852 [0.728–0.943]; AP 0.740 [0.419–0.899] | AUROC 0.650 [0.596–0.701]; AP 0.651 [0.591–0.708] | AUROC 0.757 [0.729–0.786]; AP 0.113 [0.092–0.149] | AUROC 0.739 [0.697–0.777]; AP 0.715 [0.661–0.768] |
| **TabPFN (SQI)** | AUROC 0.962 [0.898–0.997]; AP 0.913 [0.664–0.985] | AUROC 0.749 [0.702–0.793]; AP 0.687 [0.617–0.748] | AUROC 0.858 [0.836–0.878]; AP 0.210 [0.168–0.257] | AUROC 0.861 [0.828–0.889]; AP 0.845 [0.800–0.886] |
| **HuBERT-ECG probe** | AUROC 0.972 [0.901–0.998]; AP 0.936 [0.729–0.995] | AUROC 0.562 [0.512–0.610]; AP 0.424 [0.370–0.494] | AUROC 0.907 [0.893–0.921]; AP 0.262 [0.218–0.318] | AUROC 0.916 [0.892–0.937]; AP 0.909 [0.875–0.936] |
| **d1-3B image-token probe** | AUROC 0.979 [0.945–0.996]; AP 0.934 [0.762–0.989] | AUROC 0.833 [0.795–0.869]; AP 0.809 [0.763–0.852] | AUROC 0.908 [0.894–0.921]; AP 0.264 [0.222–0.317] | AUROC 0.910 [0.883–0.932]; AP 0.900 [0.865–0.930] |

## Table S31. Distribution of each SQI by dataset and class: median [interquartile range] per window. CinC 2017 windows carry the label of their recording.

| SQI | BUT QDB usable (n = 750) | BUT QDB unusable (n = 297) | CinC 2011 acceptable (n = 368) | CinC 2011 unacceptable (n = 197) | CinC 2017 not noisy (n = 23,396) | CinC 2017 noisy (n = 589) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bsqi_detector_agreement` | 0.952 [0.88–1] | 0.364 [0.275–0.511] | 0.563 [0.438–0.667] | 0.364 [0–0.583] | 0.947 [0.903–1] | 0.667 [0.5–0.811] |
| `baseline_wander_ratio` | 0.00274 [0.00108–0.011] | 0.00193 [0.000297–0.0111] | 0.00322 [0.00136–0.00889] | 0.00271 [0–0.0135] | 0.00918 [0.00349–0.0254] | 0.0254 [0.0114–0.0457] |
| `high_frequency_noise_ratio` | 0.0081 [0.00509–0.0136] | 1.03 [0.00948–1.86] | 0.0132 [0.00679–0.0309] | 0.00523 [0–0.0174] | 0.000456 [0.000243–0.000803] | 0.000341 [0.000102–0.0014] |
| `saturation_fraction` | 0.0004 [0.0004–0.0004] | 0.0004 [0.0004–0.0004] | 0.0006 [0.0004–0.00065] | 0.0006 [0.0004–1] | 0.0004 [0.0004–0.0004] | 0.0004 [0.0004–0.0004] |
| `flatline_fraction` | 0 [0–0.0002] | 0 [0–0] | 0.282 [0.205–0.361] | 0.35 [0.231–1] | 0.0006 [0.0002–0.001] | 0.0002 [0–0.0004] |
| `qrs_energy_ratio` | 0.427 [0.364–0.473] | 0.266 [0.0677–0.332] | 0.436 [0.326–0.519] | 0.145 [0–0.384] | 0.366 [0.244–0.468] | 0.134 [0.0701–0.266] |
| `kurtosis_ksqi` | 14.6 [4.27–21.4] | 0.973 [-0.0886–5.44] | 10.1 [4.76–16.5] | 2.29 [0–9.45] | 7.75 [3.83–13.5] | 3.88 [1.36–8.82] |
| `skewness_ssqi` | 3.32 [0.862–4.21] | -0.0218 [-0.166–0.0235] | 2.03 [0.391–3.36] | 0 [0–1.85] | 1.44 [-0.0948–2.81] | 0.00185 [-0.95–0.793] |
| `zero_crossing_rate` | 0.0324 [0.0216–0.0442] | 0.205 [0.0262–0.269] | 0.0754 [0.0555–0.101] | 0.0364 [0–0.0736] | 0.0156 [0.0126–0.019] | 0.0152 [0.0106–0.0244] |
| `powerline_noise_ratio` | 0.000968 [0.000607–0.002] | 0.04 [0.00119–0.373] | 0.00149 [0.000669–0.00371] | 0.000558 [0–0.00186] | 3.37e-05 [1.87e-05–5.83e-05] | 2.92e-05 [9.01e-06–0.000123] |
| `signal_std` | 0.312 [0.253–0.412] | 0.177 [0.0443–0.646] | 0.124 [0.0929–0.164] | 0.0955 [0–0.204] | 0.163 [0.123–0.22] | 0.334 [0.186–0.54] |

## Table S32. Univariate AUROC of each SQI on BUT QDB and on CinC 2011 lead II with and without the 65 records whose lead II is constant (post hoc, descriptive). Values below 0.5: higher feature value associated with usable.

| SQI feature | BUT QDB (1,047 windows) | CinC 2011 lead II, all 565 records | CinC 2011 lead II, without the 65 constant-lead records (500) |
| :--- | :---: | :---: | :---: |
| `bsqi_detector_agreement` | 0.030 | 0.308 | 0.458 |
| `baseline_wander_ratio` | 0.421 | 0.437 | 0.653 |
| `high_frequency_noise_ratio` | 0.777 | 0.307 | 0.459 |
| `saturation_fraction` | 0.501 | 0.634 | 0.453 |
| `flatline_fraction` | 0.419 | 0.645 | 0.470 |
| `qrs_energy_ratio` | 0.132 | 0.226 | 0.338 |
| `kurtosis_ksqi` | 0.164 | 0.269 | 0.399 |
| `skewness_ssqi` | 0.093 | 0.338 | 0.402 |
| `zero_crossing_rate` | 0.727 | 0.248 | 0.371 |
| `powerline_noise_ratio` | 0.773 | 0.312 | 0.465 |
| `signal_std` | 0.385 | 0.402 | 0.600 |

## Table S33. Sensitivity of the multiplicity family: comparisons with the best model of each evaluation set, unadjusted and Holm-adjusted over all 55 pairs (primary) or over the 10 comparisons with the best model. The best model was identified from the point estimates, so the second family is a sensitivity analysis only.

| Evaluation set | Best model | Model | AUROC (best) | p, unadjusted | p_Holm, all 55 pairs | p_Holm, 10 comparisons with the best |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| BUT_QDB (5-fold subject CV) | 1D CNN (Raw Signal) | **Logistic Regression (SQI)** | 0.938 (0.989) | 0.010 | 0.450 | 0.080 |
| BUT_QDB (5-fold subject CV) | 1D CNN (Raw Signal) | **LightGBM (SQI)** | 0.964 (0.989) | 0.177 | 1.000 | 0.750 |
| BUT_QDB (5-fold subject CV) | 1D CNN (Raw Signal) | **Independent SQI MLP** | 0.939 (0.989) | 0.006 | 0.288 | 0.054 |
| BUT_QDB (5-fold subject CV) | 1D CNN (Raw Signal) | **2D CNN (Rendered Image)** | 0.989 (0.989) | 0.936 | 1.000 | 0.936 |
| BUT_QDB (5-fold subject CV) | 1D CNN (Raw Signal) | **SinglePass Multimodal Net** | 0.967 (0.989) | 0.257 | 1.000 | 0.771 |
| BUT_QDB (5-fold subject CV) | 1D CNN (Raw Signal) | **Rule-Based SQI Heuristic** | 0.977 (0.989) | 0.357 | 1.000 | 0.771 |
| BUT_QDB (5-fold subject CV) | 1D CNN (Raw Signal) | **LiquidAI d1-3B (Zero-Shot)** | 0.852 (0.989) | < 0.001 | < 0.001 | < 0.001 |
| BUT_QDB (5-fold subject CV) | 1D CNN (Raw Signal) | **d1-3B image-token probe** | 0.979 (0.989) | 0.066 | 1.000 | 0.396 |
| BUT_QDB (5-fold subject CV) | 1D CNN (Raw Signal) | **TabPFN (SQI)** | 0.962 (0.989) | 0.027 | 1.000 | 0.189 |
| BUT_QDB (5-fold subject CV) | 1D CNN (Raw Signal) | **HuBERT-ECG probe** | 0.972 (0.989) | 0.150 | 1.000 | 0.750 |
| CinC2011 lead II (565 records) | d1-3B image-token probe | **Logistic Regression (SQI)** | 0.320 (0.833) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (565 records) | d1-3B image-token probe | **LightGBM (SQI)** | 0.754 (0.833) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (565 records) | d1-3B image-token probe | **Independent SQI MLP** | 0.672 (0.833) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (565 records) | d1-3B image-token probe | **1D CNN (Raw Signal)** | 0.719 (0.833) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (565 records) | d1-3B image-token probe | **2D CNN (Rendered Image)** | 0.787 (0.833) | < 0.001 | 0.015 | < 0.001 |
| CinC2011 lead II (565 records) | d1-3B image-token probe | **SinglePass Multimodal Net** | 0.674 (0.833) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (565 records) | d1-3B image-token probe | **Rule-Based SQI Heuristic** | 0.685 (0.833) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (565 records) | d1-3B image-token probe | **LiquidAI d1-3B (Zero-Shot)** | 0.650 (0.833) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (565 records) | d1-3B image-token probe | **TabPFN (SQI)** | 0.749 (0.833) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (565 records) | d1-3B image-token probe | **HuBERT-ECG probe** | 0.562 (0.833) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | d1-3B image-token probe | **Logistic Regression (SQI)** | 0.868 (0.908) | < 0.001 | 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | d1-3B image-token probe | **LightGBM (SQI)** | 0.880 (0.908) | 0.002 | 0.043 | 0.006 |
| CinC2017 (8,512 recordings) | d1-3B image-token probe | **Independent SQI MLP** | 0.868 (0.908) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | d1-3B image-token probe | **1D CNN (Raw Signal)** | 0.773 (0.908) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | d1-3B image-token probe | **2D CNN (Rendered Image)** | 0.850 (0.908) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | d1-3B image-token probe | **SinglePass Multimodal Net** | 0.835 (0.908) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | d1-3B image-token probe | **Rule-Based SQI Heuristic** | 0.887 (0.908) | 0.036 | 0.481 | 0.072 |
| CinC2017 (8,512 recordings) | d1-3B image-token probe | **LiquidAI d1-3B (Zero-Shot)** | 0.757 (0.908) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | d1-3B image-token probe | **TabPFN (SQI)** | 0.858 (0.908) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | d1-3B image-token probe | **HuBERT-ECG probe** | 0.907 (0.908) | 0.927 | 1.000 | 0.927 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **Logistic Regression (SQI)** | 0.860 (0.916) | < 0.001 | 0.005 | 0.001 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **LightGBM (SQI)** | 0.880 (0.916) | 0.010 | 0.261 | 0.020 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **Independent SQI MLP** | 0.866 (0.916) | < 0.001 | 0.028 | 0.004 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **1D CNN (Raw Signal)** | 0.767 (0.916) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **2D CNN (Rendered Image)** | 0.859 (0.916) | < 0.001 | 0.005 | 0.001 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **SinglePass Multimodal Net** | 0.848 (0.916) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **Rule-Based SQI Heuristic** | 0.876 (0.916) | 0.006 | 0.174 | 0.019 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **LiquidAI d1-3B (Zero-Shot)** | 0.739 (0.916) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **TabPFN (SQI)** | 0.861 (0.916) | < 0.001 | 0.010 | 0.002 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **d1-3B image-token probe** | 0.910 (0.916) | 0.612 | 1.000 | 0.612 |
