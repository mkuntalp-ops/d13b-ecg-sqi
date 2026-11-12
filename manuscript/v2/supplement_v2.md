# Supplementary Material


Every table is generated from saved outputs by evaluation/build_manuscript_tables.py (results/v2_2026-11/manuscript_tables_2026-10-12.md), except Table S20, which is read from results/v2_2026-11/heuristic_refit_summary.csv, and Supplementary Methods, which are text. Positive class: unusable.

## Supplementary Methods

All definitions below were checked against the code in the repository (`features/sqi_extractor.py`, `features/qrs_detectors.py`, `models/*.py`).

### S-M1. Signal quality indices

Each 10-s window is at 500 Hz, in mV, with n = 5,000 samples. Spectral quantities use Welch's method with 2-s segments.

**The 11 tabular inputs (items 1–11).** Items 12 and 13 are constant after windowing and resampling and are not model inputs; they appear only in the text summary given to zero-shot d1-3B (S-M4).
1. **bSQI.** F1 agreement between two QRS detectors, an implementation following Pan and Tompkins and a second-difference detector (0.5-Hz high-pass, |x[n+2] − 2x[n] + x[n−2]|, 80-ms moving average). Beats are matched one to one within 50 ms, and bSQI is 0 if either detector finds no beat.
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
12. **Window duration.** In s (prompt only).
13. **Sampling rate.** In Hz (prompt only).

A window whose range is below 10^-6 receives flatline and saturation fractions of 1, and non-finite values are set to 0.

### S-M2. Network architectures

All networks were trained with AdamW (learning rate 10^-3, weight decay 10^-3) on binary cross-entropy with the positive class weighted by the usable/unusable ratio of the training data, without early stopping, and with seed 42 for the reported run.

The SQI MLP (3,073 parameters) has the layers Linear(11→64), BatchNorm, ReLU, Dropout 0.2, Linear(64→32), BatchNorm, ReLU and Linear(32→1). Its inputs are standardised with the training-set mean and SD, and it was trained for 35 epochs with batch size 64.

The 1D CNN (167,489 parameters) has four blocks of Conv1d, BatchNorm, LeakyReLU (slope 0.1), MaxPool (×4) and Dropout 0.15, with 32, 64, 128 and 128 channels and kernel sizes 15, 11, 7 and 5. Global average pooling is followed by Linear(128→32), ReLU, Dropout 0.25 and Linear(32→1). Its input is the 5,000-sample window, z-scored per window (an SD below 10^-6 is replaced by 1). It was trained for 25 epochs with batch size 32 and a cosine learning-rate schedule.

The 2D CNN (245,697 parameters) receives the rendered strip resized to 3 × 128 × 256 and normalised with mean 0.95 and SD 0.15 per channel. It has four blocks of 3 × 3 Conv2d, BatchNorm and ReLU, with 2 × 2 max-pooling after the first three and 32, 64, 128 and 128 channels, followed by global average pooling, Linear(128→32), ReLU, Dropout 0.3 and Linear(32→1). It was trained for 25 epochs with batch size 32 and a cosine schedule.

The multimodal net (131,841 parameters) combines an image encoder and an SQI encoder. The image encoder has three blocks of Conv2d, BatchNorm and ReLU (32, 64 and 128 channels), global average pooling, Linear(128→64) and ReLU; the SQI encoder has Linear(11→64), BatchNorm, ReLU, Dropout 0.2, Linear(64→64) and ReLU. The concatenated 128-d vector is multiplied element-wise by a sigmoid gate, Linear(128→128), and passed to a head with Linear(128→64), ReLU, Dropout 0.3 and Linear(64→1). It was trained for 25 epochs with batch size 32 and a cosine schedule.

### S-M3. Rule heuristic (`models/rule_heuristic_baseline.py`)

**SQI component.** With the inputs of S-M1:

logit_SQI = −3.0·(bSQI − 0.5) + 1.2·min(baseline-wander ratio, 3) + 1.5·min(high-frequency ratio, 3) + 5.0·saturation + 5.0·flatline − 0.5

p_SQI = 1 / (1 + exp(−logit_SQI))

**Image component.** The rendered strip is converted to 8-bit grey scale; *ink* is the share of pixels with value below 200, and *spread* is the SD over image columns of the per-column ink share:

logit_vis = 4.0·(ink − 0.08) − 2.0·(spread − 0.02)

p_vis = 1 / (1 + exp(−logit_vis))

**Output.** The score is p = 0.6·p_SQI + 0.4·p_vis, clipped to [0.01, 0.99]. All coefficients are hand-set. They were set during the initial prototyping of the analysis code, and their independence from BUT QDB cannot be guaranteed; the leakage-free in-domain comparison is the logistic-regression replacement fitted inside the folds (Table S20).

### S-M4. Zero-shot d1-3B input (`models/d1_real_neural_model.py`)

**Image.** The rendered strip.

**State text.** A summary of 10 inputs (S-M1, items 1–8, 12 and 13), in the format:

"ECG Signal Quality Indicators (SQIs): QRS Detector Agreement (bSQI): x (1.0 = perfect consensus, 0.0 = total disagreement); Baseline Wander Ratio; High-Frequency Noise Ratio; Saturation Fraction (%); Flatline Fraction (%); QRS Band Energy Ratio; Kurtosis and Skewness; Window Duration and Sampling Rate."

**Question.** A choice question whose instructions read: "Determine whether this electrocardiogram (ECG) recording segment is usable for reliable QRS-based rhythm analysis, or unusable due to severe artifacts, baseline wander, high-frequency noise, or saturation/flatline." The criterion for "usable" is "QRS complexes are distinguishable and suitable for rhythm analysis. Artifacts are manageable.", and the criterion for "unusable" is "ECG is severely corrupted by noise, baseline drift, saturation, or absent complexes; requires human review or re-recording."

**Output.** The probability of "unusable" from a single forward pass with zero output tokens. Model revision: 051bcc464b01b9f92942b364d9586b0ef5912432.

### S-M5. Latency measurement (`experiments/latency_all.sh`)

All latency scripts were run in one session by a single entry point with no other computation running. Latency reflects one machine and Python implementations, and TabPFN was timed on the CPU only. Table S45 and Table S8 come from different scripts: in Table S45 (`experiments/empirical_latency_benchmark.py`) the models are timed one after another in one process, whereas the script of Table S8 (`experiments/latency_extended.py`) interleaves CPU and MPS passes per window and excludes the first window. Their MPS forward times of a few milliseconds differ (1D CNN 1.31 vs 2.75 ms; 2D CNN 5.83 vs 3.09 ms). Part of the image-model difference is explained by caching: the image models cache each decoded and resized strip after first use, so decoding and resizing fall inside the timed forward pass for 19 of 25 windows in Table S45, but only in the first of three runs in Table S8. The remaining differences, including the 1D CNN's, are unexplained, so MPS forward times at this scale are treated as harness-dependent; the conclusions rest only on differences of an order of magnitude, namely rendering of about 61 ms versus the forward pass, and 831 ms for d1-3B. Every model in Table S45 is timed from the same input, the stored 500-Hz window, to its decision: models that need a strip image (the 2D CNN, the multimodal net, zero-shot d1-3B and the rule heuristic, which uses two strip-image features) are charged for rendering it. In an earlier version the heuristic read its strip from a cache, so its total (2.53 ms) omitted rendering; the session was repeated with the common boundary, and the earlier outputs are kept with the results.

### S-M6. Implementation note on HuBERT-ECG preprocessing

The reported HuBERT-ECG results use the preprocessing of the authors' public code (main text, section 2.3). An earlier implementation, run on the superseded 1,047-window BUT QDB cohort and the 565-record CinC 2011 set, used a Butterworth band-pass and max-abs scaling of each 5-s half with absent leads left at 0. It gave AUROC 0.987 (BUT QDB), 0.732 (CinC 2011) and 0.867 (CinC 2017), against 0.972, 0.562 and 0.907 with the authors' preprocessing on those data. It was not re-run on the present cohort and is not a comparison of the present results.

## Supplementary Results

These analyses were part of the main Results in an earlier version and were moved here unchanged when the paper was reframed (main text, section 2.6).

### S-R1. Class 2 relabelled as unusable, and the official BUT QDB partition

Treating Class 2 as unusable as well (431 unusable windows) left the 2D CNN unchanged (0.985) and lowered zero-shot d1-3B to 0.760 and the multimodal net to 0.852 (Table S11). Under the official partition (Table S24), AUROC ranged from 0.723 (zero-shot d1-3B) to 1.000 (1D CNN and HuBERT-ECG probe), and no comparison with the best model was significant (all p_Holm = 1.000). Unusable windows occur in only three of the seven test subjects, mostly two, so this binary task on the official test set is close to ceiling and cannot separate the models.

### S-R2. Partial fine-tuning of HuBERT-ECG

In a single run of partial fine-tuning, HuBERT-ECG reached 0.947 on BUT QDB, 0.655 on CinC 2011 and 0.910 on CinC 2017 (Table S23). These values are descriptive: they were not tested against the probe and were not repeated across seeds.

### S-R3. Quantization-insensitive flatline input (exploratory)

Replacing the flatline input by the quantization-insensitive measure (exploratory; Table S34) changed CinC 2011 AUROC from 0.675 to 0.737 for the MLP, from 0.322 to 0.431 for logistic regression and from 0.752 to 0.759 for LightGBM, with changes of at most 0.012 on BUT QDB and at most 0.011 on CinC 2017 (no confidence intervals or tests were computed). The new measure is 0 in all 988 BUT QDB windows and therefore carries no information in training, so its effect is equivalent to removing the artefact-driven input, and logistic regression remained inverted. The definition does not resolve the transfer failure.

### S-R4. Hand-set heuristic versus its logistic-regression replacement

The hand-set heuristic outperformed its logistic-regression replacement in every evaluation (Table S20): on BUT QDB, 0.971 versus 0.905 (cluster-bootstrap p = 0.001); on CinC 2011, 0.678 versus 0.330 (DeLong p < 0.001); and on CinC 2017, 0.887 versus 0.864 (DeLong p < 0.001). Because the heuristic's coefficients may not be independent of BUT QDB, only the two external comparisons are informative.

### S-R5. Paired noise stress on NSTDB

Under paired electrode-motion stress (Tables S7 and S16), every trained model lowered its usable probability as noise increased. At 0 dB, the mean paired change was −0.94 for logistic regression, −0.90 for LightGBM, −0.84 for the multimodal net, −0.75 for the 2D CNN and −0.59 for the 1D CNN. Zero-shot d1-3B changed by at most 0.09 and did not track the noise.

### S-R6. Computational cost

For the image pipelines, rendering dominated: about 61 ms (92% of the total) against 5.8 ms for the forward pass. The rule heuristic also needs the strip, so from the stored window it took 62.70 ms, of which 62.44 ms was preprocessing; only the three SQI models (1.51–2.42 ms) and the 1D CNN avoid rendering. This cost belongs to the plotting library rather than to rasterization as such: a direct raster renderer took 0.63 ms (Table S8), and the 2D CNN retrained on such strips kept its discrimination (Table S33; AUROC on BUT QDB 0.980 versus 0.985, Δ −0.005, 95% CI −0.027 to 0.005; CinC 2017 0.852 versus 0.850; on CinC 2011 it improved, 0.798 versus 0.787, Δ +0.012, 95% CI 0.003–0.021, p_Holm = 0.025). The d1-3B probe and the multimodal net were not re-evaluated on raster strips. The 1D CNN, which needs no rendering step, took 1.31 ms on the GPU from the stored 500-Hz window (Table S45). d1-3B took 896 ms per window, dominated by its 831-ms forward pass. Forward-pass times of a few milliseconds on the GPU differ between our two timing scripts (Supplementary Methods S-M5), so only order-of-magnitude differences are interpreted. Among the foundation-model approaches (Table S25), the HuBERT-ECG probe took 28.0 ms per window on MPS (60.6 ms on the CPU), of which 23.6 ms was the forward pass; the linear d1-3B probe adds 0.12 ms to the d1-3B forward pass; and TabPFN, which predicts in context over the 988 training windows, took 3.8 s per window on the CPU.

### S-R7. Quality gating of a downstream rhythm classifier

The rhythm classifier reached a macro-F1 of 0.719 on the 8,234 CinC 2017 recordings without any rejection (out-of-fold; Table S37). Rejecting the 10% of recordings with the highest quality score changed macro-F1, relative to random rejection, by 0.001 to 0.008 across the eleven quality models. The increases were largest for the d1-3B image probe (0.008, 95% CI 0.005–0.013) and the 1D CNN (0.008, 95% CI 0.004–0.013), and the only other model whose interval excluded zero was TabPFN (0.005, 95% CI 0.001–0.009). The d1-3B probe exceeded the 2D CNN by 0.007 (95% CI 0.002–0.011) and the HuBERT-ECG probe by 0.007 (95% CI 0.003–0.012). The change ranged from −0.001 to 0.007 when 5% were rejected and from 0.005 to 0.011 when 20% were rejected, where the intervals for the probe's lead over the 2D CNN and over the HuBERT-ECG probe both included zero. These intervals are not adjusted for the 11 models and three fractions. The d1-3B probe's 10% rejection removed 500 normal, 47 atrial fibrillation and 276 other-rhythm recordings, so it rejected a smaller share of atrial fibrillation (5.7% of rejected recordings against 9.2% of all recordings) and a larger share of other rhythms (33.5% against 29.3%).

## Table S1. Patient-independent 5-fold cross-validation on BUT QDB (988 windows, 15 subjects), eight approaches of the main analysis; discrimination estimates for all 11 approaches are in Table S28. 95% CIs: subject-level cluster bootstrap.

| Model Architecture | AUROC [95% CI] | AP | Brier | Fixed τ = 0.50 [TP / FP / TN / FN] | Balanced Acc. [95% CI] | Sensitivity [95% CI] | False Reject Rate [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | 0.985 [0.960 – 0.999] | 0.954 | 0.046 | [198 / 12 / 738 / 40] | 90.8% [75.9 – 99.3%] | 83.2% [53.7 – 100.0%] | 1.6% [0.5 – 2.8%] |
| **1D CNN (Raw Signal)** | 0.983 [0.938 – 0.999] | 0.960 | 0.055 | [179 / 6 / 744 / 59] | 87.2% [71.7 – 98.0%] | 75.2% [44.4 – 96.7%] | 0.8% [0.1 – 1.7%] |
| **Rule-Based SQI Heuristic** | 0.971 [0.914 – 0.998] | 0.920 | 0.108 | [137 / 9 / 741 / 101] | 78.2% [56.0 – 94.8%] | 57.6% [13.3 – 89.9%] | 1.2% [0.3 – 2.4%] |
| **SinglePass Multimodal Net** | 0.966 [0.921 – 0.998] | 0.909 | 0.088 | [128 / 15 / 735 / 110] | 75.9% [58.1 – 98.3%] | 53.8% [17.7 – 98.5%] | 2.0% [0.7 – 3.6%] |
| **LightGBM (SQI)** | 0.957 [0.883 – 0.996] | 0.894 | 0.076 | [161 / 33 / 717 / 77] | 81.6% [62.5 – 95.0%] | 67.6% [30.0 – 93.1%] | 4.4% [1.7 – 7.7%] |
| **Independent SQI MLP** | 0.904 [0.772 – 0.979] | 0.747 | 0.108 | [170 / 68 / 682 / 68] | 81.2% [63.1 – 93.8%] | 71.4% [38.8 – 93.2%] | 9.1% [2.3 – 20.1%] |
| **Logistic Regression (SQI)** | 0.899 [0.794 – 0.979] | 0.759 | 0.114 | [127 / 48 / 702 / 111] | 73.5% [59.9 – 90.5%] | 53.4% [26.1 – 87.0%] | 6.4% [1.7 – 13.7%] |
| **LiquidAI d1-3B (Zero-Shot)** | 0.828 [0.673 – 0.940] | 0.688 | 0.148 | [120 / 30 / 720 / 118] | 73.2% [55.5 – 88.6%] | 50.4% [13.8 – 81.4%] | 4.0% [1.5 – 6.9%] |

## Table S2. Nested versus post-hoc threshold selection on BUT QDB.

| Model Architecture | Target 1 (FPR ≤ 5%): nested Sens. / FPR | Target 1: post-hoc Sens. / FPR | Target 2 (Sens. ≥ 80%): nested Sens. / FPR | Target 2: post-hoc Sens. / FPR | Youden: nested Sens. / FPR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 36.6% / 5.3% | 49.2% / 4.9% | 58.4% / 15.1% | 80.3% / 16.7% | 37.4% / 8.0% |
| **LightGBM (SQI)** | 65.5% / 3.6% | 76.1% / 4.9% | 76.1% / 7.1% | 80.3% / 5.6% | 77.3% / 9.9% |
| **Independent SQI MLP** | 33.2% / 6.5% | 32.8% / 4.3% | 73.9% / 13.6% | 81.5% / 13.1% | 72.7% / 13.9% |
| **1D CNN (Raw Signal)** | 93.3% / 4.9% | 94.1% / 4.8% | 66.4% / 0.9% | 81.9% / 1.6% | 89.5% / 2.5% |
| **2D CNN (Rendered Image)** | 85.3% / 5.6% | 90.8% / 4.8% | 84.0% / 2.4% | 82.4% / 1.2% | 85.3% / 7.2% |
| **SinglePass Multimodal Net** | 64.7% / 2.7% | 86.1% / 4.9% | 54.2% / 4.3% | 81.5% / 4.4% | 79.4% / 6.5% |
| **Rule-Based SQI Heuristic** | 81.1% / 5.3% | 80.7% / 4.9% | 70.6% / 4.4% | 80.3% / 4.8% | 87.8% / 9.6% |
| **LiquidAI d1-3B (Zero-Shot)** | 50.4% / 4.0% | 50.4% / 4.0% | 81.1% / 43.5% | 82.8% / 43.3% | 54.2% / 17.5% |

## Table S3. BUT QDB precision metrics projected to the prevalence of Class 3 inside the official intervals (2.14%, primary) and over all annotated intervals including the excluded stretch (16.0%; tabulated for continuity with the earlier version, not interpreted). The projection changes the class ratio only and assumes that the score distributions within each class in the quota cohort hold for all eligible windows; the values are conditional projections, not estimates for the full eligible population (main text, section 2.4; Table S43).

| Model Architecture | AP, quota cohort (24.1%) | AP projected to all annotated intervals (16.0%) | AP projected to official intervals (2.14%) | Projected PPV at τ = 0.50, 16.0% | Projected PPV at τ = 0.50, 2.14% | Projected NPV at τ = 0.50, 2.14% | False rejections per hour of usable signal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.759 | 0.678 | 0.372 | 0.614 | 0.154 | 0.989 | 23.0 |
| **LightGBM (SQI)** | 0.894 | 0.851 | 0.623 | 0.746 | 0.252 | 0.993 | 15.8 |
| **Independent SQI MLP** | 0.747 | 0.661 | 0.345 | 0.601 | 0.147 | 0.993 | 32.6 |
| **1D CNN (Raw Signal)** | 0.960 | 0.943 | 0.811 | 0.947 | 0.673 | 0.995 | 2.9 |
| **2D CNN (Rendered Image)** | 0.954 | 0.930 | 0.718 | 0.909 | 0.532 | 0.996 | 5.8 |
| **SinglePass Multimodal Net** | 0.909 | 0.867 | 0.605 | 0.837 | 0.370 | 0.990 | 7.2 |
| **Rule-Based SQI Heuristic** | 0.920 | 0.884 | 0.674 | 0.902 | 0.512 | 0.991 | 4.3 |
| **LiquidAI d1-3B (Zero-Shot)** | 0.688 | 0.615 | 0.350 | 0.707 | 0.216 | 0.989 | 14.4 |

## Table S4. Thresholds chosen on pooled BUT QDB out-of-fold scores and applied unchanged to the external datasets. CinC 2011: all 998 records, lead II. CinC 2017: all 8,512 recordings (recording maximum).

| Model Architecture | Rule | Threshold (from BUT QDB) | BUT QDB Sens. / FPR | CinC 2011 (998 records) Sens. / FPR | CinC 2017 (8,512 recordings) Sens. / FPR |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | Target 1 (FPR ≤ 5%) | 0.5672 | 49.2% / 4.9% | 67.1% / 100.0% | 77.3% / 17.2% |
| **Logistic Regression (SQI)** | Target 2 (Sens. ≥ 80%) | 0.1762 | 80.3% / 16.7% | 67.1% / 100.0% | 95.7% / 50.0% |
| **LightGBM (SQI)** | Target 1 (FPR ≤ 5%) | 0.3569 | 76.1% / 4.9% | 52.4% / 6.7% | 79.1% / 16.7% |
| **LightGBM (SQI)** | Target 2 (Sens. ≥ 80%) | 0.2987 | 80.3% / 5.6% | 54.7% / 8.7% | 82.7% / 19.0% |
| **Independent SQI MLP** | Target 1 (FPR ≤ 5%) | 0.8164 | 32.8% / 4.3% | 32.9% / 0.0% | 48.2% / 7.6% |
| **Independent SQI MLP** | Target 2 (Sens. ≥ 80%) | 0.2989 | 81.5% / 13.1% | 32.9% / 0.0% | 88.8% / 25.6% |
| **1D CNN (Raw Signal)** | Target 1 (FPR ≤ 5%) | 0.01151 | 94.1% / 4.8% | 77.3% / 36.5% | 99.6% / 55.9% |
| **1D CNN (Raw Signal)** | Target 2 (Sens. ≥ 80%) | 0.1698 | 81.9% / 1.6% | 72.4% / 27.8% | 95.7% / 42.1% |
| **2D CNN (Rendered Image)** | Target 1 (FPR ≤ 5%) | 0.1552 | 90.8% / 4.8% | 57.8% / 10.3% | 95.3% / 48.6% |
| **2D CNN (Rendered Image)** | Target 2 (Sens. ≥ 80%) | 0.6025 | 82.4% / 1.2% | 54.7% / 6.9% | 88.1% / 33.1% |
| **SinglePass Multimodal Net** | Target 1 (FPR ≤ 5%) | 0.1186 | 86.1% / 4.9% | 100.0% / 100.0% | 96.0% / 57.3% |
| **SinglePass Multimodal Net** | Target 2 (Sens. ≥ 80%) | 0.1581 | 81.5% / 4.4% | 100.0% / 100.0% | 95.3% / 55.6% |
| **Rule-Based SQI Heuristic** | Target 1 (FPR ≤ 5%) | 0.3762 | 80.7% / 4.9% | 100.0% / 99.6% | 52.9% / 6.2% |
| **Rule-Based SQI Heuristic** | Target 2 (Sens. ≥ 80%) | 0.3801 | 80.3% / 4.8% | 100.0% / 99.6% | 51.4% / 6.0% |
| **LiquidAI d1-3B (Zero-Shot)** | Target 1 (FPR ≤ 5%) | 0.5 | 50.4% / 4.0% | 99.6% / 100.0% | 19.8% / 3.1% |
| **LiquidAI d1-3B (Zero-Shot)** | Target 2 (Sens. ≥ 80%) | 0.3486 | 82.8% / 43.3% | 100.0% / 100.0% | 58.3% / 22.5% |
| **TabPFN (SQI)** | Target 1 (FPR ≤ 5%) | 0.1068 | 76.1% / 4.8% | 59.1% / 21.7% | 84.5% / 28.0% |
| **TabPFN (SQI)** | Target 2 (Sens. ≥ 80%) | 0.06638 | 80.7% / 6.7% | 62.2% / 29.5% | 88.1% / 33.6% |
| **HuBERT-ECG probe** | Target 1 (FPR ≤ 5%) | 0.1192 | 79.0% / 4.9% | 51.1% / 45.9% | 99.6% / 58.1% |
| **HuBERT-ECG probe** | Target 2 (Sens. ≥ 80%) | 0.09571 | 80.7% / 5.6% | 51.6% / 48.4% | 99.6% / 61.4% |
| **d1-3B image-token probe** | Target 1 (FPR ≤ 5%) | 0.05944 | 86.1% / 4.9% | 57.8% / 7.8% | 97.5% / 45.3% |
| **d1-3B image-token probe** | Target 2 (Sens. ≥ 80%) | 0.12 | 81.5% / 2.8% | 55.6% / 5.7% | 94.6% / 38.8% |

## Table S5. CinC 2011 (998 records) scored on all 12 leads (aggregation rules chosen after the main analysis). Both foundation-model probes were fitted on BUT QDB only and scored after the other models' 12-lead results were known. TabPFN was not scored on 12 leads.

| Model Architecture | Lead II: AUROC [95% CI] | Maximum over 12 leads: AUROC [95% CI] | Mean over 12 leads: AUROC [95% CI] |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.322 [0.288–0.359] | 0.351 [0.318–0.382] | 0.154 [0.124–0.186] |
| **LightGBM (SQI)** | 0.752 [0.712–0.793] | 0.617 [0.579–0.654] | 0.854 [0.826–0.880] |
| **Independent SQI MLP** | 0.675 [0.640–0.708] | 0.838 [0.809–0.868] | 0.840 [0.811–0.871] |
| **1D CNN (Raw Signal)** | 0.695 [0.656–0.730] | 0.229 [0.187–0.270] | 0.789 [0.757–0.822] |
| **2D CNN (Rendered Image)** | 0.787 [0.751–0.820] | 0.718 [0.686–0.751] | 0.882 [0.858–0.903] |
| **SinglePass Multimodal Net** | 0.500 [0.500–0.500] | 0.500 [0.500–0.500] | 0.500 [0.500–0.500] |
| **Rule-Based SQI Heuristic** | 0.678 [0.628–0.720] | 0.842 [0.806–0.874] | 0.719 [0.673–0.759] |
| **LiquidAI d1-3B (Zero-Shot)** | 0.641 [0.591–0.684] | 0.813 [0.770–0.852] | 0.644 [0.594–0.692] |
| **d1-3B image-token probe** | 0.824 [0.788–0.855] | 0.951 [0.935–0.966] | 0.944 [0.926–0.959] |
| **HuBERT-ECG probe (each lead in the lead II slot)** | 0.562 [0.518–0.605] | 0.431 [0.384–0.481] | 0.510 [0.458–0.563] |

## Table S6. Univariate AUROC of each SQI across datasets (values below 0.5: higher feature value associated with usable).

| SQI feature | BUT QDB (988 windows) | CinC 2011 lead II (998 records) | CinC 2017 (first window, all recordings) |
| :--- | :---: | :---: | :---: |
| `bsqi_detector_agreement` | 0.035 | 0.309 | 0.124 |
| `baseline_wander_ratio` | 0.525 | 0.443 | 0.595 |
| `high_frequency_noise_ratio` | 0.722 | 0.307 | 0.414 |
| `saturation_fraction` | 0.501 | 0.623 | 0.505 |
| `flatline_fraction` | 0.422 | 0.634 | 0.281 |
| `qrs_energy_ratio` | 0.164 | 0.220 | 0.231 |
| `kurtosis_ksqi` | 0.129 | 0.279 | 0.401 |
| `skewness_ssqi` | 0.100 | 0.345 | 0.342 |
| `zero_crossing_rate` | 0.660 | 0.256 | 0.470 |
| `powerline_noise_ratio` | 0.717 | 0.311 | 0.426 |
| `signal_std` | 0.480 | 0.410 | 0.786 |

## Table S7. NSTDB: paired change in usable probability relative to the clean partner window (mean ± SD).

| SNR Level | Logistic Regression (SQI) | LightGBM (SQI) | 1D CNN (Raw Signal) | 2D CNN (Image) | Multimodal Net | d1-3B (zero-shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+24 dB (Low Noise)** | -0.077 ± 0.130 | -0.103 ± 0.248 | -0.026 ± 0.062 | -0.133 ± 0.212 | -0.093 ± 0.178 | -0.018 ± 0.058 |
| **+18 dB** | -0.177 ± 0.154 | -0.441 ± 0.357 | -0.082 ± 0.179 | -0.407 ± 0.351 | -0.432 ± 0.369 | -0.037 ± 0.059 |
| **+12 dB** | -0.510 ± 0.213 | -0.815 ± 0.210 | -0.266 ± 0.335 | -0.697 ± 0.299 | -0.769 ± 0.321 | -0.072 ± 0.092 |
| **+6 dB** | -0.822 ± 0.123 | -0.900 ± 0.124 | -0.488 ± 0.407 | -0.751 ± 0.292 | -0.829 ± 0.294 | -0.092 ± 0.103 |
| **0 dB (Equal Power)** | -0.937 ± 0.047 | -0.904 ± 0.111 | -0.593 ± 0.391 | -0.755 ± 0.290 | -0.843 ± 0.286 | -0.092 ± 0.119 |
| **-6 dB (Severe Noise)** | -0.957 ± 0.034 | -0.896 ± 0.116 | -0.618 ± 0.385 | -0.755 ± 0.293 | -0.837 ± 0.290 | -0.078 ± 0.113 |

## Table S8. Extended latency profile (Apple M4; `experiments/latency_extended.py`). This is a separate script from Table S45 (`experiments/empirical_latency_benchmark.py`): CPU and MPS passes are interleaved per window, and the first window of each run is excluded. Decoded image tensors are cached after first use, so image decoding and resizing are timed only in the first run; the reported median over runs reflects cached inputs. MPS forward times of a few milliseconds therefore differ from Table S45 (Supplementary Methods S-M5).

| Step | Median (three runs) | Range of run medians | p95 (median over runs) |
| :--- | :---: | :---: | :---: |
| Unit conversion (µV to mV) | 0.01 ms | 0.01–0.01 ms | 0.02 ms |
| Resampling 1,000 → 500 Hz | 0.09 ms | 0.09–0.09 ms | 0.10 ms |
| SQI extraction (13 features) | 1.67 ms | 1.66–1.68 ms | 1.79 ms |
| Rendering, Matplotlib (used by all image models) | 57.31 ms | 57.26–57.31 ms | 118.18 ms |
| Rendering, direct raster (PIL; timing only) | 0.63 ms | 0.59–0.64 ms | 0.68 ms |
| 1D CNN forward, CPU | 1.32 ms | 1.31–1.33 ms | 1.40 ms |
| 1D CNN raw-to-decision, CPU | 1.42 ms | 1.41–1.44 ms | 1.51 ms |
| 2D CNN forward, CPU | 5.14 ms | 5.08–6.84 ms | 5.29 ms |
| Multimodal net forward, CPU | 4.50 ms | 4.39–5.42 ms | 4.59 ms |
| 1D CNN forward, MPS | 2.75 ms | 2.68–2.78 ms | 2.95 ms |
| 1D CNN raw-to-decision, MPS | 2.85 ms | 2.78–2.89 ms | 3.05 ms |
| 2D CNN forward, MPS | 3.09 ms | 2.93–3.50 ms | 3.96 ms |
| Multimodal net forward, MPS | 3.33 ms | 3.13–3.76 ms | 4.39 ms |

## Table S9. Seed stability of the 1D CNN, 2D CNN and multimodal net. CinC rows refer to the 998-record CinC 2011 evaluation.

| Model | Metric | Seed 42 (reported) | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± SD (seeds 1–4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D CNN (Raw Signal)** | BUT QDB pooled AUROC | 0.983 | 0.981 | 0.980 | 0.983 | 0.978 | 0.981 ± 0.002 |
| **1D CNN (Raw Signal)** | CinC AUROC | 0.695 | 0.696 | 0.693 | 0.708 | 0.681 | 0.694 ± 0.011 |
| **1D CNN (Raw Signal)** | CinC sensitivity at τ = 0.50 | 70.7% | 69.3% | 69.8% | 67.1% | 72.0% | 69.6 ± 2.0% |
| **1D CNN (Raw Signal)** | CinC false reject at τ = 0.50 | 25.1% | 24.7% | 25.4% | 23.0% | 29.9% | 25.7 ± 2.9% |
| **2D CNN (Rendered Image)** | BUT QDB pooled AUROC | 0.985 | 0.982 | 0.985 | 0.979 | 0.985 | 0.983 ± 0.003 |
| **2D CNN (Rendered Image)** | CinC AUROC | 0.787 | 0.775 | 0.787 | 0.781 | 0.784 | 0.782 ± 0.005 |
| **2D CNN (Rendered Image)** | CinC sensitivity at τ = 0.50 | 55.1% | 56.0% | 55.6% | 25.8% | 54.2% | 47.9 ± 14.8% |
| **2D CNN (Rendered Image)** | CinC false reject at τ = 0.50 | 7.2% | 9.1% | 7.6% | 8.2% | 8.0% | 8.2 ± 0.6% |
| **SinglePass Multimodal Net** | BUT QDB pooled AUROC | 0.966 | 0.962 | 0.969 | 0.968 | 0.965 | 0.966 ± 0.003 |
| **SinglePass Multimodal Net** | CinC AUROC | 0.500 | 0.501 | 0.503 | 0.671 | 0.546 | 0.555 ± 0.080 |
| **SinglePass Multimodal Net** | CinC sensitivity at τ = 0.50 | 100.0% | 100.0% | 99.1% | 32.9% | 98.7% | 82.7 ± 33.2% |
| **SinglePass Multimodal Net** | CinC false reject at τ = 0.50 | 100.0% | 100.0% | 99.2% | 0.1% | 93.9% | 73.3 ± 48.9% |

## Table S10. Post-hoc ROC operating points on BUT QDB (exploratory; thresholds chosen on the evaluated data).

| Model Architecture | Target 1: Sens. (Actual FPR) | Threshold τ | Target 2: FPR (Actual Sens.) | Threshold τ |
| :--- | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | 90.76% (FPR 4.80%) | 0.15516 | 1.20% (Sens 82.35%) | 0.60249 |
| **1D CNN (Raw Signal)** | 94.12% (FPR 4.80%) | 0.01151 | 1.60% (Sens 81.93%) | 0.16983 |
| **Rule-Based SQI Heuristic** | 80.67% (FPR 4.93%) | 0.37622 | 4.80% (Sens 80.25%) | 0.38008 |
| **SinglePass Multimodal Net** | 86.13% (FPR 4.93%) | 0.11858 | 4.40% (Sens 81.51%) | 0.15808 |
| **LightGBM (SQI)** | 76.05% (FPR 4.93%) | 0.35693 | 5.60% (Sens 80.25%) | 0.29870 |
| **Independent SQI MLP** | 32.77% (FPR 4.27%) | 0.81639 | 13.07% (Sens 81.51%) | 0.29890 |
| **Logistic Regression (SQI)** | 49.16% (FPR 4.93%) | 0.56717 | 16.67% (Sens 80.25%) | 0.17625 |
| **LiquidAI d1-3B (Zero-Shot)** | 50.42% (FPR 4.00%) | 0.50000 | 43.33% (Sens 82.77%) | 0.34864 |

## Table S11. Sensitivity analysis with Class 2 relabelled as unusable (BUT QDB).

| Model | AUROC, class 3 unusable (238) [95% CI] | AUROC, classes 2–3 unusable (431), seed 42 [95% CI] | Sens. / FR at τ = 0.50, class 3 unusable | Sens. / FR at τ = 0.50, classes 2–3 unusable |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.899 [0.794 – 0.979] | 0.951 [0.917 – 0.976] | 53.4% / 6.4% | 86.5% / 8.8% |
| **LightGBM (SQI)** | 0.957 [0.883 – 0.996] | 0.941 [0.901 – 0.968] | 67.6% / 4.4% | 88.9% / 17.1% |
| **Independent SQI MLP** | 0.904 [0.772 – 0.979] | 0.941 [0.882 – 0.974] | 71.4% / 9.1% | 87.0% / 13.5% |
| **1D CNN (Raw Signal)** | 0.983 [0.938 – 0.999] | 0.959 [0.903 – 0.993] | 75.2% / 0.8% | 92.1% / 9.3% |
| **2D CNN (Rendered Image)** | 0.985 [0.960 – 0.999] | 0.985 [0.963 – 0.998] | 83.2% / 1.6% | 90.3% / 5.2% |
| **SinglePass Multimodal Net** | 0.966 [0.921 – 0.998] | 0.852 [0.584 – 0.993] | 53.8% / 2.0% | 77.5% / 5.6% |
| **Rule-Based SQI Heuristic** | 0.971 [0.914 – 0.998] | 0.969 [0.940 – 0.985] | 57.6% / 1.2% | 33.9% / 0.0% |
| **LiquidAI d1-3B (Zero-Shot)** | 0.828 [0.673 – 0.940] | 0.760 [0.623 – 0.858] | 50.4% / 4.0% | 31.3% / 2.7% |

## Table S12. Subject-level sensitivity on BUT QDB.

| Held-out subject (fold) | Unusable / usable windows | 1D CNN (Raw Signal) | 2D CNN (Rendered Image) | SinglePass Multimodal Net | LightGBM (SQI) | Rule-Based SQI Heuristic | LiquidAI d1-3B (Zero-Shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100 (4)** | 22 / 50 | 18% | 86% | 82% | 9% | 9% | 5% |
| **105 (3)** | 1 / 50 | 100% | 100% | 100% | 100% | 100% | 100% |
| **111 (2)** | 60 / 50 | 47% | 38% | 25% | 37% | 12% | 53% |
| **113 (0)** | 4 / 50 | 100% | 100% | 75% | 75% | 50% | 100% |
| **114 (1)** | 60 / 50 | 100% | 100% | 100% | 100% | 98% | 100% |
| **122 (0)** | 60 / 50 | 100% | 100% | 0% | 100% | 100% | 33% |
| **124 (4)** | 31 / 50 | 71% | 100% | 100% | 42% | 19% | 6% |
| **False reject, range over 15 subjects** | — / 750 | 0–6% | 0–8% | 0–10% | 0–22% | 0–8% | 0–16% |

## Table S13. CinC 2011 Set A, all 998 records, lead II.

| Model Architecture | Transfer AUROC | Transfer AP | Brier Score | Fixed τ = 0.50 [TP / FP / TN / FN] | Sensitivity | False Reject Rate | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | 0.787 | 0.519 | 0.154 | [124 / 56 / 717 / 101] | 55.1% | 7.2% | 73.9% |
| **LightGBM (SQI)** | 0.752 | 0.515 | 0.156 | [39 / 38 / 735 / 186] | 17.3% | 4.9% | 56.2% |
| **1D CNN (Raw Signal)** | 0.695 | 0.338 | 0.249 | [159 / 194 / 579 / 66] | 70.7% | 25.1% | 72.8% |
| **Rule-Based SQI Heuristic** | 0.678 | 0.581 | 0.310 | [207 / 679 / 94 / 18] | 92.0% | 87.8% | 52.1% |
| **Independent SQI MLP** | 0.675 | 0.500 | 0.151 | [74 / 0 / 773 / 151] | 32.9% | 0.0% | 66.4% |
| **LiquidAI d1-3B (Zero-Shot)** | 0.641 | 0.559 | 0.603 | [224 / 773 / 0 / 1] | 99.6% | 100.0% | 49.8% |
| **SinglePass Multimodal Net** | 0.500 | 0.225 | 0.775 | [225 / 773 / 0 / 0] | 100.0% | 100.0% | 50.0% |
| **Logistic Regression (SQI)** | 0.322 | 0.180 | 0.849 | [151 / 773 / 0 / 74] | 67.1% | 100.0% | 33.6% |

## Table S14. Post-hoc ROC operating points on CinC 2011, all 998 records (exploratory).

| Model Architecture | Target 1: Sens. (Actual FPR) | Threshold τ | Target 2: FPR (Actual Sens.) | Threshold τ |
| :--- | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | 19.56% (FPR 4.92%) | 0.86797 | 36.74% (Sens 80.00%) | 0.00116 |
| **1D CNN (Raw Signal)** | 2.22% (FPR 3.88%) | 1.00000 | 39.07% (Sens 80.00%) | 0.00306 |
| **Rule-Based SQI Heuristic** | 39.56% (FPR 3.75%) | 0.71338 | 74.39% (Sens 80.00%) | 0.55213 |
| **SinglePass Multimodal Net** | not attainable (scores saturate) | — | trivial (all predicted unusable) | 1.00000 |
| **LightGBM (SQI)** | 17.78% (FPR 4.92%) | 0.47956 | 52.26% (Sens 80.00%) | 0.03697 |
| **Independent SQI MLP** | 35.56% (FPR 0.65%) | 1.87e-36 | trivial (all predicted unusable) | 0.00000 |
| **Logistic Regression (SQI)** | not attainable (scores saturate) | — | trivial (all predicted unusable) | 0.00000 |
| **LiquidAI d1-3B (Zero-Shot)** | 39.56% (FPR 4.27%) | 0.95257 | 75.68% (Sens 80.44%) | 0.84390 |

## Table S15. Feature-set and scaler sensitivity of the tabular models under transfer. CinC 2011: all 998 records.

| Feature set | LR · StandardScaler AUROC / AP | LR · RobustScaler AUROC / AP | LR · Z-clip [-3, 3] AUROC / AP | MLP (internal scaler) AUROC / AP |
| :--- | :---: | :---: | :---: | :---: |
| All 11 model features | 0.322 / 0.180 | 0.371 / 0.232 | 0.760 / 0.635 | 0.675 / 0.500 |
| Drop signal_std | 0.392 / 0.247 | 0.367 / 0.228 | 0.730 / 0.606 | 0.504 / 0.232 |
| Drop saturation, flatline | 0.759 / 0.649 | 0.775 / 0.667 | 0.757 / 0.632 | 0.778 / 0.619 |
| Drop signal_std, saturation, flatline | 0.722 / 0.598 | 0.746 / 0.623 | 0.728 / 0.607 | 0.742 / 0.581 |

## Table S16. NSTDB: usable probability by nominal SNR.

| SNR Level | Logistic Regression (SQI) | LightGBM (SQI) | 1D CNN (Raw Signal) | 2D CNN (Image) | Multimodal Net | Rule Heuristic Baseline | d1-3B (Zero-Shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean reference (MIT-BIH base)** | 0.962 ± 0.032 | 0.972 ± 0.111 | 0.689 ± 0.377 | 0.762 ± 0.291 | 0.856 ± 0.287 | 0.740 ± 0.011 | 0.777 ± 0.104 |
| **+24 dB (Low Noise)** | 0.886 ± 0.153 | 0.869 ± 0.213 | 0.663 ± 0.367 | 0.629 ± 0.345 | 0.763 ± 0.322 | 0.717 ± 0.026 | 0.759 ± 0.100 |
| **+18 dB** | 0.785 ± 0.175 | 0.531 ± 0.342 | 0.607 ± 0.360 | 0.355 ± 0.377 | 0.424 ± 0.373 | 0.695 ± 0.025 | 0.740 ± 0.074 |
| **+12 dB** | 0.452 ± 0.226 | 0.157 ± 0.172 | 0.422 ± 0.347 | 0.066 ± 0.143 | 0.088 ± 0.183 | 0.653 ± 0.038 | 0.704 ± 0.052 |
| **+6 dB** | 0.140 ± 0.127 | 0.072 ± 0.062 | 0.201 ± 0.274 | 0.012 ± 0.039 | 0.028 ± 0.079 | 0.606 ± 0.044 | 0.685 ± 0.049 |
| **0 dB (Equal Power)** | 0.025 ± 0.033 | 0.067 ± 0.028 | 0.095 ± 0.158 | 0.008 ± 0.024 | 0.014 ± 0.035 | 0.550 ± 0.063 | 0.685 ± 0.051 |
| **-6 dB (Severe Noise)** | 0.006 ± 0.007 | 0.076 ± 0.044 | 0.071 ± 0.113 | 0.007 ± 0.019 | 0.019 ± 0.051 | 0.555 ± 0.070 | 0.699 ± 0.059 |

## Table S17. Prompt sensitivity of zero-shot d1-3B. The prompt variants were scored on a 150-record balanced CinC 2011 subset (75 acceptable, 75 unacceptable) and were not extended to all 998 records.

| Configuration | BUT QDB AUROC [95% CI] | BUT QDB Sens. / FR at τ = 0.50 | CinC 2011 (150-record subset) AUROC | CinC 2011 (150-record subset) Sens. / FR at τ = 0.50 | NSTDB clean P(usable) | NSTDB paired change, +24 dB | NSTDB paired change, 0 dB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary (SQI text + image)** | 0.828 [0.673 – 0.940] | 50.4% / 4.0% | 0.636 | 100.0% / 100.0% | 0.777 | -0.018 | -0.092 |
| **Image only (no SQI text)** | 0.918 [0.855 – 0.981] | 99.6% / 81.3% | 0.680 | 81.3% / 78.7% | 0.459 | -0.037 | -0.039 |
| **Reworded question (SQI text + image)** | 0.858 [0.708 – 0.967] | 21.8% / 0.5% | 0.627 | 100.0% / 98.7% | 0.870 | -0.017 | -0.073 |

## Table S18. Linear probes on frozen d1-3B representations.

| d1-3B representation | BUT QDB AUROC [95% CI] (5-fold subject CV) | CinC 2011 lead II, 998 records AUROC | CinC 2017 balanced subset AUROC (recording max) |
| :--- | :---: | :---: | :---: |
| **Zero-shot choice probability (primary configuration)** | 0.828 [0.673 – 0.940] | 0.641 | 0.739 |
| **Linear probe on the decision state (final norm, answer position)** | 0.967 [0.923 – 0.995] | 0.422 | 0.885 |
| **Linear probe on image tokens (projector output, mean)** | 0.971 [0.914 – 0.995] | 0.824 | 0.893 |

## Table S19. CinC 2017 balanced subset (278 noisy + 278 non-noisy recordings) including zero-shot d1-3B.

| Model Architecture | AUROC [95% CI], recording max (278 noisy + 278 non-noisy) | Sens. / FR at τ = 0.50 |
| :--- | :---: | :---: |
| **Logistic Regression (SQI)** | 0.863 [0.830 – 0.893] | 81.7% / 19.8% |
| **LightGBM (SQI)** | 0.892 [0.864 – 0.916] | 68.0% / 13.3% |
| **Independent SQI MLP** | 0.875 [0.843 – 0.903] | 79.1% / 17.3% |
| **1D CNN (Raw Signal)** | 0.761 [0.714 – 0.807] | 92.4% / 37.1% |
| **2D CNN (Rendered Image)** | 0.862 [0.829 – 0.891] | 89.2% / 33.8% |
| **SinglePass Multimodal Net** | 0.826 [0.787 – 0.858] | 92.8% / 44.2% |
| **Rule-Based SQI Heuristic** | 0.876 [0.843 – 0.905] | 18.3% / 3.2% |
| **LiquidAI d1-3B (Zero-Shot)** | 0.739 [0.697 – 0.778] | 19.8% / 4.0% |

## Table S20. Hand-set rule heuristic versus a logistic-regression replacement using the same seven input features (standardised L2 logistic regression fitted inside the BUT QDB folds and transferred like the other models). The replacement changes the functional form as well as the coefficients (Supplementary Methods S-M3). p: BUT QDB, subject-level cluster bootstrap; external sets, DeLong; unadjusted.

| Evaluation set | Hand-set heuristic AUROC | LR replacement AUROC | Δ (replacement − hand-set) [95% CI] | p | Test |
| :--- | :---: | :---: | :---: | :---: | :--- |
| BUT_QDB (5-fold subject CV) | 0.971 | 0.905 | -0.066 [-0.178 to -0.001] | 0.001 | subject cluster bootstrap |
| CinC2011 lead II (998) | 0.678 | 0.330 | -0.348 [-0.418 to -0.278] | < 0.001 | DeLong |
| CinC2017 (8,512, recording max) | 0.887 | 0.864 | -0.023 [-0.037 to -0.009] | < 0.001 | DeLong |

## Table S21. Seed repetition (seeds 42 and 1–4) of BUT-only training and leave-one-dataset-out training for the seed-dependent networks. Consistent: same sign of Δ in all five seeds and |median Δ| ≥ 0.02 (pre-specified).

| Model | Held-out dataset | Seeds | LODO AUROC, median (range) | BUT-only AUROC, median (range) | Δ, median (range) | Consistent |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Independent SQI MLP** | CinC2011 | 5 | 0.330 (0.318–0.498) | 0.669 (0.504–0.675) | -0.186 (-0.341 to -0.176) | yes |
| **1D CNN (Raw Signal)** | CinC2011 | 5 | 0.777 (0.761–0.783) | 0.695 (0.681–0.708) | +0.078 (+0.068 to +0.097) | yes |
| **2D CNN (Rendered Image)** | CinC2011 | 5 | 0.762 (0.758–0.784) | 0.784 (0.775–0.787) | -0.019 (-0.028 to +0.010) | no |
| **SinglePass Multimodal Net** | CinC2011 | 5 | 0.541 (0.498–0.747) | 0.503 (0.500–0.671) | -0.002 (-0.129 to +0.201) | no |
| **Independent SQI MLP** | CinC2017 | 5 | 0.879 (0.872–0.886) | 0.865 (0.844–0.876) | +0.013 (-0.004 to +0.042) | no |
| **1D CNN (Raw Signal)** | CinC2017 | 5 | 0.872 (0.871–0.889) | 0.767 (0.757–0.768) | +0.116 (+0.103 to +0.122) | yes |
| **2D CNN (Rendered Image)** | CinC2017 | 5 | 0.813 (0.790–0.819) | 0.840 (0.832–0.850) | -0.037 (-0.050 to -0.013) | yes |
| **SinglePass Multimodal Net** | CinC2017 | 5 | 0.873 (0.867–0.883) | 0.820 (0.798–0.845) | +0.053 (+0.038 to +0.076) | yes |
| **Independent SQI MLP** | BUT_QDB | 5 | 0.978 (0.977–0.980) | — | — | — |
| **1D CNN (Raw Signal)** | BUT_QDB | 5 | 0.992 (0.989–0.994) | — | — | — |
| **2D CNN (Rendered Image)** | BUT_QDB | 5 | 0.975 (0.973–0.990) | — | — | — |
| **SinglePass Multimodal Net** | BUT_QDB | 5 | 0.741 (0.736–0.970) | — | — | — |

## Table S22. Local recalibration on n = 25–200 randomly drawn labelled target recordings (500 draws each), evaluated on the remaining recordings. Prevalence-only Brier score: CinC 2011 0.175, CinC 2017 0.032.

| Dataset | Model | n | Transferred threshold: FPR / Sens. (median) | Local threshold: FPR median [95% range] | Local threshold: Sens. median [95% range] | Draws with FPR ≤ 5% / ≤ 7.5% (local) | Fallback draws | Brier: raw → Platt (median; Platt skipped) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CinC2011 | **Logistic Regression (SQI)** | 25 | 100.0% / 67.1% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 2 | 0.849 → 0.129 (2) |
| CinC2011 | **Logistic Regression (SQI)** | 50 | 100.0% / 67.1% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 0 | 0.848 → 0.127 (0) |
| CinC2011 | **Logistic Regression (SQI)** | 100 | 100.0% / 67.1% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 0 | 0.849 → 0.126 (0) |
| CinC2011 | **Logistic Regression (SQI)** | 200 | 100.0% / 67.0% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 0 | 0.848 → 0.126 (0) |
| CinC2011 | **LightGBM (SQI)** | 25 | 6.8% / 52.5% | 2.6% [0.0–12.7%] | 11.8% [0.0–59.3%] | 57% / 93% | 2 | 0.156 → 0.156 (2) |
| CinC2011 | **LightGBM (SQI)** | 50 | 6.8% / 52.4% | 2.7% [0.0–12.3%] | 13.0% [0.0–58.3%] | 60% / 91% | 0 | 0.156 → 0.151 (0) |
| CinC2011 | **LightGBM (SQI)** | 100 | 6.8% / 52.5% | 3.6% [0.1–8.0%] | 14.4% [0.0–54.7%] | 66% / 91% | 0 | 0.156 → 0.149 (0) |
| CinC2011 | **LightGBM (SQI)** | 200 | 6.7% / 52.5% | 3.7% [1.7–8.2%] | 15.2% [6.4–54.5%] | 67% / 92% | 0 | 0.156 → 0.148 (0) |
| CinC2011 | **Independent SQI MLP** | 25 | 0.0% / 32.9% | 0.0% [0.0–0.7%] | 32.9% [0.0–34.9%] | 100% / 100% | 2 | 0.151 → 0.129 (2) |
| CinC2011 | **Independent SQI MLP** | 50 | 0.0% / 32.9% | 0.0% [0.0–0.7%] | 33.0% [31.3–35.3%] | 100% / 100% | 0 | 0.152 → 0.127 (0) |
| CinC2011 | **Independent SQI MLP** | 100 | 0.0% / 32.9% | 0.0% [0.0–0.7%] | 33.3% [30.9–36.3%] | 100% / 100% | 0 | 0.151 → 0.126 (0) |
| CinC2011 | **Independent SQI MLP** | 200 | 0.0% / 33.0% | 0.2% [0.0–0.8%] | 33.9% [30.3–37.6%] | 100% / 100% | 0 | 0.152 → 0.126 (0) |
| CinC2011 | **1D CNN (Raw Signal)** | 25 | 36.5% / 77.3% | 0.0% [0.0–19.2%] | 0.0% [0.0–25.1%] | 86% / 87% | 2 | 0.250 → 0.176 (2) |
| CinC2011 | **1D CNN (Raw Signal)** | 50 | 36.5% / 77.3% | 0.0% [0.0–14.4%] | 0.0% [0.0–13.6%] | 83% / 86% | 0 | 0.250 → 0.171 (0) |
| CinC2011 | **1D CNN (Raw Signal)** | 100 | 36.5% / 77.3% | 0.0% [0.0–9.6%] | 0.0% [0.0–5.7%] | 87% / 91% | 0 | 0.249 → 0.168 (0) |
| CinC2011 | **1D CNN (Raw Signal)** | 200 | 36.6% / 77.3% | 2.5% [0.0–9.6%] | 0.5% [0.0–5.0%] | 90% / 94% | 0 | 0.250 → 0.167 (0) |
| CinC2011 | **2D CNN (Rendered Image)** | 25 | 10.4% / 57.8% | 3.1% [0.0–13.7%] | 8.3% [0.0–61.6%] | 61% / 92% | 2 | 0.155 → 0.148 (2) |
| CinC2011 | **2D CNN (Rendered Image)** | 50 | 10.4% / 57.7% | 4.2% [0.0–10.2%] | 15.2% [0.0–57.6%] | 58% / 93% | 0 | 0.155 → 0.142 (0) |
| CinC2011 | **2D CNN (Rendered Image)** | 100 | 10.4% / 57.8% | 4.2% [0.0–8.9%] | 14.9% [0.0–56.6%] | 63% / 96% | 0 | 0.154 → 0.140 (0) |
| CinC2011 | **2D CNN (Rendered Image)** | 200 | 10.4% / 57.8% | 4.4% [1.3–7.8%] | 17.5% [4.0–56.6%] | 62% / 93% | 0 | 0.154 → 0.139 (0) |
| CinC2011 | **SinglePass Multimodal Net** | 25 | 100.0% / 100.0% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 2 | 0.774 → 0.180 (2) |
| CinC2011 | **SinglePass Multimodal Net** | 50 | 100.0% / 100.0% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 0 | 0.774 → 0.176 (0) |
| CinC2011 | **SinglePass Multimodal Net** | 100 | 100.0% / 100.0% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 0 | 0.774 → 0.175 (0) |
| CinC2011 | **SinglePass Multimodal Net** | 200 | 100.0% / 100.0% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 0 | 0.774 → 0.175 (0) |
| CinC2011 | **Rule-Based SQI Heuristic** | 25 | 99.6% / 100.0% | 0.1% [0.0–9.7%] | 33.0% [0.0–45.5%] | 87% / 95% | 2 | 0.310 → 0.156 (2) |
| CinC2011 | **Rule-Based SQI Heuristic** | 50 | 99.6% / 100.0% | 1.5% [0.1–9.9%] | 34.6% [31.6–45.3%] | 83% / 94% | 0 | 0.310 → 0.152 (0) |
| CinC2011 | **Rule-Based SQI Heuristic** | 100 | 99.6% / 100.0% | 2.2% [0.1–9.3%] | 36.8% [31.6–44.2%] | 75% / 94% | 0 | 0.310 → 0.150 (0) |
| CinC2011 | **Rule-Based SQI Heuristic** | 200 | 99.5% / 100.0% | 3.2% [0.2–9.0%] | 38.8% [32.6–44.3%] | 66% / 96% | 0 | 0.310 → 0.149 (0) |
| CinC2011 | **LiquidAI d1-3B (Zero-Shot)** | 25 | 100.0% / 99.5% | 0.0% [0.0–12.9%] | 32.6% [0.0–42.7%] | 93% / 93% | 2 | 0.603 → 0.152 (2) |
| CinC2011 | **LiquidAI d1-3B (Zero-Shot)** | 50 | 100.0% / 99.5% | 0.0% [0.0–8.0%] | 33.5% [31.3–41.5%] | 94% / 94% | 0 | 0.603 → 0.150 (0) |
| CinC2011 | **LiquidAI d1-3B (Zero-Shot)** | 100 | 100.0% / 99.5% | 0.0% [0.0–8.0%] | 36.8% [31.1–41.3%] | 94% / 94% | 0 | 0.603 → 0.148 (0) |
| CinC2011 | **LiquidAI d1-3B (Zero-Shot)** | 200 | 100.0% / 99.5% | 2.4% [0.0–8.5%] | 37.8% [31.7–42.5%] | 93% / 94% | 0 | 0.604 → 0.148 (0) |
| CinC2011 | **d1-3B image-token probe** | 25 | 7.8% / 57.8% | 0.3% [0.0–10.8%] | 41.4% [0.0–61.6%] | 86% / 94% | 2 | 0.120 → 0.115 (2) |
| CinC2011 | **d1-3B image-token probe** | 50 | 7.8% / 57.7% | 1.2% [0.1–9.9%] | 48.1% [32.2–60.5%] | 76% / 94% | 0 | 0.120 → 0.112 (0) |
| CinC2011 | **d1-3B image-token probe** | 100 | 7.8% / 57.8% | 2.8% [0.1–9.1%] | 50.5% [37.3–58.9%] | 72% / 94% | 0 | 0.121 → 0.110 (0) |
| CinC2011 | **d1-3B image-token probe** | 200 | 7.8% / 57.7% | 3.9% [0.6–8.9%] | 52.3% [43.7–58.5%] | 61% / 94% | 0 | 0.120 → 0.109 (0) |
| CinC2011 | **TabPFN (SQI)** | 25 | 21.8% / 59.1% | 1.7% [0.0–13.2%] | 6.0% [0.0–55.8%] | 58% / 92% | 2 | 0.162 → 0.162 (2) |
| CinC2011 | **TabPFN (SQI)** | 50 | 21.7% / 59.1% | 3.1% [0.0–8.2%] | 9.4% [0.0–51.0%] | 57% / 97% | 0 | 0.162 → 0.156 (0) |
| CinC2011 | **TabPFN (SQI)** | 100 | 21.8% / 59.2% | 3.9% [0.0–10.3%] | 10.7% [0.0–51.6%] | 63% / 95% | 0 | 0.162 → 0.154 (0) |
| CinC2011 | **TabPFN (SQI)** | 200 | 21.8% / 59.1% | 4.2% [1.1–7.5%] | 11.8% [4.4–51.1%] | 67% / 98% | 0 | 0.162 → 0.153 (0) |
| CinC2011 | **HuBERT-ECG probe** | 25 | 45.9% / 51.1% | 0.0% [0.0–17.3%] | 0.0% [0.0–25.1%] | 79% / 84% | 2 | 0.281 → 0.183 (2) |
| CinC2011 | **HuBERT-ECG probe** | 50 | 46.0% / 51.2% | 0.5% [0.0–14.0%] | 0.5% [0.0–18.5%] | 78% / 84% | 0 | 0.280 → 0.178 (0) |
| CinC2011 | **HuBERT-ECG probe** | 100 | 45.9% / 51.2% | 0.7% [0.0–10.2%] | 2.0% [0.0–14.3%] | 81% / 88% | 0 | 0.280 → 0.175 (0) |
| CinC2011 | **HuBERT-ECG probe** | 200 | 45.9% / 51.1% | 2.3% [0.0–9.3%] | 3.4% [0.0–10.7%] | 77% / 91% | 0 | 0.280 → 0.174 (0) |
| CinC2017 | **Logistic Regression (SQI)** | 25 | 17.2% / 77.3% | 8.1% [0.0–17.2%] | 56.0% [0.0–77.3%] | 42% / 49% | 229 | 0.157 → 0.036 (229) |
| CinC2017 | **Logistic Regression (SQI)** | 50 | 17.2% / 77.3% | 2.7% [0.0–17.2%] | 18.2% [0.0–77.3%] | 68% / 78% | 82 | 0.157 → 0.032 (82) |
| CinC2017 | **Logistic Regression (SQI)** | 100 | 17.2% / 77.3% | 2.5% [0.0–17.2%] | 15.9% [0.0–77.3%] | 83% / 93% | 21 | 0.157 → 0.031 (21) |
| CinC2017 | **Logistic Regression (SQI)** | 200 | 17.2% / 77.3% | 3.5% [0.0–7.3%] | 25.7% [0.0–52.8%] | 82% / 98% | 0 | 0.157 → 0.030 (0) |
| CinC2017 | **LightGBM (SQI)** | 25 | 16.7% / 79.1% | 8.5% [0.0–16.7%] | 55.2% [0.0–79.1%] | 43% / 48% | 229 | 0.097 → 0.034 (229) |
| CinC2017 | **LightGBM (SQI)** | 50 | 16.7% / 79.1% | 1.5% [0.0–16.7%] | 19.9% [0.0–79.1%] | 71% / 78% | 82 | 0.097 → 0.030 (82) |
| CinC2017 | **LightGBM (SQI)** | 100 | 16.7% / 79.1% | 1.9% [0.0–16.7%] | 21.7% [0.0–79.1%] | 85% / 94% | 21 | 0.097 → 0.029 (21) |
| CinC2017 | **LightGBM (SQI)** | 200 | 16.7% / 79.1% | 3.0% [0.0–7.0%] | 28.7% [0.0–47.1%] | 89% / 99% | 0 | 0.097 → 0.029 (0) |
| CinC2017 | **Independent SQI MLP** | 25 | 7.6% / 48.2% | 7.5% [0.0–11.1%] | 48.2% [0.0–60.9%] | 45% / 50% | 229 | 0.125 → 0.034 (229) |
| CinC2017 | **Independent SQI MLP** | 50 | 7.6% / 48.2% | 2.1% [0.0–10.6%] | 14.5% [0.0–58.2%] | 70% / 79% | 82 | 0.125 → 0.031 (82) |
| CinC2017 | **Independent SQI MLP** | 100 | 7.6% / 48.2% | 1.9% [0.0–7.6%] | 13.9% [0.0–48.2%] | 86% / 95% | 21 | 0.125 → 0.030 (21) |
| CinC2017 | **Independent SQI MLP** | 200 | 7.6% / 48.2% | 3.4% [0.0–7.0%] | 25.0% [0.0–46.7%] | 85% / 98% | 0 | 0.125 → 0.030 (0) |
| CinC2017 | **1D CNN (Raw Signal)** | 25 | 55.9% / 99.6% | 3.7% [0.0–56.0%] | 0.2% [0.0–99.6%] | 50% / 50% | 229 | 0.321 → 0.033 (229) |
| CinC2017 | **1D CNN (Raw Signal)** | 50 | 55.9% / 99.6% | 0.0% [0.0–56.0%] | 0.0% [0.0–99.6%] | 82% / 83% | 82 | 0.321 → 0.032 (82) |
| CinC2017 | **1D CNN (Raw Signal)** | 100 | 55.9% / 99.6% | 0.0% [0.0–55.9%] | 0.0% [0.0–99.6%] | 95% / 95% | 21 | 0.321 → 0.032 (21) |
| CinC2017 | **1D CNN (Raw Signal)** | 200 | 55.9% / 99.6% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 99% / 100% | 0 | 0.321 → 0.032 (0) |
| CinC2017 | **2D CNN (Rendered Image)** | 25 | 48.6% / 95.3% | 9.7% [0.0–48.7%] | 44.3% [0.0–95.3%] | 45% / 48% | 229 | 0.292 → 0.034 (229) |
| CinC2017 | **2D CNN (Rendered Image)** | 50 | 48.6% / 95.3% | 0.6% [0.0–48.7%] | 7.0% [0.0–95.3%] | 73% / 78% | 82 | 0.292 → 0.032 (82) |
| CinC2017 | **2D CNN (Rendered Image)** | 100 | 48.6% / 95.3% | 0.6% [0.0–48.6%] | 6.4% [0.0–95.3%] | 89% / 93% | 21 | 0.292 → 0.031 (21) |
| CinC2017 | **2D CNN (Rendered Image)** | 200 | 48.6% / 95.2% | 2.4% [0.0–6.7%] | 17.5% [0.0–34.8%] | 93% / 99% | 0 | 0.292 → 0.030 (0) |
| CinC2017 | **SinglePass Multimodal Net** | 25 | 57.3% / 96.0% | 9.1% [0.0–57.4%] | 26.7% [0.0–96.0%] | 46% / 48% | 229 | 0.388 → 0.035 (229) |
| CinC2017 | **SinglePass Multimodal Net** | 50 | 57.3% / 96.0% | 0.0% [0.0–57.4%] | 0.0% [0.0–96.0%] | 77% / 79% | 82 | 0.388 → 0.032 (82) |
| CinC2017 | **SinglePass Multimodal Net** | 100 | 57.3% / 96.0% | 0.0% [0.0–57.3%] | 0.0% [0.0–96.0%] | 91% / 94% | 21 | 0.388 → 0.031 (21) |
| CinC2017 | **SinglePass Multimodal Net** | 200 | 57.3% / 96.0% | 1.3% [0.0–7.0%] | 5.1% [0.0–19.9%] | 90% / 99% | 0 | 0.388 → 0.031 (0) |
| CinC2017 | **Rule-Based SQI Heuristic** | 25 | 6.2% / 52.9% | 6.2% [0.0–8.7%] | 52.9% [0.0–70.1%] | 41% / 95% | 229 | 0.095 → 0.037 (229) |
| CinC2017 | **Rule-Based SQI Heuristic** | 50 | 6.2% / 52.9% | 2.9% [0.0–8.6%] | 20.6% [0.0–69.5%] | 65% / 94% | 82 | 0.095 → 0.033 (82) |
| CinC2017 | **Rule-Based SQI Heuristic** | 100 | 6.2% / 52.9% | 2.2% [0.0–7.5%] | 13.5% [0.0–62.4%] | 77% / 97% | 21 | 0.095 → 0.031 (21) |
| CinC2017 | **Rule-Based SQI Heuristic** | 200 | 6.2% / 52.9% | 3.6% [0.0–7.6%] | 25.0% [0.0–62.4%] | 76% / 97% | 0 | 0.095 → 0.031 (0) |
| CinC2017 | **LiquidAI d1-3B (Zero-Shot)** | 25 | 3.1% / 19.8% | 3.1% [0.0–7.0%] | 19.8% [0.0–33.2%] | 96% / 98% | 229 | 0.091 → 0.034 (229) |
| CinC2017 | **LiquidAI d1-3B (Zero-Shot)** | 50 | 3.1% / 19.9% | 0.0% [0.0–7.0%] | 0.0% [0.0–33.3%] | 94% / 99% | 82 | 0.091 → 0.032 (82) |
| CinC2017 | **LiquidAI d1-3B (Zero-Shot)** | 100 | 3.1% / 19.8% | 0.5% [0.0–7.0%] | 5.5% [0.0–33.2%] | 95% / 100% | 21 | 0.091 → 0.031 (21) |
| CinC2017 | **LiquidAI d1-3B (Zero-Shot)** | 200 | 3.1% / 19.8% | 2.1% [0.0–7.0%] | 16.3% [0.0–33.3%] | 95% / 100% | 0 | 0.091 → 0.031 (0) |
| CinC2017 | **TabPFN (SQI)** | 25 | 28.0% / 84.5% | 8.9% [0.0–28.0%] | 51.4% [0.0–84.5%] | 45% / 48% | 229 | 0.105 → 0.035 (229) |
| CinC2017 | **TabPFN (SQI)** | 50 | 28.0% / 84.5% | 1.5% [0.0–28.0%] | 13.9% [0.0–84.5%] | 71% / 78% | 82 | 0.105 → 0.031 (82) |
| CinC2017 | **TabPFN (SQI)** | 100 | 28.0% / 84.5% | 1.5% [0.0–28.0%] | 13.9% [0.0–84.5%] | 86% / 94% | 21 | 0.105 → 0.030 (21) |
| CinC2017 | **TabPFN (SQI)** | 200 | 28.0% / 84.5% | 2.7% [0.0–6.0%] | 23.9% [0.0–41.9%] | 86% / 99% | 0 | 0.105 → 0.029 (0) |
| CinC2017 | **HuBERT-ECG probe** | 25 | 58.1% / 99.6% | 5.8% [0.0–58.2%] | 54.4% [0.0–99.6%] | 49% / 51% | 229 | 0.176 → 0.033 (229) |
| CinC2017 | **HuBERT-ECG probe** | 50 | 58.1% / 99.6% | 2.1% [0.0–58.2%] | 25.5% [0.0–99.6%] | 72% / 81% | 82 | 0.176 → 0.030 (82) |
| CinC2017 | **HuBERT-ECG probe** | 100 | 58.1% / 99.6% | 2.2% [0.0–58.1%] | 28.2% [0.0–99.6%] | 87% / 95% | 21 | 0.176 → 0.029 (21) |
| CinC2017 | **HuBERT-ECG probe** | 200 | 58.1% / 99.6% | 3.3% [0.0–6.8%] | 36.2% [0.0–59.4%] | 87% / 99% | 0 | 0.176 → 0.028 (0) |
| CinC2017 | **d1-3B image-token probe** | 25 | 45.3% / 97.5% | 6.2% [0.0–45.3%] | 52.7% [0.0–97.5%] | 47% / 51% | 229 | 0.180 → 0.033 (229) |
| CinC2017 | **d1-3B image-token probe** | 50 | 45.3% / 97.5% | 1.4% [0.0–45.3%] | 22.0% [0.0–97.5%] | 74% / 81% | 82 | 0.180 → 0.030 (82) |
| CinC2017 | **d1-3B image-token probe** | 100 | 45.3% / 97.5% | 2.0% [0.0–45.2%] | 25.4% [0.0–97.5%] | 86% / 94% | 21 | 0.180 → 0.029 (21) |
| CinC2017 | **d1-3B image-token probe** | 200 | 45.3% / 97.4% | 2.8% [0.0–6.2%] | 34.7% [0.0–52.7%] | 87% / 99% | 0 | 0.180 → 0.028 (0) |

## Table S23. HuBERT-ECG: linear probe and partial fine-tuning (AUROC). Fine-tuning was not run in LODO.

| HuBERT-ECG variant | BUT QDB (5-fold subject CV) | CinC 2011 (BUT-only) | CinC 2017 (BUT-only) | LODO: BUT QDB | LODO: CinC 2011 | LODO: CinC 2017 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Probe, authors' preprocessing (primary)** | 0.938 | 0.562 | 0.906 | 0.988 | 0.802 | 0.879 |
| **Fine-tuned (last 4 layers + head), authors' preprocessing** | 0.947 | 0.655 | 0.910 | — | — | — |

## Table S24. BUT QDB official protocol (Smital et al. 2026): models retrained on the official training records and tested on the official test records (window-level binary task). 95% CI: subject-level cluster bootstrap; p_Holm: paired comparison with the best model.

| Model | Official train/test split, test set (442 windows, 92 unusable, 7 subjects): AUROC [95% CI] | Δ vs best | p (Holm) vs best |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.980 [0.957–1.000] | -0.020 | 1.000 |
| **LightGBM (SQI)** | 0.992 [0.940–1.000] | -0.008 | 1.000 |
| **Independent SQI MLP** | 0.973 [0.951–1.000] | -0.027 | 1.000 |
| **1D CNN (Raw Signal)** | **1.000 [1.000–1.000]** (best) | — | — |
| **2D CNN (Rendered Image)** | 0.961 [0.933–1.000] | -0.039 | 1.000 |
| **SinglePass Multimodal Net** | 0.935 [0.883–1.000] | -0.065 | 1.000 |
| **Rule-Based SQI Heuristic** | 0.993 [0.971–1.000] | -0.007 | 1.000 |
| **LiquidAI d1-3B (Zero-Shot)** | 0.723 [0.576–1.000] | -0.277 | 1.000 |
| **TabPFN (SQI)** | 0.995 [0.960–1.000] | -0.005 | 1.000 |
| **HuBERT-ECG probe** | 1.000 [1.000–1.000] | -0.000 | 1.000 |
| **d1-3B image-token probe** | 0.991 [0.984–1.000] | -0.009 | 1.000 |

## Table S25. Latency of the foundation-model probes and TabPFN (Apple M4; same 25 windows and three runs as Table S45, harness of Table S8).

| Step | Median (three runs) | Range of run medians | p95 (median over runs) |
| :--- | :---: | :---: | :---: |
| HuBERT-ECG: preprocessing (authors' pipeline) | 4.18 ms | 4.16–4.18 ms | 4.29 ms |
| HuBERT-ECG: forward pass, CPU | 56.17 ms | 56.12–56.38 ms | 57.06 ms |
| HuBERT-ECG: forward pass, MPS | 23.58 ms | 23.48–23.68 ms | 29.52 ms |
| HuBERT-ECG: linear probe | 0.25 ms | 0.24–0.26 ms | 0.28 ms |
| HuBERT-ECG probe: total, CPU | 60.57 ms | 60.53–60.80 ms | 61.48 ms |
| HuBERT-ECG probe: total, MPS | 28.01 ms | 27.92–28.12 ms | 33.91 ms |
| d1-3B image probe: linear probe (added to the d1-3B forward pass of Table S45) | 0.12 ms | 0.11–0.12 ms | 0.13 ms |
| TabPFN: fit on 988 windows (once per model) | 156.21 ms | 118.95–473.35 ms | 156.21 ms |
| TabPFN: prediction of one window, CPU | 3786.46 ms | 3773.02–3788.83 ms | 3800.81 ms |

## Table S26. CinC 2011 records with a constant lead II (73 of 225 unacceptable records; post hoc, descriptive). Number flagged at τ = 0.50 and AUROC with and without these records; models trained on BUT QDB only.

| Model | Constant-lead records flagged at τ = 0.50 | AUROC, all 998 records | AUROC without the 73 constant-lead records |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0 / 73 | 0.322 | 0.477 |
| **LightGBM (SQI)** | 0 / 73 | 0.752 | 0.665 |
| **Independent SQI MLP** | 73 / 73 | 0.675 | 0.519 |
| **1D CNN (Raw Signal)** | 73 / 73 | 0.695 | 0.660 |
| **2D CNN (Rendered Image)** | 73 / 73 | 0.787 | 0.716 |
| **SinglePass Multimodal Net** | 73 / 73 | 0.500 | 0.500 |
| **Rule-Based SQI Heuristic** | 73 / 73 | 0.678 | 0.524 |
| **LiquidAI d1-3B (Zero-Shot)** | 73 / 73 | 0.641 | 0.469 |
| **TabPFN (SQI)** | 0 / 73 | 0.731 | 0.630 |
| **HuBERT-ECG probe** | 0 / 73 | 0.562 | 0.684 |
| **d1-3B image-token probe** | 73 / 73 | 0.824 | 0.740 |

## Table S27. CinC 2017 at natural prevalence with bootstrap confidence intervals for AUROC and AP (confidence intervals of decision-level metrics: Table S32), seven main models (recording maximum; mean in the last column).

| Model Architecture | AUROC [95% CI], recording max | AP [95% CI], recording max (prevalence 3.3%) | Sens. / FR at τ = 0.50 | PPV / NPV at τ = 0.50 | AUROC, recording mean (secondary) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.870 [0.850 – 0.888] | 0.175 [0.147 – 0.210] | 81.7% / 20.8% | 0.117 / 0.992 | 0.919 |
| **LightGBM (SQI)** | 0.891 [0.875 – 0.905] | 0.227 [0.184 – 0.277] | 68.0% / 12.7% | 0.154 / 0.988 | 0.939 |
| **Independent SQI MLP** | 0.876 [0.858 – 0.893] | 0.163 [0.138 – 0.196] | 79.1% / 16.9% | 0.136 / 0.992 | 0.933 |
| **1D CNN (Raw Signal)** | 0.768 [0.754 – 0.780] | 0.062 [0.055 – 0.071] | 92.4% / 36.1% | 0.080 / 0.996 | 0.797 |
| **2D CNN (Rendered Image)** | 0.850 [0.831 – 0.867] | 0.146 [0.120 – 0.183] | 89.2% / 35.6% | 0.078 / 0.994 | 0.875 |
| **SinglePass Multimodal Net** | 0.799 [0.781 – 0.817] | 0.088 [0.076 – 0.106] | 92.8% / 45.6% | 0.064 / 0.996 | 0.882 |
| **Rule-Based SQI Heuristic** | 0.887 [0.867 – 0.905] | 0.182 [0.155 – 0.217] | 18.3% / 2.5% | 0.199 / 0.973 | 0.933 |

## Table S28. Discrimination of every model in each primary evaluation set: AUROC and average precision with 95% bootstrap confidence intervals (2,000 resamples, seed 42; resampling unit subject on BUT QDB and recording externally). Models trained on BUT QDB only; external sets scored with the pre-specified recording maximum. Intervals differ slightly from Tables S1 and S27, which used other resampling runs.

| Model | BUT_QDB (5-fold subject CV) | CinC2011 lead II (998 records) | CinC2017 (8,512 recordings) | CinC2017 balanced subset (556 recordings) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | AUROC 0.899 [0.790–0.977]; AP 0.759 [0.357–0.943] | AUROC 0.322 [0.290–0.357]; AP 0.180 [0.158–0.203] | AUROC 0.870 [0.851–0.889]; AP 0.175 [0.147–0.211] | AUROC 0.863 [0.830–0.894]; AP 0.820 [0.762–0.873] |
| **LightGBM (SQI)** | AUROC 0.957 [0.879–0.999]; AP 0.894 [0.490–0.993] | AUROC 0.752 [0.711–0.792]; AP 0.515 [0.455–0.587] | AUROC 0.891 [0.876–0.906]; AP 0.227 [0.185–0.277] | AUROC 0.892 [0.863–0.916]; AP 0.866 [0.817–0.907] |
| **Independent SQI MLP** | AUROC 0.904 [0.767–0.977]; AP 0.747 [0.243–0.944] | AUROC 0.675 [0.642–0.706]; AP 0.500 [0.448–0.549] | AUROC 0.876 [0.857–0.893]; AP 0.163 [0.138–0.196] | AUROC 0.875 [0.843–0.904]; AP 0.830 [0.776–0.881] |
| **1D CNN (Raw Signal)** | AUROC 0.983 [0.939–1.000]; AP 0.960 [0.775–1.000] | AUROC 0.695 [0.657–0.732]; AP 0.338 [0.298–0.384] | AUROC 0.768 [0.754–0.780]; AP 0.062 [0.055–0.070] | AUROC 0.761 [0.714–0.805]; AP 0.620 [0.568–0.682] |
| **2D CNN (Rendered Image)** | AUROC 0.985 [0.956–0.999]; AP 0.954 [0.817–0.996] | AUROC 0.787 [0.750–0.821]; AP 0.519 [0.460–0.590] | AUROC 0.850 [0.831–0.867]; AP 0.146 [0.122–0.183] | AUROC 0.862 [0.829–0.892]; AP 0.829 [0.780–0.877] |
| **SinglePass Multimodal Net** | AUROC 0.966 [0.915–0.998]; AP 0.909 [0.675–0.993] | AUROC 0.500 [0.500–0.500]; AP 0.225 [0.200–0.251] | AUROC 0.799 [0.781–0.819]; AP 0.088 [0.075–0.107] | AUROC 0.826 [0.789–0.857]; AP 0.779 [0.721–0.836] |
| **Rule-Based SQI Heuristic** | AUROC 0.971 [0.908–1.000]; AP 0.920 [0.621–0.997] | AUROC 0.678 [0.629–0.722]; AP 0.581 [0.520–0.638] | AUROC 0.887 [0.868–0.905]; AP 0.182 [0.155–0.218] | AUROC 0.876 [0.842–0.905]; AP 0.839 [0.785–0.889] |
| **LiquidAI d1-3B (Zero-Shot)** | AUROC 0.828 [0.685–0.939]; AP 0.688 [0.268–0.895] | AUROC 0.641 [0.592–0.684]; AP 0.559 [0.500–0.613] | AUROC 0.757 [0.729–0.786]; AP 0.113 [0.092–0.149] | AUROC 0.739 [0.697–0.777]; AP 0.715 [0.661–0.768] |
| **TabPFN (SQI)** | AUROC 0.956 [0.895–0.998]; AP 0.898 [0.547–0.992] | AUROC 0.731 [0.689–0.770]; AP 0.499 [0.434–0.573] | AUROC 0.853 [0.829–0.874]; AP 0.202 [0.162–0.247] | AUROC 0.853 [0.820–0.883]; AP 0.835 [0.787–0.877] |
| **HuBERT-ECG probe** | AUROC 0.938 [0.825–0.995]; AP 0.858 [0.545–0.982] | AUROC 0.562 [0.520–0.605]; AP 0.279 [0.242–0.329] | AUROC 0.906 [0.893–0.920]; AP 0.248 [0.207–0.302] | AUROC 0.916 [0.892–0.937]; AP 0.908 [0.872–0.936] |
| **d1-3B image-token probe** | AUROC 0.971 [0.920–0.995]; AP 0.913 [0.568–0.985] | AUROC 0.824 [0.789–0.856]; AP 0.723 [0.669–0.774] | AUROC 0.894 [0.879–0.908]; AP 0.241 [0.201–0.295] | AUROC 0.893 [0.863–0.917]; AP 0.875 [0.837–0.911] |

## Table S29. Distribution of each SQI by dataset and class: median [interquartile range] per window. CinC 2017 windows carry the label of their recording.

| SQI | BUT QDB usable (n = 750) | BUT QDB unusable (n = 238) | CinC 2011 acceptable (n = 773) | CinC 2011 unacceptable (n = 225) | CinC 2017 not noisy (n = 23,396) | CinC 2017 noisy (n = 589) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bsqi_detector_agreement` | 0.952 [0.88–1] | 0.4 [0.278–0.544] | 0.545 [0.417–0.649] | 0.35 [0–0.571] | 0.947 [0.903–1] | 0.667 [0.5–0.811] |
| `baseline_wander_ratio` | 0.00274 [0.00108–0.011] | 0.00303 [0.000909–0.0178] | 0.00356 [0.00159–0.0086] | 0.00313 [0–0.0143] | 0.00918 [0.00349–0.0254] | 0.0254 [0.0114–0.0457] |
| `high_frequency_noise_ratio` | 0.0081 [0.00509–0.0136] | 0.607 [0.00699–1.21] | 0.0146 [0.00748–0.03] | 0.00523 [0–0.0187] | 0.000456 [0.000243–0.000803] | 0.000341 [0.000102–0.0014] |
| `saturation_fraction` | 0.0004 [0.0004–0.0004] | 0.0004 [0.0004–0.0004] | 0.0006 [0.0004–0.0008] | 0.0006 [0.0004–1] | 0.0004 [0.0004–0.0004] | 0.0004 [0.0004–0.0004] |
| `flatline_fraction` | 0 [0–0.0002] | 0 [0–0] | 0.287 [0.206–0.364] | 0.348 [0.23–1] | 0.0006 [0.0002–0.001] | 0.0002 [0–0.0004] |
| `qrs_energy_ratio` | 0.427 [0.364–0.473] | 0.293 [0.208–0.345] | 0.431 [0.314–0.519] | 0.145 [0–0.375] | 0.366 [0.244–0.468] | 0.134 [0.0701–0.266] |
| `kurtosis_ksqi` | 14.6 [4.27–21.4] | 0.0626 [-0.511–2.18] | 9.58 [4.79–16.1] | 2.45 [0–9.9] | 7.75 [3.83–13.5] | 3.88 [1.36–8.82] |
| `skewness_ssqi` | 3.32 [0.862–4.21] | -0.0147 [-0.108–0.0597] | 2.01 [0.377–3.34] | 0 [0–1.87] | 1.44 [-0.0948–2.81] | 0.00185 [-0.95–0.793] |
| `zero_crossing_rate` | 0.0324 [0.0216–0.0442] | 0.139 [0.0234–0.247] | 0.0748 [0.0548–0.101] | 0.0364 [0–0.0744] | 0.0156 [0.0126–0.019] | 0.0152 [0.0106–0.0244] |
| `powerline_noise_ratio` | 0.000968 [0.000607–0.002] | 0.14 [0.000865–0.436] | 0.00162 [0.000762–0.00353] | 0.000558 [0–0.0021] | 3.37e-05 [1.87e-05–5.83e-05] | 2.92e-05 [9.01e-06–0.000123] |
| `signal_std` | 0.312 [0.253–0.412] | 0.318 [0.0484–0.719] | 0.121 [0.0912–0.161] | 0.0962 [0–0.212] | 0.163 [0.123–0.22] | 0.334 [0.186–0.54] |

## Table S30. Univariate AUROC of each SQI on BUT QDB and on CinC 2011 lead II with and without the 73 records whose lead II is constant (post hoc, descriptive). Values below 0.5: higher feature value associated with usable.

| SQI feature | BUT QDB (988 windows) | CinC 2011 lead II, all 998 records | CinC 2011 lead II, without the 73 constant-lead records (925) |
| :--- | :---: | :---: | :---: |
| `bsqi_detector_agreement` | 0.035 | 0.309 | 0.455 |
| `baseline_wander_ratio` | 0.525 | 0.443 | 0.655 |
| `high_frequency_noise_ratio` | 0.722 | 0.307 | 0.454 |
| `saturation_fraction` | 0.501 | 0.623 | 0.441 |
| `flatline_fraction` | 0.422 | 0.634 | 0.459 |
| `qrs_energy_ratio` | 0.164 | 0.220 | 0.325 |
| `kurtosis_ksqi` | 0.129 | 0.279 | 0.410 |
| `skewness_ssqi` | 0.100 | 0.345 | 0.410 |
| `zero_crossing_rate` | 0.660 | 0.256 | 0.379 |
| `powerline_noise_ratio` | 0.717 | 0.311 | 0.461 |
| `signal_std` | 0.480 | 0.410 | 0.607 |

## Table S31. Sensitivity of the multiplicity family: comparisons with the best model of each evaluation set, unadjusted and Holm-adjusted over all 55 pairs (primary) or over the 10 comparisons with the best model. The best model was identified from the point estimates, so the second family is a sensitivity analysis only.

| Evaluation set | Best model | Model | AUROC (best) | p, unadjusted | p_Holm, all 55 pairs | p_Holm, 10 comparisons with the best |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| BUT_QDB (5-fold subject CV) | 2D CNN (Rendered Image) | **Logistic Regression (SQI)** | 0.899 (0.985) | < 0.001 | 0.028 | 0.005 |
| BUT_QDB (5-fold subject CV) | 2D CNN (Rendered Image) | **LightGBM (SQI)** | 0.957 (0.985) | 0.080 | 1.000 | 0.400 |
| BUT_QDB (5-fold subject CV) | 2D CNN (Rendered Image) | **Independent SQI MLP** | 0.904 (0.985) | < 0.001 | 0.028 | 0.005 |
| BUT_QDB (5-fold subject CV) | 2D CNN (Rendered Image) | **1D CNN (Raw Signal)** | 0.983 (0.985) | 0.900 | 1.000 | 0.900 |
| BUT_QDB (5-fold subject CV) | 2D CNN (Rendered Image) | **SinglePass Multimodal Net** | 0.966 (0.985) | 0.043 | 1.000 | 0.258 |
| BUT_QDB (5-fold subject CV) | 2D CNN (Rendered Image) | **Rule-Based SQI Heuristic** | 0.971 (0.985) | 0.184 | 1.000 | 0.552 |
| BUT_QDB (5-fold subject CV) | 2D CNN (Rendered Image) | **LiquidAI d1-3B (Zero-Shot)** | 0.828 (0.985) | 0.001 | 0.051 | 0.008 |
| BUT_QDB (5-fold subject CV) | 2D CNN (Rendered Image) | **d1-3B image-token probe** | 0.971 (0.985) | 0.237 | 1.000 | 0.552 |
| BUT_QDB (5-fold subject CV) | 2D CNN (Rendered Image) | **TabPFN (SQI)** | 0.956 (0.985) | 0.093 | 1.000 | 0.400 |
| BUT_QDB (5-fold subject CV) | 2D CNN (Rendered Image) | **HuBERT-ECG probe** | 0.938 (0.985) | 0.016 | 0.640 | 0.112 |
| CinC2011 lead II (998 records) | d1-3B image-token probe | **Logistic Regression (SQI)** | 0.322 (0.824) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (998 records) | d1-3B image-token probe | **LightGBM (SQI)** | 0.752 (0.824) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (998 records) | d1-3B image-token probe | **Independent SQI MLP** | 0.675 (0.824) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (998 records) | d1-3B image-token probe | **1D CNN (Raw Signal)** | 0.695 (0.824) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (998 records) | d1-3B image-token probe | **2D CNN (Rendered Image)** | 0.787 (0.824) | < 0.001 | 0.003 | < 0.001 |
| CinC2011 lead II (998 records) | d1-3B image-token probe | **SinglePass Multimodal Net** | 0.500 (0.824) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (998 records) | d1-3B image-token probe | **Rule-Based SQI Heuristic** | 0.678 (0.824) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (998 records) | d1-3B image-token probe | **LiquidAI d1-3B (Zero-Shot)** | 0.641 (0.824) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (998 records) | d1-3B image-token probe | **TabPFN (SQI)** | 0.731 (0.824) | < 0.001 | < 0.001 | < 0.001 |
| CinC2011 lead II (998 records) | d1-3B image-token probe | **HuBERT-ECG probe** | 0.562 (0.824) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | HuBERT-ECG probe | **Logistic Regression (SQI)** | 0.870 (0.906) | < 0.001 | 0.004 | < 0.001 |
| CinC2017 (8,512 recordings) | HuBERT-ECG probe | **LightGBM (SQI)** | 0.891 (0.906) | 0.076 | 0.844 | 0.152 |
| CinC2017 (8,512 recordings) | HuBERT-ECG probe | **Independent SQI MLP** | 0.876 (0.906) | 0.001 | 0.031 | 0.006 |
| CinC2017 (8,512 recordings) | HuBERT-ECG probe | **1D CNN (Raw Signal)** | 0.768 (0.906) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | HuBERT-ECG probe | **2D CNN (Rendered Image)** | 0.850 (0.906) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | HuBERT-ECG probe | **SinglePass Multimodal Net** | 0.799 (0.906) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | HuBERT-ECG probe | **Rule-Based SQI Heuristic** | 0.887 (0.906) | 0.039 | 0.525 | 0.118 |
| CinC2017 (8,512 recordings) | HuBERT-ECG probe | **LiquidAI d1-3B (Zero-Shot)** | 0.757 (0.906) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | HuBERT-ECG probe | **TabPFN (SQI)** | 0.853 (0.906) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 (8,512 recordings) | HuBERT-ECG probe | **d1-3B image-token probe** | 0.894 (0.906) | 0.092 | 0.844 | 0.152 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **Logistic Regression (SQI)** | 0.863 (0.916) | < 0.001 | 0.009 | 0.001 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **LightGBM (SQI)** | 0.892 (0.916) | 0.062 | 1.000 | 0.123 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **Independent SQI MLP** | 0.875 (0.916) | 0.005 | 0.132 | 0.019 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **1D CNN (Raw Signal)** | 0.761 (0.916) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **2D CNN (Rendered Image)** | 0.862 (0.916) | < 0.001 | 0.009 | 0.001 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **SinglePass Multimodal Net** | 0.826 (0.916) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **Rule-Based SQI Heuristic** | 0.876 (0.916) | 0.006 | 0.164 | 0.019 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **LiquidAI d1-3B (Zero-Shot)** | 0.739 (0.916) | < 0.001 | < 0.001 | < 0.001 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **TabPFN (SQI)** | 0.853 (0.916) | < 0.001 | 0.002 | < 0.001 |
| CinC2017 balanced subset (556 recordings) | HuBERT-ECG probe | **d1-3B image-token probe** | 0.893 (0.916) | 0.070 | 1.000 | 0.123 |

## Table S32. Decision-level metrics at τ = 0.50 with 95% bootstrap confidence intervals (2,000 resamples; subject level on BUT QDB, recording level externally) and calibration intercept and slope on the logit scale. Calibration intercept is interpreted only for external sets. Not estimable: scores saturated at 0 or 1.

| Dataset | Model | Units (positive) | Sensitivity, % [95% CI] | False reject, % [95% CI] | PPV, % [95% CI] | NPV, % [95% CI] | Calibration intercept [95% CI] | Calibration slope [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BUT_QDB (5-fold subject CV) | **Logistic Regression (SQI)** | 988 (238) | 53.4 [26.5–86.7] | 6.4 [1.6–14.0] | 72.6 [33.0–91.0] | 86.3 [75.4–97.4] | 0.32 [-1.92–1.38] | 0.55 [0.34–1.03] |
| BUT_QDB (5-fold subject CV) | **LightGBM (SQI)** | 988 (238) | 67.6 [29.9–97.8] | 4.4 [1.7–8.1] | 83.0 [44.2–93.6] | 90.3 [80.4–99.7] | 0.74 [-1.40–1.62] | 0.76 [0.60–1.57] |
| BUT_QDB (5-fold subject CV) | **Independent SQI MLP** | 988 (238) | 71.4 [38.0–94.9] | 9.1 [2.1–20.4] | 71.4 [28.4–91.5] | 90.9 [81.8–98.9] | -0.11 [-2.45–0.89] | 0.66 [0.35–1.18] |
| BUT_QDB (5-fold subject CV) | **1D CNN (Raw Signal)** | 988 (238) | 75.2 [43.8–100.0] | 0.8 [0.1–1.7] | 96.8 [84.3–99.4] | 92.7 [84.5–100.0] | 3.10 [-1.42–5.44] | 0.56 [0.39–1.20] |
| BUT_QDB (5-fold subject CV) | **2D CNN (Rendered Image)** | 988 (238) | 83.2 [51.9–100.0] | 1.6 [0.5–2.8] | 94.3 [79.6–98.4] | 94.9 [86.3–100.0] | 1.25 [-1.98–2.80] | 0.77 [0.59–2.11] |
| BUT_QDB (5-fold subject CV) | **SinglePass Multimodal Net** | 988 (238) | 53.8 [16.4–98.0] | 2.0 [0.7–3.7] | 89.5 [55.4–97.8] | 87.0 [73.0–99.7] | 2.00 [-2.47–3.45] | 0.61 [0.47–0.92] |
| BUT_QDB (5-fold subject CV) | **Rule-Based SQI Heuristic** | 988 (238) | 57.6 [13.3–94.9] | 1.2 [0.3–2.5] | 93.8 [62.5–98.7] | 88.0 [76.4–99.3] | -0.64 [-1.60–-0.16] | 5.09 [3.96–8.43] |
| BUT_QDB (5-fold subject CV) | **LiquidAI d1-3B (Zero-Shot)** | 988 (238) | 50.4 [16.4–81.6] | 4.0 [1.6–6.9] | 80.0 [34.3–92.6] | 85.9 [76.0–96.3] | -0.61 [-1.72–-0.08] | 2.66 [1.64–4.45] |
| BUT_QDB (5-fold subject CV) | **TabPFN (SQI)** | 988 (238) | 58.0 [12.8–94.9] | 0.8 [0.3–1.3] | 95.8 [76.7–97.9] | 88.2 [76.5–99.3] | 2.21 [-0.25–3.32] | 0.65 [0.54–1.22] |
| BUT_QDB (5-fold subject CV) | **HuBERT-ECG probe** | 988 (238) | 55.5 [26.7–83.2] | 2.1 [0.4–4.7] | 89.2 [59.3–97.9] | 87.4 [78.6–96.9] | 2.49 [0.45–4.15] | 0.51 [0.30–1.22] |
| BUT_QDB (5-fold subject CV) | **d1-3B image-token probe** | 988 (238) | 60.1 [18.3–96.7] | 1.6 [0.1–3.9] | 92.3 [52.3–99.3] | 88.6 [77.8–99.2] | 2.42 [0.03–3.66] | 0.63 [0.39–1.15] |
| CinC2011 lead II (998 records) | **Logistic Regression (SQI)** | 998 (225) | 67.1 [60.9–73.3] | 100.0 [100.0–100.0] | 16.3 [13.9–18.8] | 0.0 [0.0–0.0] | not estimable | not estimable |
| CinC2011 lead II (998 records) | **LightGBM (SQI)** | 998 (225) | 17.3 [12.2–22.4] | 4.9 [3.4–6.5] | 50.6 [39.7–62.2] | 79.8 [77.3–82.4] | 0.77 [0.53–0.99] | 0.49 [0.40–0.62] |
| CinC2011 lead II (998 records) | **Independent SQI MLP** | 998 (225) | 32.9 [26.7–39.1] | 0.0 [0.0–0.0] | 100.0 [100.0–100.0] | 83.7 [81.2–86.1] | not estimable | not estimable |
| CinC2011 lead II (998 records) | **1D CNN (Raw Signal)** | 998 (225) | 70.7 [64.6–76.5] | 25.1 [22.0–28.3] | 45.0 [39.9–50.1] | 89.8 [87.3–92.0] | -2.60 [-3.65–-1.78] | 0.08 [0.06–0.10] |
| CinC2011 lead II (998 records) | **2D CNN (Rendered Image)** | 998 (225) | 55.1 [48.5–61.5] | 7.2 [5.4–9.2] | 68.9 [62.4–75.7] | 87.7 [85.2–89.8] | 2.07 [1.26–2.88] | 0.22 [0.18–0.26] |
| CinC2011 lead II (998 records) | **SinglePass Multimodal Net** | 998 (225) | 100.0 [100.0–100.0] | 100.0 [100.0–100.0] | 22.5 [20.0–25.1] | — | not estimable | not estimable |
| CinC2011 lead II (998 records) | **Rule-Based SQI Heuristic** | 998 (225) | 92.0 [88.4–95.3] | 87.8 [85.4–90.0] | 23.4 [20.6–26.1] | 83.9 [76.6–90.4] | -1.74 [-1.89–-1.61] | 2.16 [1.64–2.70] |
| CinC2011 lead II (998 records) | **LiquidAI d1-3B (Zero-Shot)** | 998 (225) | 99.6 [98.6–100.0] | 100.0 [100.0–100.0] | 22.5 [19.9–25.0] | 0.0 [0.0–0.0] | not estimable | 0.90 [0.67–1.13] |
| CinC2011 lead II (998 records) | **TabPFN (SQI)** | 998 (225) | 12.0 [7.9–16.4] | 4.0 [2.7–5.4] | 46.6 [33.9–60.0] | 78.9 [76.4–81.5] | 1.12 [0.87–1.35] | 0.38 [0.29–0.48] |
| CinC2011 lead II (998 records) | **HuBERT-ECG probe** | 998 (225) | 40.0 [33.8–46.7] | 27.0 [23.8–30.2] | 30.1 [25.0–35.4] | 80.7 [77.7–83.6] | -0.92 [-1.30–-0.54] | 0.07 [0.03–0.12] |
| CinC2011 lead II (998 records) | **d1-3B image-token probe** | 998 (225) | 49.8 [43.2–56.2] | 2.7 [1.6–3.9] | 84.2 [78.0–90.4] | 86.9 [84.6–89.1] | 3.40 [2.79–4.03] | 0.25 [0.22–0.28] |
| CinC2017 (8,512 recordings) | **Logistic Regression (SQI)** | 8,512 (278) | 81.7 [77.0–86.0] | 20.8 [19.9–21.7] | 11.7 [10.3–13.1] | 99.2 [99.0–99.4] | -4.63 [-4.89–-4.39] | 0.46 [0.43–0.50] |
| CinC2017 (8,512 recordings) | **LightGBM (SQI)** | 8,512 (278) | 68.0 [62.4–73.3] | 12.7 [12.0–13.4] | 15.4 [13.4–17.5] | 98.8 [98.5–99.0] | -3.17 [-3.33–-3.01] | 0.64 [0.59–0.70] |
| CinC2017 (8,512 recordings) | **Independent SQI MLP** | 8,512 (278) | 79.1 [74.4–83.8] | 16.9 [16.1–17.7] | 13.6 [12.0–15.4] | 99.2 [98.9–99.4] | -3.55 [-3.71–-3.40] | 0.62 [0.56–0.68] |
| CinC2017 (8,512 recordings) | **1D CNN (Raw Signal)** | 8,512 (278) | 92.4 [89.1–95.3] | 36.1 [35.1–37.1] | 8.0 [7.1–9.0] | 99.6 [99.4–99.8] | -14.75 [-14.93–-14.57] | 0.10 [0.09–0.11] |
| CinC2017 (8,512 recordings) | **2D CNN (Rendered Image)** | 8,512 (278) | 89.2 [85.3–92.7] | 35.6 [34.6–36.6] | 7.8 [6.9–8.8] | 99.4 [99.2–99.6] | -6.97 [-7.18–-6.77] | 0.35 [0.32–0.39] |
| CinC2017 (8,512 recordings) | **SinglePass Multimodal Net** | 8,512 (278) | 92.8 [89.7–95.8] | 45.6 [44.5–46.7] | 6.4 [5.7–7.3] | 99.6 [99.3–99.7] | -10.55 [-10.94–-10.17] | 0.19 [0.18–0.21] |
| CinC2017 (8,512 recordings) | **Rule-Based SQI Heuristic** | 8,512 (278) | 18.3 [14.1–23.2] | 2.5 [2.2–2.8] | 19.9 [15.1–25.3] | 97.3 [96.9–97.6] | -2.53 [-2.66–-2.41] | 2.87 [2.66–3.10] |
| CinC2017 (8,512 recordings) | **LiquidAI d1-3B (Zero-Shot)** | 8,512 (278) | 19.8 [15.3–24.6] | 3.1 [2.7–3.5] | 17.7 [13.5–22.2] | 97.3 [96.9–97.6] | -2.47 [-2.60–-2.35] | 1.66 [1.47–1.87] |
| CinC2017 (8,512 recordings) | **TabPFN (SQI)** | 8,512 (278) | 66.5 [61.0–72.2] | 13.5 [12.7–14.2] | 14.3 [12.4–16.4] | 98.7 [98.4–99.0] | -3.50 [-3.68–-3.32] | 0.46 [0.42–0.50] |
| CinC2017 (8,512 recordings) | **HuBERT-ECG probe** | 8,512 (278) | 92.4 [89.2–95.5] | 26.3 [25.3–27.2] | 10.6 [9.4–12.0] | 99.7 [99.5–99.8] | -3.88 [-4.02–-3.74] | 0.93 [0.85–1.02] |
| CinC2017 (8,512 recordings) | **d1-3B image-token probe** | 8,512 (278) | 86.3 [82.4–90.1] | 23.5 [22.6–24.4] | 11.0 [9.7–12.4] | 99.4 [99.2–99.6] | -5.61 [-5.83–-5.41] | 0.44 [0.41–0.48] |
| CinC2017 balanced subset (556 recordings) | **Logistic Regression (SQI)** | 556 (278) | 81.7 [76.9–86.0] | 19.8 [15.2–24.8] | 80.5 [75.7–85.0] | 81.4 [76.3–85.8] | -0.15 [-0.43–0.14] | 0.63 [0.51–0.79] |
| CinC2017 balanced subset (556 recordings) | **LightGBM (SQI)** | 556 (278) | 68.0 [62.6–73.8] | 13.3 [9.6–17.2] | 83.6 [78.6–88.1] | 73.0 [68.2–77.8] | 0.94 [0.62–1.26] | 0.74 [0.64–0.87] |
| CinC2017 balanced subset (556 recordings) | **Independent SQI MLP** | 556 (278) | 79.1 [74.1–83.9] | 17.3 [13.0–21.9] | 82.1 [77.2–86.5] | 79.9 [75.0–84.3] | 0.45 [0.16–0.78] | 0.69 [0.58–0.82] |
| CinC2017 balanced subset (556 recordings) | **1D CNN (Raw Signal)** | 556 (278) | 92.4 [89.1–95.6] | 37.1 [31.6–43.0] | 71.4 [66.5–76.3] | 89.3 [84.5–93.5] | -2.91 [-3.64–-2.20] | 0.15 [0.12–0.19] |
| CinC2017 balanced subset (556 recordings) | **2D CNN (Rendered Image)** | 556 (278) | 89.2 [85.3–92.5] | 33.8 [28.2–39.7] | 72.5 [67.4–77.4] | 86.0 [80.8–90.4] | -1.79 [-2.25–-1.30] | 0.43 [0.37–0.51] |
| CinC2017 balanced subset (556 recordings) | **SinglePass Multimodal Net** | 556 (278) | 92.8 [89.6–95.6] | 44.2 [38.6–50.2] | 67.7 [62.7–72.4] | 88.6 [83.6–92.9] | -2.85 [-3.32–-2.36] | 0.30 [0.25–0.36] |
| CinC2017 balanced subset (556 recordings) | **Rule-Based SQI Heuristic** | 556 (278) | 18.3 [13.8–23.1] | 3.2 [1.4–5.6] | 85.0 [75.0–93.3] | 54.2 [49.6–58.8] | 0.67 [0.51–0.84] | 4.39 [3.41–5.76] |
| CinC2017 balanced subset (556 recordings) | **LiquidAI d1-3B (Zero-Shot)** | 556 (278) | 19.8 [15.0–24.5] | 4.0 [1.8–6.4] | 83.3 [73.9–91.7] | 54.5 [49.9–59.1] | 0.80 [0.63–0.97] | 1.56 [1.23–1.95] |
| CinC2017 balanced subset (556 recordings) | **TabPFN (SQI)** | 556 (278) | 66.5 [60.9–72.1] | 12.9 [9.2–17.2] | 83.7 [78.8–88.4] | 72.2 [67.1–77.1] | 1.40 [0.98–1.84] | 0.48 [0.40–0.56] |
| CinC2017 balanced subset (556 recordings) | **HuBERT-ECG probe** | 556 (278) | 92.4 [89.1–95.4] | 27.7 [22.8–33.2] | 76.9 [72.2–81.6] | 90.5 [86.5–94.2] | -0.43 [-0.65–-0.19] | 1.16 [1.00–1.38] |
| CinC2017 balanced subset (556 recordings) | **d1-3B image-token probe** | 556 (278) | 86.3 [82.2–90.1] | 21.2 [16.8–26.3] | 80.3 [75.6–84.7] | 85.2 [80.4–89.5] | -0.51 [-1.00–-0.01] | 0.49 [0.42–0.56] |

## Table S33. 2D CNN retrained on strips drawn by a direct raster renderer versus the plotting-library strips, same folds and BUT-only transfer. Secondary analysis; Holm-adjusted over the three comparisons.

| Evaluation set | Units (positive) | Matplotlib strips AUROC | Raster strips AUROC | Δ (raster − Matplotlib) [95% CI] | p (Holm) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| BUT QDB (5-fold subject CV) | 988 (238) | 0.985 | 0.980 | -0.005 [-0.027 to +0.005] | 0.843 |
| CinC 2011 lead II | 998 (225) | 0.787 | 0.798 | +0.012 [+0.003 to +0.021] | 0.025 |
| CinC 2017 (recording max) | 8,512 (278) | 0.850 | 0.852 | +0.002 [-0.003 to +0.007] | 0.843 |

## Table S34. Exploratory quantization-insensitive flatline input (main text, sections 2.4 and 2.6). AUROC with the original and the new flatline input; no confidence intervals or tests were computed. Distinct scores: number of distinct predicted values, an indicator of saturation.

| Dataset | Model | Units | AUROC, original flatline | AUROC, robust flatline | Distinct scores, original | Distinct scores, robust |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| BUT_QDB | Independent SQI MLP | 988 | 0.904 | 0.916 | 931 | 931 |
| BUT_QDB | LightGBM (SQI) | 988 | 0.957 | 0.956 | 697 | 683 |
| BUT_QDB | Logistic Regression (SQI) | 988 | 0.899 | 0.903 | 929 | 931 |
| CinC2011 | Independent SQI MLP | 998 | 0.675 | 0.737 | 15 | 909 |
| CinC2011 | LightGBM (SQI) | 998 | 0.752 | 0.759 | 900 | 884 |
| CinC2011 | Logistic Regression (SQI) | 998 | 0.322 | 0.431 | 28 | 925 |
| CinC2017 | Independent SQI MLP | 8,512 | 0.876 | 0.887 | 8,510 | 8,504 |
| CinC2017 | LightGBM (SQI) | 8,512 | 0.891 | 0.889 | 6,573 | 6,042 |
| CinC2017 | Logistic Regression (SQI) | 8,512 | 0.870 | 0.879 | 8,510 | 8,510 |

## Table S35. CinC 2011 scored on lead II and on 12 leads (maximum and mean over leads), with and without the 135 records that have at least one constant lead (signal SD 0; 129 of them unacceptable). Post hoc and descriptive; no tests. Without these records: 863 records, 96 unacceptable. Number of constant leads alone (all 998 records): AUROC 0.784.

| Model Architecture | Lead II: all 998 | Lead II: without constant-lead records | Maximum: all 998 | Maximum: without constant-lead records | Mean: all 998 | Mean: without constant-lead records |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.322 | 0.467 | 0.351 | 0.500 | 0.154 | 0.355 |
| **LightGBM (SQI)** | 0.752 | 0.715 | 0.617 | 0.753 | 0.854 | 0.837 |
| **Independent SQI MLP** | 0.675 | 0.527 | 0.838 | 0.631 | 0.840 | 0.631 |
| **1D CNN (Raw Signal)** | 0.695 | 0.707 | 0.229 | 0.307 | 0.789 | 0.769 |
| **2D CNN (Rendered Image)** | 0.787 | 0.759 | 0.718 | 0.854 | 0.882 | 0.878 |
| **SinglePass Multimodal Net** | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| **Rule-Based SQI Heuristic** | 0.678 | 0.517 | 0.842 | 0.650 | 0.719 | 0.513 |
| **LiquidAI d1-3B (Zero-Shot)** | 0.641 | 0.424 | 0.813 | 0.570 | 0.644 | 0.401 |
| **d1-3B image-token probe** | 0.824 | 0.795 | 0.951 | 0.902 | 0.944 | 0.912 |
| **HuBERT-ECG probe** | 0.562 | 0.736 | 0.431 | 0.683 | 0.510 | 0.805 |

## Table S36. Diagnosis of the constant multimodal-net score on CinC 2011 lead II (seed 42; descriptive, no tests). Top: inputs standardised with the BUT QDB mean and SD. Bottom: rescoring with the same fitted net.

| Input | BUT QDB mean | BUT QDB SD | CinC 2011 median z | CinC 2011 max abs z | Records with abs z > 10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| bsqi_detector_agreement | 0.7889 | 0.2551 | -1.04 | 3.09 | 0.0% |
| baseline_wander_ratio | 0.01298 | 0.0257 | -0.37 | 52.62 | 0.6% |
| high_frequency_noise_ratio | 3.233 | 13.4 | -0.24 | 0.24 | 0.0% |
| saturation_fraction | 0.0004004 | 8.999e-06 | 22.18 | 111075.10 | 57.7% |
| flatline_fraction | 6.903e-05 | 0.0001272 | 2286.40 | 7863.77 | 100.0% |
| qrs_energy_ratio | 0.3715 | 0.1203 | 0.17 | 3.09 | 0.0% |
| kurtosis_ksqi | 11.91 | 12.08 | -0.29 | 3.15 | 0.0% |
| skewness_ssqi | 2.059 | 2.181 | -0.20 | 5.53 | 0.0% |
| zero_crossing_rate | 0.05729 | 0.06857 | 0.18 | 5.79 | 0.0% |
| powerline_noise_ratio | 3.092 | 13.05 | -0.24 | 0.24 | 0.0% |
| signal_std | 0.4655 | 0.9998 | -0.35 | 40.87 | 1.5% |

| Variant | Records | Distinct scores | AUROC |
| :--- | :---: | :---: | :---: |
| Original inputs | 998 | 1 | 0.500 |
| Image stream only | 998 | 930 | 0.801 |
| SQI stream only | 998 | 1 | 0.500 |
| Flatline input replaced by quantization-insensitive measure | 998 | 798 | 0.719 |

## Table S37. Effect of rejecting recordings by quality score on rhythm classification, CinC 2017 (8,234 recordings with a rhythm label; macro-F1 without rejection 0.719). Values are the change in macro-F1 minus the mean change under random rejection of the same fraction, with 95% bootstrap interval; intervals are not adjusted for multiplicity and no tests were run. Quality-score ties were broken at random (1D CNN: 1,159 recordings share a value with another recording, so its figures depend slightly on the tie-break; a deterministic tie-break changed its macro-F1 by at most 0.0002). Bottom: paired differences in the change.

| Quality model | Reject 5% | Reject 10% (primary) | Reject 20% | Rejected at 10%: N / A / O |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | +0.0012 (-0.0019 to +0.0039) | +0.0028 (-0.0013 to +0.0069) | +0.0092 (+0.0032 to +0.0154) | 506 / 63 / 254 |
| **LightGBM (SQI)** | +0.0006 (-0.0022 to +0.0035) | +0.0029 (-0.0014 to +0.0067) | +0.0073 (+0.0019 to +0.0130) | 518 / 61 / 244 |
| **Independent SQI MLP** | +0.0010 (-0.0016 to +0.0039) | +0.0028 (-0.0008 to +0.0070) | +0.0091 (+0.0033 to +0.0149) | 528 / 61 / 234 |
| **1D CNN (Raw Signal)** | +0.0070 (+0.0037 to +0.0098) | +0.0082 (+0.0037 to +0.0127) | +0.0107 (+0.0046 to +0.0175) | 507 / 85 / 231 |
| **2D CNN (Rendered Image)** | -0.0012 (-0.0043 to +0.0020) | +0.0017 (-0.0024 to +0.0062) | +0.0055 (-0.0004 to +0.0114) | 442 / 55 / 326 |
| **SinglePass Multimodal Net** | -0.0005 (-0.0034 to +0.0024) | +0.0010 (-0.0033 to +0.0050) | +0.0057 (-0.0007 to +0.0114) | 441 / 50 / 332 |
| **Rule-Based SQI Heuristic** | +0.0012 (-0.0018 to +0.0043) | +0.0023 (-0.0019 to +0.0067) | +0.0057 (-0.0001 to +0.0118) | 476 / 71 / 276 |
| **LiquidAI d1-3B (Zero-Shot)** | +0.0013 (-0.0018 to +0.0040) | +0.0023 (-0.0021 to +0.0064) | +0.0064 (+0.0000 to +0.0126) | 461 / 75 / 287 |
| **TabPFN (SQI)** | +0.0017 (-0.0007 to +0.0045) | +0.0050 (+0.0010 to +0.0090) | +0.0055 (-0.0005 to +0.0105) | 544 / 53 / 226 |
| **HuBERT-ECG probe** | -0.0007 (-0.0039 to +0.0029) | +0.0012 (-0.0036 to +0.0056) | +0.0073 (+0.0001 to +0.0141) | 389 / 73 / 361 |
| **d1-3B image-token probe** | +0.0063 (+0.0036 to +0.0092) | +0.0082 (+0.0046 to +0.0126) | +0.0107 (+0.0054 to +0.0169) | 500 / 47 / 276 |

| Comparison | Reject 5% | Reject 10% | Reject 20% |
| :--- | :---: | :---: | :---: |
| d1-3B image-token probe minus 2D CNN | +0.0075 (+0.0042 to +0.0110) | +0.0066 (+0.0024 to +0.0110) | +0.0052 (-0.0003 to +0.0112) |
| d1-3B image-token probe minus HuBERT-ECG probe | +0.0070 (+0.0031 to +0.0108) | +0.0071 (+0.0026 to +0.0125) | +0.0035 (-0.0028 to +0.0103) |

## Table S38. Frozen test on PTB-XL v1.0.3 (one record per patient; models fitted on BUT QDB only). AUROC with 95% bootstrap interval. Global versus clean: 600 records, 300 global. Partial versus clean: 600 records, 300 partial. The outcome is annotated noise, a proxy for unusability, and the sampled prevalence is artificial. No record had a constant lead, so the planned analysis without such records is identical.

| Model Architecture | Global, lead II | Global, maximum | Global, mean | Partial, maximum | Partial, mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.394 (0.370–0.418) | 0.493 (0.486–0.498) | 0.299 (0.267–0.331) | 0.500 (0.500–0.500) | 0.436 (0.408–0.463) |
| **LightGBM (SQI)** | 0.631 (0.586–0.677) | 0.689 (0.647–0.732) | 0.694 (0.651–0.736) | 0.603 (0.560–0.647) | 0.610 (0.566–0.653) |
| **Independent SQI MLP** | 0.555 (0.538–0.574) | 0.622 (0.594–0.651) | 0.622 (0.594–0.651) | 0.528 (0.509–0.547) | 0.528 (0.509–0.547) |
| **1D CNN (Raw Signal)** | 0.584 (0.536–0.628) | 0.457 (0.412–0.501) | 0.574 (0.530–0.621) | 0.535 (0.492–0.578) | 0.580 (0.532–0.627) |
| **2D CNN (Rendered Image)** | 0.497 (0.447–0.543) | 0.501 (0.451–0.546) | 0.521 (0.473–0.569) | 0.479 (0.434–0.526) | 0.482 (0.437–0.529) |
| **SinglePass Multimodal Net** | 0.500 (0.500–0.500) | 0.500 (0.500–0.500) | 0.502 (0.500–0.505) | 0.500 (0.500–0.500) | 0.502 (0.500–0.505) |
| **Rule-Based SQI Heuristic** | 0.429 (0.385–0.475) | 0.408 (0.359–0.455) | 0.392 (0.349–0.438) | 0.437 (0.393–0.481) | 0.428 (0.380–0.473) |
| **LiquidAI d1-3B (Zero-Shot)** | 0.231 (0.196–0.269) | 0.224 (0.190–0.261) | 0.184 (0.151–0.219) | 0.419 (0.375–0.466) | 0.276 (0.236–0.318) |
| **d1-3B image-token probe** | 0.623 (0.576–0.667) | 0.579 (0.533–0.628) | 0.592 (0.547–0.641) | 0.515 (0.468–0.566) | 0.519 (0.473–0.568) |
| **HuBERT-ECG probe** | 0.645 (0.597–0.689) | 0.736 (0.694–0.775) | 0.694 (0.649–0.736) | 0.678 (0.637–0.720) | 0.643 (0.599–0.687) |

## Table S39. Paired comparisons of AUROC on PTB-XL, global noise versus clean (DeLong). Primary: d1-3B image probe versus 2D CNN, Holm over two tests. Secondary: d1-3B image probe versus the other nine models under both rules, Holm over 18 tests.

| Family | Comparator | Rule | AUROC probe | AUROC comparator | Difference (95% CI) | p | p_Holm |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| primary | 2D CNN (Rendered Image) | Maximum | 0.579 | 0.501 | +0.078 (+0.031 to +0.125) | 0.001 | 0.001 |
| primary | 2D CNN (Rendered Image) | Mean | 0.592 | 0.521 | +0.071 (+0.031 to +0.111) | <0.001 | 0.001 |
| secondary | Logistic Regression (SQI) | Maximum | 0.579 | 0.493 | +0.086 (+0.039 to +0.133) | <0.001 | 0.002 |
| secondary | LightGBM (SQI) | Maximum | 0.579 | 0.689 | -0.110 (-0.157 to -0.063) | <0.001 | <0.001 |
| secondary | Independent SQI MLP | Maximum | 0.579 | 0.622 | -0.042 (-0.089 to +0.004) | 0.074 | 0.223 |
| secondary | 1D CNN (Raw Signal) | Maximum | 0.579 | 0.457 | +0.123 (+0.057 to +0.188) | <0.001 | 0.002 |
| secondary | 2D CNN (Rendered Image) | Maximum | 0.579 | 0.501 | +0.078 (+0.031 to +0.125) | 0.001 | 0.005 |
| secondary | SinglePass Multimodal Net | Maximum | 0.579 | 0.500 | +0.079 (+0.034 to +0.125) | <0.001 | 0.003 |
| secondary | Rule-Based SQI Heuristic | Maximum | 0.579 | 0.408 | +0.171 (+0.118 to +0.224) | <0.001 | <0.001 |
| secondary | LiquidAI d1-3B (Zero-Shot) | Maximum | 0.579 | 0.224 | +0.355 (+0.290 to +0.420) | <0.001 | <0.001 |
| secondary | HuBERT-ECG probe | Maximum | 0.579 | 0.736 | -0.156 (-0.200 to -0.113) | <0.001 | <0.001 |
| secondary | Logistic Regression (SQI) | Mean | 0.592 | 0.299 | +0.294 (+0.234 to +0.354) | <0.001 | <0.001 |
| secondary | LightGBM (SQI) | Mean | 0.592 | 0.694 | -0.102 (-0.147 to -0.057) | <0.001 | <0.001 |
| secondary | Independent SQI MLP | Mean | 0.592 | 0.622 | -0.029 (-0.076 to +0.017) | 0.217 | 0.433 |
| secondary | 1D CNN (Raw Signal) | Mean | 0.592 | 0.574 | +0.018 (-0.022 to +0.059) | 0.377 | 0.433 |
| secondary | 2D CNN (Rendered Image) | Mean | 0.592 | 0.521 | +0.071 (+0.031 to +0.111) | <0.001 | 0.003 |
| secondary | SinglePass Multimodal Net | Mean | 0.592 | 0.502 | +0.091 (+0.045 to +0.136) | <0.001 | <0.001 |
| secondary | Rule-Based SQI Heuristic | Mean | 0.592 | 0.392 | +0.201 (+0.144 to +0.257) | <0.001 | <0.001 |
| secondary | LiquidAI d1-3B (Zero-Shot) | Mean | 0.592 | 0.184 | +0.408 (+0.348 to +0.468) | <0.001 | <0.001 |
| secondary | HuBERT-ECG probe | Mean | 0.592 | 0.694 | -0.101 (-0.136 to -0.067) | <0.001 | <0.001 |

## Table S40. CinC 2011 lead II without the 73 records whose lead II is constant (925 records, 152 unacceptable): comparisons with the highest-AUROC model (DeLong). Post hoc sensitivity analysis added after an independent audit; models trained on BUT QDB only. Holm adjustment over all 55 pairs, as in the primary family, and over the 10 comparisons with the best model.

| Model | AUROC | Difference from the best [95% CI] | p, unadjusted | p_Holm, all 55 pairs | p_Holm, 10 comparisons with the best |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **d1-3B image-token probe** (best) | 0.740 | — | — | — | — |
| **2D CNN (Rendered Image)** | 0.716 | -0.024 [-0.050 to +0.002] | 0.069 | 0.833 | 0.069 |
| **HuBERT-ECG probe** | 0.684 | -0.056 [-0.082 to -0.030] | < 0.001 | < 0.001 | < 0.001 |
| **LightGBM (SQI)** | 0.665 | -0.076 [-0.114 to -0.037] | < 0.001 | 0.003 | < 0.001 |
| **1D CNN (Raw Signal)** | 0.660 | -0.080 [-0.112 to -0.047] | < 0.001 | < 0.001 | < 0.001 |
| **TabPFN (SQI)** | 0.630 | -0.110 [-0.157 to -0.063] | < 0.001 | < 0.001 | < 0.001 |
| **Rule-Based SQI Heuristic** | 0.524 | -0.216 [-0.281 to -0.152] | < 0.001 | < 0.001 | < 0.001 |
| **Independent SQI MLP** | 0.519 | -0.222 [-0.267 to -0.176] | < 0.001 | < 0.001 | < 0.001 |
| **SinglePass Multimodal Net** | 0.500 | -0.240 [-0.286 to -0.195] | < 0.001 | < 0.001 | < 0.001 |
| **Logistic Regression (SQI)** | 0.477 | -0.263 [-0.317 to -0.209] | < 0.001 | < 0.001 | < 0.001 |
| **LiquidAI d1-3B (Zero-Shot)** | 0.469 | -0.272 [-0.347 to -0.196] | < 0.001 | < 0.001 | < 0.001 |

## Table S41. BUT QDB: AUROC of the out-of-fold scores inside each subject that has both classes (unusable / all windows of the subject in parentheses), pooled over these seven subjects, and over all 15 subjects (primary estimate). Descriptive; no tests. Subject 105 has one unusable window.

| Model | Subject 100 (22 / 72) | Subject 105 (1 / 51) | Subject 111 (60 / 110) | Subject 113 (4 / 54) | Subject 114 (60 / 110) | Subject 122 (60 / 110) | Subject 124 (31 / 81) | These subjects pooled (238 / 588) | All 15 subjects (238 / 988) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | 0.998 | 1.000 | 0.959 | 1.000 | 1.000 | 1.000 | 0.997 | 0.988 | 0.985 |
| **1D CNN (Raw Signal)** | 0.971 | 1.000 | 0.961 | 1.000 | 1.000 | 1.000 | 1.000 | 0.986 | 0.983 |
| **Rule-Based SQI Heuristic** | 0.942 | 1.000 | 0.704 | 1.000 | 1.000 | 1.000 | 0.948 | 0.960 | 0.971 |
| **d1-3B image-token probe** | 0.994 | 1.000 | 0.920 | 1.000 | 1.000 | 1.000 | 0.935 | 0.978 | 0.971 |
| **SinglePass Multimodal Net** | 0.996 | 1.000 | 0.932 | 1.000 | 1.000 | 0.961 | 1.000 | 0.978 | 0.966 |
| **LightGBM (SQI)** | 0.924 | 1.000 | 0.757 | 0.985 | 1.000 | 1.000 | 0.909 | 0.950 | 0.957 |
| **TabPFN (SQI)** | 0.951 | 1.000 | 0.811 | 0.980 | 0.999 | 1.000 | 0.845 | 0.941 | 0.956 |
| **HuBERT-ECG probe** | 0.985 | 1.000 | 0.689 | 1.000 | 1.000 | 0.961 | 0.995 | 0.948 | 0.938 |
| **Independent SQI MLP** | 0.897 | 1.000 | 0.765 | 0.955 | 1.000 | 0.936 | 0.863 | 0.925 | 0.904 |
| **Logistic Regression (SQI)** | 0.947 | 1.000 | 0.760 | 0.975 | 1.000 | 0.933 | 0.917 | 0.908 | 0.899 |
| **LiquidAI d1-3B (Zero-Shot)** | 0.540 | 1.000 | 0.752 | 1.000 | 1.000 | 0.755 | 0.885 | 0.820 | 0.828 |

## Table S42. CinC 2017: AUROC by number of 10-s windows per recording (noisy / all recordings in parentheses) for the recording maximum (primary rule) and mean; with one window the two rules coincide. Last row: AUROC of the number of windows alone. Descriptive; no tests. Models trained on BUT QDB only.

| Model | All recordings (278 / 8512), maximum | All recordings (278 / 8512), mean | 1 window (110 / 600) | 2 windows (25 / 351), maximum | 2 windows (25 / 351), mean | 3 windows (143 / 7561), maximum | 3 windows (143 / 7561), mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HuBERT-ECG probe** | 0.906 | 0.941 | 0.867 | 0.901 | 0.922 | 0.953 | 0.959 |
| **d1-3B image-token probe** | 0.894 | 0.935 | 0.874 | 0.877 | 0.912 | 0.924 | 0.949 |
| **LightGBM (SQI)** | 0.891 | 0.939 | 0.893 | 0.875 | 0.911 | 0.923 | 0.956 |
| **Rule-Based SQI Heuristic** | 0.887 | 0.933 | 0.856 | 0.881 | 0.899 | 0.951 | 0.972 |
| **Independent SQI MLP** | 0.876 | 0.933 | 0.884 | 0.864 | 0.913 | 0.908 | 0.945 |
| **Logistic Regression (SQI)** | 0.870 | 0.919 | 0.835 | 0.856 | 0.894 | 0.929 | 0.954 |
| **TabPFN (SQI)** | 0.853 | 0.890 | 0.861 | 0.845 | 0.868 | 0.880 | 0.900 |
| **2D CNN (Rendered Image)** | 0.850 | 0.875 | 0.832 | 0.848 | 0.848 | 0.884 | 0.880 |
| **SinglePass Multimodal Net** | 0.799 | 0.882 | 0.836 | 0.782 | 0.869 | 0.831 | 0.879 |
| **1D CNN (Raw Signal)** | 0.768 | 0.797 | 0.751 | 0.766 | 0.798 | 0.799 | 0.812 |
| **LiquidAI d1-3B (Zero-Shot)** | 0.757 | 0.816 | 0.796 | 0.667 | 0.707 | 0.818 | 0.821 |
| Number of windows alone (fewer windows scored higher) | 0.698 | — | — | — | — | — | — |

## Table S43. BUT QDB: Class 3 candidate windows per subject over all annotated intervals and inside the official intervals, and unusable windows in the cohort. The quota of up to 60 unusable windows per subject was drawn over all annotated intervals before the windows outside the official intervals were removed. Subjects without Class 3 windows are omitted.

| Subject | Class 3 candidate windows, all annotated intervals | Class 3 candidate windows, inside the official intervals | Unusable windows in the cohort | Unusable windows a quota drawn inside the official intervals would give |
| :--- | :---: | :---: | :---: | :---: |
| 100 | 22 | 22 | 22 | 22 |
| 105 | 4,734 | 54 | 1 | 54 |
| 111 | 292 | 292 | 60 | 60 |
| 113 | 4 | 4 | 4 | 4 |
| 114 | 83 | 83 | 60 | 60 |
| 122 | 119 | 119 | 60 | 60 |
| 124 | 31 | 31 | 31 | 31 |
| **Total** | 5,285 | 605 | 238 | 291 |

## Table S44. AUROC of each model in the four evaluation sets with its rank among the 11 models (1 = highest), and Spearman correlations of the AUROCs between sets. Descriptive and post hoc; the unit is the model (n = 11), so the correlations are imprecise. The two CinC 2017 sets are not independent of each other.

| Model | BUT QDB: AUROC (rank) | CinC2011 lead II: AUROC (rank) | CinC2017: AUROC (rank) | CinC2017 balanced subset: AUROC (rank) |
| :--- | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | 0.985 (1) | 0.787 (2) | 0.850 (8) | 0.862 (7) |
| **1D CNN (Raw Signal)** | 0.983 (2) | 0.695 (5) | 0.768 (10) | 0.761 (10) |
| **Rule-Based SQI Heuristic** | 0.971 (3) | 0.678 (6) | 0.887 (4) | 0.876 (4) |
| **d1-3B image-token probe** | 0.971 (4) | 0.824 (1) | 0.894 (2) | 0.893 (2) |
| **SinglePass Multimodal Net** | 0.966 (5) | 0.500 (10) | 0.799 (9) | 0.826 (9) |
| **LightGBM (SQI)** | 0.957 (6) | 0.752 (3) | 0.891 (3) | 0.892 (3) |
| **TabPFN (SQI)** | 0.956 (7) | 0.731 (4) | 0.853 (7) | 0.853 (8) |
| **HuBERT-ECG probe** | 0.938 (8) | 0.562 (9) | 0.906 (1) | 0.916 (1) |
| **Independent SQI MLP** | 0.904 (9) | 0.675 (7) | 0.876 (5) | 0.875 (5) |
| **Logistic Regression (SQI)** | 0.899 (10) | 0.322 (11) | 0.870 (6) | 0.863 (6) |
| **LiquidAI d1-3B (Zero-Shot)** | 0.828 (11) | 0.641 (8) | 0.757 (11) | 0.739 (11) |

| Evaluation sets compared | Models | Spearman ρ of AUROC | p (descriptive) |
| :--- | :---: | :---: | :---: |
| BUT QDB vs CinC2011 lead II | 11 | +0.61 | 0.047 |
| BUT QDB vs CinC2017 | 11 | -0.02 | 0.958 |
| BUT QDB vs CinC2017 balanced subset | 11 | +0.04 | 0.915 |
| CinC2011 lead II vs CinC2017 | 11 | +0.19 | 0.574 |
| CinC2011 lead II vs CinC2017 balanced subset | 11 | +0.21 | 0.537 |
| CinC2017 vs CinC2017 balanced subset | 11 | +0.99 | < 0.001 |

## Table S45. Per-window latency on an Apple M4 (median and 95th percentile of three runs of 25 windows), separating preprocessing from the forward pass. Every model is timed from the stored 500-Hz window to its decision; rendering is included for every model that needs a strip image, including the rule heuristic. (Table 5 of the main text in an earlier version.)

| Model Architecture | Parameters | Disk Weights | Execution Device | Preproc Median (p95) | Forward Median (p95) | Total Median (p95) | Total Median, Range over Runs | Preproc Share (Median of Ratios) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based SQI Heuristic** | 0 | 0.00 MiB | CPU | 62.44 ms (142.96 ms) | 0.25 ms (0.28 ms) | 62.70 ms (143.21 ms) | 61.91–63.83 ms | 99.6% |
| **Logistic Regression (SQI)** | 12 | 0.01 MiB | CPU | 1.35 ms (1.51 ms) | 0.18 ms (0.24 ms) | 1.53 ms (1.76 ms) | 1.51–1.54 ms | 88.3% |
| **LightGBM (SQI)** | 1,094 | 0.14 MiB | CPU | 1.34 ms (1.56 ms) | 0.18 ms (0.25 ms) | 1.51 ms (1.78 ms) | 1.49–1.52 ms | 87.7% |
| **Independent SQI MLP** | 3,073 | 0.01 MiB | Apple M4 (MPS) | 1.37 ms (1.68 ms) | 1.03 ms (1.75 ms) | 2.42 ms (3.38 ms) | 2.39–2.62 ms | 56.2% |
| **1D CNN (Raw Signal)** | 167,489 | 0.64 MiB | Apple M4 (MPS) | 0.00 ms (0.00 ms) | 1.31 ms (2.10 ms) | 1.31 ms (2.11 ms) | 1.29–1.56 ms | 0.0% |
| **2D CNN (Rendered Image)** | 245,697 | 0.94 MiB | Apple M4 (MPS) | 60.72 ms (131.51 ms) | 5.83 ms (6.25 ms) | 65.92 ms (141.08 ms) | 65.41–66.29 ms | 91.4% |
| **SinglePass Multimodal Net** | 131,841 | 0.47 MiB | Apple M4 (MPS) | 62.88 ms (131.44 ms) | 5.60 ms (6.17 ms) | 68.32 ms (137.26 ms) | 67.34–71.56 ms | 91.9% |
| **LiquidAI d1-3B (Zero-Shot)** | 3,120,000,000 | 5958.00 MiB | Apple M4 (MPS) | 64.16 ms (153.21 ms) | 830.84 ms (834.07 ms) | 896.00 ms (984.59 ms) | 891.65–896.13 ms | 7.2% |

## Table S46. Harmonised-pipeline comparison (post hoc, secondary; a results-informed sensitivity analysis whose definition was committed before any harmonised score was computed but after the earlier external results were known). AUROC with the original pipeline and with three input safeguards: G1, constant-input rule (a window with zero signal SD is scored unusable); G2, quantization-insensitive flatline input; G3, standardised inputs clipped to [−3, 3] with training statistics. Δ with 95% CI: subject cluster bootstrap on BUT QDB, DeLong externally; no multiplicity adjustment. Models marked "G1 only" were not refitted; the SQI values in the zero-shot d1-3B prompt were left unchanged. Seed 42; rows ordered by harmonised AUROC on CinC 2011. BUT QDB contains no constant window.

| Model | Safeguards applied | BUT QDB (988 windows): original → harmonised (Δ [95% CI]) | CinC 2011 lead II (998 records): original → harmonised (Δ [95% CI]) | CinC 2017 (8,512 recordings): original → harmonised (Δ [95% CI]) | Distinct scores on CinC 2011: original → harmonised |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **d1-3B image-token probe** | G1 only | 0.971 → 0.971 (unchanged) | 0.824 → 0.824 (+0.000 [-0.000 to +0.001]) | 0.894 → 0.894 (unchanged) | 930 → 926 |
| **2D CNN (Rendered Image)** | G1 only | 0.985 → 0.985 (unchanged) | 0.787 → 0.808 (+0.022 [+0.015 to +0.029]) | 0.850 → 0.850 (unchanged) | 930 → 926 |
| **HuBERT-ECG probe** | G1 only | 0.938 → 0.938 (unchanged) | 0.562 → 0.787 (+0.224 [+0.180 to +0.268]) | 0.906 → 0.906 (unchanged) | 926 → 926 |
| **LightGBM (SQI)** | G1, G2, G3 | 0.957 → 0.956 (-0.001 [-0.009 to +0.005]) | 0.752 → 0.779 (+0.026 [+0.016 to +0.037]) | 0.891 → 0.888 (-0.004 [-0.007 to -0.001]) | 900 → 832 |
| **1D CNN (Raw Signal)** | G1 only | 0.983 → 0.983 (unchanged) | 0.695 → 0.770 (+0.075 [+0.058 to +0.092]) | 0.768 → 0.768 (unchanged) | 911 → 909 |
| **Logistic Regression (SQI)** | G1, G2, G3 | 0.899 → 0.912 (+0.013 [+0.000 to +0.029]) | 0.322 → 0.766 (+0.443 [+0.380 to +0.507]) | 0.870 → 0.890 (+0.020 [+0.013 to +0.026]) | 28 → 926 |
| **Independent SQI MLP** | G1, G2, G3 | 0.904 → 0.899 (-0.005 [-0.048 to +0.071]) | 0.675 → 0.761 (+0.086 [+0.051 to +0.122]) | 0.876 → 0.889 (+0.013 [+0.001 to +0.025]) | 15 → 913 |
| **TabPFN (SQI)** | G1, G2, G3 | 0.956 → 0.955 (-0.001 [-0.008 to +0.004]) | 0.731 → 0.741 (+0.010 [-0.002 to +0.022]) | 0.853 → 0.857 (+0.004 [+0.001 to +0.007]) | 926 → 926 |
| **Rule-Based SQI Heuristic** | G1, G2 | 0.971 → 0.971 (-0.000 [-0.000 to +0.000]) | 0.678 → 0.738 (+0.060 [+0.032 to +0.089]) | 0.887 → 0.888 (+0.001 [+0.000 to +0.001]) | 926 → 926 |
| **SinglePass Multimodal Net** | G1, G2, G3 | 0.966 → 0.941 (-0.024 [-0.062 to +0.000]) | 0.500 → 0.731 (+0.231 [+0.186 to +0.276]) | 0.799 → 0.858 (+0.058 [+0.047 to +0.070]) | 1 → 914 |
| **LiquidAI d1-3B (Zero-Shot)** | G1 only | 0.828 → 0.828 (unchanged) | 0.641 → 0.641 (unchanged) | 0.757 → 0.757 (unchanged) | 48 → 48 |

## Table S47. Harmonised-pipeline comparison, continued. Top: Spearman correlation of the 11 AUROCs between evaluation sets, original and harmonised (descriptive; n = 11 models). Middle: seed range of the refitted networks under BUT-only transfer. Bottom: share of windows whose standardised input lies beyond 3 SD of the BUT QDB distribution, with the median absolute z-score, before clipping.

| Evaluation sets compared | Models | Spearman ρ, original AUROCs | Spearman ρ, harmonised AUROCs |
| :--- | :---: | :---: | :---: |
| BUT QDB vs CinC 2011 lead II | 11 | +0.61 | +0.49 |
| BUT QDB vs CinC 2017 | 11 | -0.02 | -0.19 |
| CinC 2011 lead II vs CinC 2017 | 11 | +0.19 | +0.43 |

| Model (harmonised, BUT-only transfer) | Evaluation set | AUROC, seed 42 | AUROC, range over seeds 42 and 1–4 | Distinct scores, range over seeds |
| :--- | :--- | :---: | :---: | :---: |
| **Independent SQI MLP** | CinC 2011 lead II (998 records) | 0.761 | 0.686–0.776 | 913–914 |
| **SinglePass Multimodal Net** | CinC 2011 lead II (998 records) | 0.731 | 0.706–0.768 | 905–914 |
| **Independent SQI MLP** | CinC 2017 (8,512 recordings) | 0.889 | 0.873–0.891 | 8499–8509 |
| **SinglePass Multimodal Net** | CinC 2017 (8,512 recordings) | 0.858 | 0.821–0.858 | 8421–8497 |

| Input (standardised with BUT QDB mean and SD) | BUT QDB: share of windows beyond 3 SD (median absolute z) | CinC 2011 lead II: share of windows beyond 3 SD (median absolute z) | CinC 2017: share of windows beyond 3 SD (median absolute z) |
| :--- | :---: | :---: | :---: |
| bsqi_detector_agreement | 0.1% (0.828) | 7.8% (1.04) | 0.1% (0.641) |
| baseline_wander_ratio | 2.4% (0.441) | 2.5% (0.419) | 3.3% (0.384) |
| high_frequency_noise_ratio | 4.6% (0.241) | 0.0% (0.24) | 0.0% (0.241) |
| saturation_fraction | 0.2% (0.045) | 57.7% (22.2) | 3.1% (0.045) |
| flatline_fraction | 1.0% (0.543) | 100.0% (2.29e+03) | 52.6% (4.18) |
| qrs_energy_ratio | 0.1% (0.71) | 7.7% (1.08) | 0.1% (0.924) |
| kurtosis_ksqi | 0.7% (0.796) | 1.2% (0.549) | 0.5% (0.52) |
| skewness_ssqi | 0.4% (0.902) | 1.1% (0.72) | 0.9% (0.648) |
| zero_crossing_rate | 3.5% (0.439) | 0.2% (0.366) | 0.0% (0.608) |
| powerline_noise_ratio | 4.6% (0.237) | 0.0% (0.237) | 0.0% (0.237) |
| signal_std | 0.8% (0.191) | 2.0% (0.354) | 0.0% (0.303) |
| flatline_fraction_robust (replaces flatline_fraction; G2) | 0.0% (0) | 0.0% (0) | 0.0% (0) |
