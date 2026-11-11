# Supplementary Material

<!-- Template filled by evaluation/assemble_q1_manuscript.py (writes manuscript/v2/supplement_v2.md). -->

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

<!-- source: Table 1 -->
{{Table 1}}

## Table S2. Nested versus post-hoc threshold selection on BUT QDB.

<!-- source: Table 1c -->
{{Table 1c}}

## Table S3. BUT QDB precision metrics projected to the prevalence of Class 3 inside the official intervals (2.14%, primary) and over all annotated intervals including the excluded stretch (16.0%; tabulated for continuity with the earlier version, not interpreted). The projection changes the class ratio only and assumes that the score distributions within each class in the quota cohort hold for all eligible windows; the values are conditional projections, not estimates for the full eligible population (main text, section 2.4; Table S43).

<!-- source: Table 1e -->
{{Table 1e}}

## Table S4. Thresholds chosen on pooled BUT QDB out-of-fold scores and applied unchanged to the external datasets. CinC 2011: all 998 records, lead II. CinC 2017: all 8,512 recordings (recording maximum).

<!-- source: Table 2f -->
{{Table 2f}}

## Table S5. CinC 2011 (998 records) scored on all 12 leads (aggregation rules chosen after the main analysis). Both foundation-model probes were fitted on BUT QDB only and scored after the other models' 12-lead results were known. TabPFN was not scored on 12 leads.

<!-- source: Table 2e -->
{{Table 2e}}

## Table S6. Univariate AUROC of each SQI across datasets (values below 0.5: higher feature value associated with usable).

<!-- source: Table 2g -->
{{Table 2g}}

## Table S7. NSTDB: paired change in usable probability relative to the clean partner window (mean ± SD).

<!-- source: Table 3b -->
{{Table 3b}}

## Table S8. Extended latency profile (Apple M4; `experiments/latency_extended.py`). This is a separate script from Table S45 (`experiments/empirical_latency_benchmark.py`): CPU and MPS passes are interleaved per window, and the first window of each run is excluded. Decoded image tensors are cached after first use, so image decoding and resizing are timed only in the first run; the reported median over runs reflects cached inputs. MPS forward times of a few milliseconds therefore differ from Table S45 (Supplementary Methods S-M5).

<!-- source: Table 4b -->
{{Table 4b}}

## Table S9. Seed stability of the 1D CNN, 2D CNN and multimodal net. CinC rows refer to the 998-record CinC 2011 evaluation.

<!-- source: Table 5 -->
{{Table 5}}

## Table S10. Post-hoc ROC operating points on BUT QDB (exploratory; thresholds chosen on the evaluated data).

<!-- source: Table 1b -->
{{Table 1b}}

## Table S11. Sensitivity analysis with Class 2 relabelled as unusable (BUT QDB).

<!-- source: Table 1d -->
{{Table 1d}}

## Table S12. Subject-level sensitivity on BUT QDB.

<!-- source: Table 1f -->
{{Table 1f}}

## Table S13. CinC 2011 Set A, all 998 records, lead II.

<!-- source: Table 2 -->
{{Table 2}}

## Table S14. Post-hoc ROC operating points on CinC 2011, all 998 records (exploratory).

<!-- source: Table 2b -->
{{Table 2b}}

## Table S15. Feature-set and scaler sensitivity of the tabular models under transfer. CinC 2011: all 998 records.

<!-- source: Table 2c -->
{{Table 2c}}

## Table S16. NSTDB: usable probability by nominal SNR.

<!-- source: Table 3 -->
{{Table 3}}

## Table S17. Prompt sensitivity of zero-shot d1-3B. The prompt variants were scored on a 150-record balanced CinC 2011 subset (75 acceptable, 75 unacceptable) and were not extended to all 998 records.

<!-- source: Table 7 -->
{{Table 7}}

## Table S18. Linear probes on frozen d1-3B representations.

<!-- source: Table 7b -->
{{Table 7b}}

## Table S19. CinC 2017 balanced subset (278 noisy + 278 non-noisy recordings) including zero-shot d1-3B.

<!-- source: Table 8b -->
{{Table 8b}}

## Table S20. Hand-set rule heuristic versus a logistic-regression replacement using the same seven input features (standardised L2 logistic regression fitted inside the BUT QDB folds and transferred like the other models). The replacement changes the functional form as well as the coefficients (Supplementary Methods S-M3). p: BUT QDB, subject-level cluster bootstrap; external sets, DeLong; unadjusted.

{{CSV:heuristic_refit}}

## Table S21. Seed repetition (seeds 42 and 1–4) of BUT-only training and leave-one-dataset-out training for the seed-dependent networks. Consistent: same sign of Δ in all five seeds and |median Δ| ≥ 0.02 (pre-specified).

{{Table 10b}}

## Table S22. Local recalibration on n = 25–200 randomly drawn labelled target recordings (500 draws each), evaluated on the remaining recordings. Prevalence-only Brier score: CinC 2011 0.175, CinC 2017 0.032.

{{Table 11b}}

## Table S23. HuBERT-ECG: linear probe and partial fine-tuning (AUROC). Fine-tuning was not run in LODO.

{{Table 12}}

## Table S24. BUT QDB official protocol (Smital et al. 2026): models retrained on the official training records and tested on the official test records (window-level binary task). 95% CI: subject-level cluster bootstrap; p_Holm: paired comparison with the best model.

{{Table 13}}

## Table S25. Latency of the foundation-model probes and TabPFN (Apple M4; same 25 windows and three runs as Table S45, harness of Table S8).

{{Table 4c}}

## Table S26. CinC 2011 records with a constant lead II (73 of 225 unacceptable records; post hoc, descriptive). Number flagged at τ = 0.50 and AUROC with and without these records; models trained on BUT QDB only.

{{Table 2h}}

## Table S27. CinC 2017 at natural prevalence with bootstrap confidence intervals for AUROC and AP (confidence intervals of decision-level metrics: Table S32), seven main models (recording maximum; mean in the last column).

{{Table 8}}

## Table S28. Discrimination of every model in each primary evaluation set: AUROC and average precision with 95% bootstrap confidence intervals (2,000 resamples, seed 42; resampling unit subject on BUT QDB and recording externally). Models trained on BUT QDB only; external sets scored with the pre-specified recording maximum. Intervals differ slightly from Tables S1 and S27, which used other resampling runs.

{{Table 1g}}

## Table S29. Distribution of each SQI by dataset and class: median [interquartile range] per window. CinC 2017 windows carry the label of their recording.

{{Table 2i}}

## Table S30. Univariate AUROC of each SQI on BUT QDB and on CinC 2011 lead II with and without the 73 records whose lead II is constant (post hoc, descriptive). Values below 0.5: higher feature value associated with usable.

{{Table 2j}}

## Table S31. Sensitivity of the multiplicity family: comparisons with the best model of each evaluation set, unadjusted and Holm-adjusted over all 55 pairs (primary) or over the 10 comparisons with the best model. The best model was identified from the point estimates, so the second family is a sensitivity analysis only.

{{Table 9b}}

## Table S32. Decision-level metrics at τ = 0.50 with 95% bootstrap confidence intervals (2,000 resamples; subject level on BUT QDB, recording level externally) and calibration intercept and slope on the logit scale. Calibration intercept is interpreted only for external sets. Not estimable: scores saturated at 0 or 1.

{{Table 1h}}

## Table S33. 2D CNN retrained on strips drawn by a direct raster renderer versus the plotting-library strips, same folds and BUT-only transfer. Secondary analysis; Holm-adjusted over the three comparisons.

{{Table 14}}

## Table S34. Exploratory quantization-insensitive flatline input (main text, sections 2.4 and 2.6). AUROC with the original and the new flatline input; no confidence intervals or tests were computed. Distinct scores: number of distinct predicted values, an indicator of saturation.

{{Table 15}}

## Table S35. CinC 2011 scored on 12 leads, with and without the 135 records that have at least one constant lead (signal SD 0; 129 of them unacceptable). Post hoc and descriptive; no tests. Without these records: 863 records, 96 unacceptable. Number of constant leads alone (all 998 records): AUROC 0.784.

{{Table 16}}

## Table S36. Diagnosis of the constant multimodal-net score on CinC 2011 lead II (seed 42; descriptive, no tests). Top: inputs standardised with the BUT QDB mean and SD. Bottom: rescoring with the same fitted net.

{{Table 17}}

## Table S37. Effect of rejecting recordings by quality score on rhythm classification, CinC 2017 (8,234 recordings with a rhythm label; macro-F1 without rejection 0.719). Values are the change in macro-F1 minus the mean change under random rejection of the same fraction, with 95% bootstrap interval; intervals are not adjusted for multiplicity and no tests were run. Quality-score ties were broken at random (1D CNN: 1,159 recordings share a value with another recording, so its figures depend slightly on the tie-break; a deterministic tie-break changed its macro-F1 by at most 0.0002). Bottom: paired differences in the change.

{{Table 18}}

## Table S38. Frozen test on PTB-XL v1.0.3 (one record per patient; models fitted on BUT QDB only). AUROC with 95% bootstrap interval. Global versus clean: 600 records, 300 global. Partial versus clean: 600 records, 300 partial. The outcome is annotated noise, a proxy for unusability, and the sampled prevalence is artificial. No record had a constant lead, so the planned analysis without such records is identical.

{{Table 19}}

## Table S39. Paired comparisons of AUROC on PTB-XL, global noise versus clean (DeLong). Primary: d1-3B image probe versus 2D CNN, Holm over two tests. Secondary: d1-3B image probe versus the other nine models under both rules, Holm over 18 tests.

{{Table 20}}

## Table S40. CinC 2011 lead II without the 73 records whose lead II is constant (925 records, 152 unacceptable): comparisons with the highest-AUROC model (DeLong). Post hoc sensitivity analysis added after an independent audit; models trained on BUT QDB only. Holm adjustment over all 55 pairs, as in the primary family, and over the 10 comparisons with the best model.

{{Table 21}}

## Table S41. BUT QDB: AUROC of the out-of-fold scores inside each subject that has both classes (unusable / all windows of the subject in parentheses), pooled over these seven subjects, and over all 15 subjects (primary estimate). Descriptive; no tests. Subject 105 has one unusable window.

{{Table 22}}

## Table S42. CinC 2017: AUROC by number of 10-s windows per recording (noisy / all recordings in parentheses) for the recording maximum (primary rule) and mean; with one window the two rules coincide. Last row: AUROC of the number of windows alone. Descriptive; no tests. Models trained on BUT QDB only.

{{Table 23}}

## Table S43. BUT QDB: Class 3 candidate windows per subject over all annotated intervals and inside the official intervals, and unusable windows in the cohort. The quota of up to 60 unusable windows per subject was drawn over all annotated intervals before the windows outside the official intervals were removed. Subjects without Class 3 windows are omitted.

{{Table 24}}

## Table S44. AUROC of each model in the four evaluation sets with its rank among the 11 models (1 = highest), and Spearman correlations of the AUROCs between sets. Descriptive and post hoc; the unit is the model (n = 11), so the correlations are imprecise. The two CinC 2017 sets are not independent of each other.

{{Table 25}}

## Table S45. Per-window latency on an Apple M4 (median and 95th percentile of three runs of 25 windows), separating preprocessing from the forward pass. Every model is timed from the stored 500-Hz window to its decision; rendering is included for every model that needs a strip image, including the rule heuristic. (Table 5 of the main text in an earlier version.)

{{Table 4}}

## Table S46. Harmonised-pipeline comparison (post hoc, secondary; definition committed before any harmonised score was computed). AUROC with the original pipeline and with three input safeguards: G1, constant-input rule (a window with zero signal SD is scored unusable); G2, quantization-insensitive flatline input; G3, standardised inputs clipped to [−3, 3] with training statistics. Δ with 95% CI: subject cluster bootstrap on BUT QDB, DeLong externally; no multiplicity adjustment. Models marked "G1 only" were not refitted. Seed 42; rows ordered by harmonised AUROC on CinC 2011. BUT QDB contains no constant window.

{{Table 26}}

## Table S47. Harmonised-pipeline comparison, continued. Top: Spearman correlation of the 11 AUROCs between evaluation sets, original and harmonised (descriptive; n = 11 models). Middle: seed range of the refitted networks under BUT-only transfer. Bottom: share of windows whose standardised input lies beyond 3 SD of the BUT QDB distribution, with the median absolute z-score, before clipping.

{{Table 27}}
