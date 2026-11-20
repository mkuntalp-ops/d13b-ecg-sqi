# Supplementary Information

**In a retrospective benchmark, in-domain performance did not identify the electrocardiogram quality models that ranked best on heterogeneous external datasets**

Mehmet Kuntalp, Department of Electrical and Electronics Engineering, Dokuz Eylül University, İzmir, Turkey

<!-- Scientific Reports version. Template filled by evaluation/assemble_q1_manuscript.py --variant srep (writes manuscript/srep/supplement_srep.md). -->

Every table is generated from saved outputs by evaluation/build_manuscript_tables.py (results/v2_2026-11/manuscript_tables_2026-10-12.md), except Supplementary Table S20, which is read from results/v2_2026-11/heuristic_refit_summary.csv, Supplementary Table S51, which summarises model documentation checked by hand, and Supplementary Table S61, an inventory of the analyses compiled by hand from the Methods. Where a table is computed by a separate script, its caption names the script and its output file (for example Supplementary Tables S50, S52–S60 and S62). Supplementary Methods, Supplementary Results and the Supplementary Note are text. Positive class: unusable.

## Supplementary Methods

All definitions below were checked against the code in the repository (`features/sqi_extractor.py`, `features/qrs_detectors.py`, `models/*.py`).

### Signal quality indices

Each 10-s window is at 500 Hz, in mV, with n = 5,000 samples. Spectral quantities use Welch's method with 2-s segments.

**The 11 tabular inputs (items 1–11).** Items 12 and 13 are constant after windowing and resampling and are not model inputs; they appear only in the text summary given to zero-shot d1-3B (see Zero-shot d1-3B input).
1. **bSQI.** F1 agreement between two QRS detectors, a simplified re-implementation of the Pan–Tompkins detector (its band-pass filtering, differentiation, squaring and moving-window integration, but a single global threshold in place of its adaptive thresholds and search-back) and a second-difference detector (0.5-Hz high-pass, |x[n+2] − 2x[n] + x[n−2]|, 80-ms moving average). Beats are matched one to one within 50 ms, and bSQI is 0 if either detector finds no beat.
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

### Network architectures

All networks were trained with AdamW (learning rate 10^-3, weight decay 10^-3) on binary cross-entropy with the positive class weighted by the usable/unusable ratio of the training data, without early stopping, and with seed 42 for the reported run.

The SQI MLP (3,073 parameters) has the layers Linear(11→64), BatchNorm, ReLU, Dropout 0.2, Linear(64→32), BatchNorm, ReLU and Linear(32→1). Its inputs are standardised with the training-set mean and SD, and it was trained for 35 epochs with batch size 64.

The 1D CNN (167,489 parameters) has four blocks of Conv1d, BatchNorm, LeakyReLU (slope 0.1), MaxPool (×4) and Dropout 0.15, with 32, 64, 128 and 128 channels and kernel sizes 15, 11, 7 and 5. Global average pooling is followed by Linear(128→32), ReLU, Dropout 0.25 and Linear(32→1). Its input is the 5,000-sample window, z-scored per window (an SD below 10^-6 is replaced by 1). It was trained for 25 epochs with batch size 32 and a cosine learning-rate schedule.

The 2D CNN (245,697 parameters) receives the rendered strip resized to 3 × 128 × 256 and normalised with mean 0.95 and SD 0.15 per channel. It has four blocks of 3 × 3 Conv2d, BatchNorm and ReLU, with 2 × 2 max-pooling after the first three and 32, 64, 128 and 128 channels, followed by global average pooling, Linear(128→32), ReLU, Dropout 0.3 and Linear(32→1). It was trained for 25 epochs with batch size 32 and a cosine schedule.

The multimodal net (131,841 parameters) combines an image encoder and an SQI encoder. The image encoder has three blocks of Conv2d, BatchNorm and ReLU (32, 64 and 128 channels), global average pooling, Linear(128→64) and ReLU; the SQI encoder has Linear(11→64), BatchNorm, ReLU, Dropout 0.2, Linear(64→64) and ReLU. The concatenated 128-d vector is multiplied element-wise by a sigmoid gate, Linear(128→128), and passed to a head with Linear(128→64), ReLU, Dropout 0.3 and Linear(64→1). It was trained for 25 epochs with batch size 32 and a cosine schedule.

### Rule heuristic (`models/rule_heuristic_baseline.py`)

**SQI component.** With the inputs listed under Signal quality indices:

logit_SQI = −3.0·(bSQI − 0.5) + 1.2·min(baseline-wander ratio, 3) + 1.5·min(high-frequency ratio, 3) + 5.0·saturation + 5.0·flatline − 0.5

p_SQI = 1 / (1 + exp(−logit_SQI))

**Image component.** The rendered strip is converted to 8-bit grey scale; *ink* is the share of pixels with value below 200, and *spread* is the SD over image columns of the per-column ink share:

logit_vis = 4.0·(ink − 0.08) − 2.0·(spread − 0.02)

p_vis = 1 / (1 + exp(−logit_vis))

**Output.** The score is p = 0.6·p_SQI + 0.4·p_vis, clipped to [0.01, 0.99]. All coefficients are hand-set. They were set during the initial prototyping of the analysis code, and their independence from BUT QDB cannot be guaranteed; the leakage-free in-domain comparison is the logistic-regression replacement fitted inside the folds (Supplementary Table S20).

### Zero-shot d1-3B input (`models/d1_real_neural_model.py`)

**Image.** The rendered strip.

**State text.** A summary of 10 inputs (Signal quality indices, items 1–8, 12 and 13), in the format:

"ECG Signal Quality Indicators (SQIs): QRS Detector Agreement (bSQI): x (1.0 = perfect consensus, 0.0 = total disagreement); Baseline Wander Ratio; High-Frequency Noise Ratio; Saturation Fraction (%); Flatline Fraction (%); QRS Band Energy Ratio; Kurtosis and Skewness; Window Duration and Sampling Rate."

**Question.** A choice question whose instructions read: "Determine whether this electrocardiogram (ECG) recording segment is usable for reliable QRS-based rhythm analysis, or unusable due to severe artifacts, baseline wander, high-frequency noise, or saturation/flatline." The criterion for "usable" is "QRS complexes are distinguishable and suitable for rhythm analysis. Artifacts are manageable.", and the criterion for "unusable" is "ECG is severely corrupted by noise, baseline drift, saturation, or absent complexes; requires human review or re-recording."

**Output.** The probability of "unusable" from a single forward pass with zero output tokens. Model revision: 051bcc464b01b9f92942b364d9586b0ef5912432.

### Latency measurement (`experiments/latency_all.sh`)

All latency scripts were run in one session by a single entry point with no other computation running. Latency reflects one machine and Python implementations, and TabPFN was timed on the CPU only. Supplementary Table S45 and Supplementary Table S8 come from different scripts: in Supplementary Table S45 (`experiments/empirical_latency_benchmark.py`) the models are timed one after another in one process, whereas the script of Supplementary Table S8 (`experiments/latency_extended.py`) interleaves CPU and MPS passes per window and excludes the first window. Their MPS forward times of a few milliseconds differ (1D CNN 1.31 vs 2.75 ms; 2D CNN 5.83 vs 3.09 ms). Part of the image-model difference is explained by caching: the image models cache each decoded and resized strip after first use, so decoding and resizing fall inside the timed forward pass for 19 of 25 windows in Supplementary Table S45, but only in the first of three runs in Supplementary Table S8. The remaining differences, including the 1D CNN's, are unexplained, so MPS forward times at this scale are treated as harness-dependent; the conclusions rest only on differences of an order of magnitude, namely rendering of about 61 ms versus the forward pass, and 831 ms for d1-3B. Every model in Supplementary Table S45 is timed from the same input, the stored 500-Hz window, to its decision: models that need a strip image (the 2D CNN, the multimodal net, zero-shot d1-3B and the rule heuristic, which uses two strip-image features) are charged for rendering it. In an earlier version the heuristic read its strip from a cache, so its total (2.53 ms) omitted rendering; the session was repeated with the common boundary, and the earlier outputs are kept with the results.

### Implementation note on HuBERT-ECG preprocessing

The reported HuBERT-ECG results use the preprocessing of the authors' public code (main text, Methods, Models). An earlier implementation, run on the superseded 1,047-window BUT QDB cohort and the 565-record CinC 2011 set, used a Butterworth band-pass and max-abs scaling of each 5-s half with absent leads left at 0. It gave AUROC 0.987 (BUT QDB), 0.732 (CinC 2011) and 0.867 (CinC 2017), against 0.972, 0.562 and 0.907 with the authors' preprocessing on those data. It was not re-run on the present cohort and is not a comparison of the present results.

## Supplementary Results

These analyses were part of the main Results in an earlier version and were moved here unchanged when the paper was reframed (main text, Methods, Deviations from the analysis plan).

### Class 2 relabelled as unusable, and the official BUT QDB partition

Treating Class 2 as unusable as well (431 unusable windows) left the 2D CNN unchanged (0.985) and lowered zero-shot d1-3B to 0.760 and the multimodal net to 0.852 (Supplementary Table S11). Under the official partition (Supplementary Table S24), AUROC ranged from 0.723 (zero-shot d1-3B) to 1.000 (1D CNN and HuBERT-ECG probe), and no comparison with the best model was significant (all p_Holm = 1.000). Unusable windows occur in only three of the seven test subjects, mostly two, so this binary task on the official test set is close to ceiling and cannot separate the models.

### Partial fine-tuning of HuBERT-ECG

In a single run of partial fine-tuning, HuBERT-ECG reached 0.947 on BUT QDB, 0.655 on CinC 2011 and 0.910 on CinC 2017 (Supplementary Table S23). These values are descriptive: they were not tested against the probe and were not repeated across seeds.

### Quantization-insensitive flatline input (exploratory)

Replacing the flatline input by the quantization-insensitive measure (exploratory; Supplementary Table S34) changed CinC 2011 AUROC from 0.675 to 0.737 for the MLP, from 0.322 to 0.431 for logistic regression and from 0.752 to 0.759 for LightGBM, with changes of at most 0.012 on BUT QDB and at most 0.011 on CinC 2017 (no confidence intervals or tests were computed). The new measure is 0 in all 988 BUT QDB windows and therefore carries no information in training, so its effect is equivalent to removing the artefact-driven input, and logistic regression remained inverted. The definition does not resolve the transfer failure.

### Hand-set heuristic versus its logistic-regression replacement

The hand-set heuristic outperformed its logistic-regression replacement in every evaluation (Supplementary Table S20): on BUT QDB, 0.971 versus 0.905 (cluster-bootstrap p = 0.001); on CinC 2011, 0.678 versus 0.330 (DeLong p < 0.001); and on CinC 2017, 0.887 versus 0.864 (DeLong p < 0.001). Because the heuristic's coefficients may not be independent of BUT QDB, only the two external comparisons are informative.

### Paired noise stress on NSTDB

Under paired electrode-motion stress (Supplementary Tables S7 and S16), each of the five trained models evaluated on NSTDB (logistic regression, LightGBM, both CNNs and the multimodal net) lowered its usable probability as noise increased; the MLP, TabPFN and the two foundation-model probes were not evaluated on NSTDB, and the heuristic appears in Supplementary Table S16 only. At 0 dB, the mean paired change was −0.94 for logistic regression, −0.90 for LightGBM, −0.84 for the multimodal net, −0.75 for the 2D CNN and −0.59 for the 1D CNN. Zero-shot d1-3B changed by at most 0.09 and did not track the noise.

### Computational cost

For the image pipelines, rendering dominated: about 61 ms (92% of the total) against 5.8 ms for the forward pass. The rule heuristic also needs the strip, so from the stored window it took 62.70 ms, of which 62.44 ms was preprocessing; only the three SQI models (1.51–2.42 ms) and the 1D CNN avoid rendering. This cost belongs to the plotting library rather than to rasterization as such: a direct raster renderer took 0.63 ms (Supplementary Table S8), and the 2D CNN retrained on such strips kept its discrimination (Supplementary Table S33; AUROC on BUT QDB 0.980 versus 0.985, Δ −0.005, 95% CI −0.027 to 0.005; CinC 2017 0.852 versus 0.850; on CinC 2011 it improved, 0.798 versus 0.787, Δ +0.012, 95% CI 0.003–0.021, p_Holm = 0.025). The d1-3B probe and the multimodal net were not re-evaluated on raster strips. The 1D CNN, which needs no rendering step, took 1.31 ms on the GPU from the stored 500-Hz window (Supplementary Table S45). d1-3B took 896 ms per window, dominated by its 831-ms forward pass. Forward-pass times of a few milliseconds on the GPU differ between our two timing scripts (Supplementary Methods, Latency measurement), so only order-of-magnitude differences are interpreted. Among the foundation-model approaches (Supplementary Table S25), the HuBERT-ECG probe took 28.0 ms per window on MPS (60.6 ms on the CPU), of which 23.6 ms was the forward pass; the linear d1-3B probe adds 0.12 ms to the d1-3B forward pass; and TabPFN, which predicts in context over the 988 training windows, took 3.8 s per window on the CPU.

### Quality gating of a downstream rhythm classifier

The rhythm classifier reached a macro-F1 of 0.719 on the 8,234 CinC 2017 recordings without any rejection (out-of-fold; Supplementary Table S37). Rejecting the 10% of recordings with the highest quality score changed macro-F1, relative to random rejection, by 0.001 to 0.008 across the eleven quality models. The increases were largest for the d1-3B image probe (0.008, 95% CI 0.005–0.013) and the 1D CNN (0.008, 95% CI 0.004–0.013), and the only other model whose interval excluded zero was TabPFN (0.005, 95% CI 0.001–0.009). The d1-3B probe exceeded the 2D CNN by 0.007 (95% CI 0.002–0.011) and the HuBERT-ECG probe by 0.007 (95% CI 0.003–0.012). The change ranged from −0.001 to 0.007 when 5% were rejected and from 0.005 to 0.011 when 20% were rejected, where the intervals for the probe's lead over the 2D CNN and over the HuBERT-ECG probe both included zero. These intervals are not adjusted for the 11 models and three fractions. The d1-3B probe's 10% rejection removed 500 normal, 47 atrial fibrillation and 276 other-rhythm recordings, so it rejected a smaller share of atrial fibrillation (5.7% of rejected recordings against 9.2% of all recordings) and a larger share of other rhythms (33.5% against 29.3%).

### Diagnostics added in the last review

**Bootstrap with 20,000 resamples.** All 20,000 resamples were valid in every one of the 55 BUT QDB pairs, and the first 2,000 reproduced every stored p-value exactly (Supplementary Table S54). Under the plus-one convention, which has no 1/n floor, four pairs had p_Holm < 0.05: the MLP against the 1D CNN (0.0055), against the 2D CNN and against the rule heuristic (both 0.022), and logistic regression against the 2D CNN (0.031). The two comparisons with the best model that were significant at the floor with 2,000 resamples therefore remained significant without it (p_Holm = 0.022 and 0.031), whereas logistic regression against the rule heuristic, 0.028 at the floor, gave 0.098, and the MLP against the heuristic, 0.051 with 2,000 resamples, gave 0.022. The next value was 0.051 (2D CNN against zero-shot d1-3B). With 20,000 resamples the smallest attainable Holm-adjusted plus-one value over 55 pairs is 0.0055.

**bSQI detectors.** On acceptable CinC 2011 lead II records, the Pan–Tompkins re-implementation detected a median of 12 beats per record and agreed with the wfdb XQRS detector (median agreement 1.000), whereas the second-difference detector detected 22, about two peaks per beat, so the stored bSQI had a median of 0.545 (Supplementary Table S55). After a 40-Hz low-pass filter the second-difference detector detected 13 beats, and bSQI computed with it had a median of 1.000 in these records. In usable BUT QDB windows the second-difference detector detected 14 beats (13 after the filter). The filter therefore removes the extra detections, which points to high-frequency content of the CinC 2011 recordings that the BUT QDB windows lack; its source (acquisition bandwidth, quantization or resampling) was not separated. Like the quantization repeats in the flatline fraction, this is an implementation artefact of a feature. With the filtered detector, the univariate AUROC of bSQI rose from 0.965 to 0.993 on BUT QDB and from 0.691 to 0.748 on CinC 2011 lead II (from 0.545 to 0.628 without the 73 constant-lead records). Two validated detectors are not a simple replacement: in usable BUT QDB windows GQRS detected 24 peaks against 13 for XQRS, and the AUROC of their agreement there was 0.307. The SQI models were not refitted with a corrected feature, so how much of their CinC 2011 performance this artefact explains is not known.

**SQI inputs on PTB-XL.** On PTB-XL lead II (Supplementary Table S56), the flatline fraction was higher in clean than in globally noisy records (median 0.355 against 0.198; univariate AUROC for global noise 0.225). Standardised with the BUT QDB windows, whose flatline fraction is at most 0.0008, these medians lie 2,793 and 1,554 SDs above the BUT QDB mean. As on CinC 2011, the measure counts repeated sample values from amplitude quantization, and these were more frequent in the clean records. This is compatible with the AUROCs below 0.5 of logistic regression, the rule heuristic, whose formula adds 5.0 × flatline to the logit, and zero-shot d1-3B, whose prompt contains the flatline value (Supplementary Table S38), but it was not tested model by model. bSQI was lower in noisy records, the expected direction (AUROC 0.288), and the high-frequency ratio and the zero-crossing rate were higher (0.688 and 0.753). The mean over the 12 leads gave similar results. This diagnostic is descriptive.

### Analyses added in the final review

**Subject influence.** With each of the 15 BUT QDB subjects left out of the pooled evaluation in turn, without refitting (Supplementary Table S57), the 2D CNN kept the highest AUROC in 13 of 15 cases; without subject 100 or subject 125 the 1D CNN was highest and the 2D CNN second. The differences of logistic regression from the 2D CNN ranged from −0.116 to −0.062 and those of the MLP from −0.111 to −0.049, so neither changed sign, and the Spearman correlation of the 11 AUROCs with those of the full cohort was at least 0.88. This shows how much the pooled estimates depend on single subjects' windows, not how models trained without a subject would behave.

**PTB-XL test fold.** On the 73 records of PTB-XL fold 10 (40 global noise, 33 clean), which the HuBERT-ECG developers' description implies was not used in its pre-training (Supplementary Table S51), the HuBERT-ECG probe reached an AUROC of 0.791 (95% CI 0.679–0.893) with the maximum, against 0.729 on the 527 records of folds 1–9, and LightGBM 0.739 (Supplementary Table S58). The fold 10 intervals are wide and cannot show that pre-training exposure had no effect, but they give no indication that the HuBERT-ECG result depends on the exposed records.

**Flat records in the CinC 2011 margin.** Because all 73 flat records are unacceptable, the AUROC splits exactly into the contributions of flat and non-flat unacceptable records ranked against the same 773 acceptable records (Supplementary Table S59). The d1-3B probe's margin of 0.037 over the 2D CNN consists of 0.021 from the flat records, against which the probe reached 0.999 and the 2D CNN 0.933, and 0.016 from the non-flat records (0.740 against 0.716). Ranking and decision differ: LightGBM flagged none of the flat records at τ = 0.50 (Supplementary Table S26) yet ranked them above the acceptable records with an AUROC of 0.935.

**Thresholds with the 1,041-window cohort.** With the models of the quota sensitivity cohort (Supplementary Table S60), the same two of 22 Target 1 thresholds met their target externally: that of the MLP on CinC 2011 (false reject rate 0.1%, sensitivity 33.3%) and that of zero-shot d1-3B on CinC 2017 (3.1%, 19.8%), whose scores are the same in both cohorts. False reject rates ranged from 3.1% to 56.9% on CinC 2017 and from 0.1% to 100.0% on CinC 2011. The transfer reproduced the stored primary-cohort thresholds exactly.

## Supplementary Note: extended results

The main text of this version was shortened for the journal after all results were known. The paragraphs below are the full-length Results text of the earlier version, reproduced unchanged except for cross-references; no analysis was added or removed (main text, Methods, Deviations from the analysis plan). Table numbers refer to the main text of this version.

### In-domain AUROCs clustered above 0.93 and did not separate the models

On BUT QDB, eight models reached a pooled out-of-fold AUROC of 0.938–0.985 (Table 1; per-model details in Supplementary Table S1; 95% CIs for all 11 models in every evaluation set in Supplementary Table S28). The 2D CNN reached 0.985, the 1D CNN 0.983, the rule heuristic and the d1-3B image probe 0.971, the multimodal net 0.966, LightGBM 0.957, TabPFN 0.956 and the HuBERT-ECG probe 0.938. None of the seven below the 2D CNN differed significantly from it after Holm adjustment (all p_Holm ≥ 0.640), which reflects limited power as much as similarity: without multiplicity adjustment the multimodal net and the HuBERT-ECG probe were below the 2D CNN (p = 0.043 and 0.016). The AUROC differences of these seven from the 2D CNN ranged from −0.046 to −0.002, and the lower limits of their unadjusted 95% cluster-bootstrap intervals ranged from −0.142 to −0.047, so differences of that size cannot be excluded with 15 subjects (Supplementary Table S50). The MLP (0.904; difference −0.081, 95% CI −0.204 to −0.017) and logistic regression (0.899; −0.086, 95% CI −0.184 to −0.021) were below the best at the smallest attainable p-value (both p_Holm = 0.028): no resample reversed the sign, so the unadjusted p is the 1/n floor of 0.0005. This significance depends on the reporting convention, because with one added to the tail count and to the number of resamples no BUT QDB pair could reach p_Holm < 0.05 with 2,000 resamples (smallest attainable value 0.055; Supplementary Table S53; with 20,000 resamples, Supplementary Table S54). Their 95% intervals, which do not depend on this convention, exclude zero but are wide. Only 7 subjects have both classes, and inside them the models differed mainly in subject 111 (Supplementary Table S41): there both CNNs kept an AUROC of 0.959–0.961, whereas the heuristic, LightGBM, logistic regression, the MLP, zero-shot d1-3B and the HuBERT-ECG probe had 0.689–0.765. Subject 105 has a single unusable window, which every model ranked first.

Because the coefficients of the rule heuristic may not be independent of BUT QDB, its in-domain estimate should be read with its leakage-free replacement: logistic regression on the same seven inputs, fitted inside the folds, reached 0.905 (Supplementary Table S20).

Zero-shot d1-3B reached 0.828. Its deficit relative to the 2D CNN (0.157; 95% CI 0.045–0.304) was significant without adjustment (p = 0.001) and with the ten-comparison family (p_Holm = 0.008), but not over all 55 pairs (p_Holm = 0.051). The linear probe on the same model's frozen image tokens reached 0.971, a gain of 0.143 (95% CI 0.040–0.264; unadjusted p = 0.010, p_Holm = 0.440). The zero-shot prompt mattered: the image-only variant reached 0.918, but at τ = 0.50 it flagged 81.3% of usable windows (Supplementary Table S17).

The models differed at the decision level more than in ranking (Supplementary Table S32). Projected to the 2.14% prevalence of the official intervals under the assumption stated in main text, Methods, Evaluation protocols, the positive predictive value at τ = 0.5 was 0.67 for the 1D CNN, 0.53 for the 2D CNN, 0.51 for the heuristic, and 0.15–0.25 for the logistic regression, LightGBM and MLP models (Supplementary Table S3). Sensitivity intervals were wide (for example 2D CNN 83.2%, 95% CI 51.9–100.0%) because unusable windows come from 7 subjects, and sensitivity varied widely between them (Supplementary Table S12). Under nested threshold selection for Target 1, the 1D CNN and the heuristic stayed within 1 percentage point of the post-hoc sensitivity, whereas the 2D CNN, LightGBM, logistic regression and the multimodal net lost 5–21 points (Supplementary Table S2).


### In-domain rank did not reliably identify the models that transferred

On CinC 2011 (Supplementary Table S28), where lead II is scored against 12-lead labels, the d1-3B image-token probe ranked recordings best (AUROC 0.824, 95% CI 0.789–0.856) and was significantly better than all ten other models, including the 2D CNN (0.787; difference 0.037, 95% CI 0.018–0.057; p_Holm = 0.003). The other foundation-model probe, HuBERT-ECG, was close to chance (0.562). The secondary probe of the d1-3B decision state was inverted (0.422; Supplementary Table S18), which suggests that the useful information is in the image tokens rather than in the decision state. The multimodal net gave every record the same score (0.500), and the SQI logistic regression was inverted (0.322), as was its replacement for the heuristic (0.330). Across seeds 1–4 the multimodal net reached 0.501–0.671 (Supplementary Table S9), so this value is seed-dependent. The constant score has an identifiable cause (descriptive diagnosis; Supplementary Table S36). Flatline fractions in BUT QDB are near zero (SD 0.00013) because of the Fourier resampling described below, so the quantization repeats of CinC 2011 lead II gave standardised flatline inputs with a median of 2,286 SDs, all 998 records lying beyond 10 SDs. With the SQI stream alone the net again returned one score for every record. With the image stream alone the same fitted net reached 0.801, and replacing only the flatline input by the quantization-insensitive measure gave 798 distinct scores and 0.719. The value of 0.500 is therefore an input-scaling artefact in this seed and says nothing about the image stream.

On CinC 2017 (Table 2 and Supplementary Table S28), a single-lead label at natural prevalence, the HuBERT-ECG probe had the highest AUROC (0.906, 95% CI 0.893–0.920). Here, as throughout CinC 2017, the intervals and tests treat recordings as independent, which cannot be verified without patient identifiers (Methods). Three models were not significantly different from it: the d1-3B probe (0.894, p_Holm = 0.844), LightGBM (0.891, p_Holm = 0.844) and the rule heuristic (0.887, p_Holm = 0.525). All others were significantly worse: the MLP (0.876, p_Holm = 0.031), logistic regression (0.870, p_Holm = 0.004), TabPFN (0.853), the 2D CNN (0.850), the multimodal net (0.799), the 1D CNN (0.768) and zero-shot d1-3B (0.757). The 1D CNN had been second in-domain. With the recording mean instead of the maximum, every model's AUROC rose, by 0.025–0.083 (Table 2); the HuBERT-ECG probe (0.941), LightGBM (0.939) and the d1-3B probe (0.935) were highest. Recording length is confounded with the label under the maximum: 110 of the 278 noisy recordings but only 490 of the 8,234 others contributed a single window, and the number of windows alone gave an AUROC of 0.698 (Supplementary Table S42; descriptive). Among the 7,561 recordings with three windows (143 noisy), the maximum gave 0.953 for the HuBERT-ECG probe, 0.951 for the heuristic, 0.929 for logistic regression, 0.924 for the d1-3B probe and 0.923 for LightGBM, so both the absolute values and the order of the leading models depend on recording length as well as on the rule. On the balanced 556-recording subset (Supplementary Table S28), the HuBERT-ECG probe (0.916) was best, and the d1-3B probe (0.893), LightGBM (0.892), the heuristic (0.876) and the MLP (0.875) were not significantly different from it.

By observed rank, the d1-3B image probe was fourth in-domain, first on CinC 2011 and second on CinC 2017, the only model in the top four of all three sets; with the 53 omitted development windows restored, the HuBERT-ECG probe was too (third, fourth and fourth; Supplementary Table S49). The probe was not significantly different from the best in-domain or on CinC 2017, but failing to reject a difference does not show equivalence, no margin of acceptable loss was specified, and the balanced CinC 2017 subset is part of the full set, so these results do not establish consistency or non-inferiority. On CinC 2011, the one set where the probe was significantly best, that significance depends on the 73 constant-lead records (Supplementary Table S40). It does not extend to the PTB-XL test (main text, PTB-XL results).


### Flat leads, the lead–label mismatch and input scaling moved the external rankings

A descriptive check showed that 73 of the 225 unacceptable CinC 2011 records have a constant lead II (Supplementary Table S26). At τ = 0.50, all 73 were flagged by the MLP, both CNNs, the multimodal net, the heuristic, zero-shot d1-3B and the d1-3B probe, and none by logistic regression, LightGBM, TabPFN or the HuBERT-ECG probe. Without these records, AUROC fell for most models, the d1-3B probe remaining highest (0.740), followed by the 2D CNN (0.716) and the HuBERT-ECG probe (0.684), whose AUROC rose. When the paired tests were repeated on the remaining 925 records (152 unacceptable; Supplementary Table S40; post hoc), the probe's margin over the 2D CNN was no longer significant (difference 0.024, 95% CI −0.002 to 0.050; unadjusted p = 0.069, p_Holm = 0.833), while the other nine models stayed significantly below the probe (all p_Holm ≤ 0.003). The probe's significant advantage over the 2D CNN on lead II therefore rests on records that a flat-line check alone would identify. BUT QDB contains no constant windows, so the trained models never saw this failure during development.

Scoring all 12 leads changed the picture (Supplementary Table S5; aggregation rules chosen after the main analysis; logistic regression, LightGBM, MLP, both CNNs, the multimodal net, the heuristic, zero-shot d1-3B and the two foundation-model probes; TabPFN was not scored on 12 leads). Averaging the leads gave 0.882 for the 2D CNN, 0.854 for LightGBM and 0.840 for the MLP, and the maximum gave 0.842 for the heuristic, 0.838 for the MLP and 0.813 for zero-shot d1-3B. The d1-3B image probe, fitted on BUT QDB only and scored on 12 leads after these results were known, reached 0.951 (95% CI 0.935–0.966) with the maximum and 0.944 (95% CI 0.926–0.959) with the mean. This exceeded the best other model under each rule: the heuristic with the maximum (difference 0.109, 95% CI 0.076–0.143) and the 2D CNN with the mean (difference 0.062, 95% CI 0.044–0.080; both p_Holm < 0.001, DeLong, Holm over four tests that also compared each rule with the probe's own lead II result, both also p_Holm < 0.001). Records with a flat lead contribute to this: 135 of the 998 records had at least one constant lead (129 of them unacceptable), and the number of constant leads alone gave an AUROC of 0.784. In a descriptive analysis without these records (863 records, 96 unacceptable; Supplementary Table S35) the probe reached 0.902 with the maximum and 0.912 with the mean, against 0.854 and 0.878 for the 2D CNN. The HuBERT-ECG probe, scored on each lead in the same way, did not improve on its lead II result of 0.562: it reached 0.431 with the maximum and 0.510 with the mean, 0.520 and 0.434 below the d1-3B probe (both p_Holm < 0.001). Without the records with a constant lead its AUROC was 0.683 and 0.805. Given the real 12 leads in one input, it reached 0.267, below chance; its probe had been fitted on single-lead inputs only, so it was applied here to a kind of input it had not been fitted on. Logistic regression remained inverted under every aggregation. The gains indicate that part of the CinC 2011 gap is the single-lead versus 12-lead label mismatch.

Several SQIs reversed their association with unusability between datasets (Supplementary Table S6; distributions by dataset and class in Supplementary Table S29). As univariate AUROC on BUT QDB versus CinC 2011, the high-frequency ratio went from 0.722 to 0.307, the powerline ratio from 0.717 to 0.311 and the zero-crossing rate from 0.660 to 0.256. Two mechanisms contribute. The first is asymmetric preprocessing: BUT QDB windows are Fourier-resampled from 1,000 Hz, which removes exact repeats of consecutive sample values, so their flatline fraction is near zero, whereas CinC 2011 lead II is used at its native 500 Hz with 5-µV steps, so its flatline fraction largely counts quantization repeats, and acceptable records already have a median of 0.287. The second is the constant leads, which have high-frequency, powerline and zero-crossing values of 0 and so look like a clean signal under the BUT QDB associations (Supplementary Table S30). Without those records the three AUROCs were 0.454, 0.461 and 0.379: the strong reversals largely disappear, but the features do not regain their BUT QDB direction. The SQI logistic regression likewise rose from 0.322 to 0.477 without them (Supplementary Table S26). A descriptive input-sensitivity check on CinC 2011 (Supplementary Table S15) points the same way: without the saturation and flatline inputs, logistic regression reached 0.759 and the MLP 0.778, and clipping standardised inputs to [−3, 3] alone gave logistic regression 0.760, so out-of-range inputs drive much of its failure.

The harmonised pipeline (Supplementary Table S46; a post hoc, results-informed sensitivity analysis, main text, Methods, Evaluation protocols) shows how much of the CinC 2011 spread came from these input incompatibilities. With the three safeguards, logistic regression rose from 0.322 to 0.766, the multimodal net from 0.500 to 0.731 (seeds: 0.706–0.768) and the MLP from 0.675 to 0.761 (seeds: 0.686–0.776); the constant-input rule alone raised the HuBERT-ECG probe from 0.562 to 0.787 and the 1D CNN from 0.695 to 0.770. The ten models other than zero-shot d1-3B then spanned 0.731–0.824 instead of 0.322–0.824; the d1-3B probe (0.824) and the 2D CNN (0.808) remained the two highest, and zero-shot d1-3B was unchanged (0.641). In-domain AUROCs changed by at most 0.024 and CinC 2017 AUROCs by at most 0.058 (multimodal net, 0.799 to 0.858). The rank agreement between in-domain and external AUROC did not improve: the Spearman correlation was 0.49 on CinC 2011 and −0.19 on CinC 2017, against 0.61 and −0.02 with the original pipeline, while the agreement between the two external datasets rose from 0.19 to 0.43 (Supplementary Table S47). The low AUROCs of several models on CinC 2011 were therefore largely failures of the input pipeline, whereas the weak relation between in-domain and external ranking was not.

### Multi-source training helps some models and harms others

Changing the training sources changed AUROC substantially, in both directions (Table 3). Across five training seeds (Supplementary Table S21), five effects of the networks were consistent (medians over seeds): the 1D CNN gained 0.078 on CinC 2011 and 0.116 on CinC 2017, the multimodal net gained 0.053 on CinC 2017, the 2D CNN lost 0.037 on CinC 2017, and the MLP lost 0.186 on CinC 2011. The 2D CNN on CinC 2011 (median −0.019) and the multimodal net on CinC 2011 (median −0.002, range −0.129 to +0.201) were not consistent. On the BUT QDB test, the multimodal net varied widely between seeds (0.736–0.970).

Among the other models, logistic regression improved on CinC 2011 (+0.076), although it remained below 0.5, and slightly on CinC 2017 (+0.020); LightGBM lost 0.115 on CinC 2011; and TabPFN lost 0.049 on CinC 2011. With only CinC data for training, logistic regression ranked BUT QDB windows poorly (0.671, 95% CI 0.278–0.963). The HuBERT-ECG probe gained most on CinC 2011, from 0.562 to 0.802 (p_Holm < 0.001), and lost slightly on CinC 2017 (−0.028, p_Holm < 0.001). The d1-3B probe was stable on CinC 2011 (0.816, Δ −0.008, p_Holm = 0.286) and gained on CinC 2017 (0.922, Δ +0.028, p_Holm < 0.001).


### Operating points, calibration and prevalence

Transferred operating points rarely met their target. Thresholds chosen on the pooled BUT QDB scores for a false reject rate of at most 5% (Target 1) gave false reject rates of 3.1–58.1% across the 11 models on CinC 2017 and 0.0–100.0% on CinC 2011 (Supplementary Table S4). Two of these 22 thresholds met the target: that of the MLP on CinC 2011 (false reject rate 0.0%, sensitivity 32.9%) and that of zero-shot d1-3B on CinC 2017 (3.1%, 19.8%), so meeting the false-reject target did not imply useful detection. On CinC 2017 the transfer also changed the evaluation unit from a window to the maximum over up to three windows of a recording, so these results do not separate dataset shift from the change of unit.

At the 3.3% prevalence of CinC 2017, the positive predictive value at the default threshold was low for all 11 models (0.064–0.199; Table 2). The heuristic had the highest value (0.199), but at 18.3% sensitivity. Of the other Target 1 thresholds (Supplementary Table S4), those on CinC 2017 gave false reject rates of 7.6–58.1% for the trained models and 6.2% for the heuristic, whose sensitivity fell to 52.9%. On CinC 2011, the same thresholds gave sensitivities between 32.9% (MLP) and 100% at a false reject rate of 99.6–100% (heuristic and multimodal net).

Calibration did not transfer either (Supplementary Table S32). On CinC 2017 the calibration intercept was negative for all 11 models (−2.47 to −14.75), indicating systematic overestimation of unusable risk after moving from 24.1% to 3.3% prevalence. Slopes ranged from 0.10 (1D CNN) to 2.87 (heuristic), and the HuBERT-ECG probe had the slope closest to 1 (0.93). On CinC 2011 the intercept ranged from −15.05 (multimodal net, which gave every record the same score) to 12.18 (MLP); the slope was not identifiable for the multimodal net and not estimable for logistic regression and the MLP: in both, every record off one clipping bound was unusable while the records at that bound were of both classes (logistic regression: all 76 records below the upper bound unusable, 149 of the 922 at it; MLP: all 75 records above the lower bound unusable, 150 of the 923 at it), a quasi-complete separation under which the maximum-likelihood slope does not exist.

Re-setting the threshold on a small labelled target sample kept false rejections near the target in most draws, but often by rejecting nothing (Supplementary Table S48). On CinC 2011, with n = 100, the false reject rate was at most 5% in 63–100% and at most 7.5% in 88–100% of draws, depending on the model, and the median sensitivity was 50% for the d1-3B probe, 37% for the heuristic and 14% for LightGBM; it was 0% for logistic regression, the 1D CNN and the multimodal net. On CinC 2017, with n = 100, the false reject rate was at most 5% in 77–95% and at most 7.5% in at least 93% of draws, but the median sensitivity was only 0–28%. A draw of 100 recordings contains about three noisy recordings, so the threshold rests on very few positives, and 21 of 500 draws contained none. Even with n = 200, the median sensitivity remained 0–36% (Supplementary Table S22).

Platt scaling on the drawn recordings lowered the Brier score of every model. On CinC 2017, however, the recalibrated Brier scores (0.029–0.032 at n = 100) were close to that of a constant prediction at the prevalence (0.032). Much of the gain therefore reflects a corrected probability level; this comparison does not establish that the recalibrated probabilities are well calibrated.


### On PTB-XL the probe exceeded the 2D CNN but discriminated annotated noise weakly

In the frozen PTB-XL test (Supplementary Table S38; 300 records with global noise and 300 clean; no record had a constant lead), the d1-3B image probe reached an AUROC of 0.579 (95% CI 0.533–0.628) with the maximum and 0.592 (95% CI 0.547–0.641) with the mean, against 0.501 and 0.521 for the 2D CNN. The probe was higher under both rules (difference 0.078, 95% CI 0.031–0.125, p_Holm = 0.001; difference 0.071, 95% CI 0.031–0.111, p_Holm = 0.001), so the hypothesis fixed in the plan was supported. The probe was nevertheless not the best model, and its absolute discrimination of this proxy outcome was weak.

In the secondary comparisons (Supplementary Table S39; Holm over 18 tests), the HuBERT-ECG probe (0.736 with the maximum, 0.694 with the mean) and LightGBM (0.689 and 0.694) were significantly above the d1-3B probe (all p_Holm < 0.001). The MLP (0.622 under both rules; p_Holm = 0.223 and 0.433) was not significantly different from it, nor was the 1D CNN under the mean (0.574; p_Holm = 0.433). The probe was significantly above logistic regression, the multimodal net, the heuristic and zero-shot d1-3B under both rules, and above the 1D CNN under the maximum (0.457). Zero-shot d1-3B was inverted (0.224 and 0.184), as was the heuristic (0.408 and 0.392) and, under the mean, logistic regression (0.299). The multimodal net again gave almost no discrimination (0.500 and 0.502). For partial noise versus clean, the d1-3B probe reached 0.515 and 0.519, with intervals that included 0.5, and the highest values were those of the HuBERT-ECG probe (0.678 and 0.643) and LightGBM (0.603 and 0.610). Under the maximum rule no model exceeded an AUROC of 0.736 on this outcome.

## Supplementary Table S1. Patient-independent 5-fold cross-validation on BUT QDB (988 windows, 15 subjects), eight approaches of the main analysis; discrimination estimates for all 11 approaches are in Supplementary Table S28. 95% CIs: subject-level cluster bootstrap.

<!-- source: Table 1 -->
{{Table 1}}

## Supplementary Table S2. Nested versus post-hoc threshold selection on BUT QDB.

<!-- source: Table 1c -->
{{Table 1c}}

## Supplementary Table S3. BUT QDB precision metrics projected to the prevalence of Class 3 inside the official intervals (2.14%, primary) and over all annotated intervals including the excluded stretch (16.0%; tabulated for continuity with the earlier version, not interpreted). The projection changes the class ratio only and assumes that the score distributions within each class in the quota cohort hold for all eligible windows; the values are conditional projections, not estimates for the full eligible population (main text, Methods, Evaluation protocols; Supplementary Table S43). Eight approaches; TabPFN and the two foundation-model probes are not included.

<!-- source: Table 1e -->
{{Table 1e}}

## Supplementary Table S4. Thresholds chosen on pooled BUT QDB out-of-fold scores and applied unchanged to the external datasets. CinC 2011: all 998 records, lead II. CinC 2017: all 8,512 recordings (recording maximum). Supplementary Table S60 repeats Target 1 with the models of the 1,041-window cohort.

<!-- source: Table 2f -->
{{Table 2f}}

## Supplementary Table S5. CinC 2011 (998 records) scored on all 12 leads (aggregation rules chosen after the main analysis). Both foundation-model probes were fitted on BUT QDB only and scored after the other models' 12-lead results were known. TabPFN was not scored on 12 leads.

<!-- source: Table 2e -->
{{Table 2e}}

## Supplementary Table S6. Univariate AUROC of each SQI across datasets (values below 0.5: higher feature value associated with usable).

<!-- source: Table 2g -->
{{Table 2g}}

## Supplementary Table S7. NSTDB: paired change in usable probability relative to the clean partner window (mean ± SD). Six approaches; the MLP, TabPFN, the rule heuristic and the two foundation-model probes are not included.

<!-- source: Table 3b -->
{{Table 3b}}

## Supplementary Table S8. Extended latency profile (Apple M4; `experiments/latency_extended.py`). This is a separate script from Supplementary Table S45 (`experiments/empirical_latency_benchmark.py`): CPU and MPS passes are interleaved per window, and the first window of each run is excluded. Decoded image tensors are cached after first use, so image decoding and resizing are timed only in the first run; the reported median over runs reflects cached inputs. MPS forward times of a few milliseconds therefore differ from Supplementary Table S45 (Supplementary Methods, Latency measurement).

<!-- source: Table 4b -->
{{Table 4b}}

## Supplementary Table S9. Seed stability of the 1D CNN, 2D CNN and multimodal net. CinC rows refer to the 998-record CinC 2011 evaluation.

<!-- source: Table 5 -->
{{Table 5}}

## Supplementary Table S10. Post-hoc ROC operating points on BUT QDB (exploratory; thresholds chosen on the evaluated data).

<!-- source: Table 1b -->
{{Table 1b}}

## Supplementary Table S11. Sensitivity analysis with Class 2 relabelled as unusable (BUT QDB).

<!-- source: Table 1d -->
{{Table 1d}}

## Supplementary Table S12. Subject-level sensitivity on BUT QDB.

<!-- source: Table 1f -->
{{Table 1f}}

## Supplementary Table S13. CinC 2011 Set A, all 998 records, lead II. Eight approaches; AUROC and average precision of all 11 are in Supplementary Table S28.

<!-- source: Table 2 -->
{{Table 2}}

## Supplementary Table S14. Post-hoc ROC operating points on CinC 2011, all 998 records (exploratory).

<!-- source: Table 2b -->
{{Table 2b}}

## Supplementary Table S15. Feature-set and scaler sensitivity of the tabular models under transfer. CinC 2011: all 998 records.

<!-- source: Table 2c -->
{{Table 2c}}

## Supplementary Table S16. NSTDB: usable probability by nominal SNR. Seven approaches; the MLP, TabPFN and the two foundation-model probes are not included.

<!-- source: Table 3 -->
{{Table 3}}

## Supplementary Table S17. Prompt sensitivity of zero-shot d1-3B. The prompt variants were scored on a 150-record balanced CinC 2011 subset (75 acceptable, 75 unacceptable) and were not extended to all 998 records.

<!-- source: Table 7 -->
{{Table 7}}

## Supplementary Table S18. Linear probes on frozen d1-3B representations.

<!-- source: Table 7b -->
{{Table 7b}}

## Supplementary Table S19. CinC 2017 balanced subset (278 noisy + 278 non-noisy recordings) including zero-shot d1-3B. Eight approaches; AUROC of all 11 is in Supplementary Table S28.

<!-- source: Table 8b -->
{{Table 8b}}

## Supplementary Table S20. Hand-set rule heuristic versus a logistic-regression replacement using the same seven input features (standardised L2 logistic regression fitted inside the BUT QDB folds and transferred like the other models). The replacement changes the functional form as well as the coefficients (Supplementary Methods, Rule heuristic). p: BUT QDB, subject-level cluster bootstrap; external sets, DeLong; unadjusted.

{{CSV:heuristic_refit}}

## Supplementary Table S21. Seed repetition (seeds 42 and 1–4) of BUT-only training and leave-one-dataset-out training for the seed-dependent networks. Consistent: same sign of Δ in all five seeds and |median Δ| ≥ 0.02 (pre-specified).

{{Table 10b}}

## Supplementary Table S22. Local recalibration on n = 25–200 randomly drawn labelled target recordings (500 draws each), evaluated on the remaining recordings. Prevalence-only Brier score: CinC 2011 0.175, CinC 2017 0.032. Zero-rejection draws: share of draws whose local threshold flagged no held-out recording, which meets the false-reject target trivially. FPR ≤ 5% and Sens. > 0: share of draws meeting the target while flagging at least one unusable recording (both columns post hoc, added in review). Draws counted under FPR ≤ 5% but in neither of these two columns rejected only usable recordings (0 < FPR ≤ 5%, sensitivity 0).

{{Table 11b}}

## Supplementary Table S23. HuBERT-ECG: linear probe and partial fine-tuning (AUROC). Fine-tuning was not run in LODO.

{{Table 12}}

## Supplementary Table S24. BUT QDB official protocol (Smital et al. 2026): models retrained on the official training records and tested on the official test records (window-level binary task). 95% CI: subject-level cluster bootstrap; p_Holm: paired comparison with the best model.

{{Table 13}}

## Supplementary Table S25. Latency of the foundation-model probes and TabPFN (Apple M4; same 25 windows and three runs as Supplementary Table S45, harness of Supplementary Table S8).

{{Table 4c}}

## Supplementary Table S26. CinC 2011 records with a constant lead II (73 of 225 unacceptable records; post hoc, descriptive). Number flagged at τ = 0.50 and AUROC with and without these records; fitted models trained on BUT QDB only (the rule heuristic and zero-shot d1-3B were not fitted).

{{Table 2h}}

## Supplementary Table S27. CinC 2017 at natural prevalence with bootstrap confidence intervals for AUROC and AP (confidence intervals of decision-level metrics: Supplementary Table S32), seven main models (recording maximum; mean in the last column); all 11 are in Table 2 and Supplementary Table S28.

{{Table 8}}

## Supplementary Table S28. Discrimination of every model in each primary evaluation set: AUROC and average precision with 95% bootstrap confidence intervals (2,000 resamples, seed 42; resampling unit subject on BUT QDB and recording externally). Fitted models trained on BUT QDB only (the rule heuristic and zero-shot d1-3B were not fitted); external sets scored with the pre-specified recording maximum. Intervals differ slightly from Supplementary Tables S1 and S27, which used other resampling runs. On CinC 2017, 422 recordings have a 1D CNN score of exactly 1.0, all of them usable; because these tied scores belong to one class, AUROC and average precision do not depend on their order and would be the same on the logit scale.

{{Table 1g}}

## Supplementary Table S29. Distribution of each SQI by dataset and class: median [interquartile range] per window. CinC 2017 windows carry the label of their recording. The low bSQI of acceptable CinC 2011 records is examined in Supplementary Table S55.

{{Table 2i}}

## Supplementary Table S30. Univariate AUROC of each SQI on BUT QDB and on CinC 2011 lead II with and without the 73 records whose lead II is constant (post hoc, descriptive). Values below 0.5: higher feature value associated with usable.

{{Table 2j}}

## Supplementary Table S31. Sensitivity of the multiplicity family: comparisons with the best model of each evaluation set, unadjusted and Holm-adjusted over all 55 pairs (primary) or over the 10 comparisons with the best model. The best model was identified from the point estimates, so the second family is a sensitivity analysis only.

{{Table 9b}}

## Supplementary Table S32. Decision-level metrics at τ = 0.50 with 95% bootstrap confidence intervals (2,000 resamples; subject level on BUT QDB, recording level externally) and calibration intercept and slope on the logit scale. BUT QDB intervals differ slightly from Supplementary Table S1, which used another resampling run. Calibration intercept is interpreted only for external sets. The intercept is the root of its score equation and is finite whenever both classes occur. Slope: not identifiable (constant score) when a model gives every unit the same score; not estimable (complete or quasi-complete separation) when a direct check of scores and labels finds a cut v with all unusable units on one side of v and all usable units on the other, with no unit at v (complete separation) or with units of either or both classes at v (quasi-complete separation; Albert and Anderson 1984); the log-likelihood then keeps increasing as the slope tends to +∞ or −∞ and approaches a finite supremum that no finite slope attains, so no finite maximum-likelihood slope exists. On CinC 2011 this holds for logistic regression (v at the upper clipping bound: all 76 records below it unusable, none above it, 149 unusable and 773 usable at it; slope → −∞) and the MLP (v at the lower bound: all 75 records above it unusable, none below it, 150 unusable and 773 usable at it; slope → +∞). The constant-score check precedes the separation check. A fit failing without separation would be labelled not converged (optimisation), which did not occur. Interval bounds are percentiles over the resamples with a finite estimate and are therefore conditional on those resamples; where some had none (in every case a diagnosed separation), the number with an estimate is given in parentheses, and the excluded resamples are not represented in the interval. Corrected after review: an earlier undamped Newton solver had reported four estimable CinC 2011 intercepts as not estimable.

{{Table 1h}}

## Supplementary Table S33. 2D CNN retrained on strips drawn by a direct raster renderer versus the plotting-library strips, same folds and BUT-only transfer. Secondary analysis; Holm-adjusted over the three comparisons.

{{Table 14}}

## Supplementary Table S34. Exploratory quantization-insensitive flatline input (main text, Methods). AUROC with the original and the new flatline input; no confidence intervals or tests were computed. Distinct scores: number of distinct predicted values, an indicator of saturation.

{{Table 15}}

## Supplementary Table S35. CinC 2011 scored on lead II and on 12 leads (maximum and mean over leads), with and without the 135 records that have at least one constant lead (signal SD 0; 129 of them unacceptable). Post hoc and descriptive; no tests. Without these records: 863 records, 96 unacceptable. Number of constant leads alone (all 998 records): AUROC 0.784.

{{Table 16}}

## Supplementary Table S36. Diagnosis of the constant multimodal-net score on CinC 2011 lead II (seed 42; descriptive, no tests). Top: inputs standardised with the BUT QDB mean and SD. Bottom: rescoring with the same fitted net.

{{Table 17}}

## Supplementary Table S37. Effect of rejecting recordings by quality score on rhythm classification, CinC 2017 (8,234 recordings with a rhythm label; macro-F1 without rejection 0.719). Values are the change in macro-F1 minus the mean change under random rejection of the same fraction, with 95% bootstrap interval; intervals are not adjusted for multiplicity and no tests were run. Quality-score ties were broken at random (1D CNN: 1,159 recordings share a value with another recording, so its figures depend slightly on the tie-break; a deterministic tie-break changed its macro-F1 by at most 0.0002). Last column: number of recordings rejected at 10% in each rhythm class of the dataset (N, normal rhythm; A, atrial fibrillation; O, other rhythm). Bottom: paired differences in the change.

{{Table 18}}

## Supplementary Table S38. Frozen test on PTB-XL v1.0.3 (recordings made between October 1989 and June 1996; one record per patient; models fitted on BUT QDB only). AUROC with 95% bootstrap interval. Global versus clean: 600 records, 300 global. Partial versus clean: 600 records, 300 partial. The outcome is annotated noise, a proxy for unusability, and the sampled prevalence is artificial. No record had a constant lead, so the planned analysis without such records is identical. The SQI inputs of this sample are described in Supplementary Table S56, and AUROCs by PTB-XL fold are in Supplementary Table S58.

{{Table 19}}

## Supplementary Table S39. Paired comparisons of AUROC on PTB-XL, global noise versus clean (DeLong). Primary: d1-3B image probe versus 2D CNN, Holm over two tests. Secondary: d1-3B image probe versus the other nine models under both rules, Holm over 18 tests. The secondary family includes the 2D CNN, so this comparison is tested in both families.

{{Table 20}}

## Supplementary Table S40. CinC 2011 lead II without the 73 records whose lead II is constant (925 records, 152 unacceptable): comparisons with the highest-AUROC model (DeLong). Supplementary Table S59 splits the all-record AUROC into the contributions of flat and non-flat records. Post hoc sensitivity analysis added after an independent audit; fitted models trained on BUT QDB only (the rule heuristic and zero-shot d1-3B were not fitted). Holm adjustment over all 55 pairs, as in the primary family, and over the 10 comparisons with the best model.

{{Table 21}}

## Supplementary Table S41. BUT QDB: AUROC of the out-of-fold scores inside each subject that has both classes (unusable / all windows of the subject in parentheses), pooled over these seven subjects, and over all 15 subjects (primary estimate). Descriptive; no tests. Subject 105 has one unusable window.

{{Table 22}}

## Supplementary Table S42. CinC 2017: AUROC by number of 10-s windows per recording (noisy / all recordings in parentheses) for the recording maximum (primary rule) and mean; with one window the two rules coincide. Last row: AUROC of the number of windows alone. Descriptive; no tests. Fitted models trained on BUT QDB only (the rule heuristic and zero-shot d1-3B were not fitted).

{{Table 23}}

## Supplementary Table S43. BUT QDB: Class 3 candidate windows per subject over all annotated intervals and inside the official intervals, and unusable windows in the cohort. The quota of up to 60 unusable windows per subject was drawn over all annotated intervals before the windows outside the official intervals were removed. Subjects without Class 3 windows are omitted.

{{Table 24}}

## Supplementary Table S44. AUROC of each model in the four evaluation sets with its rank among the 11 models (1 = highest), and Spearman correlations of the AUROCs between sets. Descriptive and post hoc; the unit is the model (n = 11), so the correlations are imprecise. The two CinC 2017 sets are not independent of each other. In the bootstrap rows, the leader of a set is its model with the highest AUROC; when several models share that value exactly, the first in the fixed model order of the analysis code is taken (primary rule). Because this rule depends on that order, the number of resamples with such a tie and the same-leader count under the least and the most favourable tie-break are also given; the AUROC lost changed by at most 0.002 at either interval bound between the best and the worst tie-break. The rows without the d1-3B image-token probe (10 models), which was chosen as the primary probe after both probes' results were known, and without the probe and the heuristic (9 models) were added in review from the same resamples (evaluation/rank_sensitivity_probe.py, which first reproduced the two other model sets exactly; results/v2_2026-11/rank_sensitivity_probe.csv).

{{Table 25}}

## Supplementary Table S45. Per-window latency on an Apple M4 (median and 95th percentile of three runs of 25 windows), separating preprocessing from the forward pass. Every model is timed from the stored 500-Hz window to its decision; rendering is included for every model that needs a strip image, including the rule heuristic. (Table 5 of the main text in an earlier version.)

{{Table 4}}

## Supplementary Table S46. Harmonised-pipeline comparison (post hoc, secondary; a results-informed sensitivity analysis whose definition was committed before any harmonised score was computed but after the earlier external results were known). AUROC with the original pipeline and with three input safeguards: G1, constant-input rule (a window with zero signal SD is scored unusable); G2, quantization-insensitive flatline input; G3, standardised inputs clipped to [−3, 3] with training statistics. Δ with 95% CI: subject cluster bootstrap on BUT QDB, DeLong externally; no multiplicity adjustment. Models marked "G1 only" were not refitted; the SQI values in the zero-shot d1-3B prompt were left unchanged. Seed 42; rows ordered by harmonised AUROC on CinC 2011. BUT QDB contains no constant window.

{{Table 26}}

## Supplementary Table S47. Harmonised-pipeline comparison, continued. Top: Spearman correlation of the 11 AUROCs between evaluation sets, original and harmonised (descriptive; n = 11 models). Middle: seed range of the refitted networks under BUT-only transfer. Bottom: share of windows whose standardised input lies beyond 3 SD of the BUT QDB distribution, with the median absolute z-score, before clipping.

{{Table 27}}

## Supplementary Table S48. Target 1 thresholds (maximum sensitivity at FPR ≤ 5%) re-set on n = 100 randomly drawn labelled target recordings and evaluated on the remaining recordings (500 draws; median and 2.5–97.5 percentiles over draws). Achieved FPR is given as the share of draws with FPR ≤ 5% and ≤ 7.5%; the 7.5% level is a tolerance for the sampling error of a threshold chosen on n recordings. Zero-rejection draws and the share with FPR ≤ 5% and Sens. > 0 are defined in Supplementary Table S22. Fallback: draws without a positive recording, for which the transferred threshold was kept. Brier: uncalibrated BUT-only score versus Platt scaling fitted on the same draws; the median is over the draws in which Platt scaling could be fitted, and the number skipped (single-class draws) is in parentheses. All sample sizes: Supplementary Table S22.

<!-- source: Table 11 -->
{{Table 11}}

## Supplementary Table S49. Quota sensitivity analysis for BUT QDB subject 105 (post hoc, requested in review; the plan was written before any model scored the added windows, with all primary results known). The primary cohort (988 windows) has 1 unusable window of subject 105. The sensitivity cohort (1,041 windows, 291 unusable) adds the 53 unusable windows that a quota drawn inside the official intervals selects (Supplementary Table S43) and is otherwise identical; folds are unchanged. The nine fitted models were refitted with the code, hyperparameters and seed of the primary analysis; the rule heuristic and zero-shot d1-3B are not fitted, so their external scores are the same. BUT QDB: pooled out-of-fold AUROC, with the subject-level cluster-bootstrap 95% CI for the sensitivity cohort, and the Holm-adjusted p-value (all 55 pairs) of the comparison with the 2D CNN, the best model in both cohorts. In the sensitivity cohort, as in the primary one, every BUT QDB pair with p_Holm < 0.05 has p = 0.0005, the 1/n floor reached when no resample reversed the sign, so these decisions depend on that convention (Supplementary Table S53). External sets: Δ = sensitivity − primary, DeLong 95% CI for paired scores, no multiplicity adjustment. Ranks are among the 11 models (1 = highest AUROC). Rows are ordered by in-domain AUROC in the sensitivity cohort. Bottom: Spearman correlation of the 11 AUROCs between evaluation sets (descriptive; n = 11 models). Discrimination only; threshold transfer is repeated in Supplementary Table S60, and calibration and the other analyses were not repeated.

{{Table 28}}

## Supplementary Table S50. BUT QDB: resample counts and unrounded p-values of the paired cluster-bootstrap comparisons with the best in-domain model (added in review from saved out-of-fold predictions; no model refitted). Each of 2,000 resamples draws 15 subjects with replacement; a resample whose subjects contain only one class would be discarded, which did not occur. Δ = AUROC of the model − AUROC of the 2D CNN. A resample with Δ = 0 is counted in both columns. p = max(2 × min(share with Δ ≤ 0, share with Δ ≥ 0), 1/n), the percentile-bootstrap p-value used throughout: an approximate percentile-tail calculation that does not resample under a null hypothesis (null-centred p-values from the same resamples: Supplementary Table S62). A p of 0.0005 is the 1/n floor reached when no resample reversed the sign, a reporting convention rather than a calibrated exact value; Supplementary Table S53 gives all 55 pairs and the plus-one convention, and Supplementary Table S54 repeats them with 20,000 resamples. The recomputed p-values equal the stored ones exactly. p_Holm: Holm adjustment over all 55 pairs of the BUT QDB family (the criterion used in the text) and over the ten comparisons with the best model (sensitivity analysis). The intervals and tests are conditional on the fitted models and do not include refitting variability. The 95% CI of Δ is the percentile interval stored with the original comparison, unadjusted for multiplicity and independent of the p-value convention. -0.000 denotes a negative value that rounds to zero: the upper limit for the multimodal net is -0.0001, so its interval excludes zero. Source: results/v2_2026-11/butqdb_bootstrap_p_audit.csv (evaluation/bootstrap_p_audit.py) and, for the intervals, results/v2_2026-11/paired_comparisons.csv.

{{Table 29}}

## Supplementary Table S51. Documented pre-training data of the pre-trained checkpoints and their exposure to the evaluation datasets (model documentation checked on 18 November 2026; for HuBERT-ECG, the developers' preprint, version 4, checked on 20 November 2026). "Unknown" means that the documentation neither lists nor excludes the dataset; no exact or near-duplicate search against the pre-training data was possible. CinC 2017 has no patient identifiers, so repeated recordings of one person cannot be excluded.

| Checkpoint | Documented pre-training data | Exposure to BUT QDB, CinC 2011, CinC 2017, PTB-XL and NSTDB |
| :--- | :--- | :--- |
| HuBERT-ECG base (revision 8ffa1ad3593366ca45f5351d75317d4ff8f14465) | 9.1 million 12-lead ECGs from four countries (model card). The developers' preprint (Coppola, E. et al. HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications. *medRxiv*, preprint, not peer reviewed, https://doi.org/10.1101/2024.11.14.24317328, version 4, Methods, Data and Preprocessing) gives the same total and lists the self-supervised pre-training data: Ribeiro (CODE, except its public test set), CPSC and CPSC-Extra, PTB, PTB-XL, Georgia, Chapman-Shaoxing, Ningbo, the Tianchi arrhythmia competition data, Shandong Provincial Hospital and MIMIC-IV-ECG. The preprint does not name the checkpoint revision used here; labels were removed for pre-training | PTB-XL: listed. The developers follow the original PTB-XL split and test on partitions not used in pre-training, so the test fold (strat_fold 10) was probably not used; of the 600 records of the primary PTB-XL comparison (global noise versus clean), 73 are in fold 10 (40 global noise, 33 clean; AUROCs in Supplementary Table S58) and 527 in folds 1–9, 68 of them in fold 9, the validation fold, whose use is not stated (results/v2_2026-11/ptbxl_strat_fold_counts.csv, evaluation/ptbxl_fold_counts.py). BUT QDB, CinC 2011, CinC 2017 and NSTDB: not listed |
| LiquidAI d1-3B (revision 051bcc464b01b9f92942b364d9586b0ef5912432), post-trained from LFM2.5-VL-3B | Training data not described on the model card | Unknown for all five |
| TabPFN-3 classifier checkpoint tabpfn-v3-classifier-v3_default.ckpt | "Trained purely on synthetic tabular tasks" (model card) | None documented: no real data in pre-training |
| Models fitted in this study (tabular models, networks, probe heads) | No pre-training | BUT QDB development windows only; in leave-one-dataset-out training also the two non-held-out datasets (CinC 2017: 1,278 recordings) |

## Supplementary Table S52. CinC 2017 label file: the file read by the extraction script compared record by record with the four label versions published with the challenge (https://physionet.org/files/challenge-2017/1.0.0/, downloaded 18 November 2026). The file used and REFERENCE-v3.csv give the same label for all 8,528 recordings; their SHA-256 values differ because the files are not byte-identical. The table shows the first 16 hexadecimal digits; the source file gives the full SHA-256 values. The saved labels of the 8,512 analysed recordings equal those of the file used. Source: results/v2_2026-11/cinc2017_label_provenance.csv (evaluation/cinc2017_label_provenance.py).

{{Table 30}}

## Supplementary Table S53. BUT QDB, primary cohort (988 windows): all 55 paired cluster-bootstrap comparisons (added in review from saved out-of-fold predictions; no model refitted). Same resamples as Supplementary Table S50 (seed 42, pair order of the analysis); all 2,000 resamples were valid in every pair. Δ = AUROC of model A − AUROC of model B; a resample with Δ = 0 is counted in both columns. p (1/n floor) = max(2 × min(share with Δ ≤ 0, share with Δ ≥ 0), 1/2,000), the value used in the text; p (plus one) = 2 × (smaller count + 1) / 2,001, a convention-sensitivity value computed from the same resamples. Holm adjustment over the 55 pairs under each convention. Four pairs reach p_Holm = 0.028 under the floor because no resample reversed the sign; none is below 0.05 under the plus-one convention, whose smallest attainable Holm value with 2,000 resamples is 0.055. The recomputed floor p-values and Holm values equal the stored ones exactly. Supplementary Table S54 repeats all 55 pairs with 20,000 resamples. Conditional on the fitted models. Source: results/v2_2026-11/butqdb_bootstrap_p_audit_all_pairs.csv (evaluation/bootstrap_p_audit.py).

{{Table 31}}

## Supplementary Table S54. BUT QDB, primary cohort (988 windows): all 55 paired cluster-bootstrap comparisons repeated with 20,000 resamples (post hoc, added in review from saved out-of-fold predictions; no model refitted). Same generator, seed and pair order as Supplementary Table S53, so the first 2,000 resamples are those of the primary analysis; their counts reproduced every stored p-value exactly. All 20,000 resamples were valid in every pair. Δ = AUROC of model A − AUROC of model B; a resample with Δ = 0 is counted in both columns. p (plus one) = min(1, 2 × (smaller count + 1) / 20,001), which has no 1/n floor, with Holm adjustment over the 55 pairs; the Holm values with the 1/n floor at 20,000 resamples and those of the primary analysis are given for comparison. Rows are ordered by p (plus one). Conditional on the fitted models. Source: results/v2_2026-11/butqdb_bootstrap_b20000_sensitivity.csv (evaluation/bootstrap_b20000_sensitivity.py).

{{Table 33}}

## Supplementary Table S55. bSQI diagnostic on BUT QDB (988 windows) and CinC 2011 lead II (998 records; post hoc, added in review; stored features and models unchanged). Beats detected per 10-s window and agreement (F1 of matched beats) between detectors: the two detectors of bSQI (Supplementary Methods, item 1), recomputed from the cached signals and equal to the stored feature within 10^-6; the XQRS and GQRS detectors of the wfdb package; and the second-difference detector after a 40-Hz low-pass filter (fourth-order Butterworth, zero phase). Matching tolerance 50 ms, as in the stored feature, or 150 ms where stated. Bottom: univariate AUROC, positive class unusable, with lower agreement scored as more likely unusable. Source: results/v2_2026-11/bsqi_diagnostic_summary.csv and bsqi_diagnostic_windows.csv (evaluation/bsqi_diagnostic.py).

{{Table 34}}

## Supplementary Table S56. SQI inputs on the frozen PTB-XL sample, global noise (300 records) versus clean (300 records): median of each stored SQI and its univariate AUROC for global noise (values below 0.5: higher value associated with clean), for lead II and for the record mean over the 12 leads (post hoc, descriptive, added in review). Last row: flatline fraction standardised with the mean and SD of the 988 BUT QDB windows, approximately the scale on which the tabular models, standardised with BUT QDB training statistics, receive it. Source: results/v2_2026-11/ptbxl_sqi_diagnostic.csv (evaluation/ptbxl_sqi_diagnostic.py).

{{Table 35}}

## Supplementary Table S57. BUT QDB, primary cohort: influence of single subjects on the pooled out-of-fold comparison (post hoc, descriptive, added in review). Each row leaves one subject's windows out of the pooled evaluation; no model is refitted. Highest AUROC among the 11 models, rank of the 2D CNN (1 = highest), AUROC differences of logistic regression and the MLP from the 2D CNN, Spearman correlation of the 11 AUROCs with those of the full cohort, and the AUROCs of four models (all 11 are in the source file). Source: results/v2_2026-11/butqdb_subject_influence.csv (evaluation/subject_influence.py).

{{Table 36}}

## Supplementary Table S58. PTB-XL, global noise versus clean: AUROC [95% bootstrap CI] on the records of strat_fold 10, the PTB-XL test fold, which the HuBERT-ECG developers' description implies was not used in its pre-training (Supplementary Table S51), on folds 1–9 and on all 600 records (post hoc, descriptive, added in review; saved window scores re-aggregated, no model rescored). 12-lead maximum and mean; recording bootstrap with 2,000 resamples, seed 42. The values for all records equal Supplementary Table S38. TabPFN was not scored on PTB-XL. Source: results/v2_2026-11/ptbxl_fold10_subset.csv (evaluation/ptbxl_fold10_subset.py).

{{Table 37}}

## Supplementary Table S59. CinC 2011 lead II: AUROC split into the contributions of the 73 flat (constant lead II) and the 152 non-flat unacceptable records, each ranked against the 773 acceptable records (post hoc, descriptive, added in review). AUROC = (73/225) × AUROC of the flat records + (152/225) × AUROC of the non-flat records; the second AUROC equals the AUROC without the constant-lead records (Supplementary Tables S26 and S40). Ties count one half. Source: results/v2_2026-11/cinc2011_flat_lead_decomposition.csv (evaluation/flat_lead_decomposition.py).

{{Table 38}}

## Supplementary Table S60. Target 1 thresholds (maximum sensitivity at a false reject rate ≤ 5% on the pooled BUT QDB out-of-fold scores) chosen and transferred with the models of the primary cohort (988 windows) and of the quota sensitivity cohort (1,041 windows; Supplementary Table S49); post hoc, added in review, saved predictions only. FR: false reject rate; Sens.: sensitivity. External sets are scored as in Supplementary Table S4 (CinC 2017: recording maximum). Bold: target met externally. The rule heuristic and zero-shot d1-3B are not fitted, so their thresholds can differ between cohorts only through the BUT QDB selection set. The primary-cohort values reproduce Supplementary Table S4. Source: results/v2_2026-11/threshold_transfer_quota_sensitivity.csv (evaluation/threshold_transfer_quota_sensitivity.py).

{{Table 39}}

## Supplementary Table S61. Inventory of the reported analyses: development cohort, when each analysis was decided (item numbers of Deviations from the analysis plan, Methods) and its role. Compiled by hand from the Methods (added in review). The original analyses had no written protocol. Analyses listed with the 988-window cohort only were not reassessed with the 1,041-window cohort.

| Analysis (tables) | Development cohort | When decided (Deviations item) | Role |
| :--- | :--- | :--- | :--- |
| In-domain discrimination and paired comparisons (Table 1, Fig. 2; S1, S28, S50, S53, S54, S57, S62) | 988; 1,041 in Table 1, S49 and S62 | Original analyses; cohort corrected (1); S50, S53, S54 and S57 added in review (14, 15, 17); null-centred p-values (18) | Primary (988); S49 sensitivity; S54, S57 and S62 sensitivity and descriptive |
| External discrimination on CinC 2011 and CinC 2017 (Tables 1 and 2; S13, S27, S28) | 988; 1,041 in Table 1 and S49 | Original analyses; a CinC 2011 subset used during development was replaced by all 998 records (2) | Primary (988) |
| TabPFN and the HuBERT-ECG probe | 988; 1,041 in Table 1 and S49 | Added after the single-source results were known (5) | Part of the primary comparison |
| d1-3B image-token probe as the primary probe (S18) | 988; 1,041 in Table 1 and S49 | Introduced after the zero-shot results and chosen after both probes' results were known (4) | Primary probe |
| Rank agreement and in-domain selection (S44) | 988; correlations for 1,041 in S49 | Question placed first after all results were known (9); bootstrap intervals added in review (12); rows without the d1-3B probe (18) | Descriptive, post hoc |
| Constant-lead analyses (S26, S35, S40, S59) | 988 | After the first results (5, 8, 17) | Descriptive; S40 sensitivity |
| 12-lead scoring (S5) | 988 | Aggregation rules chosen after the main analysis (5) | Secondary |
| Multi-source training (Table 3, Fig. 3; S21, S62) | 988 | Added after the single-source results were known (5); 1/n floor and null-centred p-values (18) | Secondary |
| Threshold transfer (S4; S60) | 988; 1,041 in S60 | Original analysis; S60 added in review (17) | Secondary |
| Local threshold re-setting and recalibration (Fig. 4; S22, S48) and calibration (S32) | 988 | Original analyses; zero-rejection columns and calibration solver revised in review (12) | Secondary |
| Quantization-insensitive flatline input (S34) | 988 | After the CinC 2011 results (6) | Exploratory |
| Harmonised pipeline (S46, S47) | 988 | Definition committed before any harmonised score, designed with knowledge of the external results (10) | Results-informed sensitivity |
| PTB-XL frozen test (S38, S39) | 988 | Written plan before scoring (7) | Secondary |
| PTB-XL diagnostics (S56, S58) | 988 | Added in review (15, 17) | Descriptive |
| Quality gating of a rhythm classifier (S37) and multimodal-net diagnosis (S36) | 988 | Written plan (7) | Secondary, descriptive |
| NSTDB stress (S7, S16) | 988 | Original analyses | Secondary |
| Quota sensitivity analysis (S49) | 1,041 | Plan written before the added windows were scored, with the primary results known (13) | Sensitivity |
| bSQI diagnostic (S55) | 988 | Added in review (15) | Descriptive |

## Supplementary Table S62. Null-centred bootstrap p-values beside the percentile p-values used in the text (post hoc, added in review; saved predictions only, no model refitted). evaluation/bootstrap_null_centred.py re-runs the resamples of the primary analyses (same generator, seed and pair order) and reproduced every stored p-value and difference exactly. Δ is the observed and Δ* a resampled AUROC difference. Percentile p = max(2 × min(share with Δ* ≤ 0, share with Δ* ≥ 0), 1/n), the value used in the text; null-centred p = max(share with ∣Δ* − Δ∣ ≥ ∣Δ∣, 1/n), in which Δ* − Δ approximates the distribution of the estimate under the null hypothesis (main text, Statistical analysis); n = valid resamples of 2,000. The percentile p-value is the primary procedure and the null-centred p-value a sensitivity analysis; with 15 clusters neither is known to control the type I error rate, and the main text does not regard BUT QDB differences that are significant with only one of them as established. Holm adjustment within the families of the main analyses: all 55 BUT QDB pairs of each cohort (shown: the ten comparisons with the 2D CNN, the best model in both cohorts, and the number of the 55 pairs significant under each p-value) and the nine comparisons of each held-out dataset in Table 3. Resampling unit: BUT QDB subjects, also when BUT QDB is held out in Table 3; recordings for CinC 2011 and CinC 2017 held out. The Table 3 percentile values carry the 1/n floor from round V23 on; before, the unfloored values were adjusted and p-values of 0 were printed as p_Holm < 0.001 (no decision at the 0.05 level changed). Source: results/v2_2026-11/bootstrap_null_centred.csv.

{{Table 40}}
