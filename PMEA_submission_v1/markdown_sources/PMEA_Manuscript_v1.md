# In-domain rankings did not identify the best-transferring model in ECG usability screening: a patient-independent, multi-dataset benchmark of task-specific, tabular and foundation models

Mehmet Kuntalp

Department of Electrical and Electronics Engineering, Dokuz Eylül University, İzmir, Turkey. Postal address: Tınaztepe Campus, Buca, 35160 İzmir, Turkey

Corresponding author: Mehmet Kuntalp, mehmet.kuntalp@deu.edu.tr

ORCID: Mehmet Kuntalp, https://orcid.org/0000-0002-3381-9026


## Abstract

**Objective.** To measure how much in-domain performance of ECG usability models survives transfer to new devices and labels.

**Approach.** Eleven models (SQI logistic regression, LightGBM, MLP and TabPFN; raw-signal and image CNNs and a multimodal network; a rule-based heuristic; zero-shot d1-3B, a 3.1-billion-parameter vision-language model; and linear probes on frozen d1-3B and HuBERT-ECG features) were developed on BUT QDB (1,047 single-lead 10-s windows, 15 subjects) with subject-grouped cross-validation and applied unchanged to CinC 2011 (565 records) and CinC 2017 (8,512 recordings, 3.3% noisy); seven were also stress-tested on NSTDB. Leave-one-dataset-out training (five seeds) tested multi-source development; thresholds were re-set on small labelled target samples. Paired AUROC tests were Holm-adjusted.

**Main results.** In-domain, ten models reached AUROC 0.938–0.989, none significantly below the best; zero-shot d1-3B (0.852) was. Externally, rankings changed: on CinC 2011, the d1-3B image-token probe was best (0.833 vs 0.787; adjusted p = 0.015); on CinC 2017, the d1-3B probe (0.908), the HuBERT-ECG probe (0.907) and the heuristic (0.887) formed the top tier, and the 1D CNN fell to 0.773. Across five seeds, multi-source training improved the 1D CNN (median +0.066 and +0.102) and degraded the 2D CNN. Development-data thresholds did not transfer. At 3.3% prevalence, thresholds re-set on 100 labelled recordings controlled false rejections but kept median sensitivity below 30%.

**Significance.** In-domain benchmarks did not identify the model that transferred best. A probe on frozen vision-language features was never significantly behind the best. Operating points need local calibration with sufficient positives.

**Keywords:** electrocardiogram; signal quality; external validation; dataset shift; foundation models; benchmark

**Abbreviations:** AP, average precision; AUROC, area under the receiver operating characteristic curve; CI, confidence interval; CNN, convolutional neural network; FPR, false-positive rate; FR, false reject rate; LODO, leave-one-dataset-out; MLP, multilayer perceptron; NPV, negative predictive value; PPV, positive predictive value; SNR, signal-to-noise ratio; SQI, signal quality index.

## 1. Introduction

Automated ECG interpretation is only as reliable as the signal it is given. In ambulatory, wearable and handheld recordings, motion artefact, electrode contact loss, baseline wander and muscle noise can make a segment unusable for rhythm analysis. A quality gate that rejects such segments, or prompts a new recording, is the first step of most analysis pipelines (Clifford et al 2012, Orphanidou et al 2015, Satija et al 2018).

Quality assessment has moved through three stages:
- hand-crafted signal quality indices (SQIs) and rule-based fusion (Li et al 2008, Clifford et al 2012, Orphanidou et al 2015);
- learned models on raw signals or images (Herraiz et al 2020, Liu et al 2022, Huang et al 2024, Hu et al 2025);
- most recently, general-purpose multimodal foundation models that can be prompted without task-specific training (Liquid AI 2026).

Published results are hard to compare, and they rarely tell a practitioner what will happen on their own device. Studies differ in three ways:
- the quality definition: QRS detectability, full-waveform usability or 12-lead diagnostic acceptability (PhysioNet 2011, Silva et al 2011, Smital et al 2020);
- whether windows from the same subject appear in both training and test data;
- whether models are tested on any data other than their own.

When models are moved to new data, ranking, calibration and operating points can all change (Moeyersons et al 2019, Subbaswamy and Saria 2020, Guo et al 2022). Cross-dataset evaluation of ECG quality models, including multi-source training, has been reported (Moeyersons et al 2019). What is less often available is a comparison that combines these elements:
- many model families, including foundation-model probes, under one patient-independent protocol;
- formal paired tests;
- external datasets whose reference labels differ;
- an examination of whether operating points transfer. Two recent developments make this benchmark timely. Tabular foundation models such as TabPFN (Hollmann et al 2025) remove most of the tuning burden from SQI classifiers. ECG foundation models such as HuBERT-ECG (Coppola et al 2024) and ECG-FM (McKeen et al 2025) offer frozen representations that can be read out with a linear probe.

This study provides that benchmark for single-lead usability screening. Its contributions are:
1. **Model comparison.** A patient-independent comparison of eleven models from six families, with formal paired tests and multiplicity control:
   - tabular SQI models, including a tabular foundation model;
   - raw-signal and image CNNs, and a multimodal network;
   - a rule-based heuristic;
   - a 3.1-billion-parameter vision-language decision model used zero-shot;
   - linear probes on two frozen foundation models.
2. **Transfer evaluation.**
   - Three external evaluations whose reference labels differ: 12-lead acceptability, single-lead rhythm classifiability at natural prevalence, and paired electrode-motion stress.
   - Leave-one-dataset-out multi-source training.
   - Together, these show how much of each model's in-domain advantage survives transfer.
3. **Analyses of transfer failure.**
   - The single-lead versus 12-lead label mismatch.
   - Changes in feature meaning between datasets, including an amplitude-quantization artefact in a common SQI.
   - Whether the rule heuristic's coefficients act as transferable priors.
   - The effect of prevalence.
   - Whether operating points chosen on the development data transfer.
4. **Information in the zero-shot model.** An analysis of whether the zero-shot model's frozen representation contains task-relevant information that its zero-shot decision configuration does not exploit.
5. **Cost profile.** A cost profile that separates model inference from preprocessing and rendering.

## 2. Methods

### 2.1 Datasets

**BUT QDB (development).** The Brno University of Technology ECG Quality Database (Nemcova et al 2020, Smital et al 2026) contains 18 long-term single-lead recordings from 15 subjects, sampled at 1,000 Hz.
- *Annotation.* Three experts annotated the recordings into three quality classes (Smital et al 2020):
  - Class 1: full-waveform analysis possible.
  - Class 2: only QRS detection reliable.
  - Class 3: QRS detection not possible.

  The usability target was Class 3 versus Classes 1–2. We used the consensus annotations (99.39 h of the 478.7 h recorded).
- *Candidate pool.* All non-overlapping 10-s windows inside the annotation intervals formed a candidate pool of 32,956 windows (16.0% Class 3).
- *Sampling.* Per subject, up to 50 usable and up to 60 unusable windows were sampled evenly in time. This gave 1,047 windows (297 unusable) from 17 records and 15 subjects. Class 3 occurred in 7 subjects.
- *Resampling.* Windows were resampled to 500 Hz (Fourier method).
- *Relation to the official protocol.* The recently published data descriptor defines a subject-independent train/test split (11 records from 8 subjects versus 7 records from 7 subjects) and a sample-level three-class evaluation (Smital et al 2026). It also excludes a 13-h stretch of missing signal at the start of record 105001, which is included in our 99.39 h. Our cohort and folds were defined without reference to that protocol and do not follow it. According to the repository history, the cohort was created in October 2026, after the descriptor was published. 59 of our 297 unusable windows lie in the excluded stretch. Two sensitivity analyses address this (Section 2.4). They were specified after this count was known, but before their results were computed.

**CinC 2011 Set A (external, 12-lead label).** These are 10-s 12-lead recordings at 500 Hz, labelled acceptable or unacceptable for 12-lead interpretation (PhysioNet 2011, Silva et al 2011).
- The external set is the union of six balanced subsets drawn for the transfer analyses: 565 records (197 unacceptable, 368 acceptable).
- Five of the six subsets are random; one consists of the first 75 records of each sorted list.
- Lead II was scored in the primary analysis; all 12 leads were scored in a sensitivity analysis.

**CinC 2017 (external, single-lead label at natural prevalence).** The dataset contains 8,528 handheld single-lead recordings (300 Hz, 9–61 s), labelled normal rhythm, atrial fibrillation, other rhythm, or too noisy to classify (Clifford et al 2017). The "too noisy" label marks recordings whose rhythm cannot be classified from the lead; it was the unusable target.
- We excluded 16 recordings shorter than 10 s, leaving 8,512 recordings (278 noisy, 3.3%).
- Each recording was split into up to three 10-s windows (23,985 windows) and resampled to 500 Hz.
- The recording score is the maximum window score (pre-specified).

**MIT-BIH NSTDB (controlled stress).** Electrode-motion noise was added to MIT-BIH records 118 and 119 at nominal SNRs of +24 to −6 dB (Moody et al 1984).
- Noise segments (300–420 s and 540–660 s, verified against the signals) provided 264 noisy windows: 2 records × 6 SNR labels × 2 segments × 11 windows (10 s, starting 5 s into each segment).
- Each noisy window was paired with the clean base-record window at the same start time.

### 2.2 Inputs

**Tabular inputs.** Thirteen tabular inputs were computed per window: 11 SQIs (Li et al 2008, Clifford et al 2012, Orphanidou et al 2015) and two constants (window duration and sampling rate). The 11 SQIs are:
- bSQI, the agreement of the Pan–Tompkins (Pan and Tompkins 1985) and a difference-based QRS detector, with a 50-ms tolerance;
- spectral baseline-wander, high-frequency, QRS-band and powerline ratios;
- saturation and flatline fractions;
- kurtosis and skewness;
- zero-crossing rate;
- amplitude standard deviation.

Amplitudes were harmonised to mV before feature extraction.

**Edge cases.**
- *Undefined values:* the extractor sets undefined (non-finite) values to 0 and clips the spectral ratios and moments to fixed ranges (Supplementary Methods). Clipping affected 3 (BUT QDB), 6 (CinC 2011) and 122 (CinC 2017) kurtosis or skewness values.
- *Constant windows:* a constant window receives flatline and saturation fractions of 1 and bSQI 0. This occurred in 65 CinC 2011 lead-II records, all unacceptable, and in no other dataset.
- *Short recordings:* recordings shorter than one window were excluded (Section 2.1).
- *Raw-signal input:* per-window z-scoring divides by 1 when the SD is below 10^-6.

Full SQI definitions, network architectures, the heuristic formula and the d1-3B prompt are given in the Supplementary Methods.

**Image inputs.** Image inputs are rendered 10-s strips (513 × 257 px) with a light grid and a per-strip vertical scale (1st–99th percentile, at least 1 mV).

**Raw-signal inputs.** Raw-signal models receive the 5,000-sample window, z-scored per window.

### 2.3 Models

**Tabular.**
- L2 logistic regression on standardised inputs.
- LightGBM (100 trees, depth 4, learning rate 0.05).
- A two-layer MLP: 64 and 32 units, each followed by batch normalisation and ReLU, dropout 0.2 after the first layer, 3,201 parameters.
- TabPFN-3 (Grinsztajn et al 2026), a pre-trained tabular foundation model building on earlier TabPFN versions (Hollmann et al 2025), used without tuning.
  - *Package and checkpoint:* `tabpfn` package 9.0.0 with the checkpoint file `tabpfn-v3-classifier-v3_default.ckpt` (model name TabPFN-v3, architecture `tabpfn_v3`; SHA-256 prefix d0d865d5), loaded explicitly by path. No automatic model-version selection was used.
  - *Inference settings:* package defaults (number of estimators and softmax temperature `auto`, no probability balancing), random state 42, CPU.

**Raw signal.** A four-block 1D CNN (167,489 parameters).

**Image.** A four-block 2D CNN (245,697 parameters).

**Multimodal.** A gated fusion of a three-block image encoder and an SQI branch (131,969 parameters).

**Rule heuristic.** A fixed logistic combination of five SQIs and two strip features with hand-set coefficients (no trainable parameters; formula in the Supplementary Methods).

**Zero-shot vision-language model.** LiquidAI d1-3B (3.12 × 10^9 parameters) (Liquid AI 2026).
- *Input:* the strip, a text summary of 10 of the 13 tabular inputs, and a usability question.
- *Output:* class probabilities from a single forward pass, without task-specific training.
- *Sensitivity analysis:* two alternative prompt configurations.

**Frozen foundation-model probes.**

*Probe protocol.* Both probes used the same classifier: a standardised L2 logistic regression whose penalty (C ∈ {10^-3, 10^-2, 10^-1, 1}) was chosen by inner subject-grouped cross-validation within the training data.

*d1-3B image-token probe.* The feature is the mean image-token embedding after the multimodal projector, captured in the primary zero-shot forward pass. The final hidden state at the answer position was probed as a secondary representation.

*HuBERT-ECG probe.* The feature is the time-averaged last hidden state of HuBERT-ECG base (Coppola et al 2024), a self-supervised ECG foundation model (93 M parameters).
- *Input format:* the model takes 12 leads × 5 s at 500 Hz, flattened and decimated to 6,000 samples. The single lead was placed in the lead II slot, with the other leads set to zero, and the embeddings of the two 5-s halves of each window were averaged.
- *Preprocessing (primary):* the preprocessing of the authors' public code, re-implemented and checked against it. This is a 151-tap FIR band-pass filter (0.05–47 Hz, forward-backward) followed by per-lead min-max scaling to [−1, 1] over the 10-s window. Absent leads are not addressed by the authors; their scaling maps the zero leads to −1, and this was kept.
- *Preprocessing (sensitivity):* the version we ran first, before checking the authors' code: a Butterworth band-pass and max-abs scaling of each 5-s half, with absent leads left at 0.

*HuBERT-ECG partial fine-tuning (secondary).* The last four transformer layers and a linear head were trained, with the rest of the model frozen. The configuration was fixed in the analysis plan: AdamW, learning rates 10^-5 and 10^-3, 10 epochs, batch size 16, class-weighted loss, seed 42, primary preprocessing. It was evaluated in the same folds and by BUT-only transfer, not in LODO.

**Training.**
- *Neural networks:* all trained networks used AdamW (learning rate and weight decay 10^-3), class-weighted binary cross-entropy, 25 epochs (MLP: 35), no early stopping and seed 42.
- *Logistic regression and LightGBM:* balanced class weights.
- *Hyperparameter choice:* hyperparameters of the task-specific models were fixed before any evaluation and were not tuned. TabPFN was added as a tuning-free reference for the tabular models (Section 4).
- *Test-set isolation:* for the primary unchanged-transfer analyses, target evaluation data were not used for model fitting, hyperparameters, features or threshold selection. In LODO, the held-out dataset was used only for testing.
  - *Local adaptation:* in the local-adaptation experiments, labelled target recordings were split into calibration and evaluation sets within each draw.
  - *Post-hoc operating points:* these are reported only as descriptive reference values.

### 2.4 Evaluation protocols

**Patient-independent cross-validation.** We used five subject-grouped folds on BUT QDB (three subjects per fold). Scalers, probes and thresholds were fitted on training folds only. No subject appears in more than one fold, and windows do not overlap.

**Official BUT QDB protocol (sensitivity analyses).**
- *F1a:* the pooled out-of-fold AUROC was recomputed without the 59 windows from the excluded 105001 stretch (988 windows, 238 unusable). This is evaluation-only; the models had been trained with these windows.
- *F1b:* the models were retrained on the cohort windows of the official training records and tested on those of the official test records, using only windows inside the officially evaluated intervals. The task is window-level and binary (Class 3 versus Classes 1–2), so it does not use the official sample-level three-class metric. TabPFN was not run.

Both analyses used paired comparisons with the best model (subject-level cluster bootstrap, Holm).

**Single-source transfer.** Models were trained on all of BUT QDB and applied unchanged to the external datasets. Scalers were fitted on BUT QDB only.

**Leave-one-dataset-out (LODO).** Models were trained on two of BUT QDB, CinC 2011 and CinC 2017 and tested on the third, with the same fixed hyperparameters.
- When CinC 2017 was a training source, we used all noisy recordings and 1,000 random non-noisy recordings (seed 42), and copied the recording label to its windows.
- *Training-set sizes:* the CinC 2017 training subset comprised 1,278 recordings (3,437 windows, 589 unusable). The three training sets were:
  - BUT QDB held out: 4,002 windows, 786 unusable;
  - CinC 2011 held out: 4,484 windows, 886 unusable;
  - CinC 2017 held out: 1,612 windows, 494 unusable.
- The rule heuristic is not trained and is reported unchanged.
- The d1-3B probe was run in the same splits, with d1-3B features extracted for all 23,985 CinC 2017 windows.
- *Probe tuning in LODO:* the inner cross-validation that chose the probes' penalty grouped training windows by subject (BUT QDB) or recording (CinC).
- *Seeds:* the seed-dependent networks (MLP, both CNNs, multimodal net) were repeated with training seeds 1–4 in addition to 42, both for BUT-only training and for LODO. A LODO effect was called consistent if its sign agreed in all five seeds and its median magnitude was at least 0.02 (pre-specified). Logistic regression and LightGBM use a fixed random state; the probes are deterministic given their features.

**Noise stress.** We computed the usable probability per SNR label and its paired change relative to the clean partner window.

**Operating points.** Besides the default threshold (0.5), two rules were applied:
- Target 1: maximum sensitivity at FPR ≤ 5%.
- Target 2: minimum FPR at sensitivity ≥ 80%.

Thresholds were chosen in four ways:
- (a) post hoc on the evaluated data (reference only);
- (b) by nested cross-validation within the training subjects of each fold;
- (c) on pooled BUT QDB out-of-fold scores, transferred unchanged to the external datasets;
- (d) locally, on n = 25, 50, 100 or 200 randomly drawn labelled target recordings (500 draws each), with evaluation on the remaining recordings. If a draw contained only one class, the transferred threshold was used and the draw was counted.

In (d), Platt scaling fitted on the same draws was compared with the uncalibrated scores by the Brier score. This analysis is descriptive.

**Prevalence.**
- BUT QDB precision metrics were re-weighted from the sampled prevalence (28.4%) to the natural prevalence of the candidate pool (16.0%). They were also re-weighted to 2.8%, the prevalence without the one record that contributes 90% of Class 3 windows.
- CinC 2017 was evaluated at its natural prevalence of 3.3%.

**Heuristic versus a logistic-regression replacement.** The hand-set heuristic was compared with a logistic-regression replacement using the same seven input features. The replacement was a standardised L2 logistic regression fitted inside the same folds and transferred like the other models. The replacement changes the functional form as well as the coefficients: the heuristic mixes two separate sigmoid outputs and caps two inputs (Supplementary Methods S-M3). The comparison therefore does not isolate the effect of the hand-set coefficients.

### 2.5 Statistical analysis

**Metrics.** The positive class was unusable.
- Discrimination: area under the receiver operating characteristic curve (AUROC) and average precision (AP).
- Calibration: Brier score.
- Decisions: sensitivity and false reject rate (FR, the proportion of usable recordings flagged as unusable). With unusable as the positive class, FR equals the false-positive rate (FPR).

**Confidence intervals.** We used a subject-level cluster bootstrap on BUT QDB, where windows of a subject are correlated, and a recording-level bootstrap externally (1,000–2,000 resamples).

**Paired comparisons.** Models were compared pairwise on identical units:
- on BUT QDB, with a subject-level cluster bootstrap of the AUROC difference (2,000 resamples);
- on the external datasets, where recordings are independent, with DeLong's test (DeLong et al 1988).

**Multiplicity.** Within each evaluation set, all pairwise comparisons formed one family and were Holm-adjusted (Holm 1979). The families were BUT QDB, CinC 2011, CinC 2017 and the balanced 556-recording CinC 2017 subset, each with 11 models (55 comparisons). LODO effects (multi-source versus BUT-only) were Holm-adjusted within each held-out dataset.

**Secondary tests.**
- *Official-protocol analyses:* each was Holm-adjusted over its own pairwise comparisons.
- *Heuristic versus its logistic-regression replacement:* the three comparisons (one per dataset) are reported unadjusted.
- *Descriptive analyses:* the recording-mean aggregation, local recalibration, fine-tuning and the constant-lead check were not tested.

**Reporting rule.** Rankings are reported only where the adjusted comparison supports them. External rankings refer to the pre-specified recording-maximum aggregation; the recording mean is a post hoc sensitivity analysis.

**Reporting guideline.** The study is reported following TRIPOD+AI; the completed checklist is provided as a supplementary file.

**Implementation check.** On one pair, the DeLong implementation was checked against a recording bootstrap; the two 95% intervals for the AUROC difference were 0.015–0.060 and 0.014–0.059.

**Pre-specification.** The seed repetition, the full CinC 2017 run of d1-3B, the local recalibration, the HuBERT-ECG preprocessing check, the fine-tuning and the official-protocol analyses were specified in a dated analysis plan, committed to version control before they were run. Amendments made while they ran are recorded there. The study as a whole was not registered, and the original analyses had no written protocol. The plan also contained a decision rule: the statement that the d1-3B probe was never significantly behind the best would be withdrawn if the full CinC 2017 analysis contradicted it.

### 2.6 Computational cost and reproducibility

**Latency.** Latency was measured on an Apple M4 (10-core CPU, 10-core GPU, 16 GB) with synchronised GPU timing, in three runs of 25 windows. The measurement separated these steps:
- rendering;
- SQI extraction;
- resampling from the native 1,000-Hz signal;
- forward passes on CPU and GPU.

**Reproducibility.**
- Code, fixed seeds, pinned dependency versions, saved predictions and the scripts that generate every table are available.
- Every table is produced from saved outputs by script.
- Unit tests cover the metrics, the paired tests, fold disjointness and the feature definitions.

## 3. Results

### 3.1 In-domain performance does not separate the leading models

On BUT QDB, ten models reached pooled out-of-fold AUROC 0.938–0.989 (Table 1; per-model details in Table S1; 95% CIs for all 11 models in every evaluation set in Table S30):
- 1D and 2D CNN: 0.989;
- d1-3B image probe: 0.979;
- rule heuristic: 0.977;
- HuBERT-ECG probe: 0.972;
- multimodal net: 0.967;
- LightGBM: 0.964;
- TabPFN: 0.962;
- MLP: 0.939;
- logistic regression: 0.938.

**No differences from the best.** None of these ten differed significantly from the best model after Holm adjustment (all p_Holm ≥ 0.29).

**Zero-shot d1-3B.** Zero-shot d1-3B (0.852) was significantly worse than the best model, the 2D CNN, the multimodal net and the d1-3B probe (all p_Holm < 0.001). The linear probe on the same model's frozen image tokens reached 0.979, an AUROC gain of 0.127 (95% CI 0.041–0.242). This shows that the representation contains useful task-relevant information that the zero-shot decision configuration does not exploit effectively.

**Decision-level behaviour.** The models differed at the decision level, not in ranking.
- *Natural prevalence.* Re-weighted to the 2.8% natural prevalence, positive predictive value at τ = 0.5 was 0.77 for the 1D CNN, 0.71 for the 2D CNN, 0.61 for the heuristic and 0.21–0.34 for the SQI models (Table S3).
- *Nested threshold selection.* Under nested threshold selection, the CNNs kept their operating points; the SQI models did not (Table S2).
- *Subjects.* Sensitivity varied widely between the seven subjects with unusable windows.

**Official protocol (Table S26).**
- *F1a:* without the 59 windows from the excluded 105001 stretch, every model's AUROC fell slightly, for example:
  - 1D CNN: 0.989 to 0.987;
  - logistic regression: 0.938 to 0.923;
  - zero-shot d1-3B: 0.852 to 0.828.

  No model differed significantly from the best. The significant deficit of zero-shot d1-3B therefore depended in part on these near-flat windows (p_Holm = 0.368 without them).
- *F1b:* on the official test set, AUROC ranged from 0.723 (zero-shot d1-3B) to 1.000 (1D CNN and HuBERT-ECG probe), and no comparison was significant. Unusable windows occur in only three of the seven test subjects, mostly two. This window-level binary task on the official test set is close to ceiling and cannot separate the models.

### 3.2 External rankings differ between datasets

*Table 1. Pooled AUROC of each model per evaluation set; in parentheses, the Holm-adjusted p-value of the paired comparison with the best model of that set (bold). BUT QDB: subject-level cluster bootstrap; external sets: DeLong test. Models trained on BUT QDB only. HuBERT-ECG: primary (authors') preprocessing. Positive class: unusable.*

| Model | BUT_QDB (5-fold subject CV) | CinC2011 lead II (565 records) | CinC2017 (8,512 recordings) | CinC2017 balanced subset (556 recordings) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.938 (p = 0.450) | 0.320 (p < 0.001) | 0.868 (p = 0.001) | 0.860 (p = 0.005) |
| **LightGBM (SQI)** | 0.964 (p = 1.000) | 0.754 (p < 0.001) | 0.880 (p = 0.043) | 0.880 (p = 0.261) |
| **Independent SQI MLP** | 0.939 (p = 0.288) | 0.672 (p < 0.001) | 0.868 (p < 0.001) | 0.866 (p = 0.028) |
| **1D CNN (Raw Signal)** | **0.989** (best) | 0.719 (p < 0.001) | 0.773 (p < 0.001) | 0.767 (p < 0.001) |
| **2D CNN (Rendered Image)** | 0.989 (p = 1.000) | 0.787 (p = 0.015) | 0.850 (p < 0.001) | 0.859 (p = 0.005) |
| **SinglePass Multimodal Net** | 0.967 (p = 1.000) | 0.674 (p < 0.001) | 0.835 (p < 0.001) | 0.848 (p < 0.001) |
| **Rule-Based SQI Heuristic** | 0.977 (p = 1.000) | 0.685 (p < 0.001) | 0.887 (p = 0.481) | 0.876 (p = 0.174) |
| **LiquidAI d1-3B (Zero-Shot)** | 0.852 (p < 0.001) | 0.650 (p < 0.001) | 0.757 (p < 0.001) | 0.739 (p < 0.001) |
| **TabPFN (SQI)** | 0.962 (p = 1.000) | 0.749 (p < 0.001) | 0.858 (p < 0.001) | 0.861 (p = 0.010) |
| **HuBERT-ECG probe** | 0.972 (p = 1.000) | 0.562 (p < 0.001) | 0.907 (p = 1.000) | **0.916** (best) |
| **d1-3B image-token probe** | 0.979 (p = 1.000) | **0.833** (best) | **0.908** (best) | 0.910 (p = 1.000) |

![](/Volumes/KIOXIA/Projects/d13b-ecg-sqi/figures/q1_figure1.png)

*Figure 1. AUROC of each model in the four evaluation sets. The marker shows the outcome of the Holm-adjusted paired comparison with the best model of that set: best (diamond), not significantly different (filled) or significantly worse (open). Rows are grouped into tabular models, task-specific networks, the rule heuristic and the foundation-model approaches. Dashed line: chance. Source: Table 1.*

**CinC 2011 (lead II against 12-lead labels).** The d1-3B image-token probe ranked recordings best (AUROC 0.833). It was significantly better than every other model, including the 2D CNN (0.787; difference 0.046, 95% CI 0.020–0.073; p_Holm = 0.015). The HuBERT-ECG probe was close to chance (0.562).
- *SQI models.* The SQI logistic regression was inverted (0.320), and so was the logistic-regression replacement of the heuristic (0.325).
- *Label mismatch.* Averaging the scores of all 12 leads raised AUROC for every trained model except logistic regression. Over the random subsets, the mean was 0.852 for the 2D CNN and 0.877 for LightGBM (Table S5). This aggregation rule was chosen after the main analysis. It indicates that part of the CinC 2011 gap is the single-lead versus 12-lead label mismatch.

- *Constant leads (post hoc, descriptive).* 65 of the 197 unacceptable records have a constant lead II. At τ = 0.5 these were flagged by:
  - the heuristic, the multimodal net, zero-shot d1-3B and the d1-3B probe: 65 of 65;
  - the 1D CNN: 64 of 65;
  - the 2D CNN: 1 of 65;
  - the four tabular models and the HuBERT-ECG probe: none.

  Without these records, every model's AUROC fell except logistic regression and the HuBERT-ECG probe, and the d1-3B probe remained highest (0.751; Table S28). BUT QDB contains no constant windows, so several trained models never saw this failure during development.

**CinC 2017 (single-lead label, natural prevalence).** The d1-3B image probe had the highest AUROC (0.908). Two models were not significantly different from it:
- the HuBERT-ECG probe, 0.907 (p_Holm = 1.000);
- the rule heuristic, 0.887 (difference 0.020, 95% CI 0.001–0.039; p_Holm = 0.481).

All other models were significantly worse:
- LightGBM 0.880 (p_Holm = 0.043);
- logistic regression and MLP, both 0.868;
- TabPFN 0.858;
- 2D CNN 0.850;
- multimodal net 0.835;
- 1D CNN 0.773;
- zero-shot d1-3B 0.757.

The 1D CNN had been jointly best in-domain.

**Aggregation.** With the post hoc recording mean instead of the maximum, every model's AUROC rose, by 0.03–0.07 (Table 3). The two probes remained highest (0.949 and 0.942), followed by the MLP (0.934), the heuristic (0.933) and LightGBM (0.930). Transfer performance therefore depends on the recording-level aggregation as well as on the model.

**Balanced subset.** On the balanced 556-recording subset, the HuBERT-ECG probe (0.916), the d1-3B probe (0.910), LightGBM and the heuristic were not significantly different from one another.

**The d1-3B probe across all evaluations.** The d1-3B image probe was the only model that was best, or not significantly different from the best, in all four evaluation sets. This statement was made on the 556-recording subset before the full CinC 2017 analysis, and the pre-specified rule would have withdrawn it; the full analysis confirmed it.

**HuBERT-ECG depends on preprocessing and benefits from fine-tuning.** With our first preprocessing, the HuBERT-ECG probe gave 0.987 on BUT QDB, 0.732 on CinC 2011 and 0.867 on CinC 2017. With the authors' preprocessing, it gave 0.972, 0.562 and 0.907 (Table S25). In a single run, partial fine-tuning with the authors' preprocessing gave 0.969, 0.706 and 0.912. These values are descriptive: they were not tested against the probe and were not repeated across seeds.

**Feature meaning shifts between datasets.** Several SQIs reverse their association with unusability between datasets (Table S6; distributions by dataset and class in Table S31). Measured as univariate AUROC on BUT QDB versus CinC 2011:
- high-frequency ratio: 0.777 versus 0.307;
- powerline ratio: 0.773 versus 0.312;
- zero-crossing rate: 0.727 versus 0.248.

bSQI is informative again in CinC 2017 (0.124, i.e., low agreement marks noise). The flatline fraction in CinC 2011 largely counts repeats caused by 5-µV amplitude quantization: acceptable records already have a median of 0.28. In BUT QDB, which is resampled from 1,000 Hz and takes continuous values, the flatline fraction is near zero.

**The hand-set heuristic outperformed its logistic-regression replacement.** The replacement, which used the same seven inputs (Table S22), had a lower AUROC in every evaluation:
- BUT QDB: 0.925 versus 0.977 (cluster-bootstrap p = 0.001);
- CinC 2011: 0.325 versus 0.685 (DeLong p < 0.001);
- CinC 2017: 0.864 versus 0.887 (DeLong p < 0.001).

Whether the hand-set coefficients were chosen after inspecting BUT QDB outcomes is not documented (Section 4).

**Noise stress.** Under paired electrode-motion stress (Table S7), every trained model lowered its usable probability as noise increased. At 0 dB, the mean paired change was −0.93 for logistic regression, −0.90 for LightGBM, −0.85 for the multimodal net, −0.68 for the 1D CNN and −0.67 for the 2D CNN. Zero-shot d1-3B changed by at most 0.09 and did not track the noise.

### 3.3 Multi-source training helps some models and harms others

*Table 2. Leave-one-dataset-out AUROC [95% bootstrap CI] when training on the two other datasets (seed 42). Δ: paired difference from training on BUT QDB alone, with Holm adjustment within each held-out dataset. The rule heuristic has no trainable parameters. Seed repetition of the networks: Table S23.*

| Model | BUT QDB (train: CinC 2011 + CinC 2017) | CinC 2011 (train: BUT QDB + CinC 2017) | CinC 2017 (train: BUT QDB + CinC 2011) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.457 [0.141–0.939]; Δ −0.481 (p_Holm = 0.536) | 0.385 [0.329–0.438]; Δ +0.065 (p_Holm = 0.021) | 0.895 [0.876–0.911]; Δ +0.027 (p_Holm < 0.001) |
| **LightGBM (SQI)** | 0.949 [0.906–0.982]; Δ −0.015 (p_Holm = 1.000) | 0.679 [0.632–0.725]; Δ −0.075 (p_Holm < 0.001) | 0.896 [0.880–0.909]; Δ +0.016 (p_Holm < 0.001) |
| **Independent SQI MLP** | 0.980 [0.922–0.999]; Δ +0.041 (p_Holm = 0.108) | 0.325 [0.290–0.357]; Δ −0.347 (p_Holm < 0.001) | 0.873 [0.858–0.886]; Δ +0.005 (p_Holm = 0.864) |
| **1D CNN (Raw Signal)** | 0.989 [0.973–1.000]; Δ +0.000 (p_Holm = 1.000) | 0.785 [0.743–0.826]; Δ +0.066 (p_Holm < 0.001) | 0.875 [0.859–0.891]; Δ +0.102 (p_Holm < 0.001) |
| **2D CNN (Rendered Image)** | 0.975 [0.954–0.994]; Δ −0.014 (p_Holm = 1.000) | 0.768 [0.728–0.810]; Δ −0.019 (p_Holm = 0.221) | 0.768 [0.744–0.793]; Δ −0.082 (p_Holm < 0.001) |
| **SinglePass Multimodal Net** | 0.989 [0.957–1.000]; Δ +0.023 (p_Holm = 1.000) | 0.518 [0.505–0.531]; Δ −0.156 (p_Holm < 0.001) | 0.857 [0.839–0.874]; Δ +0.022 (p_Holm < 0.001) |
| **Rule-Based SQI Heuristic** | 0.977 (fixed rule) | 0.685 (fixed rule) | 0.887 (fixed rule) |
| **LiquidAI d1-3B (Zero-Shot)** | — | — | — |
| **TabPFN (SQI)** | 0.880 [0.806–0.957]; Δ −0.082 (p_Holm = 0.536) | 0.699 [0.654–0.747]; Δ −0.050 (p_Holm = 0.008) | 0.842 [0.818–0.866]; Δ −0.015 (p_Holm = 0.003) |
| **HuBERT-ECG probe** | 0.982 [0.955–0.999]; Δ +0.010 (p_Holm = 1.000) | 0.804 [0.764–0.846]; Δ +0.242 (p_Holm < 0.001) | 0.871 [0.853–0.890]; Δ −0.036 (p_Holm < 0.001) |
| **d1-3B image-token probe** | 0.947 [0.922–0.982]; Δ −0.032 (p_Holm = 1.000) | 0.818 [0.781–0.857]; Δ −0.015 (p_Holm = 0.021) | 0.908 [0.893–0.922]; Δ +0.000 (p_Holm = 0.974) |

![](/Volumes/KIOXIA/Projects/d13b-ecg-sqi/figures/q1_figure2.png)

*Figure 2. Effect of multi-source training: AUROC when training on the two other datasets minus AUROC when training on BUT QDB alone. Markers: seed 42, filled if Holm-adjusted p < 0.05. Bars: range over five training seeds for the seed-dependent networks. Source: Table 2 and Table S23.*

Adding a second training source changed AUROC substantially, in both directions (Table 2). Across five training seeds (Table S23), these effects of the networks were consistent:
- 1D CNN: +0.066 on CinC 2011 and +0.102 on CinC 2017;
- 2D CNN: −0.023 on CinC 2011 and −0.069 on CinC 2017;
- MLP: −0.353 on CinC 2011;
- multimodal net: +0.053 on CinC 2017.

The seed-42 drop of the multimodal net on CinC 2011 (0.518) was not reproducible. Across seeds, its difference ranged from −0.156 to +0.269, and its BUT-only result was itself unstable (0.500–0.674). On the BUT QDB test, the 2D CNN and the multimodal net varied widely between seeds (0.705–0.989 and 0.741–0.992).

**Other models.**
- *Tabular models:* logistic regression and LightGBM improved slightly on CinC 2017, and TabPFN was slightly worse on both external targets. With only CinC data for training, logistic regression no longer ranked BUT QDB windows (0.457, wide interval).
- *HuBERT-ECG probe:* it gained most on CinC 2011, from 0.562 to 0.804 (p_Holm < 0.001), and lost slightly on CinC 2017 (−0.036).
- *d1-3B probe:* it changed little (0.947, 0.818 and 0.908 on the three held-out datasets).

### 3.4 Operating points and prevalence

*Table 3. CinC 2017 at natural prevalence (8,512 recordings, 278 noisy), all 11 models trained on BUT QDB only. Recording maximum (pre-specified); the last column uses the recording mean (post hoc). Sens./FR: sensitivity and false reject rate at τ = 0.50; PPV/NPV: positive/negative predictive value. Bootstrap CIs for the seven main models: Table S29.*

| Model | AUROC, recording max (pre-specified) | AP, recording max | Sens. / FR at τ = 0.50 | PPV / NPV at τ = 0.50 | AUROC, recording mean (post hoc) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.868 | 0.173 | 80.2% / 18.9% | 0.126 / 0.992 | 0.916 |
| **LightGBM (SQI)** | 0.880 | 0.234 | 67.6% / 14.6% | 0.135 / 0.987 | 0.930 |
| **Independent SQI MLP** | 0.868 | 0.140 | 86.3% / 24.0% | 0.108 / 0.994 | 0.934 |
| **1D CNN (Raw Signal)** | 0.773 | 0.064 | 92.4% / 34.5% | 0.083 / 0.996 | 0.801 |
| **2D CNN (Rendered Image)** | 0.850 | 0.137 | 89.9% / 33.8% | 0.082 / 0.995 | 0.879 |
| **SinglePass Multimodal Net** | 0.835 | 0.107 | 88.8% / 35.7% | 0.078 / 0.994 | 0.888 |
| **Rule-Based SQI Heuristic** | 0.887 | 0.182 | 18.3% / 2.5% | 0.199 / 0.973 | 0.933 |
| **LiquidAI d1-3B (Zero-Shot)** | 0.757 | 0.113 | 19.8% / 3.1% | 0.177 / 0.973 | 0.816 |
| **TabPFN (SQI)** | 0.858 | 0.210 | 68.7% / 14.2% | 0.140 / 0.988 | 0.896 |
| **HuBERT-ECG probe** | 0.907 | 0.262 | 93.2% / 26.2% | 0.107 / 0.997 | 0.942 |
| **d1-3B image-token probe** | 0.908 | 0.264 | 87.8% / 23.2% | 0.113 / 0.995 | 0.949 |

**Default threshold.** At the 3.3% prevalence of CinC 2017, positive predictive value at the default threshold was low for all 11 models (0.08–0.20; Table 3). The heuristic had the highest positive predictive value (0.199), but at 18.3% sensitivity.

**Transferred thresholds.** Thresholds chosen on BUT QDB did not transfer (Table S4). The Target 1 rule (false reject rate ≤ 5% on BUT QDB) gave false reject rates on CinC 2017 of:
- 15.7–54.5% for the trained models;
- 6.2% for the heuristic, whose sensitivity fell to 52.9%.

On CinC 2011, the same thresholds gave sensitivities between 0% (MLP) and 100% at a 100% false reject rate (heuristic).

*Table 4. Target 1 thresholds (maximum sensitivity at FPR ≤ 5%) re-set on n = 100 randomly drawn labelled target recordings, evaluated on the remaining recordings (500 draws; median and 2.5–97.5 percentiles over draws).*
- *Achieved FPR:* the share of draws with FPR ≤ 5% and ≤ 7.5%. The 7.5% level was pre-specified as a tolerance for the sampling error of a threshold chosen on n recordings.
- *Fallback:* draws without a positive recording, for which the transferred threshold was kept.
- *Brier:* uncalibrated BUT-only score versus Platt scaling fitted on the same draws. The median is over the draws in which Platt scaling could be fitted; the number skipped (single-class draws) is in parentheses.
- *All sample sizes:* Table S24.

| Dataset | Model | n | Transferred threshold: FPR / Sens. (median) | Local threshold: FPR median [95% range] | Local threshold: Sens. median [95% range] | Draws with FPR ≤ 5% / ≤ 7.5% (local) | Fallback draws | Brier: raw → Platt (median; Platt skipped) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CinC2011 | **Logistic Regression (SQI)** | 100 | 100.0% / 66.5% | 0.0% [0.0–0.0%] | 0.0% [0.0–0.0%] | 100% / 100% | 0 | 0.768 → 0.171 (0) |
| CinC2011 | **LightGBM (SQI)** | 100 | 4.7% / 51.2% | 4.7% [0.3–11.7%] | 50.0% [6.3–57.7%] | 65% / 82% | 0 | 0.219 → 0.180 (0) |
| CinC2011 | **Independent SQI MLP** | 100 | 0.0% / 0.0% | 1.0% [0.3–1.0%] | 36.2% [32.7–38.9%] | 100% / 100% | 0 | 0.348 → 0.228 (0) |
| CinC2011 | **1D CNN (Raw Signal)** | 100 | 30.5% / 76.1% | 2.6% [0.0–13.0%] | 1.2% [0.0–16.0%] | 67% / 81% | 0 | 0.243 → 0.202 (0) |
| CinC2011 | **2D CNN (Rendered Image)** | 100 | 10.7% / 60.6% | 4.6% [0.7–11.3%] | 18.3% [1.9–59.9%] | 61% / 74% | 0 | 0.244 → 0.174 (0) |
| CinC2011 | **SinglePass Multimodal Net** | 100 | 0.3% / 34.1% | 0.3% [0.0–0.3%] | 34.2% [31.0–37.2%] | 100% / 100% | 0 | 0.232 → 0.173 (0) |
| CinC2011 | **Rule-Based SQI Heuristic** | 100 | 100.0% / 100.0% | 3.7% [0.0–12.5%] | 40.3% [32.0–47.8%] | 66% / 83% | 0 | 0.278 → 0.195 (0) |
| CinC2011 | **LiquidAI d1-3B (Zero-Shot)** | 100 | 100.0% / 99.4% | 2.7% [0.0–7.9%] | 38.5% [31.4–43.4%] | 82% / 93% | 0 | 0.510 → 0.199 (0) |
| CinC2011 | **d1-3B image-token probe** | 100 | 11.7% / 65.0% | 3.6% [0.2–11.7%] | 53.5% [44.5–63.5%] | 70% / 85% | 0 | 0.161 → 0.142 (0) |
| CinC2011 | **TabPFN (SQI)** | 100 | 24.3% / 61.4% | 3.7% [0.2–11.1%] | 44.4% [3.1–53.2%] | 61% / 90% | 0 | 0.214 → 0.186 (0) |
| CinC2011 | **HuBERT-ECG probe** | 100 | 40.0% / 48.8% | 3.6% [0.0–13.2%] | 4.9% [0.0–18.9%] | 64% / 78% | 0 | 0.319 → 0.228 (0) |
| CinC2017 | **Logistic Regression (SQI)** | 100 | 15.7% / 74.5% | 2.5% [0.0–15.7%] | 15.6% [0.0–74.5%] | 83% / 94% | 21 | 0.145 → 0.031 (21) |
| CinC2017 | **LightGBM (SQI)** | 100 | 19.2% / 79.5% | 1.7% [0.0–19.1%] | 20.9% [0.0–79.5%] | 88% / 93% | 21 | 0.105 → 0.029 (21) |
| CinC2017 | **Independent SQI MLP** | 100 | 16.5% / 74.5% | 1.8% [0.0–16.5%] | 7.3% [0.0–74.5%] | 83% / 93% | 21 | 0.175 → 0.031 (21) |
| CinC2017 | **1D CNN (Raw Signal)** | 100 | 54.5% / 99.6% | 0.0% [0.0–54.5%] | 0.0% [0.0–99.6%] | 95% / 96% | 21 | 0.308 → 0.032 (21) |
| CinC2017 | **2D CNN (Rendered Image)** | 100 | 50.4% / 96.4% | 0.4% [0.0–50.3%] | 3.3% [0.0–96.4%] | 86% / 93% | 21 | 0.284 → 0.031 (21) |
| CinC2017 | **SinglePass Multimodal Net** | 100 | 43.1% / 92.8% | 0.0% [0.0–43.0%] | 0.0% [0.0–92.8%] | 87% / 94% | 21 | 0.303 → 0.031 (21) |
| CinC2017 | **Rule-Based SQI Heuristic** | 100 | 6.2% / 52.9% | 2.2% [0.0–7.5%] | 13.5% [0.0–62.4%] | 77% / 97% | 21 | 0.095 → 0.031 (21) |
| CinC2017 | **LiquidAI d1-3B (Zero-Shot)** | 100 | 3.1% / 19.8% | 0.5% [0.0–7.0%] | 5.5% [0.0–33.2%] | 95% / 100% | 21 | 0.091 → 0.031 (21) |
| CinC2017 | **TabPFN (SQI)** | 100 | 29.2% / 87.7% | 1.1% [0.0–29.2%] | 13.1% [0.0–87.8%] | 87% / 94% | 21 | 0.109 → 0.030 (21) |
| CinC2017 | **HuBERT-ECG probe** | 100 | 51.6% / 99.6% | 2.2% [0.0–51.6%] | 28.5% [0.0–99.6%] | 87% / 95% | 21 | 0.175 → 0.028 (21) |
| CinC2017 | **d1-3B image-token probe** | 100 | 51.1% / 98.9% | 1.7% [0.0–51.1%] | 25.3% [0.0–98.9%] | 83% / 94% | 21 | 0.162 → 0.028 (21) |

**Local thresholds (Table 4).** Re-setting the threshold on a small labelled target sample brought false rejections close to the target in most draws.
- *CinC 2011:* with n = 100, the false reject rate was at most 5% in 61–100% and at most 7.5% in 74–100% of draws, depending on the model. Median sensitivity was 54% for the d1-3B probe, 50% for LightGBM and 40% for the heuristic.
- *CinC 2017:* with n = 100, the false reject rate was at most 5% in 77–95% and at most 7.5% in at least 93% of draws, but median sensitivity was 0–29%. A draw of 100 recordings contains about three noisy recordings, so the threshold rests on very few positives, and 21 of 500 draws contained none. Even with n = 200, median sensitivity remained 0–37% (Table S24).

![](/Volumes/KIOXIA/Projects/d13b-ecg-sqi/figures/q1_figure3.png)

*Figure 3. Thresholds re-set on n labelled target recordings (Target 1, false reject rate ≤ 5%), evaluated on the remaining recordings: median over 500 draws for each model (grey) and the median over models (blue). On CinC 2017, 229 of 500 draws at n = 25 and 82 at n = 50 contained no noisy recording and kept the transferred threshold, which explains the values at n = 25. Source: Table S24.*

**Calibration.** Platt scaling on the drawn recordings lowered the Brier score of every model. On CinC 2017, however, the recalibrated Brier scores (0.028–0.032 at n = 100) were close to that of a constant prediction at the prevalence (0.032). This comparison suggests that much of the gain reflects a corrected probability level; it does not by itself establish that the recalibrated probabilities are well calibrated.

### 3.5 Computational cost

*Table 5. Per-window latency on an Apple M4 (median and 95th percentile of three runs of 25 windows), separating preprocessing from the forward pass.*

| Model Architecture | Parameters | Disk Weights | Execution Device | Preproc Median (p95) | Forward Median (p95) | Total Median (p95) | Total Median, Range over Runs | Preproc Share (Median of Ratios) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based SQI Heuristic** | 0 | 0.00 MiB | CPU | 2.58 ms (3.04 ms) | 0.22 ms (0.24 ms) | 2.81 ms (3.27 ms) | 2.60–2.83 ms | 92.3% |
| **Logistic Regression (SQI)** | 14 | 0.01 MiB | CPU | 1.34 ms (1.52 ms) | 0.18 ms (0.22 ms) | 1.52 ms (1.73 ms) | 1.51–1.54 ms | 88.4% |
| **LightGBM (SQI)** | 1,092 | 0.14 MiB | CPU | 1.32 ms (1.49 ms) | 0.17 ms (0.25 ms) | 1.48 ms (1.72 ms) | 1.48–1.55 ms | 88.4% |
| **Independent SQI MLP** | 3,201 | 0.01 MiB | Apple M4 (MPS) | 1.38 ms (1.76 ms) | 1.13 ms (1.77 ms) | 2.53 ms (3.62 ms) | 2.49–2.54 ms | 54.5% |
| **1D CNN (Raw Signal)** | 167,489 | 0.64 MiB | Apple M4 (MPS) | 0.00 ms (0.00 ms) | 1.47 ms (2.28 ms) | 1.47 ms (2.28 ms) | 1.36–1.49 ms | 0.0% |
| **2D CNN (Rendered Image)** | 245,697 | 0.94 MiB | Apple M4 (MPS) | 60.23 ms (137.53 ms) | 5.12 ms (5.63 ms) | 65.24 ms (142.53 ms) | 64.74–66.39 ms | 92.4% |
| **SinglePass Multimodal Net** | 131,969 | 0.47 MiB | Apple M4 (MPS) | 61.41 ms (128.75 ms) | 5.56 ms (6.10 ms) | 67.09 ms (134.26 ms) | 66.59–69.04 ms | 92.4% |
| **LiquidAI d1-3B (Zero-Shot)** | 3,120,000,000 | 5958.00 MiB | Apple M4 (MPS) | 63.37 ms (155.48 ms) | 828.47 ms (834.71 ms) | 891.19 ms (984.62 ms) | 891.06–893.27 ms | 7.1% |

**Image pipelines.** For the image pipelines, rendering dominated: about 60 ms (92% of the total) against 5 ms for the forward pass. This cost belongs to the plotting library rather than to rasterization as such: a direct raster renderer took 0.63 ms (Table S8). The image models, however, were not evaluated on those strips.

**1D CNN.** The 1D CNN took 1.41 ms on the CPU from the native 1,000-Hz window, with no rendering step.

**d1-3B.** d1-3B took 891 ms per window, dominated by its 828-ms forward pass.

**Harness dependence.** Forward-pass times of a few milliseconds on the GPU differ between our two timing scripts (Limitation 8); only order-of-magnitude differences are interpreted.

**Foundation-model probes and TabPFN (Table S27).**
- *HuBERT-ECG probe:* 28.0 ms per window on MPS (64.1 ms on the CPU), of which 23.4 ms was the forward pass.
- *d1-3B probe:* the linear probe adds 0.13 ms to the d1-3B forward pass.
- *TabPFN:* it predicts in context over the 1,047 training windows and took 2.5 s per window on the CPU.

## 4. Discussion

**Main finding.** The main finding is negative and practical: on a patient-independent in-domain benchmark, ten of eleven models did not differ significantly from the highest-AUROC model after multiplicity adjustment, and their external rankings disagree with one another. With 15 subjects, the absence of a significant difference does not establish equivalent performance. In-domain accuracy is necessary, but it does not select a model for a new device.

**The CNNs.** The 1D and 2D CNNs were jointly best on BUT QDB, yet the 1D CNN was the worst trained model on CinC 2017. The 2D CNN was second on CinC 2011 but significantly behind the heuristic, LightGBM and both probes on CinC 2017.

**Feature reversal.** The SQI analyses explain part of this pattern. Reversals in feature–outcome associations can undermine unchanged transfer, particularly for classifiers that rely strongly on those associations. TabPFN, which removes the tuning question for the tabular models, did not improve on their external results. These findings suggest that the selected feature representation contributes to the observed transfer limitations.

**The heuristic.** The hand-set rule outperformed a logistic-regression replacement using the same seven inputs. The results are consistent with the rule providing useful inductive biases: low detector agreement, high noise power, saturation and flatline imply poor quality.

Two caveats limit this interpretation:
- *Functional form:* the replacement also changes the functional form, so the performance difference cannot be attributed solely to the hand-set coefficients.
- *Development history:* the rule's development history is undocumented, which prevents a definitive assessment of whether BUT QDB influenced it.

**The quantization artefact.** The quantization artefact in the flatline fraction is a concrete example of a feature whose meaning depends on the acquisition chain. It should be redefined (for example, on a minimum duration of constant signal at a quantization-aware tolerance) before it is used across devices.

**Foundation models: information present but not exploited.** The zero-shot vision-language model was among the weakest models in-domain and did not track controlled noise. Yet a linear probe on the image tokens of the same forward pass was never significantly behind the best model. It was best on CinC 2011 and on the full CinC 2017 set. Its representation therefore contains task-relevant information that the zero-shot decision configuration does not exploit effectively. The probe shows that useful information is present; it does not identify every reason for the zero-shot failure.

**HuBERT-ECG.** The HuBERT-ECG probe matched the best models in-domain and on CinC 2017, but was near chance on CinC 2011 with the authors' preprocessing. Two observations bear on this:
- *Preprocessing:* changing the filter and scaling moved its CinC 2011 AUROC from 0.732 to 0.562 and its CinC 2017 AUROC from 0.867 to 0.907. For frozen foundation models, preprocessing is a model parameter that must be reported, and it can change conclusions.
- *Fine-tuning:* in a single, untested run, partial fine-tuning gave 0.706 on CinC 2011, above the probe's 0.562. Whether this reproduces across seeds is unknown.

Both observations are plausible for a model pre-trained on 12-lead recordings and given one lead among eleven constant leads. A multi-lead input would test this directly.

**Recommendation for foundation models.** Probing frozen representations is cheap and is a reasonable first adaptation step before fine-tuning. Its inference cost, however, is that of the full foundation model.

**Multi-source training is not a general remedy.** Multi-source development is often recommended for robustness (Moeyersons et al 2019). Here it helped the raw-signal CNN substantially and consistently across seeds, but it harmed the 2D CNN on both external targets and the MLP on CinC 2011. Two features of the design contribute:
- the training sources carry different labels (single-lead classifiability, 12-lead acceptability, expert three-class quality);
- the training sources have very different sizes.

For a deployment, the training sources and their label definitions matter as much as their number.

**Deployment implications.** Four implications follow:
- Thresholds must be set on the target population. Neither post-hoc nor nested thresholds from BUT QDB carried over. At low prevalence, however, local recalibration needs enough labelled positives, not a fixed number of recordings: 100–200 recordings controlled false rejections but left sensitivity poorly determined. An enriched calibration sample would be needed, for example one that oversamples flagged recordings for review.
- Precision at realistic prevalence is low unless sensitivity is sacrificed. A quality gate should therefore be judged by its false reject rate per hour of usable signal and by the cost of a missed unusable segment, not by AUROC.
- The rendering step of image-based quality models is a removable engineering cost, not an intrinsic one.
- A deployed quality gate needs an explicit check for constant or disconnected signals. Several trained models, including the tabular models and the 2D CNN, scored completely flat CinC 2011 leads as usable, because no such signal occurred in the development data.

### 4.1 Limitations

**1. Development cohort size.** The development cohort has 15 subjects, and unusable windows occur in only 7. Subject-grouped folds prevent identity leakage, but they cannot add diversity. In-domain non-significance therefore partly reflects limited power: confidence intervals of the AUROC differences are wide.

**2. Sampling and protocol.** Windows were sampled by quota (28.4% unusable), so precision is reported re-weighted. The primary analysis does not follow the official BUT QDB protocol (Smital et al 2026). It includes 59 unusable windows from a stretch of missing signal that the protocol excludes; these windows slightly inflate in-domain AUROC (Table S26). Our models output window-level binary scores, so the official sample-level three-class metrics were not computed.

**3. Label definitions differ by design.** CinC 2011 labels judge the whole 12-lead recording, while we score one lead. CinC 2017 labels judge rhythm classifiability per recording. No large multi-lead dataset with lead-wise usability labels was available.

**4. Fixed hyperparameters.** The task-specific models were not tuned. TabPFN addresses this for the tabular models. For the networks, untuned hyperparameters may understate in-domain and transfer performance.

**5. Seeds and repeated runs.**
- Seed 42 is the reported run.
- Seeds 1–4 of the CNNs changed in-domain AUROC by at most 0.02, but changed default-threshold sensitivity on CinC substantially. The multimodal net gave constant outputs on CinC 2011 in three of four additional seeds (Table S9).
- LODO was repeated over five seeds for the networks (Table S23). TabPFN and the fine-tuned HuBERT-ECG are single runs.
- The probes are deterministic given their features.
- The CinC 2017 training subset of LODO was drawn once.

**6. Foundation-model scope.**
- Only two foundation models were probed with a linear read-out.
- HuBERT-ECG was also partially fine-tuned with one fixed configuration.
- d1-3B was not fine-tuned, and few-shot prompting was not tested.
- HuBERT-ECG preprocessing follows the authors' code, but the placement of a single lead among absent leads is our choice, and the results depend on preprocessing (Table S25).
- ECG-FM (McKeen et al 2025) was not evaluated.
- HuBERT-ECG weights are licensed for non-commercial use.
- The pre-training corpora of d1-3B and HuBERT-ECG (9.1 million 12-lead ECGs) were not checked record by record. Overlap with the public evaluation datasets, as unlabelled signals, could not be verified or excluded.

**7. Post-hoc analyses.** The 12-lead aggregation rules and the CinC 2017 window mean were chosen after the main analysis. They are reported as sensitivity analyses. The heuristic's logistic-regression replacement, TabPFN, the HuBERT-ECG probe and LODO were added during revision, after the single-source results were known. They did not change any earlier result. The later additions were committed to a dated analysis plan before being run: seeds, full CinC 2017 d1-3B, local recalibration, HuBERT-ECG preprocessing check and fine-tuning. The HuBERT-ECG primary result changed as a consequence of the pre-specified preprocessing rule.

**8. Latency.** Latency reflects one machine and Python implementations, and TabPFN was timed on the CPU only. Table 5 and Table S8 come from different scripts and runs:
- *Table 5:* the models are timed one after another in one process.
- *Table S8:* a later script interleaves CPU and MPS passes per window and excludes the first window.

Their MPS forward times of 1–6 ms differ by up to 2.3 ms in either direction (1D CNN 1.47 vs 2.80 ms; 2D CNN 5.12 vs 2.92 ms).

Part of the image-model difference is explained by caching:
- the image models cache each decoded and resized strip after first use;
- in Table 5, decoding and resizing therefore fall inside the timed forward pass for 19 of 25 windows;
- in Table S8 they fall inside it only in the first of three runs (2D CNN on MPS: run median 3.75 ms, against 2.90 ms in the cached runs).

The remaining differences, including the 1D CNN's, are unexplained, so MPS forward times at this scale are treated as harness-dependent. The conclusions rest only on differences of an order of magnitude: rendering of about 60 ms versus the forward pass, and 828 ms for d1-3B.

**9. NSTDB.** The NSTDB SNR labels are nominal, and there is one noisy record per label and base record.

**10. Aggregation and edge cases.**
- *Aggregation:* external rankings depend on the recording-level aggregation (Table 3).
- *Constant leads:* part of the CinC 2011 discrimination comes from 65 records with a constant lead II (Table S28).

**11. Demographics and fairness.** Performance was not assessed across demographic subgroups.
- *BUT QDB:* it reports the sex and age of its 15 healthy subjects (10 female and 8 male recordings, 21–83 years) (Smital et al 2026). With unusable windows in only 7 subjects, a subgroup analysis would not be informative.
- *CinC datasets:* demographic information was not used.

Whether performance differs by sex, age or device population is therefore unknown.

## 5. Conclusions

**Summary.** Across eleven models and three external evaluations, in-domain performance on a patient-independent benchmark did not identify the model that transferred best:
- the 1D CNN, jointly best in-domain, was the worst trained model on CinC 2017;
- a hand-set rule outperformed a logistic-regression replacement using the same inputs and had a higher AUROC on CinC 2017 than every task-specific model;
- a linear probe on a vision-language model's frozen image tokens was the only model never significantly behind the best;
- an ECG foundation model's probe was sensitive to its preprocessing.

**Multi-source training.** Multi-source training had model-specific effects, consistent across seeds for the CNNs.

**Operating points.** Operating points chosen on development data did not transfer. At low prevalence, local recalibration on 100–200 labelled recordings controlled false rejections but not sensitivity.

**Recommendation.** ECG quality models should be evaluated externally, on data whose labels match the intended use, with thresholds set locally.

## Acknowledgements

The author thanks the investigators and institutions that created and made available the BUT QDB, the PhysioNet/Computing in Cardiology Challenge 2011 and 2017 datasets and the MIT-BIH Noise Stress Test Database, and the developers of the openly released models used in this study.

## AI disclosure statement

[AUTHOR TO CONFIRM] Generative-AI tools were used in three distinct roles.

- *Antigravity (Google):* used to write an initial version of the analysis code and of the manuscript draft.
- *Claude Code (Anthropic):* used to audit and correct that code, including independent recomputation of the reported metrics and re-execution checks; to write the later analysis, table-generation, figure-generation and verification scripts; and to revise and edit the manuscript text.
- *Perplexity:* used as an LLM-as-a-judge. It reviewed successive versions of the manuscript, Supplementary Material and reporting checklist and commented on methodological, statistical and reporting problems, numerical consistency and reference accuracy.

Each critique was checked against the stored study outputs before any change was accepted, and suggestions not supported by the data were not adopted.

The author defined the research question, scientific objectives, datasets, evaluation protocol and interpretation framework; executed all code locally; reviewed the implementation, analytic outputs, reported results and references; and retains full responsibility for the study design, data handling, methods, analyses, results, interpretations, originality and integrity of this manuscript.

## Funding

[AUTHOR TO CONFIRM] This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

## Ethics approval and consent

[AUTHOR TO CONFIRM] This study is a secondary analysis of publicly available, de-identified ECG datasets, used under the terms of their public releases: BUT QDB (collected under IRB protocol EC:EK:05b/2018 of Brno University of Technology), the PhysioNet/Computing in Cardiology Challenge 2011 and 2017 datasets, and the MIT-BIH Noise Stress Test Database. No new data were collected from human participants, and the author had no access to identifiable information. Ethical approval was not required for this secondary analysis of public de-identified data.

## Conflicts of interest

[AUTHOR TO CONFIRM] The author declares no competing interests.

## Author contributions

[AUTHOR TO CONFIRM] Mehmet Kuntalp (sole author): Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review & editing.

## Data availability statement

The source ECG data are publicly available from PhysioNet: BUT QDB (Nemcova et al 2020), the PhysioNet/Computing in Cardiology Challenge 2011 (Silva et al 2011) and 2017 (Clifford et al 2017) datasets, and the MIT-BIH Noise Stress Test Database (Moody et al 1984).

The repository at https://github.com/mkuntalp-ops/d13b-ecg-sqi (MIT licence) contains:
- the analysis code;
- the dated analysis plan;
- the cohort and fold manifests;
- the saved predictions of every model;
- the scripts that regenerate every table and figure from those predictions.

The following are not redistributed:
- raw ECG waveforms;
- the weights of LiquidAI/d1-3B, HuBERT-ECG and TabPFN, which are available from their providers under their own licences (HuBERT-ECG: CC BY-NC 4.0).

[AUTHOR TO ADD: archive DOI of the release, if archived.]

## References

Clifford G D, Behar J, Li Q and Rezek I 2012 Signal quality indices and data fusion for determining clinical acceptability of electrocardiograms *Physiological Measurement* 33 1419–1433 https://doi.org/10.1088/0967-3334/33/9/1419

Clifford G D et al 2017 AF classification from a short single lead ECG recording: the PhysioNet/Computing in Cardiology Challenge 2017 *Computing in Cardiology* 44 https://doi.org/10.22489/CinC.2017.065-469

Coppola E, Savardi M, Massussi M, Adamo M, Metra M and Signoroni A 2024 HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications *medRxiv* (preprint) https://doi.org/10.1101/2024.11.14.24317328

DeLong E R, DeLong D M and Clarke-Pearson D L 1988 Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach *Biometrics* 44 837–845 https://doi.org/10.2307/2531595

Grinsztajn L et al 2026 TabPFN-3: technical report *arXiv* 2605.13986 https://arxiv.org/abs/2605.13986

Guo L L et al 2022 Evaluation of domain generalization and adaptation on improving model robustness to temporal dataset shift in clinical medicine *Scientific Reports* 12 2726 https://doi.org/10.1038/s41598-022-06484-1

Herraiz Á H, Martínez-Rodrigo A, Bertomeu-González V, Quesada A, Rieta J J and Alcaraz R 2020 A deep learning approach for featureless robust quality assessment of intermittent atrial fibrillation recordings from portable and wearable devices *Entropy* 22 733 https://doi.org/10.3390/e22070733

Hollmann N et al 2025 Accurate predictions on small data with a tabular foundation model *Nature* 637 319–326 https://doi.org/10.1038/s41586-024-08328-6

Holm S 1979 A simple sequentially rejective multiple test procedure *Scandinavian Journal of Statistics* 6 65–70

Hu Y, Liu Q, Zhou Z, Xu W and Tang H 2025 A paralleled multi-task learning-based framework for single-lead ECG fine-grained noise localization, denoising and signal quality assessment *Sensors* 25 7152 https://doi.org/10.3390/s25237152

Huang X, Zhang F, Fan H, Chang H, Zhou B and Li Z 2024 Pseudo anomalies enhanced deep support vector data description for electrocardiogram quality assessment *Computers in Biology and Medicine* 170 107928 https://doi.org/10.1016/j.compbiomed.2024.107928

Li Q, Mark R G and Clifford G D 2008 Robust heart rate estimation from multiple asynchronous noisy sources using signal quality indices and a Kalman filter *Physiological Measurement* 29 15–32 https://doi.org/10.1088/0967-3334/29/1/002

Liquid AI 2026 LiquidAI/d1-3B: open-weight edge multimodal decision model (Hugging Face model repository, revision 051bcc464b01b9f92942b364d9586b0ef5912432) https://huggingface.co/LiquidAI/d1-3B

Liu F et al 2022 Wearable electrocardiogram quality assessment using wavelet scattering and LSTM *Frontiers in Physiology* 13 905447 https://doi.org/10.3389/fphys.2022.905447

McKeen K, Masood S, Toma A, Rubin B and Wang B 2025 ECG-FM: an open electrocardiogram foundation model *JAMIA Open* 8 ooaf122 https://doi.org/10.1093/jamiaopen/ooaf122

Moeyersons J et al 2019 Artefact detection and quality assessment of ambulatory ECG signals *Computer Methods and Programs in Biomedicine* 182 105050 https://doi.org/10.1016/j.cmpb.2019.105050

Moody G B, Muldrow W E and Mark R G 1984 A noise stress test for arrhythmia detectors *Computers in Cardiology* 11 381–384

Nemcova A, Smisek R, Opravilová K, Vitek M, Smital L and Maršánová L 2020 Brno University of Technology ECG Quality Database (BUT QDB) (version 1.0.0) *PhysioNet* https://doi.org/10.13026/kah4-0w24

Orphanidou C, Bonnici T, Charlton P, Clifton D, Vallance D and Tarassenko L 2015 Signal-quality indices for the electrocardiogram and photoplethysmogram: derivation and applications to wireless monitoring *IEEE Journal of Biomedical and Health Informatics* 19 832–838 https://doi.org/10.1109/JBHI.2014.2338351

Pan J and Tompkins W J 1985 A real-time QRS detection algorithm *IEEE Transactions on Biomedical Engineering* BME-32 230–236 https://doi.org/10.1109/TBME.1985.325532

PhysioNet 2011 Improving the quality of ECGs collected using mobile phones: the PhysioNet/Computing in Cardiology Challenge 2011 (challenge description) https://moody-challenge.physionet.org/2011/

Satija U, Ramkumar B and Manikandan M S 2018 A review of signal processing techniques for electrocardiogram signal quality assessment *IEEE Reviews in Biomedical Engineering* 11 36–52 https://doi.org/10.1109/RBME.2018.2810957

Silva I, Moody G B and Celi L 2011 Improving the quality of ECGs collected using mobile phones: the PhysioNet/Computing in Cardiology Challenge 2011 *Computing in Cardiology* 38 273–276

Smital L et al 2020 Real-time quality assessment of long-term ECG signals recorded by wearables in free-living conditions *IEEE Transactions on Biomedical Engineering* 67 2721–2734 https://doi.org/10.1109/TBME.2020.2969719

Smital L et al 2026 Real-world 24h+ ECG dataset with quality annotations and motion context *Scientific Data* https://doi.org/10.1038/s41597-026-07905-w

Subbaswamy A and Saria S 2020 From development to deployment: dataset shift, causality, and shift-stable models in health AI *Biostatistics* 21 345–352 https://doi.org/10.1093/biostatistics/kxz041
