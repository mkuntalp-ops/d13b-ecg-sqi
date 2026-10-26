# Zero-shot and task-specific models for ECG usability screening: patient-independent performance, cross-dataset transfer, and computational cost

## Abstract

**Background:** Automated electrocardiogram (ECG) interpretation algorithms degrade when applied to corrupted recordings. Quality control triage is critical to accept clean traces, reject unusable segments, and prompt for re-recording. While deep multimodal decision models and lightweight convolutional networks offer potential solutions, their comparative accuracy, zero-shot transfer, and computational overhead require rigorous characterization under patient-disjoint outer validation.

**Methods:** We benchmarked eight models across six methodological paradigms: classical tabular classifiers (Logistic Regression, LightGBM, MLP on 13 input features including 11 Signal Quality Indices [SQIs] and 2 metadata parameters), deep learning architectures (1D CNN on raw 500 Hz waveforms, 2D CNN on rendered visual strips, and a single-pass multimodal network combining SQIs and visual features), an expert heuristic baseline, and the 3.12-billion-parameter LiquidAI/d1-3B multimodal edge model evaluated zero-shot with zero output tokens. Models were trained on the Brno University of Technology ECG Quality Database (BUT QDB; 1,047 windows sampled with per-subject quotas [750 usable, 297 unusable; 28.4%], 15 subjects) under patient-disjoint 5-fold GroupKFold cross-validation [7] and evaluated without retraining on the PhysioNet/Computing in Cardiology Challenge 2011 Set A (150 balanced records, lead II, with five random subsets and all 12 leads as sensitivity analyses) [2, 6], the 2017 Challenge (8,512 single-lead recordings, 3.3% too noisy to classify) [22], and the MIT-BIH Noise Stress Test Database (NSTDB; 264 electrode-motion windows at six nominal SNR labels, each paired with its clean base-record window) [3]. Thresholds were also chosen by nested cross-validation and transferred unchanged, BUT QDB metrics were re-weighted to the natural prevalence, frozen d1-3B representations were evaluated with linear probes, and latency was profiled on an Apple M4 over three runs.

**Results:** On BUT QDB (seed 42), the 1D CNN (AUROC 0.989; AP 0.975) and 2D CNN (AUROC 0.989; AP 0.971) gave the highest discrimination (means 0.976 and 0.986 over four further seeds), and nested threshold selection reproduced their post-hoc operating points (1D CNN 97.3% sensitivity at 3.5% FPR). Re-weighted to the natural prevalence without one dominant record (2.8%), their positive predictive value at τ = 0.50 was 0.71–0.77. Sensitivity varied widely between the seven subjects with unusable windows (1D CNN 14–100%). Multimodal fusion did not improve on the unimodal networks (AUROC 0.967). Zero-shot LiquidAI/d1-3B reached AUROC 0.852 (0.869–0.913 with two alternative prompts), whereas a linear probe on its frozen image tokens reached 0.979. No representation transferred best everywhere. On CinC 2011 the 2D CNN ranked recordings best (AUROC 0.746 on the reported subset; 0.713–0.884 across five random subsets) but flagged only 17.3–28.0% of unacceptable recordings at the default threshold; averaging all 12 leads raised its AUROC to 0.852 on average, and the d1-3B image probe reached 0.756–0.907. On CinC 2017 the rule heuristic (0.887) and the SQI models (0.868–0.880) ranked best and the 1D CNN worst (0.773), and at the natural 3.3% prevalence no model combined useful sensitivity with useful precision at the default threshold. Thresholds chosen on BUT QDB did not carry over (2D CNN, Target 1: 52.0% sensitivity at 10.7% false rejection on CinC 2011; 50.4% false rejection on CinC 2017). Under NSTDB electrode-motion noise the usable probability of every trained model fell with the nominal SNR (below 0.25 at +6 dB), whereas zero-shot d1-3B moved by at most 0.09. Matplotlib rendering took about 60 ms per strip (about 92% of image-pipeline latency) against 0.63 ms for a direct raster renderer, and the 1D CNN took 1.41 ms on the CPU from the native 1,000 Hz signal.

**Conclusion:** In-domain, the raw-signal and image CNNs gave the best discrimination and precision, and the 1D CNN was the cheapest to run. Transfer, not in-domain accuracy, limited every model: the best representation changed with the external dataset and its label, operating points did not transfer, and precision at natural prevalence was low, so thresholds must be set and validated on each target population. Zero-shot d1-3B lagged the task-specific models, but a linear read-out of its frozen image representation, trained on BUT QDB, matched the CNNs in-domain and transferred better than the supervised models on lead II of CinC 2011 and on the CinC 2017 subset, which points to adaptation rather than capacity as its limit.

---

## 1. Introduction & Research Framework

Automated electrocardiogram (ECG) interpretation algorithms represent a cornerstone of contemporary cardiovascular medicine, powering telemetry monitors, ambulatory Holter systems, and mobile health devices. However, automated diagnostic pipelines degrade substantially when exposed to real-world biosignal corruption. Common physical noise sources—including electrode motion artifact, baseline wander from patient respiration, electromyographic (EMG) interference from skeletal muscle contraction, and ambient powerline noise—can distort P-wave, QRS, and T-wave morphologies. As highlighted in foundational stress testing protocols [3], electrode motion artifacts are particularly pernicious because their spectral power overlaps directly with genuine ventricular depolarization, frequently mimicking ventricular arrhythmias and resisting conventional linear bandpass filtering. Automated signal quality assessment (SQA) is therefore indispensable as an initial triage gate to prevent spurious clinical alarms, prevent erroneous diagnostic conclusions, and prompt timely electrode re-attachment or signal re-recording.

### 1.1 Distinct Clinical Task Formulation

A critical methodological challenge in ECG quality assessment literature is the frequent conflation of distinct clinical task formulations. Specifically, "QRS-based rhythm usability", "morphological precision usability", and "overall 12-lead diagnostic acceptability" represent fundamentally distinct reference standards with disparate clinical stringencies. For instance, in the Brno University of Technology ECG Quality Database (BUT QDB) [1], annotations are stratified into three distinct tiers: Class 1 denotes pristine diagnostic quality; Class 2 designates records where noise precludes fine morphological measurement (such as ST-segment deviation or QT interval duration) yet permits reliable QRS complex detection and heart rate determination; and Class 3 designates recordings where severe corruption prevents even basic QRS identification. Conversely, community challenges such as the PhysioNet/Computing in Cardiology Challenge 2011 [2, 6] evaluated overall 12-lead diagnostic acceptability as judged by expert cardiologists. In this investigation, we strictly operationalize the primary clinical quality triage task as binary unusable detection (Class 3 vs. Classes 1 & 2 in BUT QDB), designed to identify uninterpretable waveforms that must be rejected from automated analysis.

### 1.2 Related Work in ECG Quality Assessment

Related Work in ECG Signal Quality Assessment spans several historical and emerging paradigms, as synthesized in comprehensive reviews of ECG quality evaluation technologies [21]. Classical engineering approaches have relied primarily on handcrafted Signal Quality Indices (SQIs) and multi-index decision heuristics. Seminal work by Clifford et al. [18] and Orphanidou et al. [20] demonstrated that statistical measures, spectral energy ratios, and detector consistency can triage clinical acceptability in ambulatory and mobile recordings. Handcrafted SQIs quantify statistical properties (e.g., skewness, kurtosis), spectral energy ratios, and consistency across parallel QRS detectors [1, 2, 18]. While computationally lightweight and interpretable, tabular feature engineering often fails to capture complex spatiotemporal artifact patterns. More recently, learned representations tailored for wearable biosensors have demonstrated improved feature extraction; for example, combining wavelet scattering transforms with Long Short-Term Memory (LSTM) networks captures multiscale transient dynamics while mitigating intra-patient morphological variability [15]. Furthermore, contemporary frameworks have explored joint noise localization and quality estimation, such as parallel multi-task convolutional architectures that simultaneously denoise, locate artifact boundaries, and classify segment quality [17], or specialized transient noise detectors designed to suppress false positive arrhythmia alarms in atrial fibrillation screening [16]. In parallel, the broader machine learning community has seen the emergence of general-purpose multimodal foundation models and vision-language decision architectures capable of zero-shot classification [5]. However, prior studies frequently evaluate these paradigms under disparate datasets, inconsistent validation splits, or optimistic random window sampling, preventing direct and reproducible comparison across architectural families.

### 1.3 Research Gaps and Primary Questions

To address these gaps, this study establishes a benchmark structured along four methodological axes:
1. Patient-Independent Generalization: Evaluating models under patient-disjoint GroupKFold cross-validation to isolate subject identities [7], preventing intra-patient data leakage.
2. Out-of-Distribution Transfer and Dataset Shift: Characterizing cross-dataset performance across disparate patient demographics and clinical acquisition modalities, grounded in established dataset shift principles in health AI [4, 14].
3. Operating Threshold Behavior: Differentiating fixed default decision rules (τ = 0.50) from post-hoc exploratory ROC operating points.
4. End-to-End Computational Profiling: Measuring the execution budget on edge hardware across CPU and GPU/MPS devices, quantifying the specific latency overhead introduced by visual waveform rasterization.

Specifically, we investigate four primary Research Questions (RQs):
* **RQ1 (Unimodal vs. Multimodal Fusion):** Does fusing visual ECG waveforms with tabular SQIs yield superior patient-independent quality triage compared to specialized unimodal 1D signal or 2D image models?
* **RQ2 (Zero-Shot Edge VLM vs. Supervised Baselines):** How does a contemporary 3.12-billion-parameter multimodal decision model (LiquidAI/d1-3B [5]), evaluated zero-shot via choice-probability scoring without task-specific training, compare against task-specialized supervised neural networks and expert rule heuristics?
* **RQ3 (Cross-Dataset Transfer Resilience):** How effectively do tabular, visual, and raw signal representations transfer out-of-distribution to external ECG recordings (CinC 2011 Set A and CinC 2017) and calibrated noise stress conditions (NSTDB), and what mechanisms govern feature shift?
* **RQ4 (The Visual Rendering Bottleneck):** What are the empirical hardware latencies and memory footprints across architectures, and does waveform rasterization impose a substantial cost for resource-constrained edge clinical deployment?

---

## 2. Data and Stratified Sampling Protocols

### 2.1 BUT ECG Quality Database (BUT QDB) [1]
The primary development cohort was derived from the BUT ECG Quality Database, curated by Brno University of Technology and hosted on PhysioNet [1]. BUT QDB comprises 18 long-term single-lead ECG recordings acquired across 15 distinct subjects. Waveforms were digitized at a sampling rate of 1,000 Hz with 16-bit resolution. The 18 recordings total 478.7 hours, of which 99.39 hours (357,799 seconds) carry consensus quality annotations (Classes 1–3); the rest of the signal is unannotated and was not used. The candidate pool was every non-overlapping 10-s window inside the consensus annotation intervals of the 18 records (32,956 windows, 5,285 of them Class 3). From this pool we sampled per subject up to 50 usable (Class 1 or 2) and up to 60 unusable (Class 3) windows, evenly spaced in time over that subject's candidate windows. This gave 1,047 windows (750 usable, 297 unusable; 28.4% unusable) from 17 records: record 100002 contributed no window, because subject 100's quota was spread over both of its records and fell entirely on record 100001. The quota prevalence is not the natural one. Among the candidate windows Class 3 makes up 16.0%, but 4,734 of the 5,285 Class 3 windows come from one record (105001); without that record the share is 2.8%, and the median record contains none. Section 4.1 reports metrics re-weighted to these prevalences. Windows were resampled from 1,000 Hz to 500 Hz by Fourier-domain resampling (scipy.signal.resample). Class 2 (193 windows) is treated as usable in the main analysis; relabelling it as unusable is evaluated in Table 1d.

### 2.2 PhysioNet / Computing in Cardiology Challenge 2011 (CinC 2011 Set A) [2, 6]
To evaluate external zero-shot transfer, we utilized Set A of the PhysioNet/Computing in Cardiology Challenge 2011 [2, 6]. The challenge dataset consists of 10-second, 12-lead ECG recordings digitized at 500 Hz. Expert cardiologists provided overall 12-lead diagnostic acceptability annotations. The reported evaluation set is a balanced 150-record cohort (75 acceptable, 75 unacceptable) formed from the first records of the sorted PhysioNet lists, evaluated strictly on Lead II signals. Because this selection is not random, five random balanced subsets (75 + 75, seeds 42, 1, 2, 3, 4) were also evaluated with the same models (Table 2d). It is vital to note that this introduces an inherent clinical target mismatch: models evaluated on Lead II are tasked with predicting a whole-record 12-lead diagnostic decision, providing a stringent real-world test of covariate and label shift [4]. To measure how much of the transfer gap comes from this single-lead versus 12-lead mismatch, all 12 leads of the 565 records used in the transfer analyses were also scored, and the record score was taken as the lead II score, the maximum over the 12 leads, or their mean (Table 2e).

### 2.3 MIT-BIH Noise Stress Test Database (NSTDB) [3]
To benchmark noise-stress robustness across graded noise levels, we evaluated models on the electrode-motion records of the MIT-BIH Noise Stress Test Database [3]. The NSTDB noisy records (118e24 to 119e_6) are MIT-BIH base records 118 and 119 with electrode-motion ('em') noise added at nominal SNR labels of +24, +18, +12, +6, 0 and -6 dB. The NSTDB documentation states that noise is added in two-minute segments after the first five minutes of each record, alternating with clean segments. We confirmed this schedule against the signals: in each 720-s record the noise lies in 300–420 s and 540–660 s, and outside these segments the noisy record equals its base record up to a constant offset (residual standard deviation below 10^-6 mV after resampling). We therefore use a paired design. Noisy windows were taken inside the noise segments, at start times 305 + 10i and 545 + 10i seconds (i = 0–10; 22 per record, 44 per SNR label, 264 in total). Each noisy window is paired with the clean base-record window at the same start time, so both members of a pair contain the same beats and no offset (44 clean reference windows, 22 per record). NSTDB defines the signal power from the peak-to-peak amplitude of the QRS complexes and the noise power as a frequency-weighted measurement over the first 300 s of the noise record [3]. We did not reproduce these definitions, so the nominal labels are used as published. As a diagnostic, the full-band SNR of each window (clean power over residual power) lies about 14 dB below its label for every label, and the labels step by 6 dB.

### 2.4 Patient-Independent Grouped Cross-Validation Design [7]
To enforce patient-disjoint outer evaluation on BUT QDB, we applied 5-fold GroupKFold cross-validation [7], grouping windows strictly by patient identifier (15 subjects). This ensures that no patient contributes data to both the training and test partitions within any fold. While GroupKFold prevents patient identity leakage, the small overall patient count (15 subjects, with unusable Class 3 artifacts present in only 7 subjects) remains an important empirical limitation. Table 0 summarizes the patient and sample distribution across all five outer cross-validation folds. Model-specific feature scaling policies were strictly isolated to each fold's 12 training subjects: Logistic Regression utilized scikit-learn's StandardScaler; Independent SQI MLP and the tabular branch of SinglePass Multimodal Net utilized empirical z-score standardization fit strictly on the training partition; LightGBM operated on raw unscaled SQI features (scale-invariant decision trees); 1D CNN processed 500 Hz biosignals normalized to zero mean and unit variance within each window; and the 2D CNN and the image branch of the multimodal net processed RGB strips normalized with a fixed per-channel mean of 0.95 and standard deviation of 0.15. No MinMax scaling was applied.

#### Table 0: Patient-Independent 5-Fold GroupKFold Cross-Validation Partitioning on BUT QDB
*Windows were grouped strictly by patient ID so that no patient appears in both training and test sets in any fold.*

| Fold ID | Patient Subject IDs (N=3/fold) | Total Windows | Usable Windows (Class 0) | Unusable Windows (Class 1) | Unusable Prevalence (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 103, 113, 122 | 214 | 150 | 64 | 29.9% |
| **Fold 1** | 114, 118, 126 | 210 | 150 | 60 | 28.6% |
| **Fold 2** | 111, 115, 125 | 210 | 150 | 60 | 28.6% |
| **Fold 3** | 104, 105, 123 | 210 | 150 | 60 | 28.6% |
| **Fold 4** | 100, 121, 124 | 203 | 150 | 53 | 26.1% |
| **Total Cohort** | **15 Unique Patients** | **1,047** | **750** | **297** | **28.4%** |

### 2.5 PhysioNet / Computing in Cardiology Challenge 2017 (CinC 2017) [22]
As a second external cohort, at natural prevalence and with a single-lead label, we used the training set of the 2017 PhysioNet/Computing in Cardiology Challenge [22]: 8,528 single-lead recordings from a handheld AliveCor device (300 Hz, 9 to 61 s) labelled normal rhythm, atrial fibrillation, other rhythm, or too noisy to classify. The noisy label states that the rhythm cannot be classified from that lead, which is close to the BUT QDB Class 3 definition, so it was used as the unusable target and the other three labels as usable. Recordings shorter than 10 s (16) were excluded, leaving 8,512 recordings, 278 (3.3%) of them noisy. Each recording was split into non-overlapping 10-s windows from its start (at most three, i.e. the first 30 s; 23,985 windows), resampled to 500 Hz by Fourier-domain resampling, and scored by the models trained on the full BUT QDB cohort. The recording score is the maximum window score (primary) or the mean window score. Patient identifiers are not part of the public release. Because of its inference time, d1-3B was evaluated on a balanced subset: all 278 noisy recordings and 278 randomly chosen non-noisy recordings (seed 42); the other models are reported on that subset as well.

---

## 3. Model Architectures & Experimental Setup

### 3.1 Signal Quality Indicators (SQIs) and Input Representations
For tabular and multimodal models, we extracted 13 input features comprising 11 domain-expert Signal Quality Indices (SQIs) [18, 20, 21] and 2 recording metadata parameters:
1. `bsqi_detector_agreement`: F1-matching score between Pan-Tompkins [19] and difference-based QRS detectors (temporal tolerance: 50 ms) [18].
2. `baseline_wander_ratio`: Relative Welch spectral power in the baseline band (<0.5 Hz) relative to the ECG band (0.5-40 Hz) [18].
3. `high_frequency_noise_ratio`: Relative Welch spectral power in the electromyographic band (>40 Hz up to min(fs/2, 100 Hz)) relative to the ECG band [18].
4. `saturation_fraction`: Proportion of samples hitting dynamic extrema (ADC rail clipping).
5. `flatline_fraction`: Proportion of consecutive samples with zero difference (amplifier disconnection or saturation).
6. `qrs_energy_ratio`: Relative Welch spectral power in the QRS depolarization band (5-15 Hz) relative to the ECG band.
7. `kurtosis_ksqi`: Sample kurtosis of the 10-second signal distribution (scipy.stats.kurtosis) [18].
8. `skewness_ssqi`: Sample skewness of the 10-second signal distribution (scipy.stats.skew) [18].
9. `zero_crossing_rate`: Rate of sign zero-crossings after median subtraction.
10. `powerline_noise_ratio`: Relative Welch spectral power in powerline notch bands (49-51 Hz and 59-61 Hz) relative to the ECG band.
11. `signal_std`: Standard deviation of signal amplitude in mV.
12. `duration_sec`: Window duration in seconds (constant metadata: 10.0 s in BUT QDB).
13. `sampling_rate`: Sampling frequency in Hz (constant metadata: 500 Hz standardized input).

In the development cohort (1,047 windows), duration_sec and sampling_rate are constant metadata, while saturation_fraction (mean 4.0 × 10^-4, σ ≈ 8.7 × 10^-6) and flatline_fraction (mean 6.7 × 10^-5, σ ≈ 1.3 × 10^-4) have near-zero variance. When applied out-of-distribution without regularization, unconstrained standard scaling caused numerical saturation, motivating our retrospective sensitivity analysis on robust scaling. Two of these definitions depend on the acquisition chain. The saturation fraction counts samples equal to the window minimum or maximum, so its floor is 2/5,000 = 4 × 10^-4 and it rises only when extremes repeat (clipping). The flatline fraction counts consecutive samples whose difference is below 10^-6 mV, so it also counts repeats caused by amplitude quantization: BUT QDB windows are resampled from 1,000 Hz and take continuous values (mean 6.7 × 10^-5), whereas CinC 2011 lead II is used at its native 500 Hz with 5 µV steps, and acceptable CinC records already have a median flatline fraction of 0.28 (unacceptable: 0.35).

### 3.2 Supervised Model Architectures
We implemented and evaluated six supervised architectures:
* **Logistic Regression (SQI):** Linear baseline trained on standardized 13-dimensional feature vectors with L2 regularization (14 parameters: 13 weights, 1 intercept).
* **LightGBM (SQI):** Gradient-boosted decision tree ensemble trained with 100 estimators, max depth 4, learning rate 0.05.
* **Independent SQI MLP:** Multi-layer perceptron comprising two hidden layers (64 and 32 units, each followed by batch normalization [BatchNorm1d] and ReLU activations, with dropout 0.2; 3,201 trainable parameters: 3,009 dense weights and biases + 192 affine batch normalization parameters), trained via AdamW (lr=1e-3, weight_decay=1e-3) for 35 epochs of mini-batch gradient descent (batch size 64).
* **1D CNN (Raw Signal):** Unimodal convolutional network operating directly on raw 500 Hz 1D ECG biosignal waveforms (1 x 5,000 samples; 167,489 trainable parameters). Features four 1D convolutional blocks (filters: 32, 64, 128, 128; kernel sizes: 15, 11, 7, 5; each block applies batch normalization, LeakyReLU, max-pooling by 4 and dropout 0.15), global average pooling, and a two-layer classification head (128 → 32 → 1).
* **2D CNN (Rendered Image):** Unimodal visual network operating on rendered RGB waveform strips (513 x 257 px at 100 dpi, resized to 3 x 128 x 256 before the network; 245,697 trainable parameters). The backbone has four convolutional blocks (channels: 32, 64, 128, 128; 3 x 3 kernels; batch normalization and ReLU; max-pooling in the first three blocks), followed by global average pooling and a two-layer classifier (128 → 32 → 1, dropout 0.3).
* **SinglePass Multimodal Net:** Multimodal network combining the visual convolutional backbone (processing the identical 3-channel RGB image tensor) with an auxiliary MLP branch processing the 13 input features (131,969 trainable parameters). Visual embeddings (64-d) and tabular embeddings (64-d) are combined via gated fusion and mapped through a decision head (64 units, dropout 0.3).

All neural networks were trained with AdamW (learning rate 10^-3, weight decay 10^-3) on binary cross-entropy with the positive class weighted by the training ratio of usable to unusable windows. The CNNs and the multimodal net used batch size 32, 25 epochs and cosine learning-rate annealing; there was no early stopping and no validation split, so no model selection used test data. Logistic regression and LightGBM used balanced class weights; LightGBM sampled 80% of the features per tree (row subsampling was configured but inactive, because no bagging frequency was set).

The image models and the rule heuristic see each window as a rendered strip: the full 10 s on a 5.12 × 2.56 inch canvas at 100 dpi with a light clinical-style grid. The vertical axis is scaled per strip (1st–99th percentile plus a margin, at least 1 mV), so strips do not have a fixed 10 mm/mV gain. All signals are converted to millivolts before rendering. An earlier version of the renderer inferred the unit from the amplitude and divided any signal exceeding 20 by 1,000; this drew 71 CinC 2011 records (all unacceptable) and 22 NSTDB windows at −6 dB as flat lines. The renderer was corrected and every result that uses those strips was recomputed (Limitation 10).

For external evaluation on CinC 2011, CinC 2017 and NSTDB, supervised models were trained on the complete 1,047-window BUT QDB development cohort using the identical hyperparameters and random seed (42).

### 3.3 Expert Heuristic Baseline: Rule-Based SQI Heuristic
As a deterministic clinical benchmark, we evaluated an established expert rule-based heuristic grounded in multi-SQI fusion protocols [18, 20] (0 trainable parameters) that computes a continuous unusability score from a fixed logistic combination of SQIs as implemented in models/rule_heuristic_baseline.py: logit_SQI = −3.0·(bSQI − 0.5) + 1.2·min(basSQI, 3) + 1.5·min(hfSQI, 3) + 5.0·saturation + 5.0·flatline − 0.5, and p_SQI = σ(logit_SQI). The visual component converts each rendered strip to 8-bit grayscale (mode 'L') and computes the ink density (fraction of pixels < 200) and the column-wise spread of ink: logit_vis = 4.0·(ink − 0.08) − 2.0·(spread − 0.02), p_vis = σ(logit_vis). The final score is p = 0.6·p_SQI + 0.4·p_vis, clipped to [0.01, 0.99], with τ = 0.50. The coefficients were set by hand (they are not fitted). An earlier draft described hard thresholds (bSQI < 0.60, basSQI > 0.40, hfSQI > 0.35, flatline > 1.5 s); that description did not match the implemented code and has been removed. Because this rule is evaluated on the same BUT QDB cohort, its coefficients should be read as expert priors; whether they were chosen after inspecting BUT QDB outcomes is not documented.

### 3.4 Multimodal Decision Model: Liquid AI d1-3B [5]
To investigate the zero-shot capabilities of frontier multimodal models, we evaluated `LiquidAI/d1-3B` [5], an open-weight 3.12-billion-parameter multimodal edge decision model (model revision commit: `051bcc464b01b9f92942b364d9586b0ef5912432`). The model was evaluated strictly zero-shot (meaning no fine-tuning or training on ECG signals was conducted) via the `system_one` inference interface shipped with that revision on Apple Silicon MPS (device: mps, dtype: torch.bfloat16). The vision encoder processes the rendered ECG strip alongside a structured clinical task prompt:

Task Question: "Determine whether this electrocardiogram (ECG) recording segment is usable for reliable QRS-based rhythm analysis, or unusable due to severe artifacts, baseline wander, high-frequency noise, or saturation/flatline."
Criteria: usable: "QRS complexes are distinguishable and suitable for rhythm analysis. Artifacts are manageable."; unusable: "ECG is severely corrupted by noise, baseline drift, saturation, or absent complexes; requires human review or re-recording."

The model is configured with `output_tokens: 0`, returning choice probabilities over categorical options from `raw_response['answers']['usability']['probabilities']['unusable']`. We explicitly note that these choice probabilities represent uncalibrated model outputs. The SQI values were also written into the d1-3B prompt as text (a fixed summary of the SQI values for each window). The text contains 10 of the 13 input features (bSQI, baseline wander, high-frequency noise, saturation, flatline, QRS energy, kurtosis, skewness, window duration and sampling rate); zero-crossing rate, powerline ratio and signal standard deviation were not included. The two multimodal models therefore received overlapping, not identical, information, and in different forms (numerical features for the Multimodal Net, prompt text for d1-3B). Two alternative configurations (image only, and a reworded question) were evaluated as sensitivity analyses (Section 4.6).

### 3.5 Evaluation Metrics, Threshold Selection, and Reproducibility Protocols
Discrimination was quantified using Area Under the Receiver Operating Characteristic (AUROC) and Average Precision (AP, non-interpolated) via scikit-learn [8]. We strictly differentiate two decision threshold regimes: (1) default fixed threshold τ = 0.50; and (2) exploratory post-hoc ROC sweeps (Target 1: maximum sensitivity at FPR ≤ 5% on BUT QDB and ≤ 4% on CinC; Target 2: minimum FPR at sensitivity ≥ 80%; ties resolved deterministically, Target 1 by maximum sensitivity then minimum FPR, Target 2 by minimum FPR then maximum sensitivity) [8]. Tables 1b and 2b present these exploratory analyses on the evaluated partitions and do not represent prospectively validated deployment thresholds. To measure what choosing thresholds on the reported data costs, the same rules were also applied with thresholds chosen without it: (a) nested selection on BUT QDB, where within each outer fold the 12 training subjects were split into four inner subject folds, every trainable model was retrained on 9 subjects and scored on the other 3, and the thresholds chosen on these inner out-of-fold scores were applied to the 3 held-out subjects (Table 1c); and (b) thresholds chosen on the pooled BUT QDB out-of-fold scores and applied unchanged to CinC 2011 and CinC 2017 (Table 2f). For external transfer experiments, feature scalers (including RobustScaler) were fit strictly on the BUT QDB source development cohort (N=1,047) and applied frozen to CinC 2011 to evaluate transfer without target data leakage. Calibration was assessed via the Brier score. Subject-level cluster bootstrapping (1,000 iterations, seed 42) was used to construct 95% confidence intervals on BUT QDB. Hardware latency was profiled on an Apple M4 Mac mini (10-core CPU with 4 performance and 6 efficiency cores, 10-core GPU, 16 GB unified memory) with explicit `torch.mps.synchronize()` barriers in three sequential runs of 25 paired test trials each.

---

## 4. Experimental Results

In this section, we present the empirical findings of our multi-paradigm benchmark structured along four evaluation axes: (1) patient-independent cross-validation on BUT QDB, including exploratory ROC operating points; (2) cross-dataset out-of-distribution transfer to CinC 2011 Set A, including a feature-set sensitivity analysis, all 12 leads and thresholds chosen on BUT QDB; (3) noise stress testing under calibrated electrode motion artifact on MIT-BIH NSTDB; and (4) hardware-synchronized edge latency, execution device, and weight storage profiling, quantifying the waveform rendering bottleneck; followed by the seed stability analysis, the d1-3B prompt and probe analyses, and (Section 4.7) external validation on CinC 2017 at natural prevalence. Nested thresholds, natural-prevalence re-weighting and subject-level results are in Section 4.1. All numbers were regenerated on 2026-10-12, and the CinC, NSTDB and new analyses on 2026-10-13 after the rendering fix, from seeded training runs (seed 42); every table is generated by `evaluation/build_manuscript_tables.py`.

### 4.1 Patient-Independent 5-Fold Cross-Validation on BUT QDB

Table 1 and Figure 1 summarize patient-independent 5-fold cross-validation performance across the eight models on BUT QDB (1,047 windows, 15 subjects), reported for seed 42. Seed-to-seed variation of the CNN and multimodal estimates is in Section 4.5. The two unimodal convolutional networks had the highest discrimination: the 1D CNN on raw 500 Hz waveforms (AUROC 0.989, 95% CI 0.964–0.999; AP 0.975; Brier 0.058) and the 2D CNN on rendered strips (AUROC 0.989, 95% CI 0.967–0.999; AP 0.971; Brier 0.052). The expert heuristic reached AUROC 0.977 (AP 0.948) with poorer calibration (Brier 0.105). The SinglePass Multimodal Net did not improve on the unimodal networks (AUROC 0.967, AP 0.928, Brier 0.094). The tabular models reached AUROC 0.938–0.964. The zero-shot LiquidAI/d1-3B model reached AUROC 0.852 (95% CI 0.717–0.944; AP 0.740; Brier 0.154), reflecting uncalibrated choice probabilities and the absence of domain-specific adaptation.

#### Table 1: Patient-Independent 5-Fold Cross-Validation Performance on BUT QDB (1,047 Windows, 15 Subjects)
*Positive Class = Unusable (297 windows), Negative Class = Usable (750 windows). 95% CIs derived via 1,000 subject-level cluster bootstrap iterations (seed 42).*

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

At the default threshold (τ = 0.50), the 1D CNN identified 76.8% of unusable windows with a 0.7% false reject rate (balanced accuracy 88.1%), and the 2D CNN identified 78.8% with a 0.9% false reject rate (balanced accuracy 88.9%). Table 1b reports exploratory post-hoc operating points under two constraints: Target 1 (maximum sensitivity with FPR ≤ 5%) and Target 2 (minimum FPR with sensitivity ≥ 80%). Under Target 1 the 1D CNN reached 95.96% sensitivity at 3.47% FPR, and the 2D CNN 94.28% at 4.67% FPR. Under Target 2 the zero-shot d1-3B model required a 32.67% false reject rate to reach 80.13% sensitivity. These sweeps are computed on the same partition they are reported on and are not validated deployment thresholds.

#### Table 1b: Exploratory ROC Operating-Point Analysis on BUT QDB
*Decision Rule: Positive class (Unusable) if p ≥ τ. Target 1: maximum sensitivity with FPR ≤ 5% (ties: minimum FPR). Target 2: minimum FPR with sensitivity ≥ 80% (ties: maximum sensitivity). Post-hoc exploratory sweeps on the reported partition, not validated deployment thresholds.*

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

Table 1c repeats the operating-point analysis with thresholds chosen without the evaluated subjects. For the CNNs the nested thresholds reproduced the post-hoc operating points almost exactly (1D CNN, Target 1: 97.3% sensitivity at 3.5% FPR against 96.0% at 3.5% post hoc; 2D CNN: 94.6% at 4.7%). For the SQI models the post-hoc points were optimistic: LightGBM's Target 1 sensitivity fell from 80.5% to 70.4%, the logistic regression's Target 1 threshold exceeded the FPR constraint on the held-out subjects (6.7%), and under Target 2 these models reached 80% sensitivity only at a higher FPR (LightGBM 92.9% at 9.5%, logistic regression 87.5% at 13.1%). The 2D CNN, the multimodal net, the rule heuristic and d1-3B fell short of 80% sensitivity under nested Target 2 (69.0–78.1%).

#### Table 1c: Nested Threshold Selection on BUT QDB
*Nested: within each outer fold, thresholds were chosen on the inner out-of-fold scores of the 12 training subjects (four inner subject folds; models retrained on 9 subjects) and applied to the 3 held-out subjects; values are pooled over the five outer folds (seed 42). Post-hoc: the same rule applied to the pooled test scores (Table 1b). The rule heuristic and d1-3B have no training; their thresholds were chosen on the training subjects' scores.*

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

Figure 1 displays the complete cross-validation ROC and PR curves with non-interpolated Average Precision baselines.

![Figure 1: BUT QDB Cross-Validation Curves](../figures/figure_1_butqdb_combined_roc_pr.png)
*Figure 1: Patient-independent cross-validation performance on BUT QDB (1,047 windows, 15 subjects). (A) Receiver Operating Characteristic (ROC) curves with chance baseline (AUROC = 0.500). (B) Precision-Recall curves with prevalence baseline (AP = 0.284). Average Precision (AP) strictly computed via non-interpolated scikit-learn metrics [8].*

#### Table 1d: Sensitivity Analysis with Class 2 Relabelled as Unusable (BUT QDB)
*Retrospective sensitivity analysis. Five-fold subject-independent CV with identical folds, settings and seed (42); only the training and evaluation labels change. Rule heuristic and d1-3B do not use training labels and keep their scores. Sens. / FR use τ = 0.50. 95% CIs from 1,000 subject-level bootstrap iterations.*

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

Relabelling the 193 class-2 windows as unusable (490 unusable windows, 46.8%) leaves the CNN ranking unchanged. The 2D CNN (AUROC 0.984 for seed 42; 0.984–0.986 across seeds 1–4) and the 1D CNN (0.961; 0.959–0.964) remain the strongest networks, and logistic regression and LightGBM keep AUROC above 0.95. The multimodal net (0.865 for seed 42) and the SQI MLP (0.837) depend strongly on the training seed under this label definition: across seeds 1–4 the multimodal net ranges from 0.867 to 0.984 and the MLP from 0.825 to 0.942. Their decrease for seed 42 is therefore not a stable property of the models. The rule heuristic (0.973) and d1-3B (0.785) do not use training labels. Because class 2 denotes noise that still permits QRS detection (Section 1.1), this analysis shows which conclusions depend on the class boundary.

The quota cohort over-represents Class 3 (28.4%). Table 1e re-weights the pooled out-of-fold predictions to the natural prevalence of the candidate pool: 16.0% over all records, and 2.8% without record 105001, which alone contributes 90% of the Class 3 windows (Section 2.1). AUROC, sensitivity and specificity do not depend on prevalence, but precision does. At 2.8%, the CNNs keep an AP of 0.81–0.83 and a positive predictive value of 0.71–0.77 at τ = 0.50, with 2.4–3.4 false rejections per hour of usable signal. The SQI models fall to an AP of 0.55–0.73 (PPV 0.21–0.34; 15–31 false rejections per hour) and zero-shot d1-3B to 0.35 (PPV 0.28).

#### Table 1e: BUT QDB Metrics Re-weighted to the Natural Prevalence of Class 3
*Pooled out-of-fold predictions (seed 42) re-weighted to the Class 3 prevalence of the candidate pool of Section 2.1 (sample weights on the usable windows). AUROC, sensitivity and specificity are unchanged by re-weighting (Table 1). PPV and NPV use the pooled sensitivity and specificity at τ = 0.50. False rejections per hour = false reject rate × 360 ten-second windows of usable signal.*

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

Table 1f shows that the pooled sensitivities hide large differences between subjects. Of the seven subjects with Class 3 windows, the 1D CNN detected 98–100% of them in four subjects but 14% in subject 100 and 32% in subject 111; the 2D CNN detected 27% in subject 111 and 73% in subject 122, and none of the models in Table 1f detected more than 53% of subject 111's unusable windows. False reject rates of the CNNs stayed between 0% and 6% in every subject. With seven affected subjects, the sensitivity estimates rest on few patients, as the wide subject-level confidence intervals of Table 1 already indicate.

#### Table 1f: Subject-Level Sensitivity on BUT QDB
*Sensitivity at τ = 0.50 for each held-out subject with Class 3 windows (pooled out-of-fold predictions, seed 42), and the range of false reject rates over all 15 subjects. Logistic regression and the MLP are omitted for space (results/butqdb_subject_heterogeneity.csv).*

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

### 4.2 External Out-of-Distribution Transfer to CinC 2011 Set A

Models trained on the complete 1,047-window BUT QDB cohort were transferred zero-shot to CinC 2011 Set A (150 balanced records, Lead II). As detailed in Section 2.2, this evaluation involves a clinical target mismatch: models evaluated on single-lead Lead II waveforms are benchmarked against labels reflecting overall 12-lead diagnostic acceptability. Table 2 presents the reported subset with the seed-42 models. By AUROC, the 2D CNN (0.746), LightGBM (0.731) and 1D CNN (0.726) ranked recordings best, followed by the rule heuristic (0.700), the zero-shot d1-3B (0.636), the SinglePass Multimodal Net (0.627) and the Independent SQI MLP (0.617). Table 2d repeats the transfer on five random balanced subsets. The 2D CNN had the highest AUROC in every subset (0.713–0.884), whereas the 1D CNN ranged from 0.660 to 0.816 and LightGBM from 0.700 to 0.821, so the ordering on the reported subset is not robust. Ranking did not translate into detection at the default threshold. The 1D CNN identified 61.3% of unacceptable recordings on the reported subset (58.7–76.0% across random subsets) at 14.7% false rejection. The 2D CNN identified 17.3% on the reported subset and 17.3–28.0% across random subsets. Its default-threshold sensitivity also depended on the training seed (17.3% for seed 42; 20.0–42.7% for seeds 1–4 on the reported subset; Section 4.5). No model reached 80% sensitivity while keeping false rejection at or below 4% on the reported subset (Table 2b).

#### Table 2: Cross-Dataset Transfer Benchmark on CinC 2011 Set A (150 Records)
*Positive Class = Unacceptable (75 records), Negative Class = Acceptable (75 records). Seed 42 models on the reported subset; random subsets are in Table 2d and training seeds 1–4 in Section 4.5. Models evaluated strictly on Lead II signals zero-shot without target-domain fine-tuning. Reference labels reflect 12-lead diagnostic acceptability. At τ = 0.50 the MLP output was saturated near zero (no recording above threshold) and logistic regression was saturated near one for most recordings.*

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

The tabular results do not have a single cause. The logistic regression (AUROC 0.383; 0.291–0.372 across random subsets) ranks acceptable recordings above unacceptable ones on this target. Two mechanisms are visible in the data. First, saturation and flatline fractions are almost constant in BUT QDB (σ ≈ 9 × 10^-6 and 1.3 × 10^-4) but large in CinC (mean flatline fraction 0.37), so standard scaling converts CinC values into very large z-scores. Much of the CinC flatline fraction is quantization rather than lead disconnection: CinC lead II keeps its 5 µV steps at 500 Hz while BUT QDB windows are resampled to continuous values, and acceptable CinC records already have a median flatline fraction of 0.28 (Section 3.1). Second, several SQIs carry opposite associations in the two databases: the univariate AUROC of the high-frequency noise ratio is 0.777 in BUT QDB but 0.329 in CinC (Limitation 9). Table 2c shows that removing the saturation, flatline and constant-metadata features restores logistic regression transfer to AUROC 0.746, whereas removing only the amplitude feature does not. Because this feature subset was chosen after inspecting CinC feature behaviour, the reduced-set results are exploratory and are not evidence of validated transfer.

#### Table 2b: Exploratory Post-Hoc ROC Operating-Point Analysis on CinC 2011 Set A
*Decision Rule: Positive class (Unusable) if p ≥ τ. Target 1: maximum sensitivity with FPR ≤ 4% (3 of 75 acceptable records; ties: minimum FPR). Target 2: minimum FPR with sensitivity ≥ 80% (ties: maximum sensitivity). "Not attainable" marks models whose scores saturate so that no threshold meets the constraint; "trivial" marks thresholds at which every recording is flagged. The multimodal net and the MLP output saturated scores (binary or near zero; thresholds below 10^-3 are not graded cut-offs), so their rows describe a single decision rule rather than a threshold curve. Exploratory sweeps on the target domain, not validated deployment thresholds.*

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

Table 2f applies thresholds chosen on BUT QDB, without any CinC data, to the external sets. The operating points did not carry over. With its BUT QDB Target 1 threshold the 2D CNN flagged 52.0% of unacceptable CinC 2011 recordings at 10.7% false rejection, and 96.4% of noisy CinC 2017 recordings at 50.4% false rejection; the 1D CNN reached 70.7% at 28.0% on CinC 2011 and 99.6% at 54.5% on CinC 2017. Among the models run on the whole CinC 2017 cohort, only the rule heuristic kept false rejection near 5% (6.2% and 5.2%, at sensitivities of 52.9% and 43.5%). Operating points therefore have to be re-established on each target population.

#### Table 2f: Thresholds Chosen on BUT QDB Applied Unchanged to the External Datasets
*Thresholds chosen on the pooled BUT QDB out-of-fold scores (seed 42) by the rules of Table 1b. CinC 2011: lead II; the random-subset column gives the range of sensitivity over the five subsets. CinC 2017: all 8,512 recordings, recording maximum; † d1-3B on the balanced subset of Table 8b. The BUT QDB scores come from the fold models, the external scores from the model trained on the full cohort.*

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
| **LiquidAI d1-3B (Zero-Shot)** | Target 1 (FPR ≤ 5%) | 0.5 | 53.2% / 4.0% | 100.0% / 100.0% | 98.7–100.0% | 19.8% / 4.0% † |
| **LiquidAI d1-3B (Zero-Shot)** | Target 2 (Sens. ≥ 80%) | 0.3775 | 80.1% / 32.7% | 100.0% / 100.0% | 100.0–100.0% | 40.6% / 14.4% † |

#### Table 2c: Feature-Set and Scaler Sensitivity of Tabular Models under Transfer
*Retrospective sensitivity analysis (BUT QDB → CinC 2011). Scalers and models are fit on BUT QDB only and applied frozen to CinC 2011. Feature subsets were selected after inspecting CinC feature behaviour and are therefore exploratory. The MLP row uses its internal standardization and cannot take an external scaler.*

| Feature set | LR · StandardScaler AUROC / AP | LR · RobustScaler AUROC / AP | LR · Z-clip [-3, 3] AUROC / AP | MLP (internal scaler) AUROC / AP |
| :--- | :---: | :---: | :---: | :---: |
| All 13 features | 0.383 / 0.451 | 0.377 / 0.442 | 0.749 / 0.791 | 0.617 / 0.627 |
| Drop signal_std | 0.431 / 0.529 | 0.370 / 0.441 | 0.723 / 0.775 | 0.627 / 0.627 |
| Drop saturation, flatline, constants | 0.746 / 0.796 | 0.746 / 0.805 | 0.751 / 0.792 | 0.750 / 0.749 |
| Drop signal_std, saturation, flatline, constants | 0.713 / 0.770 | 0.721 / 0.781 | 0.723 / 0.774 | 0.729 / 0.762 |

#### Table 2d: Transfer to Five Random Balanced CinC Subsets
*Models trained once on the BUT QDB cohort (seed 42), as in Table 2. The reported subset is the first 75 + 75 records of the sorted PhysioNet lists; random subsets are 75 + 75 records drawn without replacement (seeds 42, 1, 2, 3, 4) from 773 acceptable and 225 unacceptable records. Sensitivity and false reject use τ = 0.50.*

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

Table 2e scores all 12 leads of each CinC record. Averaging the 12 lead scores raised AUROC for every trained model except logistic regression: on the random subsets the 2D CNN rose from a mean of 0.781 with lead II alone to 0.852, LightGBM from 0.750 to 0.877, the 1D CNN from 0.707 to 0.801 and the multimodal net from 0.668 to 0.841. Taking the maximum over leads helped the rule heuristic (0.683 to 0.860), the multimodal net, the MLP and, on the reported subset, d1-3B (0.636 to 0.800), but not the CNNs: on the reported subset the 1D CNN flagged at least one lead of every record at τ = 0.50, and its maximum score ranked records almost in reverse (0.214 on the random subsets). Part of the transfer gap is therefore the single-lead versus 12-lead target mismatch. These aggregation rules were chosen after the main analysis and are a sensitivity analysis, not a validated deployment rule.

#### Table 2e: CinC 2011 Scored on All 12 Leads
*Models trained on BUT QDB (seed 42) score each of the 12 leads separately; the record score is the lead II score, the maximum, or the mean over the 12 leads. Random subsets: the five balanced subsets of Table 2d. d1-3B was run on the 12 leads of the reported subset only. Values are AUROC.*

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

Table 2g lists the univariate AUROC of every SQI in each database. The high-frequency noise, powerline and zero-crossing ratios and the flatline fraction change direction between BUT QDB and CinC 2011, and bSQI keeps its direction but loses most of its strength (0.030 in BUT QDB, 0.305 on the reported CinC subset; 0.5 is chance, and values below 0.5 mean lower values in unusable recordings). In CinC 2017, bSQI is again strongly associated with the label (0.124), which is consistent with the better transfer of the SQI models there (Section 4.7).

#### Table 2g: Univariate AUROC of Each SQI Across Datasets
*AUROC of the raw SQI value used as a score for the unusable class (0.5 = no association; below 0.5 = lower in unusable windows). Duration and sampling rate are constant and omitted. CinC 2017: first window of each of the 8,512 recordings.*

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

![Figure 2: CinC 2011 Transfer Curves](../figures/figure_2_cinc2011_combined_roc_pr.png)
*Figure 2: Out-of-distribution transfer performance on PhysioNet/CinC 2011 Set A (150 balanced records, Lead II). (A) Receiver Operating Characteristic curves with chance baseline (AUROC = 0.500). (B) Precision-Recall curves with prevalence baseline (AP = 0.500).*

### 4.3 Controlled Noise Stress Testing on MIT-BIH NSTDB

Controlled noise stress testing was conducted on NSTDB with the paired design of Section 2.3: 264 noisy windows (44 per nominal SNR label), each compared with the clean base-record window at the same start time. Table 3 gives the mean predicted usable probability, P(usable) = 1 − P(unusable), and Table 3b the paired change for each window relative to its partner; Figure 3 plots the same means. At +24 dB the usable probability is already lower than on the clean reference for logistic regression (0.967 to 0.900), LightGBM (0.975 to 0.867), the 2D CNN (0.685 to 0.575) and the multimodal net (0.857 to 0.764), whereas the 1D CNN (0.788 to 0.787) and d1-3B (0.777 to 0.759) are essentially unchanged. From +24 dB to +6 dB the usable probability falls for every learned model. At +6 dB it is below 0.25 for logistic regression (0.176), LightGBM (0.074), the 1D CNN (0.231), the 2D CNN (0.019) and the multimodal net (0.017), where the paired change is −0.56 to −0.90. At 0 dB and −6 dB all five learned models are at or below 0.11, so these levels do not separate severe from moderate noise. The rule heuristic declines more gradually (0.740 to 0.550 at 0 dB, 0.555 at −6 dB). The zero-shot d1-3B changes by at most 0.09 at any level (paired change −0.018 at +24 dB and −0.092 at 0 dB; 0.699 at −6 dB), so it does not track the noise. Between 0 dB and −6 dB the image networks stay near zero (2D CNN 0.017 to 0.012, multimodal net 0.006 to 0.006), while LightGBM (0.072 to 0.081), the rule heuristic (0.550 to 0.555) and d1-3B (0.685 to 0.699) rise slightly. An earlier version of this analysis showed clear rises at −6 dB for the image models; they came from the rendering error described in Section 3.2, which drew half of the −6 dB windows as flat lines. The heuristic column was computed by the current implementation; an earlier draft reported 0.998 at clean control, which could not be reproduced.

#### Table 3: Continuous Usability Probability Degradation Across Calibrated SNR Levels
*Mean ± SD of predicted usable probability, P(usable) = 1 − P(unusable). Clean reference: 44 windows from the MIT-BIH base records 118 and 119 (22 per record; no SNR label). Noisy: 44 windows per SNR label (22 per record; 264 in total), each paired with the clean window at the same start time (Table 3b). Models trained on the full BUT QDB cohort (seed 42).*

| SNR Level | Logistic Regression (SQI) | LightGBM (SQI) | 1D CNN (Raw Signal) | 2D CNN (Image) | Multimodal Net | Rule Heuristic Baseline | d1-3B (Zero-Shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean reference (MIT-BIH base)** | 0.967 ± 0.028 | 0.975 ± 0.082 | 0.788 ± 0.326 | 0.685 ± 0.358 | 0.857 ± 0.271 | 0.740 ± 0.011 | 0.777 ± 0.104 |
| **+24 dB (Low Noise)** | 0.900 ± 0.136 | 0.867 ± 0.217 | 0.787 ± 0.314 | 0.575 ± 0.390 | 0.764 ± 0.310 | 0.717 ± 0.026 | 0.759 ± 0.100 |
| **+18 dB** | 0.812 ± 0.159 | 0.538 ± 0.334 | 0.742 ± 0.311 | 0.367 ± 0.390 | 0.391 ± 0.409 | 0.695 ± 0.025 | 0.740 ± 0.074 |
| **+12 dB** | 0.501 ± 0.225 | 0.163 ± 0.169 | 0.521 ± 0.349 | 0.083 ± 0.181 | 0.082 ± 0.187 | 0.653 ± 0.038 | 0.704 ± 0.052 |
| **+6 dB** | 0.176 ± 0.143 | 0.074 ± 0.051 | 0.231 ± 0.307 | 0.019 ± 0.078 | 0.017 ± 0.057 | 0.606 ± 0.044 | 0.685 ± 0.049 |
| **0 dB (Equal Power)** | 0.038 ± 0.046 | 0.072 ± 0.036 | 0.105 ± 0.188 | 0.017 ± 0.064 | 0.006 ± 0.018 | 0.550 ± 0.063 | 0.685 ± 0.051 |
| **-6 dB (Severe Noise)** | 0.011 ± 0.013 | 0.081 ± 0.060 | 0.075 ± 0.139 | 0.012 ± 0.041 | 0.006 ± 0.020 | 0.555 ± 0.070 | 0.699 ± 0.059 |

#### Table 3b: Paired Change in Usable Probability Relative to the Clean Partner Window
*Each noisy window minus its clean base-record partner at the same start time; mean ± SD over the 44 pairs per SNR label. Negative values mean the usable probability fell. Computed from the seed-42 models on the same windows as Table 3.*

| SNR Level | Logistic Regression (SQI) | LightGBM (SQI) | 1D CNN (Raw Signal) | 2D CNN (Image) | Multimodal Net | d1-3B (zero-shot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+24 dB (Low Noise)** | -0.067 ± 0.117 | -0.107 ± 0.240 | -0.002 ± 0.073 | -0.110 ± 0.252 | -0.094 ± 0.183 | -0.018 ± 0.058 |
| **+18 dB** | -0.155 ± 0.141 | -0.436 ± 0.344 | -0.046 ± 0.159 | -0.318 ± 0.336 | -0.466 ± 0.402 | -0.037 ± 0.059 |
| **+12 dB** | -0.467 ± 0.214 | -0.811 ± 0.187 | -0.267 ± 0.364 | -0.602 ± 0.354 | -0.776 ± 0.307 | -0.072 ± 0.092 |
| **+6 dB** | -0.791 ± 0.139 | -0.901 ± 0.093 | -0.557 ± 0.417 | -0.666 ± 0.366 | -0.840 ± 0.276 | -0.092 ± 0.103 |
| **0 dB (Equal Power)** | -0.930 ± 0.055 | -0.902 ± 0.087 | -0.683 ± 0.373 | -0.668 ± 0.365 | -0.852 ± 0.271 | -0.092 ± 0.119 |
| **-6 dB (Severe Noise)** | -0.956 ± 0.032 | -0.893 ± 0.098 | -0.713 ± 0.356 | -0.673 ± 0.363 | -0.851 ± 0.273 | -0.078 ± 0.113 |

![Figure 3: NSTDB Degradation](../figures/figure_revised_nstdb_degradation_curve.png)
*Figure 3: Predicted usability probability on MIT-BIH Noise Stress Test Database (NSTDB Records 118 & 119). (A) Clean reference windows from the MIT-BIH base records (N = 44, mean ± 1 SD). (B) Degradation across noisy SNR labels (N = 44 windows per label, ±1 SD shaded). The zero-shot d1-3B (grey) changes by at most 0.09 (paired) across the noisy levels.*

### 4.4 Edge Computational Profiling and the Rendering Bottleneck

Edge deployment was profiled on an Apple M4 Mac mini (10-core CPU with 4 performance and 6 efficiency cores, 10-core GPU, 16 GB unified memory) with explicit `torch.mps.synchronize()` barriers in three sequential runs of 25 paired test trials each. Parameter counts were computed from the PyTorch model definitions: 1D CNN 167,489 (0.64 MiB disk), 2D CNN 245,697 (0.94 MiB), SinglePass Multimodal Net 131,969 (0.47 MiB), Independent SQI MLP 3,201 (0.01 MiB), logistic regression 14, LightGBM 1,092 split nodes across 100 trees of the cohort-fitted model, and the expert rule heuristic 0. The zero-shot LiquidAI/d1-3B model has 3.12 billion parameters (5,958 MiB on disk).

#### Table 4: Synchronized Hardware Latency, Execution Device, and Weight Storage on Apple M4
*Each cell is the median over three sequential runs of 25 paired trials, with the median over runs of the per-run p95 in parentheses; the last column gives the range of the three per-run total medians. Latencies were measured with `torch.mps.synchronize()` where the model ran on MPS. The execution device is the one the profiling run used: classifiers constructed without an explicit device default to MPS when available, so the MLP, 1D CNN, 2D CNN and multimodal net ran on MPS, while logistic regression, LightGBM and the rule heuristic ran on CPU (scikit-learn/NumPy). Preprocessing share is the median of per-run ratios (preprocessing / total). The 1D CNN preprocessing time is zero because its inputs are pre-resampled cached signals.*

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

Profiling shows that for the image-based models, waveform rasterization is the bottleneck. For the 2D CNN, Matplotlib rendering took a median 60.23 ms (p95 137.53 ms), 92.4% of the 65.24 ms median total (range 64.74–66.39 ms over three runs), while the forward pass took 5.12 ms. For the multimodal net, preprocessing took 61.41 ms (92.4% of 67.09 ms total) and the forward pass 5.56 ms. The d1-3B model is dominated by neural inference (828.47 ms forward pass of 891.19 ms total; preprocessing share 7.1%). The unimodal 1D CNN ran in 1.47 ms median (p95 2.28 ms) on cached 500 Hz inputs with no rendering step. The MLP's forward pass on MPS (1.13 ms) is dominated by device dispatch overhead for a model this small, so its preprocessing share (54.5%) should not be compared with the other models.

Table 4b adds the steps that Table 4 leaves out. Starting from the native 1,000 Hz window held in memory, unit conversion and resampling to 500 Hz add about 0.1 ms, so the 1D CNN's raw-to-decision time was 1.41 ms on the CPU and 2.90 ms on MPS; acquisition and buffering on a device were not measured. The 1D CNN forward pass was faster on the CPU (1.30 ms) than on MPS (2.80 ms in this harness, against 1.47 ms in Table 4, which indicates the precision of sub-3 ms MPS timings), whereas the 2D CNN and the multimodal net were faster on MPS (2.92 and 3.17 ms) than on the CPU (5.11 and 4.39 ms). The rendering cost is an implementation choice: a direct raster renderer drew a strip of the same size in 0.63 ms, against 58.4 ms for Matplotlib. The image models were trained on Matplotlib strips and were not evaluated on raster strips, so the accuracy of a faster image pipeline is untested. SQI extraction took 1.72 ms.

#### Table 4b: Extended Latency Profile
*Same 25 BUT QDB windows as Table 4, three runs (Apple M4), first window of each run excluded as warm-up. Raw-to-decision = unit conversion + resampling of the in-memory 1,000 Hz window + forward pass (acquisition excluded). Forward passes use untrained copies of the same architectures (cost does not depend on the weights). MPS timings are synchronised.*

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

![Figure 4: Latency vs AUROC Frontier](../figures/latency_vs_auroc.png)
*Figure 4: Diagnostic discrimination (AUROC) versus measured end-to-end inference latency on edge hardware (log scale). Marker positions reflect measured median execution times on Apple M4 hardware (median over three runs).*

### 4.5 Seed Stability of the Neural Models

Because the CNN and multimodal models are trained with stochastic optimization, we repeated the BUT QDB cross-validation and the CinC transfer with seeds 1–4 (Table 5). The in-domain ranking was stable: pooled out-of-fold AUROC changed by at most 0.02 from seed 42 for each model (1D CNN 0.971–0.979 for seeds 1–4 versus 0.989 for seed 42; 2D CNN 0.983–0.993 versus 0.989; multimodal net 0.967–0.975 versus 0.967). Transfer rankings were stable for the CNNs (1D CNN AUROC 0.710–0.719; 2D CNN 0.748–0.767), but default-threshold sensitivity on CinC was not. On the reported subset the 2D CNN flagged 20.0–42.7% of unacceptable recordings across seeds 1–4 (41–43% for three of them), against 17.3% for seed 42, so its default operating point depends on the seed and the seed-42 value should not be taken as representative. The multimodal net gave constant outputs on CinC for seeds 1, 2 and 3 (AUROC 0.500, every recording flagged). For seed 42 its outputs take only two distinct values, so its AUROC of 0.627 reflects a single operating point, and seed 4 reproduces the same AUROC, sensitivity and false rejection. Table 6 separates the effect of the training seed from that of the sampled CinC subset across five random balanced subsets (Table 2d). For the CNNs the subset explains most of the variation: the 1D and 2D CNN AUROC changes by 0.007–0.009 across training seeds within a subset, but by 0.064–0.066 across subsets. The multimodal net varies mainly with seed (SD 0.091), because seeds 1–3 produce constant outputs on every subset. The MLP outputs take only a few distinct values on CinC, so near-identical AUROCs across seeds reflect saturation rather than identical models. Logistic regression and LightGBM are deterministic under their fixed random state. Seed variation was not assessed for the NSTDB stress test or the zero-shot d1-3B, which has no training seed.

#### Table 5: Seed Stability of the 1D CNN, 2D CNN and Multimodal Net
*Seed 42 is the reported run used in Tables 1–4; seeds 1–4 were run with experiments/seed_stability.py. BUT QDB values are pooled 5-fold out-of-fold metrics; CinC values use the model trained on the full cohort. Sensitivity and false rejection use τ = 0.50.*

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


#### Table 6: Training-Seed and CinC-Subset Variation of the Supervised Models
*Seeds 42 and 1–4 retrain the cohort models; each model is scored on the five random balanced CinC subsets (the reported first-N subset is excluded here, so these statistics describe the random draws only). Values are AUROC; standard deviations use n − 1. Logistic regression and LightGBM use a fixed random state, so their seed variation is zero by construction.*

| Model | AUROC range over seeds × subsets | SD across training seeds (mean over subsets) | SD across subsets (mean over seeds) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.291–0.372 | 0.000 | 0.032 |
| **LightGBM (SQI)** | 0.700–0.821 | 0.000 | 0.049 |
| **Independent SQI MLP** | 0.613–0.707 | 0.007 | 0.033 |
| **1D CNN (Raw Signal)** | 0.643–0.825 | 0.009 | 0.066 |
| **2D CNN (Rendered Image)** | 0.713–0.904 | 0.007 | 0.064 |
| **SinglePass Multimodal Net** | 0.500–0.707 | 0.091 | 0.015 |

### 4.6 Prompt Sensitivity and Linear Probes of LiquidAI d1-3B

The main d1-3B results use one configuration: the rendered strip, a text summary of the SQIs, and the usability question of Section 3.4. To test whether the conclusions depend on that choice, we evaluated two alternatives on the same windows (`experiments/d1_prompt_variants.py`): the strip alone with the same question, and a reworded question with the same target (unusable = Class 3 on BUT QDB; unacceptable on CinC) plus the SQI text. Before these runs, the primary configuration was recomputed on 10 BUT QDB windows and reproduced the stored probabilities exactly.

Table 7 shows that discrimination depends on the configuration. Removing the SQI text raised the BUT QDB AUROC from 0.852 to 0.913, but at τ = 0.50 that configuration flagged 99.7% of unusable and 81.3% of usable windows, so its default operating point is not usable. The reworded question gave AUROC 0.869 with conservative behaviour at τ = 0.50 (sensitivity 17.5%, false reject 0.5%). On CinC 2011 the three configurations stayed in a low range (AUROC 0.627–0.680), below the 2D CNN (0.746) and LightGBM (0.731) at seed 42. The NSTDB clean-reference usable probability varied from 0.459 to 0.870 across configurations, so the output scale depends on the prompt, while the noise-induced change stayed between −0.017 and −0.092 at +24 and 0 dB in every configuration. None of the configurations approached the supervised CNNs on BUT QDB (AUROC 0.989). The prompt therefore changes the operating point and the BUT QDB ranking of d1-3B, but not its lag behind the task-specific networks.

#### Table 7: Prompt Sensitivity of LiquidAI d1-3B on the BUT QDB, CinC 2011 and NSTDB Windows
*Primary: the configuration in all main results. Image only: the strip with the same usability question and no SQI text. Reworded question: a different wording of the same target, with the SQI text. The BUT QDB AUROC interval is the subject-level bootstrap used in Table 1 (1,000 iterations, seed 42); CinC is a point estimate (150 records, no subject identifiers). NSTDB paired change is the noisy window minus its clean base-record partner for P(usable).*

| Configuration | BUT QDB AUROC [95% CI] | BUT QDB Sens. / FR at τ = 0.50 | CinC AUROC | CinC Sens. / FR at τ = 0.50 | NSTDB clean P(usable) | NSTDB paired change, +24 dB | NSTDB paired change, 0 dB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary (SQI text + image)** | 0.852 [0.717 – 0.944] | 53.2% / 4.0% | 0.636 | 100.0% / 100.0% | 0.777 | -0.018 | -0.092 |
| **Image only (no SQI text)** | 0.913 [0.855 – 0.972] | 99.7% / 81.3% | 0.680 | 81.3% / 78.7% | 0.459 | -0.037 | -0.039 |
| **Reworded question (SQI text + image)** | 0.869 [0.732 – 0.955] | 17.5% / 0.5% | 0.627 | 100.0% / 98.7% | 0.870 | -0.017 | -0.073 |

To separate capacity from adaptation, we trained linear probes on frozen d1-3B representations captured during the primary forward pass (`experiments/d1_linear_probe.py`): the final-norm hidden state at the answer position, and the mean of the image tokens after the multimodal projector (2,048 dimensions each). Each probe is an L2-regularised logistic regression whose penalty is chosen by inner subject-grouped cross-validation; it is evaluated on the five subject folds of Table 1 and, for transfer, fit on all BUT QDB windows and applied frozen. Table 7b shows that the image representation already carries the information. The image-token probe reached AUROC 0.979 [0.941–0.996] on BUT QDB, close to the CNNs (0.989), and it transferred better than any supervised model to CinC 2011 (0.777 on the reported subset; 0.756–0.907 on the random subsets) and to the CinC 2017 subset (0.910, against 0.739 zero-shot and at most 0.880 for the supervised models). The decision-state probe also fitted BUT QDB (0.974) but fell below chance on CinC 2011 (0.411–0.482); that state also encodes the SQI text, whose associations reverse in CinC 2011 (Table 2g). The zero-shot gap is therefore one of adaptation and read-out rather than of representational capacity. The probe uses BUT QDB labels and is no longer zero-shot; its transfer advantage was observed on two external datasets.

#### Table 7b: Linear Probes on Frozen d1-3B Representations
*Representations from the primary configuration (strip + SQI text + primary question). Probes: standardization + L2 logistic regression, penalty from {10^-3, 10^-2, 10^-1, 1} by inner GroupKFold over subjects. BUT QDB: pooled out-of-fold over the five subject folds, subject-level bootstrap CI. Transfer probes are fit on all 1,047 BUT QDB windows. CinC 2017: balanced subset of Table 8b, recording maximum.*

| d1-3B representation | BUT QDB AUROC [95% CI] (5-fold subject CV) | CinC 2011 reported subset AUROC | CinC 2011 random subsets AUROC, mean (min–max) | CinC 2017 balanced subset AUROC (recording max) |
| :--- | :---: | :---: | :---: | :---: |
| **Zero-shot choice probability (primary configuration)** | 0.852 [0.717 – 0.944] | 0.636 | 0.650 (0.583–0.723) | 0.739 |
| **Linear probe on the decision state (final norm, answer position)** | 0.974 [0.930 – 0.997] | 0.475 | 0.443 (0.411–0.482) | 0.883 |
| **Linear probe on image tokens (projector output, mean)** | 0.979 [0.941 – 0.996] | 0.777 | 0.825 (0.756–0.907) | 0.910 |

### 4.7 External Validation at Natural Prevalence: CinC 2017

Table 8 reports the second external cohort (Section 2.5): 8,512 single-lead handheld recordings with a 3.3% prevalence of recordings judged too noisy to classify. With the pre-specified recording maximum, the rule heuristic (AUROC 0.887, 95% CI 0.867–0.905), LightGBM (0.880), and the SQI logistic regression and MLP (0.868) ranked recordings best, followed by the 2D CNN (0.850) and the multimodal net (0.835); the 1D CNN ranked lowest (0.773). The ordering therefore differs from BUT QDB, where the 1D CNN was among the best, and from CinC 2011, where the SQI logistic regression was inverted. At this prevalence the average precision was low for every model (0.06–0.23). At τ = 0.50 the CNNs and the multimodal net flagged 89–92% of noisy recordings at a false reject rate of 34–36%, a positive predictive value of about 0.08, while the heuristic flagged 18% at 2.5% false rejection. Averaging the window scores instead of taking the maximum raised every AUROC (0.80–0.93; secondary, not pre-specified). On the balanced subset used for d1-3B (Table 8b), zero-shot d1-3B reached AUROC 0.739, below every supervised model except the 1D CNN (0.767). No single representation was best across BUT QDB, CinC 2011 and CinC 2017.

#### Table 8: External Validation on CinC 2017 (8,512 Recordings, Natural Prevalence)
*Models trained on BUT QDB (seed 42) applied to every 10-s window (at most three per recording); recording score = maximum over windows (pre-specified) or mean (secondary). 95% CIs from 1,000 recording-level bootstrap resamples. Prevalence 3.3% (278 of 8,512). d1-3B was not run on the full cohort (Table 8b).*

| Model Architecture | AUROC [95% CI], recording max | AP [95% CI], recording max (prevalence 3.3%) | Sens. / FR at τ = 0.50 | PPV / NPV at τ = 0.50 | AUROC, recording mean (secondary) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.868 [0.848 – 0.886] | 0.173 [0.146 – 0.207] | 80.2% / 18.9% | 0.126 / 0.992 | 0.916 |
| **LightGBM (SQI)** | 0.880 [0.863 – 0.895] | 0.234 [0.187 – 0.285] | 67.6% / 14.6% | 0.135 / 0.987 | 0.930 |
| **Independent SQI MLP** | 0.868 [0.852 – 0.883] | 0.140 [0.120 – 0.171] | 86.3% / 24.0% | 0.108 / 0.994 | 0.934 |
| **1D CNN (Raw Signal)** | 0.773 [0.760 – 0.786] | 0.064 [0.057 – 0.073] | 92.4% / 34.5% | 0.083 / 0.996 | 0.801 |
| **2D CNN (Rendered Image)** | 0.850 [0.832 – 0.867] | 0.137 [0.115 – 0.172] | 89.9% / 33.8% | 0.082 / 0.995 | 0.879 |
| **SinglePass Multimodal Net** | 0.835 [0.818 – 0.852] | 0.107 [0.092 – 0.128] | 88.8% / 35.7% | 0.078 / 0.994 | 0.888 |
| **Rule-Based SQI Heuristic** | 0.887 [0.867 – 0.905] | 0.182 [0.155 – 0.217] | 18.3% / 2.5% | 0.199 / 0.973 | 0.933 |

#### Table 8b: CinC 2017 Balanced Subset Including d1-3B
*All 278 noisy recordings and 278 randomly chosen non-noisy recordings (seed 42); recording maximum; recording-level bootstrap CIs.*

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


---

## 5. Discussion, Limitations & Reporting Guidelines

### 5.1 Empirical Synthesis by Research Question

**RQ1 (Unimodal vs. Multimodal Fusion):** On the patient-independent BUT QDB benchmark, unimodal convolutional networks defined the discrimination frontier (1D CNN AUROC 0.989, AP 0.975; 2D CNN AUROC 0.989, AP 0.971; seed means 0.976 and 0.986). Multimodal fusion did not improve on them (AUROC 0.967, AP 0.928; seed mean 0.969). On CinC 2011 the multimodal net gave no usable transfer result: for seed 42 it ranked below both unimodal CNNs (AUROC 0.627 versus 0.726 and 0.746), and in three of four additional seeds its outputs were constant. Fusion therefore gave no accuracy or transfer benefit in this implementation.

**RQ2 (Zero-Shot Edge VLM vs. Specialized Baselines):** The 3.12-billion-parameter `LiquidAI/d1-3B` [5], evaluated zero-shot via the `system_one` interface without task-specific training, did not match the task-specialized networks in discrimination (AUROC 0.852 on BUT QDB, against 0.938–0.989 for the supervised models and 0.977 for the heuristic). At τ = 0.50 it identified 53.2% of unusable windows at a 4.0% false reject rate. On CinC 2011 it reached AUROC 0.636 and flagged all 150 recordings as unusable at τ = 0.50, and on the CinC 2017 subset it reached 0.739. Its usable probability changed by at most 0.09 across the NSTDB noise levels, so it did not track the noise. The gap is not a lack of representational capacity: a linear probe on its frozen image tokens reached AUROC 0.979 on BUT QDB and transferred better than every supervised model (CinC 2011 reported subset 0.777; CinC 2017 subset 0.910; Table 7b). What the zero-shot configuration lacks is a read-out adapted to this task; changing the prompt moved the operating point (Section 4.6) but did not close the gap. Fine-tuning and few-shot prompting were not tested.

**RQ3 (Cross-Dataset Transfer Resilience):** Transfer depended on the target. On CinC 2011 Set A (lead II scored against 12-lead labels) the 2D CNN had the highest AUROC in every balanced subset tested (0.713–0.884) and the 1D CNN ranged from 0.660 to 0.816. Default-threshold detection was poor: the 2D CNN flagged 17.3–28.0% of unacceptable recordings across random subsets, the 1D CNN 58.7–76.0% and LightGBM 8.0–20.0%, and the SQI logistic regression was inverted in every subset (AUROC 0.29–0.37). Averaging the scores of all 12 leads raised the AUROC of every trained model except logistic regression (2D CNN 0.852 and LightGBM 0.877 on average over the random subsets; Table 2e), so part of this gap is the label mismatch. On CinC 2017, a single-lead label at natural prevalence, the SQI models and the heuristic ranked best (AUROC 0.868–0.887) and the 1D CNN worst (0.773). The SQI behaviour accounts for much of this pattern: several SQIs reverse their association between BUT QDB and CinC 2011, the CinC 2011 flatline fraction largely measures amplitude quantization, and bSQI is informative again in CinC 2017 (Table 2g; Section 3.1). No representation was best across all external evaluations.

**RQ4 (Computational Complexity and the Rendering Bottleneck):** For image-based pipelines, rasterization dominated latency: 60.23 ms (92.4%) for the 2D CNN and 61.41 ms (92.4%) for the multimodal net, while neural forward passes took 5.12–5.56 ms. The 1D CNN on cached 500 Hz inputs ran in 1.47 ms median on MPS with no rendering step, and in 1.41 ms on the CPU from the native 1,000 Hz window (Table 4b). The rendering cost belongs to the Matplotlib implementation rather than to rasterization as such: a direct raster renderer took 0.63 ms, although the image models were not evaluated on its strips. The d1-3B model is dominated by neural inference (828 ms forward pass; 891 ms total).

### 5.2 Methodological Integrity, Dataset Shift, and Shortcut Learning
In clinical artificial intelligence, high retrospective performance frequently fails to generalize across hospital environments, acquisition devices, and temporal epochs [4, 14]. A major contributor to generalization failure is shortcut learning [9], wherein deep neural networks exploit spurious confounders—such as patient-specific baseline drift, hardware-specific noise signatures, or institutional acquisition artifacts—rather than genuine clinical pathophysiological features. Our experimental design mitigates shortcut learning by enforcing patient-disjoint GroupKFold cross-validation [7] and testing transfer to two independent databases with different devices and reference labels [2, 6, 22]. Nevertheless, as noted by Geirhos et al. [9], deep models remain susceptible to domain-specific inductive biases, emphasizing the need for rigorous external validation before clinical deployment.

### 5.3 Study Limitations
Several limitations must be transparently acknowledged:
1. **Limited Patient-Level Diversity:** The primary development cohort (BUT QDB) comprises only 15 distinct subjects, with unusable Class 3 noise present in only 7 individuals. While GroupKFold prevents patient identity leakage [7], it cannot overcome the limited patient-level diversity of small clinical cohorts, and the pooled sensitivities vary widely between subjects (1D CNN 14–100%; Table 1f). CinC 2017 (8,512 recordings; Section 4.7) adds a larger and more varied population, but with a different device, a recording-level label and no patient identifiers.
2. **Quota Sampling and Prevalence:** The cohort's 28.4% unusable share is set by quotas (750 usable and 297 unusable windows). The natural share in the BUT QDB candidate pool is 16.0%, or 2.8% without one dominant record. Table 1e re-weights the BUT QDB results to these prevalences and Table 8 reports CinC 2017 at its natural 3.3%; positive predictive values at these prevalences are much lower than the quota cohort suggests, and continuous monitoring may differ again.
3. **Target Label Mismatch:** CinC 2011 labels judge the whole 12-lead recording, whereas the main analysis scores lead II [4]. Scoring all 12 leads and averaging (Table 2e) raised AUROC substantially, which confirms that the mismatch costs part of the measured transfer; that aggregation rule was chosen after the main analysis. A large multi-lead dataset with lead-wise usability labels was not available to us.
4. **Zero-Shot Model Scope:** Zero-shot evaluation was limited to `LiquidAI/d1-3B` with three prompt configurations (Section 4.6). Fine-tuning and few-shot prompting were not tested. The linear probes (Table 7b) indicate that adaptation rather than capacity limits the zero-shot model, but a linear probe is a lightweight form of adaptation and not a substitute for fine-tuning.
5. **Input Representation Asymmetry:** The Multimodal Net received visual images and 13 numerical features, whereas d1-3B received the image and a text summary of 10 of those 13 features. Their performance gap therefore reflects differing input representations and training protocols as well as model capacity, and cannot be attributed to capacity alone.
6. **Hardware-Dependent Latency:** Reported latencies reflect one Apple M4 machine and Python implementations. The image pipelines' cost is dominated by Matplotlib rendering, which a direct raster renderer reduces from 58.4 ms to 0.63 ms (Table 4b), but the image models were not evaluated on raster strips. The 1D CNN's raw-to-decision time (1.41 ms on the CPU from the native 1,000 Hz window) excludes acquisition and buffering on a device, and sub-3 ms MPS timings differed by about 1.3 ms between two measurement harnesses.
7. **Threshold Selection:** The operating points of Tables 1b and 2b are post-hoc sweeps on the reported data. Nested selection (Table 1c) reproduced them for the CNNs but not for the SQI models, and thresholds chosen on BUT QDB did not carry over to CinC 2011 or CinC 2017 (Table 2f). No threshold here is prospectively validated; a deployment would need thresholds set and then tested on its own population.
8. **Seed Dependence and Subset Dependence:** Seed 42 is the reported run for every result. Seeds 1–4 were run for the 1D CNN, 2D CNN and multimodal net (Section 4.5), and the CinC transfer was repeated on five random balanced subsets (Table 2d). The in-domain ranking is stable (AUROC varies by at most 0.02 across seeds), but default-threshold sensitivity on CinC depends on both the training seed and the sampled recordings, and the multimodal net produced constant outputs on CinC in three of four additional seeds. The NSTDB stress test and the zero-shot d1-3B are single runs per configuration (the three d1-3B prompt configurations are in Section 4.6); latency was repeated over three sequential runs (Table 4). Logistic regression and LightGBM use a fixed random state; the MLP and multimodal net were seed-tested on the CinC subsets (Table 6) and the class-2 relabelling (Table 1d).
9. **Amplitude Units and Cross-Dataset Feature Meaning:** BUT QDB segments are stored in µV and converted to mV before SQI extraction, matching CinC 2011, CinC 2017 and NSTDB (mV). Harmonizing units alone does not make the tabular SQIs transferable: several features reverse their association with unusability between datasets. For example, the univariate AUROC of high-frequency noise ratio is 0.777 in BUT QDB but 0.329 in CinC, and powerline noise ratio is 0.773 versus 0.324. Tabular transfer results therefore reflect a change in the meaning of the SQIs across databases, not only a scaling problem. In addition, the flatline fraction counts repeats caused by amplitude quantization, which are frequent at CinC 2011's native 500 Hz but absent after resampling BUT QDB (Section 3.1); a quantization-aware definition would be needed before this feature is used across devices. Table 2g lists the univariate AUROC of every SQI in each dataset.
10. **Regeneration and Seeding:** All stored results were regenerated on 2026-10-12 after the corrections recorded in the project review (amplitude-unit harmonization, one-to-one bSQI matching, a fail-closed d1 wrapper, mini-batch training for the SQI MLP, aligned CNN epochs across scripts, a corrected NSTDB record filter, and seeding of torch, numpy and scikit-learn at 42). Before seeding, identical code produced different CNN and multimodal results across runs; for example, the multimodal net's NSTDB clean-control mean moved from 0.603 to 0.835 between two runs. The archived earlier outputs are kept under results/_backup_pre_fix_2026-10-12/. On 2026-10-13 a rendering error was found and corrected: the renderer inferred the unit from the amplitude and divided signals exceeding 20 by 1,000, so 71 CinC 2011 records (all unacceptable) and 22 NSTDB windows at −6 dB were drawn as flat lines. BUT QDB strips were unaffected (verified pixel-identical). All CinC and NSTDB results of the image models, the rule heuristic and d1-3B were recomputed. AUROC on the reported CinC subset changed by at most 0.001, but the −6 dB NSTDB values of the image models fell (2D CNN 0.115 to 0.012), removing an apparent rise at the lowest SNR. Outputs from before this fix are under results/_backup_pre_render_fix_2026-10-13/.
11. **Post-Hoc and Aggregation Choices:** Besides the post-hoc thresholds (Limitation 7), the record-level aggregation rules of Table 2e (mean or maximum over leads) and the mean over windows on CinC 2017 were chosen after the main analysis and are reported as sensitivity analyses; only the CinC 2017 recording maximum was specified before it was run.
12. **NSTDB Design and SNR Labels:** Clean reference windows come from the MIT-BIH base records at the same start times as the noisy windows. The clean intervals inside the noisy records (65–175 s and 425–535 s) are noise-free but carry a constant offset relative to the base record, so they were not used as the clean reference. The nominal SNR labels were not reproduced under NSTDB's own definition (Section 2.3), and the full-band SNRs measured here lie about 14 dB below the labels, so the SNR values in this paper are nominal. There is one noisy record per label and base record, and the NSTDB analysis is a single run per model seed (Section 4.5).

### 5.4 Alignment with Clinical AI Auditing Guidelines
To ensure transparency, this study was structured with reference to contemporary clinical AI auditing frameworks. We positioned TRIPOD+AI [10] as a reporting guideline and PROBAST+AI [11] as a risk-of-bias assessment tool to guide model description, patient-independent validation, and risk-of-bias mitigation. For visual waveform components and diagnostic accuracy, Relevant items from these frameworks informed the reporting audit; full checklist-based compliance was not established [12, 13]. We frame these standards not as claims of retrospective perfection, but as structured auditing criteria governing reproducible clinical AI benchmarking.

---

## 6. Conclusion

This study benchmarked eight models across six methodological paradigms for single-lead ECG quality triage, covering patient-independent development on BUT QDB, external transfer to CinC 2011 Set A and CinC 2017, and noise stress testing on MIT-BIH NSTDB. Our empirical findings are:
1. In-domain, the unimodal 1D and 2D CNNs gave the highest discrimination (AUROC 0.989), kept the highest precision when re-weighted to the natural prevalence (PPV 0.71–0.77 at 2.8%), and kept their operating points under nested threshold selection. The 1D CNN ran in 1.41 ms on the CPU from the native signal with no rendering step. Multimodal fusion did not add to them. Sensitivity varied widely between the seven subjects with unusable windows.
2. No model was best across external datasets. On CinC 2011 the 2D CNN ranked recordings best (AUROC 0.71–0.88) but accepted 72–83% of unacceptable recordings at the default threshold; scoring all 12 leads closed part of this gap. On CinC 2017 the SQI models and the rule heuristic ranked best and the 1D CNN worst, and at the natural 3.3% prevalence no model reached a useful precision at the default threshold. Thresholds chosen on BUT QDB did not carry over to either dataset.
3. Under three prompt configurations, zero-shot LiquidAI/d1-3B was less discriminative than the supervised CNNs (BUT QDB AUROC 0.852–0.913; CinC 2011 AUROC 0.627–0.680). A linear probe on its frozen image tokens reached AUROC 0.979 on BUT QDB and transferred better than the supervised models on lead II of CinC 2011 and on the CinC 2017 subset, so its limit is adaptation rather than capacity. Fine-tuning remains untested.
4. Image pipelines were dominated by Matplotlib rendering (about 60 ms per strip), which a direct raster renderer reduces to 0.63 ms; whether the image models keep their accuracy on such strips is untested.

---

## 7. Bibliography

[1] Nemcova, A., Smisek, R., Opravilová, K., Vitek, M., Smital, L., & Maršánová, L. (2020). Brno University of Technology ECG Quality Database (BUT QDB) (version 1.0.0). *PhysioNet*. https://doi.org/10.13026/kah4-0w24.  
[2] Silva, I., Moody, G. B., & Celi, L. (2011). Improving the quality of ECGs collected using mobile phones: The PhysioNet/Computing in Cardiology Challenge 2011. *Computing in Cardiology*, 38, 273–276.  
[3] Moody, G. B., Muldrow, W. E., & Mark, R. G. (1984). A noise stress test for arrhythmia detectors. *Computers in Cardiology*, 11, 381–384. https://physionet.org/physiobank/database/nstdb/. Noise generator documentation (nst, WFDB applications): https://physionet.org/physiotools/wag/nst-1.htm.  
[4] Subbaswamy, A., & Saria, S. (2020). From development to deployment: dataset shift, causality, and shift-stable models in health AI. *Biostatistics*, 21(2), 345–352. https://doi.org/10.1093/biostatistics/kxz041.  
[5] Liquid AI. (2026). LiquidAI/d1-3B: Open-weight edge multimodal decision model. *Hugging Face Model Hub*. Commit: `051bcc464b01b9f92942b364d9586b0ef5912432`. https://huggingface.co/LiquidAI/d1-3B.  
[6] Computing in Cardiology & PhysioNet. (2011). Improving the Quality of ECGs Collected Using Mobile Phones: Challenge Protocol and Diagnostic Reference Standards. *PhysioNet Challenge Archive*. https://moody-challenge.physionet.org/2011/.  
[7] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.  
[8] scikit-learn developers. (2024). Precision-recall metrics and non-interpolated average precision (average_precision_score v1.5+). *Scikit-learn Documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html.  
[9] Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665–673. https://doi.org/10.1038/s42256-020-00257-z.  
[10] Collins, G. S., Moons, K. G. M., Dhiman, P., Navarro, C. A., Beam, A. L., Boyd, K. A., ... & TRIPOD+AI Group. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use artificial intelligence. *BMJ*, 385, e078378. https://doi.org/10.1136/bmj-2023-078378.  
[11] Moons, K. G. M., Damen, J. A. A. G., Kaul, V., et al. (2025). PROBAST+AI: a tool to assess risk of bias and applicability of prediction model studies using artificial intelligence. *BMJ*, 388, e082505. https://doi.org/10.1136/bmj-2024-082505.  
[12] Tejani, A. S., Klontzas, M. E., Gatti, A. A., et al. (2024). Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update. *Radiology: Artificial Intelligence*, 6(4), e240300. https://doi.org/10.1148/ryai.240300.  
[13] Sounderajah, V., Guni, A., Liu, X., Collins, G. S., et al. (2025). The STARD-AI reporting guideline for diagnostic accuracy studies using artificial intelligence. *Nature Medicine*, 31(10), 3283–3289. https://doi.org/10.1038/s41591-025-03953-8.  
[14] Guo, Z., Ding, Y., Yue, X., & Hu, X. (2022). Evaluation of domain generalization and adaptation on improving model robustness to temporal dataset shift in clinical medicine. *Scientific Reports*, 12, 2726. https://doi.org/10.1038/s41598-022-06484-1.  
[15] Liu, F., Xia, S., Wei, S., Chen, L., Ren, Y., et al. (2022). Wearable Electrocardiogram Quality Assessment Using Wavelet Scattering and LSTM. *Frontiers in Physiology*, 13, 905447. https://doi.org/10.3389/fphys.2022.905447.  
[16] Halvaei, H., Svennberg, E., Sörnmo, L., & Stridh, M. (2021). Identification of Transient Noise to Reduce False Detections in Screening for Atrial Fibrillation. *Frontiers in Physiology*, 12, 672875. https://doi.org/10.3389/fphys.2021.672875.  
[17] Hu, Y., Liu, Q., Zhou, Z., Xu, W., & Tang, H. (2025). A Paralleled Multi-Task Learning-Based Framework for Single-Lead ECG Fine-Grained Noise Localization, Denoising and Signal Quality Assessment. *Sensors*, 25(23), 7152. https://doi.org/10.3390/s25237152.  
[18] Clifford, G. D., Behar, J., Li, Q., & Rezek, I. (2012). Signal quality indices and data fusion for determining clinical acceptability of electrocardiograms. *Physiological Measurement*, 33(9), 1419–1433. https://doi.org/10.1088/0967-3334/33/9/1419.  
[19] Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230–236. https://doi.org/10.1109/TBME.1985.325532.  
[20] Orphanidou, C., Bonnici, T., Charlton, P., Clifton, D., Vallance, D., & Tarassenko, L. (2015). Signal-quality indices for the electrocardiogram and photoplethysmogram: Derivation and applications to wireless monitoring. *IEEE Journal of Biomedical and Health Informatics*, 19(3), 832–838. https://doi.org/10.1109/JBHI.2014.2338351.  
[21] Satija, U., Ramkumar, B., & Manikandan, M. S. (2018). A review of signal quality evaluation technologies for electrocardiogram signals. *IEEE Reviews in Biomedical Engineering*, 11, 36–52. https://doi.org/10.1109/RBME.2018.2810957.  
[22] Clifford, G. D., Liu, C., Moody, B., Lehman, L. H., Silva, I., Li, Q., Johnson, A. E., & Mark, R. G. (2017). AF classification from a short single lead ECG recording: The PhysioNet/Computing in Cardiology Challenge 2017. *Computing in Cardiology*, 44. https://doi.org/10.22489/CinC.2017.065-469.  
