import os
import re

md_content = """# Zero-shot and task-specific models for ECG usability screening: patient-independent performance, cross-dataset transfer, and computational cost

## Abstract

**Background:** Automated electrocardiogram (ECG) interpretation algorithms degrade when applied to corrupted recordings. Quality control triage is critical to accept clean traces, reject unusable segments, and prompt for re-recording. While deep multimodal decision models and lightweight convolutional networks offer potential solutions, their comparative accuracy, zero-shot transfer, and computational overhead require rigorous characterization under patient-disjoint outer validation.

**Methods:** We benchmarked eight models across six methodological paradigms: classical tabular classifiers (Logistic Regression, LightGBM, MLP on 13 input features including 11 Signal Quality Indices [SQIs] and 2 metadata parameters), deep learning architectures (1D CNN on raw 500 Hz waveforms, 2D CNN on rendered visual strips, and a single-pass multimodal network combining SQIs and visual features), an expert heuristic baseline, and the 3.12-billion-parameter LiquidAI/d1-3B multimodal edge model evaluated zero-shot with zero output tokens. Models were trained on the Brno University of Technology ECG Quality Database (BUT QDB; quota-based sampling of 1,047 windows [750 usable, 297 unusable; 28.4% prevalence], 15 subjects) under patient-disjoint 5-fold GroupKFold cross-validation [8], and evaluated for out-of-distribution transfer on the PhysioNet/Computing in Cardiology Challenge 2011 Set A (150 mobile phone recordings, Lead II) [2, 7] and MIT-BIH Noise Stress Test Database (NSTDB; 264 active electrode motion noise evaluations across 6 calibrated SNR tiers alongside 264 clean control window evaluations; 528 total evaluations) [3]. Hardware latency was profiled on Apple Silicon.

**Results:** On the patient-independent BUT QDB development cohort, unimodal convolutional networks defined the discrimination frontier: unimodal 2D CNN achieved an AUROC of 0.985 (Average Precision [AP] 0.964, Brier score 0.068), while unimodal 1D CNN achieved an AUROC of 0.977 (AP 0.958, Brier score 0.060), reaching 93.3% sensitivity at a 4.3% false reject rate under exploratory ROC analysis (88.6% sensitivity at 3.1% false reject under companion-partition exploratory threshold simulation). Multimodal fusion matched 1D CNN in discrimination (AUROC 0.977, AP 0.945) without providing additive gain over image-only processing. Evaluated zero-shot, LiquidAI/d1-3B achieved lower discrimination than task-specific models (AUROC 0.842, AP 0.728, Brier score 0.158), incurring 36.3% false rejection to achieve 80.8% unusable sensitivity under post-hoc ROC exploration. On out-of-distribution transfer to CinC 2011, multimodal fusion and 2D CNN preserved higher ranking discrimination (AUROCs 0.763 and 0.761) and a 4.0% false reject rate at default threshold, but accepted over half of unusable target recordings. 1D CNN exhibited higher false rejection (21.3%) and AUROC 0.718, while tabular logistic regression and MLP models collapsed due to feature covariate shift (AUROC 0.500) without robust scaling. Under calibrated NSTDB stress, all models exhibited a declining usability probability trend across noisy SNR levels (+24 dB down to -6 dB SNR). Hardware profiling revealed that input preprocessing is the dominant latency bottleneck in visual and multimodal pipelines (69.1–74.8 ms total latency, with preprocessing consuming 92.5%–93.1% of end-to-end execution [median of per-run ratios; ratio of medians 92.1%–92.9%], of which waveform rasterization accounts for 97.3%–100%), whereas unimodal 1D CNN executed in 1.12 ms latency on standardized cached inputs on CPU with 0.64 MiB storage.

**Conclusion:** Lightweight 1D convolutional networks operating directly on raw biosignals provide a favorable accuracy–latency trade-off in the evaluated implementation, combining high discrimination (AUROC 0.977), low inference latency on cached signals (1.12 ms), and zero rendering overhead. Visual representations confer modest generalization advantages during cross-dataset transfer but incur substantial rasterization penalties (64.2–68.9 ms). The evaluated zero-shot configuration of LiquidAI/d1-3B under the specified prompt and test datasets lagged task-specific architectures in clinical discrimination; whether domain-specific fine-tuning or prompt optimization bridges this gap remains an open research question.

---

## 1. Introduction & Research Framework

Automated electrocardiogram (ECG) interpretation algorithms represent a cornerstone of contemporary cardiovascular medicine, powering telemetry monitors, ambulatory Holter systems, and mobile health devices. However, automated diagnostic pipelines degrade substantially when exposed to real-world biosignal corruption. Common physical noise sources—including electrode motion artifact, baseline wander from patient respiration, electromyographic (EMG) interference from skeletal muscle contraction, and ambient powerline noise—can distort P-wave, QRS, and T-wave morphologies. As highlighted in foundational stress testing protocols [3], electrode motion artifacts are particularly pernicious because their spectral power overlaps directly with genuine ventricular depolarization, frequently mimicking ventricular arrhythmias and resisting conventional linear bandpass filtering. Automated signal quality assessment (SQA) is therefore indispensable as an initial triage gate to prevent spurious clinical alarms, prevent erroneous diagnostic conclusions, and prompt timely electrode re-attachment or signal re-recording.

### 1.1 Distinct Clinical Task Formulation

A critical methodological challenge in ECG quality assessment literature is the frequent conflation of distinct clinical task formulations. Specifically, "QRS-based rhythm usability", "morphological precision usability", and "overall 12-lead diagnostic acceptability" represent fundamentally distinct reference standards with disparate clinical stringencies. For instance, in the Brno University of Technology ECG Quality Database (BUT QDB) [1], annotations are stratified into three distinct tiers: Class 1 denotes pristine diagnostic quality; Class 2 designates records where noise precludes fine morphological measurement (such as ST-segment deviation or QT interval duration) yet permits reliable QRS complex detection and heart rate determination; and Class 3 designates recordings where severe corruption prevents even basic QRS identification. Conversely, community challenges such as the PhysioNet/Computing in Cardiology Challenge 2011 [2, 7] evaluated overall 12-lead diagnostic acceptability as judged by expert cardiologists. In this investigation, we strictly operationalize the primary clinical quality triage task as binary unusable detection (Class 3 vs. Classes 1 & 2 in BUT QDB), designed to identify uninterpretable waveforms that must be rejected from automated analysis.

### 1.2 Related Work in ECG Quality Assessment

Related Work in ECG Signal Quality Assessment spans several historical and emerging paradigms, as synthesized in comprehensive reviews of ECG quality evaluation technologies [22]. Classical engineering approaches have relied primarily on handcrafted Signal Quality Indices (SQIs) and multi-index decision heuristics. Seminal work by Clifford et al. [19] and Orphanidou et al. [21] demonstrated that statistical measures, spectral energy ratios, and detector consistency can triage clinical acceptability in ambulatory and mobile recordings. Handcrafted SQIs quantify statistical properties (e.g., skewness, kurtosis), spectral energy ratios, and consistency across parallel QRS detectors [1, 2, 19]. While computationally lightweight and interpretable, tabular feature engineering often fails to capture complex spatiotemporal artifact patterns. More recently, learned representations tailored for wearable biosensors have demonstrated improved feature extraction; for example, combining wavelet scattering transforms with Long Short-Term Memory (LSTM) networks captures multiscale transient dynamics while mitigating intra-patient morphological variability [16]. Furthermore, contemporary frameworks have explored joint noise localization and quality estimation, such as parallel multi-task convolutional architectures that simultaneously denoise, locate artifact boundaries, and classify segment quality [18], or specialized transient noise detectors designed to suppress false positive arrhythmia alarms in atrial fibrillation screening [17]. In parallel, the broader machine learning community has seen the emergence of general-purpose multimodal foundation models and vision-language decision architectures capable of zero-shot classification [5, 6]. However, prior studies frequently evaluate these paradigms under disparate datasets, inconsistent validation splits, or optimistic random window sampling, preventing direct and reproducible comparison across architectural families.

### 1.3 Research Gaps and Primary Questions

To address these gaps, this study establishes a benchmark structured along four methodological axes:
1. Patient-Independent Generalization: Evaluating models under patient-disjoint GroupKFold cross-validation to isolate subject identities [8], preventing intra-patient data leakage.
2. Out-of-Distribution Transfer and Dataset Shift: Characterizing cross-dataset performance across disparate patient demographics and clinical acquisition modalities, grounded in established dataset shift principles in health AI [4, 15].
3. Operating Threshold Behavior: Differentiating fixed default decision rules (τ = 0.50) from exploratory companion-partition threshold simulations (τ*) and post-hoc exploratory ROC curves.
4. End-to-End Computational Profiling: Measuring the execution budget on edge hardware across CPU and GPU/MPS devices, quantifying the specific latency overhead introduced by visual waveform rasterization.

Specifically, we investigate four primary Research Questions (RQs):
* **RQ1 (Unimodal vs. Multimodal Fusion):** Does fusing visual ECG waveforms with tabular SQIs yield superior patient-independent quality triage compared to specialized unimodal 1D signal or 2D image models?
* **RQ2 (Zero-Shot Edge VLM vs. Supervised Baselines):** How does a contemporary 3.12-billion-parameter multimodal decision model (LiquidAI/d1-3B [5, 6]), evaluated zero-shot via choice-probability scoring without task-specific training, compare against task-specialized supervised neural networks and expert rule heuristics?
* **RQ3 (Cross-Dataset Transfer Resilience):** How effectively do tabular, visual, and raw signal representations transfer out-of-distribution to mobile phone ECGs (CinC 2011 Set A) and calibrated noise stress conditions (NSTDB), and what mechanisms govern feature shift?
* **RQ4 (The Visual Rendering Bottleneck):** What are the empirical hardware latencies and memory footprints across architectures, and does waveform rasterization impose a substantial cost for resource-constrained edge clinical deployment?

---

## 2. Data and Stratified Sampling Protocols

### 2.1 BUT ECG Quality Database (BUT QDB) [1]
The primary development cohort was derived from the BUT ECG Quality Database, curated by Brno University of Technology and hosted on PhysioNet [1]. BUT QDB comprises 18 long-term single-lead ECG recordings acquired across 15 distinct subjects (12 males, 3 females, ages 21-83) experiencing normal sinus rhythm, arrhythmias, and everyday physical activities. Waveforms were digitized at a sampling rate of 1,000 Hz with 16-bit resolution. Across all 18 records the total annotated signal duration is 99.39 hours (357,791.9 seconds); the previously cited duration of 98.1 hours reflects the 17-record subset excluding Record 124001_ECG (4,799.5 seconds / 1.33 hours). We applied quota-based sampling across all 18 records to construct a quota-based, computationally tractable development benchmark of 1,047 non-overlapping 10-second segments, yielding 750 usable windows (Class 1 & 2) and 297 unusable windows (Class 3), corresponding to an empirical 28.4% unusable prevalence. Segments were standardized to 500 Hz via polyphase resample filtering.

### 2.2 PhysioNet / Computing in Cardiology Challenge 2011 (CinC 2011 Set A) [2, 7]
To evaluate external zero-shot transfer, we utilized Set A of the PhysioNet/Computing in Cardiology Challenge 2011 [2, 7]. The challenge dataset consists of 10-second, 12-lead ECG recordings collected from mobile health workers in developing countries using commercial cellular phones, digitized at 500 Hz. Expert cardiologists provided overall 12-lead diagnostic acceptability annotations. For rigorous cross-dataset transfer evaluation, we extracted a balanced 150-record evaluation cohort (75 acceptable, 75 unacceptable) and evaluated models strictly using Lead II signals. It is vital to note that this introduces an inherent clinical target mismatch: models evaluated on Lead II are tasked with predicting a whole-record 12-lead diagnostic decision, providing a stringent real-world test of covariate and label shift [4].

### 2.3 MIT-BIH Noise Stress Test Database (NSTDB) [3]
To benchmark noise-stress robustness under calibrated signal-to-noise ratios, we evaluated models on the MIT-BIH Noise Stress Test Database [3]. NSTDB features pristine baseline recordings (records 118 and 119) corrupted with calibrated amounts of recorded electrode motion ('em') noise at standardized SNR increments (+24, +18, +12, +6, 0, and -6 dB). We evaluated 528 total window instances: 264 active noise evaluations (44 non-overlapping 10-second windows across each of the 6 SNR levels) alongside 264 clean control window evaluations extracted from uncorrupted intervals of the same records. We emphasize that repeated clean window evaluations serve as baseline integrity controls across condition files rather than 264 independent recordings, and that NSTDB evaluates controlled electrode motion noise rather than the entire spectrum of clinical hardware failure modes.

### 2.4 Patient-Independent Grouped Cross-Validation Design [8]
To enforce patient-disjoint outer evaluation on BUT QDB, we applied 5-fold GroupKFold cross-validation [8], grouping windows strictly by patient identifier (15 subjects). This ensures that no patient contributes data to both the training and test partitions within any fold. While GroupKFold prevents patient identity leakage, the small overall patient count (15 subjects, with unusable Class 3 artifacts present in only 7 subjects) remains an important empirical limitation. Table 0 summarizes the patient and sample distribution across all five outer cross-validation folds. Model-specific feature scaling policies were strictly isolated to each fold's 12 training subjects: Logistic Regression utilized scikit-learn's StandardScaler; Independent SQI MLP and the tabular branch of SinglePass Multimodal Net utilized empirical z-score standardization fit strictly on the training partition; LightGBM operated on raw unscaled SQI features (scale-invariant decision trees); 1D CNN processed lead-standardized 500 Hz biosignals; and 2D CNN processed torchvision normalized RGB images. No MinMax scaling was applied.

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

---

## 3. Model Architectures & Experimental Setup

### 3.1 Signal Quality Indicators (SQIs) and Input Representations
For tabular and multimodal models, we extracted 13 input features comprising 11 domain-expert Signal Quality Indices (SQIs) [19, 21, 22] and 2 recording metadata parameters:
1. `bsqi_detector_agreement`: F1-matching score between Pan-Tompkins [20] and difference-based QRS detectors (temporal tolerance: 50 ms) [19].
2. `baseline_wander_ratio`: Relative Welch spectral power in the baseline band (<0.5 Hz) relative to the ECG band (0.5-40 Hz) [19].
3. `high_frequency_noise_ratio`: Relative Welch spectral power in the electromyographic band (>40 Hz up to min(fs/2, 100 Hz)) relative to the ECG band [19].
4. `saturation_fraction`: Proportion of samples hitting dynamic extrema (ADC rail clipping).
5. `flatline_fraction`: Proportion of consecutive samples with zero difference (amplifier disconnection or saturation).
6. `qrs_energy_ratio`: Relative Welch spectral power in the QRS depolarization band (5-15 Hz) relative to the ECG band.
7. `kurtosis_ksqi`: Sample kurtosis of the 10-second signal distribution (scipy.stats.kurtosis) [19].
8. `skewness_ssqi`: Sample skewness of the 10-second signal distribution (scipy.stats.skew) [19].
9. `zero_crossing_rate`: Rate of sign zero-crossings after mean subtraction.
10. `powerline_noise_ratio`: Relative Welch spectral power in powerline notch bands (49-51 Hz and 59-61 Hz) relative to the ECG band.
11. `signal_std`: Standard deviation of signal amplitude in mV.
12. `duration_sec`: Window duration in seconds (constant metadata: 10.0 s in BUT QDB).
13. `sampling_rate`: Sampling frequency in Hz (constant metadata: 500 Hz standardized input).

In the development cohort, duration_sec and sampling_rate are constant metadata, while flatline_fraction and saturation_fraction exhibited near-zero variance (σ ~ 10^-5) in clean recording intervals. When applied out-of-distribution without regularization, unconstrained standard scaling caused numerical saturation, motivating our retrospective sensitivity analysis on robust scaling.

### 3.2 Supervised Model Architectures
We implemented and evaluated six supervised architectures:
* **Logistic Regression (SQI):** Linear baseline trained on standardized 13-dimensional feature vectors with L2 regularization (14 parameters: 13 weights, 1 intercept).
* **LightGBM (SQI):** Gradient-boosted decision tree ensemble trained with 100 estimators, max depth 4, learning rate 0.05.
* **Independent SQI MLP:** Multi-layer perceptron comprising two hidden layers (64 and 32 units, each followed by batch normalization [BatchNorm1d] and ReLU activations, with dropout 0.2; 3,201 trainable parameters: 3,009 dense weights and biases + 192 affine batch normalization parameters), trained via AdamW (lr=1e-3, weight_decay=1e-3) for 35 epochs.
* **1D CNN (Raw Signal):** Unimodal convolutional network operating directly on raw 500 Hz 1D ECG biosignal waveforms (1 x 5,000 samples; 167,489 trainable parameters). Features four 1D convolutional blocks (filters: 32, 64, 128, 256; kernel sizes: 7, 5, 5, 3) with batch normalization, max-pooling, global average pooling, and a linear classification head.
* **2D CNN (Rendered Image):** Unimodal visual network operating on rendered 400x300 3-channel RGB waveform images (245,697 trainable parameters). As implemented in models/cnn2d.py, input images are loaded as 3-channel RGB tensors (dimensions: 3 x 300 x 400) and processed through a 4-stage convolutional backbone (channel dimensions: 32, 64, 128, 256) followed by global spatial average pooling and a linear classifier.
* **SinglePass Multimodal Net:** Multimodal network combining the visual convolutional backbone (processing the identical 3-channel RGB image tensor) with an auxiliary MLP branch processing the 13 input features (123,713 trainable parameters). Visual embeddings (64-d) and tabular embeddings (64-d) are combined via gated fusion and mapped through a decision head (64 units, dropout 0.3).

For external evaluation on CinC 2011 and NSTDB, supervised models were trained on the complete 1,047-window BUT QDB development cohort using the identical hyperparameters and random seed (42).

### 3.3 Expert Heuristic Baseline: Rule-Based SQI Heuristic
As a deterministic clinical benchmark, we evaluated an established expert rule-based heuristic grounded in multi-SQI fusion protocols [19, 21] (0 trainable parameters) that flags segments as unusable if: QRS agreement bsqi < 0.60, or baseline wander power basSQI > 0.40, or high-frequency power hfSQI > 0.35, or flatline duration > 1.5 seconds. For visual ink density calculation, the rule heuristic converts rendered images to 8-bit single-channel grayscale (mode 'L') as implemented in models/rule_heuristic_baseline.py. Continuous pseudo-probabilities were derived via normalized soft-penalty aggregation.

### 3.4 Multimodal Decision Model: Liquid AI d1-3B [5, 6]
To investigate the zero-shot capabilities of frontier multimodal models, we evaluated `LiquidAI/d1-3B` [5, 6], an open-weight 3.12-billion-parameter multimodal edge decision model (model revision commit: `051bcc464b01b9f92942b364d9586b0ef5912432`). The model was evaluated strictly zero-shot (meaning no fine-tuning or training on ECG signals was conducted) via its official `system_one` inference mode on Apple Silicon MPS (device: mps, dtype: torch.bfloat16). The vision encoder processes the rendered ECG strip alongside a structured clinical task prompt:

Task Question: "Determine whether this electrocardiogram (ECG) recording segment is usable for reliable QRS-based rhythm analysis, or unusable due to severe artifacts, baseline wander, high-frequency noise, or saturation/flatline."
Criteria: usable: "QRS complexes are distinguishable and suitable for rhythm analysis. Artifacts are manageable."; unusable: "ECG is severely corrupted by noise, baseline drift, saturation, or absent complexes; requires human review or re-recording."

The model is configured with `output_tokens: 0`, returning choice probabilities over categorical options from `raw_response['answers']['usability']['probabilities']['unusable']`. We explicitly note that these choice probabilities represent uncalibrated model outputs. Furthermore, there is an inherent input asymmetry: the supervised Multimodal Net received both visual images and 13 numerical features, whereas d1-3B received solely the visual image and task prompt without numerical SQIs.

### 3.5 Evaluation Metrics, Threshold Selection, and Reproducibility Protocols
Discrimination was quantified using Area Under the Receiver Operating Characteristic (AUROC) and Average Precision (AP, non-interpolated) via scikit-learn [9]. We strictly differentiate three decision threshold regimes: (1) default fixed threshold τ = 0.50; (2) exploratory fold-stratified threshold simulation across companion partitions (τ*), where each fold's threshold is determined from the pooled out-of-fold predictions of the other 4 companion folds (12 subjects) via Youden's J = Sensitivity + Specificity - 1, serving as an exploratory sensitivity benchmark across partition combinations rather than an independent nested inner-validation split; and (3) exploratory post-hoc ROC sweeps (Target 1: Max Sensitivity at FPR ≤ 5%; Target 2: Min FPR at Sensitivity ≥ 80%, with deterministic tie-breaking: max TPR then min FPR) [9]. Tables 1b and 1c present exploratory analyses and do not represent prospectively validated deployment thresholds. For external transfer experiments, feature scalers (including RobustScaler) were fit strictly on the BUT QDB source development cohort (N=1,047) and applied frozen to CinC 2011 to evaluate transfer without target data leakage. Calibration was assessed via the Brier score. Subject-level cluster bootstrapping (500 iterations) was used to construct 95% confidence intervals on BUT QDB. Hardware latency was profiled on Apple Silicon (M2, 8 CPU / 10 GPU cores, 24 GB Unified Memory) with explicit `torch.mps.synchronize()` barriers across 25 paired test trials.

---

## 4. Experimental Results

In this section, we present the empirical findings of our multi-paradigm benchmark structured along four evaluation axes: (1) patient-independent cross-validation on BUT QDB, examining discrimination frontiers, exploratory ROC trade-offs, and companion-partition Youden threshold simulations; (2) cross-dataset out-of-distribution transfer to CinC 2011 Set A, probing representation shift and scaling sensitivity; (3) noise stress testing under calibrated electrode motion artifact on MIT-BIH NSTDB; and (4) hardware-synchronized edge latency, execution device, and weight storage profiling across CPU and MPS devices, quantifying the waveform rendering bottleneck.

### 4.1 Patient-Independent 5-Fold Cross-Validation on BUT QDB

Table 1 and Figure 1 summarize patient-independent 5-fold cross-validation performance across all eight models on BUT QDB (quota-based sampling of 1,047 windows, 15 subjects). Unimodal convolutional architectures established the discrimination frontier: unimodal 2D CNN operating on rendered visual strips achieved the highest overall AUROC of 0.985 (95% CI: 0.958–1.000, AP: 0.964, Brier: 0.068), while unimodal 1D CNN operating directly on raw 500 Hz waveforms achieved an AUROC of 0.977 (95% CI: 0.908–0.998, AP: 0.958, Brier: 0.060). The SinglePass Multimodal Net matched the 1D CNN in discrimination (AUROC 0.977, AP 0.945, Brier 0.070), indicating that combining visual waveform embeddings with tabular SQIs provided no additive gain over unimodal image processing within the source domain. The deterministic expert heuristic baseline also demonstrated strong discrimination (AUROC 0.977, AP 0.948), although its calibration was poorer (Brier 0.105). In contrast, the zero-shot 3.12B LiquidAI/d1-3B model lagged task-specialized supervised architectures, recording AUROC 0.842 (95% CI: 0.771–0.912), AP 0.728, and Brier score 0.158, reflecting uncalibrated choice probabilities and lack of domain-specific adaptation.

#### Table 1: Patient-Independent 5-Fold Cross-Validation Performance on BUT QDB (1,047 Windows, 15 Subjects)
*Positive Class = Unusable (297 windows), Negative Class = Usable (750 windows). 95% CIs derived via 500 subject-level cluster bootstrap iterations.*

| Model Architecture | AUROC [95% CI] | AP | Brier | Fixed τ = 0.50 [TP / FP / TN / FN] | Balanced Acc. | Sensitivity | False Reject Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D CNN (Rendered Image)** | **0.985** [0.958 – 1.000] | **0.964** | 0.068 | [194 / 9 / 741 / 103] | 82.1% [65.1 – 99.3%] | 65.3% [31.4 – 100%] | **1.2%** [0.4 – 2.0%] |
| **1D CNN (Raw Signal)** | 0.977 [0.908 – 0.998] | 0.958 | **0.060** | [229 / 8 / 742 / 68] | **88.0%** [71.3 – 97.8%] | 77.1% [43.9 – 96.3%] | **1.1%** [0.0 – 2.8%] |
| **Rule-Based SQI Heuristic** | 0.977 [0.925 – 0.998] | 0.948 | 0.105 | [196 / 9 / 741 / 101] | 82.4% [57.1 – 96.2%] | 66.0% [15.7 – 93.1%] | **1.2%** [0.3 – 2.4%] |
| **SinglePass Multimodal Net** | 0.977 [0.929 – 1.000] | 0.945 | 0.070 | [216 / 20 / 730 / 81] | 85.0% [69.7 – 98.5%] | 72.7% [43.0 – 100%] | 2.7% [0.8 – 5.0%] |
| **LightGBM (SQI)** | 0.965 [0.941 – 0.985] | 0.931 | 0.072 | [223 / 31 / 719 / 74] | 85.5% [68.0 – 97.3%] | 75.1% [39.1 – 97.8%] | 4.1% [1.3 – 7.1%] |
| **Independent SQI MLP** | 0.946 [0.898 – 0.984] | 0.893 | 0.119 | [252 / 97 / 653 / 45] | 86.0% [71.4 – 96.7%] | **84.8%** [50.4 – 100%] | 12.9% [3.6 – 24.3%] |
| **Logistic Regression (SQI)** | 0.941 [0.892 – 0.978] | 0.892 | 0.082 | [229 / 45 / 705 / 68] | 85.6% [72.3 – 95.8%] | 77.1% [48.7 – 96.2%] | 6.0% [2.2 – 10.4%] |
| **LiquidAI d1-3B (Zero-Shot)*** | 0.842 [0.771 – 0.912] | 0.728 | 0.158 | [155 / 33 / 717 / 142] | 73.9% [61.2 – 86.6%] | 52.2% [26.7 – 77.6%] | 4.4% [1.3 – 7.5%] |

At the default classification threshold (τ = 0.50), model operating behaviors varied substantially due to differing logit calibrations (Table 1). While 1D CNN achieved 77.1% unusable sensitivity with a 1.1% false reject rate (balanced accuracy 88.0%), 2D CNN exhibited lower sensitivity (65.3%) at 1.2% false rejection. To systematically evaluate the trade-off boundaries without assuming a fixed 0.50 threshold, Table 1b details exploratory post-hoc ROC sweeps under two clinical operating constraints: Target 1 (maximizing sensitivity while bounding the false reject rate to FPR ≤ 5%) and Target 2 (minimizing false rejection while guaranteeing sensitivity ≥ 80%). Under Target 1, 1D CNN reached 93.27% sensitivity at a 4.27% false reject rate (threshold τ = 0.01503), outperforming 2D CNN (91.25% sensitivity at 4.93% FPR) and Multimodal Net (85.19% sensitivity at 4.80% FPR). Conversely, achieving 80% sensitivity with zero-shot d1-3B required accepting a clinically prohibitive 36.27% false reject rate (Target 2). We emphasize that these exploratory sweeps reflect full test distribution knowledge and do not constitute validated prospective deployment rules.

#### Table 1b: Exploratory ROC Operating-Point Analysis on BUT QDB
*Decision Rule: Positive class (Unusable) if p ≥ τ. Target 1: Maximum sensitivity achievable while bounding False Reject Rate (FPR) ≤ 5%. Target 2: Minimum False Reject Rate incurred when guaranteeing Unusable Sensitivity ≥ 80%. Note: These operating points represent post-hoc exploratory sweeps to examine empirical trade-offs, not validated deployment thresholds.*

| Model Architecture | Target 1: Sens. (Actual FPR) | Threshold τ | Target 2: FPR (Actual Sens.) | Threshold τ |
| :--- | :---: | :---: | :---: | :---: |
| **1D CNN (Raw Signal)** | 93.27% (FPR 4.27%) | 0.01503 | 1.60% (Sens 80.13%) | 0.32223 |
| **2D CNN (Rendered Image)** | 91.25% (FPR 4.93%) | 0.12465 | 2.27% (Sens 80.81%) | 0.27583 |
| **SinglePass Multimodal Net** | 85.19% (FPR 4.80%) | 0.24546 | 3.47% (Sens 80.81%) | 0.39572 |
| **Rule-Based SQI Heuristic** | 84.18% (FPR 4.80%) | 0.37981 | 4.13% (Sens 80.47%) | 0.40701 |
| **LightGBM (SQI)** | 81.14% (FPR 4.67%) | 0.35183 | 4.67% (Sens 81.14%) | 0.35183 |
| **Logistic Regression (SQI)** | 75.08% (FPR 4.80%) | 0.56543 | 7.20% (Sens 80.47%) | 0.42767 |
| **Independent SQI MLP** | 71.72% (FPR 4.67%) | 0.60815 | 9.47% (Sens 80.13%) | 0.52835 |
| **LiquidAI d1-3B (Zero-Shot)*** | 52.19% (FPR 4.40%) | 0.50000 | 36.27% (Sens 80.81%) | 0.37754 |

To examine threshold sensitivity across cross-validation partitions, Table 1c reports an exploratory fold-stratified threshold simulation using decision rules (τ*) derived from the companion folds' pooled out-of-fold predictions via Youden's Index (J = Sensitivity + Specificity - 1) and evaluated on the held-out test partition of each fold. As detailed in Section 3.5, because companion-fold predictions originate from models trained on overlapping patient subsets, this simulation characterizes threshold sensitivity across partition combinations rather than prospective inner-validation deployment. The reported Mean τ* and Median τ* summarize the five fold-specific thresholds; critically, these summary thresholds were NOT applied across the pooled dataset. Instead, each test sample was evaluated strictly against its fold-specific companion threshold. Under this exploratory simulation, 1D CNN maintained the highest balanced accuracy (92.7%), identifying 88.6% of unusable windows while restricting false rejections to 3.1% (confusion matrix: [263 / 23 / 727 / 34]). 2D CNN captured 91.6% sensitivity but incurred a higher 7.9% false reject rate. Multimodal Net and LightGBM both reached 85.2% sensitivity with 10.1% false rejection. Zero-shot d1-3B attained 75.1% balanced accuracy (62.3% sensitivity, 12.0% false rejection). Figure 1 displays the complete cross-validation ROC and PR curves with non-interpolated Average Precision baselines.

#### Table 1c: Exploratory Fold-Stratified Youden Threshold Simulation Across Companion Partitions
*Exploratory threshold simulation across BUT QDB cross-validation partitions. For each outer fold k, τ* was determined from the pooled out-of-fold predictions of the four companion folds (12 patients) via Youden's J = Sensitivity + Specificity - 1 and evaluated on fold k's held-out test partition. The reported Mean τ* and Median τ* summarize the five fold-specific thresholds; each held-out test sample was evaluated strictly against its corresponding fold-specific companion threshold τ*. Critical Note: Because companion-fold predictions were generated by checkpoints whose training sets included fold k's patients, this analysis represents an exploratory companion-partition sensitivity simulation across cross-validation splits rather than an independent prospective inner-validation evaluation; it should not be interpreted as prospective deployment performance.*

| Model Architecture | Fold Youden Thresholds (τ*) | Mean τ* | Median τ* | OOF Diagnostic Matrix [TP / FP / TN / FN] | Balanced Acc. | Sensitivity | False Reject Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D CNN (Raw Signal)** | [0.020, 0.020, 0.059, 0.020, 0.020] | 0.028 | 0.020 | [263 / 23 / 727 / 34] | **92.7%** | 88.6% | **3.1%** |
| **2D CNN (Rendered Image)** | [0.024, 0.024, 0.118, 0.024, 0.024] | 0.043 | 0.024 | [272 / 59 / 691 / 25] | 91.9% | 91.6% | 7.9% |
| **SinglePass Multimodal Net** | [0.012, 0.019, 0.308, 0.012, 0.012] | 0.073 | 0.012 | [253 / 76 / 674 / 44] | 87.5% | 85.2% | 10.1% |
| **Rule-Based SQI Heuristic** | [0.329, 0.331, 0.352, 0.331, 0.334] | 0.335 | 0.331 | [268 / 64 / 686 / 29] | 90.9% | 90.2% | 8.5% |
| **LightGBM (SQI)** | [0.045, 0.045, 0.183, 0.045, 0.268] | 0.117 | 0.045 | [253 / 76 / 674 / 44] | 87.5% | 85.2% | 10.1% |
| **Logistic Regression (SQI)** | [0.400, 0.400, 0.400, 0.284, 0.676] | 0.432 | 0.400 | [222 / 64 / 686 / 75] | 83.1% | 74.7% | 8.5% |
| **Independent SQI MLP** | [0.510, 0.510, 0.559, 0.477, 0.569] | 0.525 | 0.510 | [227 / 84 / 666 / 70] | 82.6% | 76.4% | 11.2% |
| **LiquidAI d1-3B (Zero-Shot)*** | [0.438, 0.438, 0.469, 0.469, 0.469] | 0.457 | 0.469 | [185 / 90 / 660 / 112] | 75.1% | 62.3% | 12.0% |

![Figure 1: BUT QDB Cross-Validation Curves](../figures/figure_1_butqdb_combined_roc_pr.png)
*Figure 1: Patient-independent cross-validation performance on BUT QDB (quota-based sampling of 1,047 windows, 15 subjects). (A) Receiver Operating Characteristic (ROC) curves with chance baseline (AUROC = 0.500). (B) Precision-Recall curves with prevalence baseline (AP = 0.284). Average Precision (AP) strictly computed via non-interpolated scikit-learn metrics [9].*

### 4.2 External Out-of-Distribution Transfer to CinC 2011 Set A

To test external out-of-distribution generalization, models trained on the complete 1,047-window BUT QDB development cohort were transferred zero-shot to the PhysioNet/Computing in Cardiology Challenge 2011 Set A (150 balanced records, Lead II). As detailed in Section 2.2, this evaluation involves an inherent clinical task mismatch: models evaluated strictly on single-lead Lead II waveforms are benchmarked against ground-truth labels reflecting overall 12-lead diagnostic acceptability. Table 2 and Figure 2 present the transfer performance. Vision-based architectures demonstrated superior ranking retention: SinglePass Multimodal Net and unimodal 2D CNN preserved the highest target AUROCs (0.763 and 0.761) and Average Precisions (0.797 and 0.757). At default threshold τ = 0.50, both models maintained low false rejection rates (4.0%), but this came at the expense of sensitivity, correctly identifying fewer than half of unusable recordings (44.0% and 42.7%). Raw waveform 1D CNN yielded an AUROC of 0.718 (AP 0.670) with 68.0% sensitivity and 21.3% false rejection. Remarkably, tabular logistic regression and MLP models experienced catastrophic feature shift, collapsing to chance discrimination (AUROC 0.500).

#### Table 2: Cross-Dataset Transfer Benchmark on CinC 2011 Set A (150 Records)
*Positive Class = Unacceptable (75 records), Negative Class = Acceptable (75 records). Models evaluated strictly on Lead II signals zero-shot without target domain fine-tuning. Note: Reference labels reflect 12-lead diagnostic acceptability whereas models evaluated Lead II.*

| Model Architecture | Transfer AUROC | Transfer AP | Brier Score | Fixed τ = 0.50 [TP / FP / TN / FN] | Sensitivity | False Reject Rate | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SinglePass Multimodal Net** | **0.763** | **0.797** | 0.286 | [33 / 3 / 72 / 42] | 44.0% | **4.0%** | 70.0% |
| **2D CNN (Rendered Image)** | 0.761 | 0.757 | 0.301 | [32 / 3 / 72 / 43] | 42.7% | **4.0%** | 69.3% |
| **LightGBM (SQI)** | 0.738 | 0.780 | 0.334 | [3 / 1 / 74 / 72] | 4.0% | **1.3%** | 51.3% |
| **1D CNN (Raw Signal)** | 0.718 | 0.670 | 0.264 | [51 / 16 / 59 / 24] | 68.0% | 21.3% | **73.3%** |
| **Rule-Based SQI Heuristic** | 0.701 | 0.761 | **0.238** | [72 / 63 / 12 / 3] | 96.0% | 84.0% | 56.0% |
| **LiquidAI d1-3B (Zero-Shot)*** | 0.636 | 0.711 | 0.389 | [75 / 75 / 0 / 0] | **100.0%** | 100.0% | 50.0% |
| **Logistic Regression (SQI)** | 0.500 | 0.500 | 0.500 | [0 / 0 / 75 / 75] | 0.0% | 0.0% | 50.0% |
| **Independent SQI MLP** | 0.500 | 0.500 | 0.500 | [0 / 0 / 75 / 75] | 0.0% | 0.0% | 50.0% |

Exploratory post-hoc ROC analysis on the CinC target domain (Table 2b) further illuminates the domain transfer boundary. When constraining false rejection to ≤ 4%, Multimodal Net and 2D CNN achieved only 45.33% and 42.67% sensitivity, while 1D CNN captured just 8.00%. Conversely, ensuring 80% unusable sensitivity incurred substantial false reject penalties across all paradigms: 34.67% for 1D CNN, 44.00% for Multimodal Net, 45.33% for 2D CNN, and 73.33% for zero-shot d1-3B. These findings underscore that without target domain threshold recalibration or fine-tuning, conservative screening models either admit substantial corrupted data or reject an unacceptable fraction of usable mobile ECGs.

#### Table 2b: Exploratory Post-Hoc ROC Operating-Point Analysis on CinC 2011 Set A
*Decision Rule: Positive class (Unusable) if p ≥ τ. Target 1: Maximum sensitivity achievable while constraining False Reject Rate (FPR) ≤ 5%. Target 2: Minimum False Reject Rate incurred when guaranteeing Unusable Sensitivity ≥ 80%. Note: These operating points represent exploratory post-hoc sweeps across the target domain to examine empirical feasibility limits; they are not validated deployment thresholds.*

| Model Architecture | Target 1: Sens. (Actual FPR) | Threshold τ | Target 2: FPR (Actual Sens.) | Threshold τ |
| :--- | :---: | :---: | :---: | :---: |
| **SinglePass Multimodal Net** | 45.33% (FPR 4.00%) | 0.26647 | 44.00% (Sens 80.00%) | 0.00060 |
| **2D CNN (Rendered Image)** | 42.67% (FPR 4.00%) | 0.63397 | 45.33% (Sens 82.67%) | 0.00024 |
| **LightGBM (SQI)** | 34.67% (FPR 2.67%) | 0.36818 | 61.33% (Sens 82.67%) | 0.01912 |
| **Rule-Based SQI Heuristic** | 32.00% (FPR 4.00%) | 0.71544 | 68.00% (Sens 81.33%) | 0.53974 |
| **LiquidAI d1-3B (Zero-Shot)*** | 30.67% (FPR 2.67%) | 0.95532 | 73.33% (Sens 80.00%) | 0.85195 |
| **1D CNN (Raw Signal)** | 8.00% (FPR 4.00%) | 0.99996 | 34.67% (Sens 80.00%) | 0.00819 |
| **Logistic Regression (SQI)** | 0.00% (FPR 0.00%) | inf | 100.00% (Sens 100.00%) | 0.00000 |
| **Independent SQI MLP** | 0.00% (FPR 0.00%) | inf | 100.00% (Sens 100.00%) | 0.00000 |

To elucidate the failure mechanism of tabular logistic regression and MLP models on CinC 2011, we conducted a retrospective feature sensitivity audit (Table 2c). In the BUT QDB development cohort, flatline_fraction and saturation_fraction exhibited near-zero variance (σ ~ 10^-5) because sensor saturation was rare. Consequently, standard feature normalization (StandardScaler) divided out-of-distribution mobile phone variations by extremely small standard deviations, generating astronomical z-scores (>10^3) that pushed logistic and sigmoid activations into permanent numerical saturation. When scaled using RobustScaler (interquartile range scaling fit strictly on the BUT QDB development cohort and applied frozen to CinC 2011 to prevent target data leakage) or z-score clipping ([-3, 3]), transfer discrimination was restored to AUROC 0.684 (AP 0.732) for Logistic Regression and AUROC 0.642 (AP 0.688) for the MLP, consistent with scaling-related numerical saturation rather than representational incapacity causing the initial collapse.

#### Table 2c: Impact of Scaling on Linear/MLP Out-of-Distribution Transfer
*Retrospective Sensitivity Analysis on Linear/MLP Scaling under Out-of-Distribution Transfer. Flatline and saturation features exhibited near-zero training variance (σ ~ 10^-5) in BUT QDB, causing logit saturation under standard scaling.*

| Model Architecture | StandardScaler (Default) Transfer AUROC / AP | RobustScaler (IQR) Transfer AUROC / AP | Z-Score Clipping ([-3, 3]) Transfer AUROC / AP |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (SQI)** | 0.500 / 0.500 | 0.684 / 0.732 | 0.671 / 0.718 |
| **Independent SQI MLP** | 0.500 / 0.500 | 0.642 / 0.688 | 0.635 / 0.679 |

![Figure 2: CinC 2011 Transfer Curves](../figures/figure_2_cinc2011_combined_roc_pr.png)
*Figure 2: Out-of-distribution transfer performance on PhysioNet/CinC 2011 Set A (150 balanced records, Lead II). (A) Receiver Operating Characteristic curves with chance baseline (AUROC = 0.500). (B) Precision-Recall curves with prevalence baseline (AP = 0.500).*

### 4.3 Controlled Noise Stress Testing on MIT-BIH NSTDB

Controlled noise stress testing was conducted on the MIT-BIH Noise Stress Test Database (NSTDB) to characterize algorithmic degradation across calibrated signal-to-noise ratios (+24 dB down to -6 dB). As defined in Section 2.3, the evaluation comprises 528 total window instances: 264 active noise evaluations (44 non-overlapping 10-second windows across each of 6 SNR increments on Records 118 and 119) and 264 clean control window evaluations extracted from uncorrupted segments of the same records. Table 3 and Figure 3 depict the predicted usable probability (P(usable) = 1 - P(unusable)) across noise tiers. All evaluated models exhibited a declining usability probability trend across noisy SNR levels (+24 dB down to -6 dB). Deep convolutional networks (2D CNN and Multimodal Net) showed steep sigmoidal degradation, dropping from >0.83 at +24 dB to near-zero (<0.012) under heavy noise (-6 dB). LightGBM exhibited a shallower response curve, maintaining an elevated usable score at -6 dB (0.038); formal tree-path and feature ablation studies would be required to isolate the specific splits driving this plateau.

#### Table 3: Continuous Usability Probability Degradation Across Calibrated SNR Levels
*Usability Probability Degradation Across Calibrated Electrode Motion Noise Levels on MIT-BIH NSTDB (Mean ± Standard Deviation of P(Usable)). Clean Control reflects 264 window evaluations across clean intervals; active noise conditions evaluate 44 10-second windows each across records 118 and 119 (264 active noise evaluations total; 528 overall). Note: Zero-shot LiquidAI/d1-3B and Independent MLP were omitted from the primary benchmark table as out of scope. For d1-3B, evaluating 528 calibration windows incurred high local neural inference runtime (~834 ms per forward pass on Apple M2; >7 minutes per run) rather than token generation costs, as zero output tokens are generated in classification mode. Independent SQI MLP was excluded to maintain a concise, non-redundant primary table focusing on representative tabular (LR, LightGBM), deep neural (1D CNN, 2D CNN, Multimodal Net), and deterministic heuristic paradigms.*

| SNR Level | Logistic Regression (SQI) | LightGBM (SQI) | 1D CNN (Raw Signal) | 2D CNN (Image) | Multimodal Net | Rule Heuristic Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Control (Baseline)** | 0.999 ± 0.001 | 0.985 ± 0.042 | 0.614 ± 0.415 | 0.982 ± 0.071 | 0.978 ± 0.088 | 0.998 ± 0.004 |
| **+24 dB (Low Noise)** | 0.781 ± 0.187 | 0.942 ± 0.091 | 0.711 ± 0.317 | 0.852 ± 0.183 | 0.839 ± 0.198 | 0.998 ± 0.006 |
| **+18 dB** | 0.452 ± 0.231 | 0.812 ± 0.154 | 0.589 ± 0.342 | 0.641 ± 0.245 | 0.628 ± 0.258 | 0.991 ± 0.021 |
| **+12 dB** | 0.218 ± 0.189 | 0.583 ± 0.201 | 0.384 ± 0.311 | 0.389 ± 0.238 | 0.374 ± 0.241 | 0.932 ± 0.084 |
| **+6 dB** | 0.084 ± 0.098 | 0.312 ± 0.176 | 0.182 ± 0.219 | 0.174 ± 0.162 | 0.168 ± 0.169 | 0.714 ± 0.192 |
| **0 dB (Equal Power)** | 0.028 ± 0.041 | 0.119 ± 0.108 | 0.064 ± 0.118 | 0.052 ± 0.079 | 0.049 ± 0.076 | 0.382 ± 0.215 |
| **-6 dB (Severe Noise)** | 0.008 ± 0.015 | 0.038 ± 0.045 | 0.019 ± 0.048 | 0.012 ± 0.021 | 0.011 ± 0.019 | 0.141 ± 0.128 |

Crucially, our audit identified an important nuance regarding 1D CNN behavior across clean control baselines versus low-noise conditions. On clean control windows, the 1D CNN predicted a mean usable probability of 0.614 ± 0.415, which rose to 0.711 ± 0.317 at +24 dB SNR before steadily decreasing to 0.019 at -6 dB. Because the baseline clean control score (0.614) is lower than the +24 dB score (0.711), the claim of strict monotonic degradation from clean controls is factually incorrect for the 1D CNN. Instead, monotonic degradation holds specifically across the active noise SNR hierarchy (+24 dB down to -6 dB). As illustrated in Figure 3, presenting clean control evaluations in a dedicated baseline panel prevents misrepresenting clean intervals as an artificial continuous extension of the SNR curve.

![Figure 3: NSTDB Degradation](../figures/figure_revised_nstdb_degradation_curve.png)
*Figure 3: Predicted usability probability on MIT-BIH Noise Stress Test Database (NSTDB Records 118 & 119). (A) Clean Control baseline evaluations (N=264, Mean ± 1 SD). (B) Degradation across active electrode motion noise SNR tiers (N=44 windows per SNR tier, ±1 SD shaded). Note: 1D CNN usability score begins at 0.614 on clean controls and rises to 0.711 at +24 dB before declining; other models exhibit monotonic decline across noisy tiers.*

### 4.4 Edge Computational Profiling and the Rendering Bottleneck

Edge deployment feasibility was evaluated on Apple Silicon (M2, 8 CPU / 10 GPU cores, 24 GB Unified Memory) with explicit `torch.mps.synchronize()` barriers across 25 paired test trials. Table 4 and Figure 4 present parameter counts, disk weights, execution devices, and latency breakdowns partitioned into preprocessing, forward neural inference, and total end-to-end execution across all eight profiled models. Model parameter counts were programmatically audited from PyTorch model definitions: 1D CNN comprises 167,489 parameters (0.64 MiB disk; CPU execution); 2D CNN comprises 245,697 parameters (0.94 MiB disk; Apple M2 MPS); SinglePass Multimodal Net comprises 123,713 parameters (0.47 MiB disk; Apple M2 MPS); Independent SQI MLP comprises 3,201 parameters (3,009 dense + 192 batch norm; 0.01 MiB disk; CPU execution); Logistic Regression comprises 14 parameters; LightGBM comprises approximately 2,500 decision split nodes across 100 trees; and the expert rule heuristic has 0 parameters. In contrast, the zero-shot LiquidAI/d1-3B model comprises 3.12 billion parameters (5,958 MiB disk).

#### Table 4: Synchronized Hardware Latency, Execution Device, and Weight Storage on Apple M2
*Synchronized Hardware Latency, Execution Device, and Weight Storage on Apple M2 (24 GB Unified Memory). Latencies measured with torch.mps.synchronize() over 25 paired test trials. *LightGBM parameters represent approximate tree decision split nodes across 100 estimators. **1D CNN latency assumes pre-resampled 500 Hz digital signal cached in memory on CPU; it does not include analog-to-digital conversion, transmission, or disk I/O. ***LiquidAI d1-3B evaluated in bfloat16 using system_one mode with zero output tokens. Primary metric for Preprocessing Share is the median of per-run trial ratios (pre_i / total_i) computed from paired raw measurement logs (2D CNN = 93.1%, Multimodal Net = 92.5%, Heuristic = 92.4%, LR = 88.1%, LightGBM = 87.7%, MLP = 94.4%, d1-3B = 7.3%). Ratio of medians (Preproc Median / Total Median) is provided as secondary descriptive metric: 2D CNN = 92.9%, Multimodal Net = 92.1%, Heuristic = 92.4%, LR = 86.8%, LightGBM = 82.6%, MLP = 92.7%, d1-3B = 7.4%. For 2D CNN, preprocessing reflects exclusively Matplotlib waveform rendering (median 64.20 ms); for Multimodal Net, total preprocessing is 68.91 ms (measured directly in each paired trial, comprising 67.01 ms rendering and 1.84 ms SQI extraction), with rendering representing 97.3% of preprocessing; for tabular models, preprocessing represents SQI feature extraction. Note: The median of total times is not algebraically equal to the sum of medians due to sample-level variance across paired test iterations.*

| Model Architecture | Parameters | Disk Weights | Execution Device | Preproc Median (p95) | Forward Median (p95) | Total Median (p95) | Preproc Share (Median of Ratios) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based SQI Heuristic** | 0 | 0.00 MiB | CPU | 2.80 ms (3.41 ms) | 0.22 ms (0.32 ms) | 3.03 ms (3.65 ms) | 92.4% |
| **Logistic Regression (SQI)** | 14 | 0.01 MiB | CPU | 1.34 ms (1.66 ms) | 0.18 ms (0.26 ms) | 1.55 ms (1.92 ms) | 88.1% |
| **LightGBM (SQI)** | 2,500* | 0.05 MiB | CPU | 1.38 ms (2.43 ms) | 0.20 ms (0.48 ms) | 1.67 ms (2.86 ms) | 87.7% |
| **Independent SQI MLP** | 3,201 | 0.01 MiB | CPU | 1.51 ms (2.06 ms) | 0.09 ms (0.25 ms) | 1.63 ms (2.26 ms) | 94.4% |
| **1D CNN (Raw Signal)** | 167,489 | 0.64 MiB | CPU | **0.00 ms**** (0.00 ms) | **1.12 ms** (1.37 ms) | **1.12 ms** (1.37 ms) | **0.0%** |
| **2D CNN (Rendered Image)** | 245,697 | 0.94 MiB | Apple M2 (MPS) | 64.20 ms (151.38 ms) | 4.70 ms (5.64 ms) | 69.07 ms (155.93 ms) | 93.1% |
| **SinglePass Multimodal Net** | 123,713 | 0.47 MiB | Apple M2 (MPS) | 68.91 ms (148.72 ms) | 5.55 ms (7.87 ms) | 74.81 ms (155.84 ms) | 92.5% |
| **LiquidAI d1-3B (Zero-Shot)**** | 3,120,000,000 | 5,958 MiB | Apple M2 (MPS) | 66.16 ms (140.60 ms) | 834.22 ms (848.40 ms) | 898.93 ms (978.57 ms) | 7.3% |

Profiling revealed that for lightweight visual networks, waveform rasterization constitutes the primary computational bottleneck. For the 2D CNN, Matplotlib rendering required a median of 64.20 ms (p95: 151.38 ms), representing 93.1% of total latency (69.07 ms total) when calculated as the median of per-run trial ratios (92.9% as the ratio of medians). For the Multimodal Net, total preprocessing required 68.91 ms (p95: 148.72 ms; 92.5% median of per-run ratios; 92.1% ratio of medians), encompassing both visual rasterization (67.01 ms median) and tabular SQI calculation (1.84 ms median), with waveform rendering accounting for 97.3% of the preprocessing stage. Actual convolutional forward inference on Apple Silicon MPS required only 4.70 ms and 5.55 ms. Conversely, for the 3.12B d1-3B model, forward neural inference dominated (834.22 ms forward; 92.8% of 898.93 ms total latency; preprocessing share 7.3%). Bypassing visual rendering entirely, the unimodal 1D CNN executed on CPU in 1.12 ms median latency (p95: 1.37 ms) on pre-resampled 500 Hz cached inputs with zero preprocessing overhead and a compact 0.64 MiB footprint, establishing a highly favorable accuracy–latency trade-off in the evaluated edge implementation.

![Figure 4: Latency vs AUROC Frontier](../figures/latency_vs_auroc.png)
*Figure 4: Diagnostic discrimination (AUROC) versus measured end-to-end inference latency on edge hardware (log scale). Marker positions reflect measured median execution times on Apple M2 hardware.*

---

## 5. Discussion, Limitations & Reporting Guidelines

### 5.1 Empirical Synthesis by Research Question

**RQ1 (Unimodal vs. Multimodal Fusion):** On the patient-independent BUT QDB benchmark (quota-based sampling of 1,047 windows), unimodal convolutional architectures defined the high discrimination frontier. The unimodal 2D CNN achieved AUROC 0.985 (AP 0.964), while the unimodal 1D CNN achieved AUROC 0.977 (AP 0.958) with 93.3% sensitivity at 4.3% false reject rate under exploratory ROC analysis (88.6% sensitivity at 3.1% false reject under companion-partition exploratory threshold simulation). The SinglePass Multimodal Net matched the 1D CNN in discrimination (AUROC 0.977, AP 0.945) and provided comparable cross-dataset transfer resilience (CinC 2011 AUROC 0.763 vs. 0.718 for 1D CNN). However, within the source domain, multimodal fusion provided no additive accuracy gain over unimodal vision; feature redundancy between high-resolution visual representations and numerical SQIs provides a plausible explanation for this plateau.

**RQ2 (Zero-Shot Edge VLM vs. Specialized Baselines):** The genuine 3.12-billion-parameter `LiquidAI/d1-3B` multimodal decision model [5, 6], evaluated zero-shot via official `system_one` inference without task-specific training under the specified prompt and test datasets, did not match compact, task-specialized supervised architectures in discrimination. Its AUROC of 0.842 on BUT QDB trailed all supervised models (0.941-0.985) and the expert rule baseline (0.977). Clinically, d1-3B achieved 52.2% sensitivity when false rejection was constrained to 5%; achieving 80% sensitivity incurred an unviable 36.3% false rejection rate under post-hoc exploration. On CinC 2011 Set A, d1-3B degraded to AUROC 0.636, classifying 100% of signals as unusable at default τ = 0.50. These findings demonstrate that the evaluated zero-shot configuration of LiquidAI/d1-3B under the specified prompt and test datasets lags compact task-specific models under the tested prompts and datasets; whether alternative prompting strategies, larger foundation architectures, or domain-specific fine-tuning bridge this gap remains an open empirical question.

**RQ3 (Cross-Dataset Transfer and Dataset Shift):** External evaluation on CinC 2011 Set A revealed differences in domain transfer across modalities [4, 15]. Multimodal fusion and visual 2D CNN models preserved higher target AUROC values (0.763 and 0.761) and maintained low false rejection rates (4.0% at τ = 0.50). However, this low rejection rate came at the severe cost of false acceptance: both models accepted over half of unusable target recordings (unusable sensitivity 44.0% and 42.7%), illustrating the clinical trade-off between false alarm reduction and corrupted data penetration. Raw 1D signal models transferred less smoothly (21.3% false reject rate, AUROC 0.718). Tabular logistic regression and MLP models suffered catastrophic covariate shift due to near-zero training variance of flatline/saturation features in BUT QDB, causing logit saturation (AUROC 0.500) that required robust scaling to mitigate.

**RQ4 (Computational Complexity and the Rendering Bottleneck):** In lightweight vision-based pipelines, input preprocessing constitutes the primary computational bottleneck. Input preprocessing consumed 64.20–68.91 ms (92.5%–93.1% of end-to-end latency for 2D CNN and Multimodal Net [ratio of medians: 92.1%–92.9%], of which waveform rendering accounts for 97.3%–100%), whereas neural forward passes required only 4.70–5.55 ms. Conversely, for the 3.12B d1-3B model, forward neural inference dominated execution (834.22 ms forward pass; 92.8% of total 898.93 ms latency). The unimodal 1D CNN operating on standardized cached signals bypassed rasterization entirely, delivering 1.12 ms total latency on CPU and 0.64 MiB disk footprint, establishing a favorable accuracy–latency trade-off in the evaluated implementation.

### 5.2 Methodological Integrity, Dataset Shift, and Shortcut Learning
In clinical artificial intelligence, high retrospective performance frequently fails to generalize across hospital environments, acquisition devices, and temporal epochs [4, 15]. A major contributor to generalization failure is shortcut learning [10], wherein deep neural networks exploit spurious confounders—such as patient-specific baseline drift, hardware-specific noise signatures, or institutional acquisition artifacts—rather than genuine clinical pathophysiological features. Our experimental design mitigates shortcut learning by enforcing patient-disjoint GroupKFold cross-validation [8] and testing cross-dataset transfer on distinct mobile phone hardware [2, 7]. Nevertheless, as noted by Geirhos et al. [10], deep models remain susceptible to domain-specific inductive biases, emphasizing the need for rigorous external validation before clinical deployment.

### 5.3 Study Limitations
Several limitations must be transparently acknowledged:
1. **Limited Patient-Level Diversity:** The primary development cohort (BUT QDB) comprises only 15 distinct subjects, with unusable Class 3 noise present in only 7 individuals. While GroupKFold prevents patient identity leakage [8], it cannot overcome the limited patient-level diversity of small clinical cohorts.
2. **Quota Sampling:** Quota-based sampling (750 usable and 297 unusable windows) was used to establish an equitable benchmark, but this alters the empirical prevalence of noise compared to continuous naturalistic monitoring.
3. **Target Label Mismatch:** The transfer dataset (CinC 2011) features expert labels reflecting overall 12-lead diagnostic acceptability, whereas our models evaluated solely Lead II. A recording with clean Lead II but corrupted peripheral leads represents an unavoidable label mismatch [4].
4. **Single Zero-Shot Prompt and Architecture:** Zero-shot evaluation was limited to `LiquidAI/d1-3B` using a single task prompt without few-shot examples or fine-tuning.
5. **Input Modality Asymmetry:** The Multimodal Net received visual images and 13 numerical features, whereas d1-3B received solely images and text prompts; their performance gap reflects differing input modalities and training protocols rather than model capacity alone.
6. **Hardware-Dependent Latency:** Reported latencies reflect Apple M2 hardware and Python/Matplotlib rendering; compiled C++ or embedded graphics implementations would alter rendering overhead. Furthermore, the 1.12 ms latency of the 1D CNN assumes standardized cached signals on CPU and does not represent complete acquisition-to-decision latency.
7. **Exploratory Post-Hoc Thresholds:** Operating points in Table 1b and Table 2b represent exploratory sweeps across test distributions and are not prospectively validated deployment thresholds.
8. **Fixed Random Seed and Multi-Seed Variance:** All neural and tabular models were trained and evaluated under a single fixed initialization seed (seed 42); multi-seed variance and stochastic training stability across random initializations remain unassessed.

### 5.4 Alignment with Clinical AI Auditing Guidelines
To ensure transparency, this study was structured with reference to contemporary clinical AI auditing frameworks. We positioned TRIPOD+AI [11] as a reporting guideline and PROBAST+AI [12] as a risk-of-bias assessment tool to guide model description, patient-independent validation, and risk-of-bias mitigation. For visual waveform components and diagnostic accuracy, Relevant items from these frameworks informed the reporting audit; full checklist-based compliance was not established [13, 14]. We frame these standards not as claims of retrospective perfection, but as structured auditing criteria governing reproducible clinical AI benchmarking.

---

## 6. Conclusion

This study conducted a benchmark of eight models across six methodological paradigms for single-lead ECG quality triage across patient-independent development, external cross-dataset transfer, and calibrated noise stress testing. Our empirical findings demonstrate that:
1. Lightweight 1D convolutional networks operating directly on raw biosignals provide a favorable accuracy–latency trade-off in the evaluated implementation (AUROC 0.977, 88.6% sensitivity at 3.1% false reject under companion-partition threshold simulation), ultra-low latency on cached signals on CPU (1.12 ms), and zero rendering overhead.
2. Visual and multimodal representations provide modest generalization benefits during cross-dataset transfer (AUROC 0.763 vs. 0.718) and preserve low false rejection (4.0%), but accept over half of corrupted target recordings and incur a substantial 64.2–68.9 ms waveform rasterization bottleneck.
3. The evaluated zero-shot configuration of LiquidAI/d1-3B under the specified prompt and test datasets underperformed compact task-specific models and exhibited threshold instability under transfer; whether domain-specific fine-tuning or prompt optimization bridges this gap remains an open research question.

---

## 7. Bibliography

[1] Nemcova, A., Jordanova, I., Vondra, V., Smisek, R., & Vitek, M. (2020). Brno University of Technology ECG Quality Database (BUT QDB) (version 1.0.0). *PhysioNet*. https://doi.org/10.13026/kah4-0w24.  
[2] Silva, I., Moody, G. B., & Celi, L. (2011). Improving the quality of ECGs collected using mobile phones: The PhysioNet/Computing in Cardiology Challenge 2011. *Computing in Cardiology*, 38, 273–276.  
[3] Moody, G. B., Muldrow, W. E., & Mark, R. G. (1984). A noise stress test for arrhythmia detectors. *Computers in Cardiology*, 11, 381–384. https://physionet.org/physiobank/database/nstdb/.  
[4] Subbaswamy, A., & Saria, S. (2020). From development to deployment: dataset shift, causality, and shift-stable models in health AI. *Biostatistics*, 21(2), 345–352. https://doi.org/10.1093/biostatistics/kxz041.  
[5] Liquid AI. (2026). LiquidAI/d1-3B: Open-weight edge multimodal decision model. *Hugging Face Model Hub*. Commit: `051bcc464b01b9f92942b364d9586b0ef5912432`. https://huggingface.co/LiquidAI/d1-3B.  
[6] Liquid AI. (2026). Liquid Foundation Models: d1-3B Architecture, system_one mode, and choice-probability API specification. *Liquid AI Documentation*. https://docs.liquid.ai/lfm/models/d1-3b.  
[7] Computing in Cardiology & PhysioNet. (2011). Improving the Quality of ECGs Collected Using Mobile Phones: Challenge Protocol and Diagnostic Reference Standards. *PhysioNet Challenge Archive*. https://moody-challenge.physionet.org/2011/.  
[8] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.  
[9] scikit-learn developers. (2024). Precision-recall metrics and non-interpolated average precision (average_precision_score v1.5+). *Scikit-learn Documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html.  
[10] Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665–673. https://doi.org/10.1038/s42256-020-00257-z.  
[11] Collins, G. S., Moons, K. G. M., Dhiman, P., Navarro, C. A., Beam, A. L., Boyd, K. A., ... & TRIPOD+AI Group. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use artificial intelligence. *BMJ*, 385, e078378. https://doi.org/10.1136/bmj-2023-078378.  
[12] Moons, K. G. M., Damen, J. A. A. G., Kaul, V., et al. (2025). PROBAST+AI: a tool to assess risk of bias and applicability of prediction model studies using artificial intelligence. *BMJ*, 388, e082505. https://doi.org/10.1136/bmj-2024-082505.  
[13] Tejani, A. S., Klontzas, M. E., Gatti, A. A., et al. (2024). Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update. *Radiology: Artificial Intelligence*, 6(4), e240300. https://doi.org/10.1148/ryai.240300.  
[14] Sounderajah, V., Guni, A., Liu, X., Collins, G. S., et al. (2025). The STARD-AI reporting guideline for diagnostic accuracy studies using artificial intelligence. *Nature Medicine*, 31(10), 3283–3289. https://doi.org/10.1038/s41591-025-03953-8.  
[15] Guo, Z., Ding, Y., Yue, X., & Hu, X. (2022). Evaluation of domain generalization and adaptation on improving model robustness to temporal dataset shift in clinical medicine. *Scientific Reports*, 12, 2726. https://doi.org/10.1038/s41598-022-06484-1.  
[16] Liu, F., Xia, S., Wei, S., Chen, L., Ren, Y., et al. (2022). Wearable Electrocardiogram Quality Assessment Using Wavelet Scattering and LSTM. *Frontiers in Physiology*, 13, 905447. https://doi.org/10.3389/fphys.2022.905447.  
[17] Halvaei, H., Svennberg, E., Sörnmo, L., & Stridh, M. (2021). Identification of Transient Noise to Reduce False Detections in Screening for Atrial Fibrillation. *Frontiers in Physiology*, 12, 672875. https://doi.org/10.3389/fphys.2021.672875.  
[18] Hu, Y., Liu, Q., Zhou, Z., Xu, W., & Tang, H. (2025). A Paralleled Multi-Task Learning-Based Framework for Single-Lead ECG Fine-Grained Noise Localization, Denoising and Signal Quality Assessment. *Sensors*, 25(23), 7152. https://doi.org/10.3390/s25237152.  
[19] Clifford, G. D., Behar, J., Li, Q., & Rezek, I. (2012). Signal quality indices and data fusion for determining clinical acceptability of electrocardiograms. *Physiological Measurement*, 33(9), 1419–1433. https://doi.org/10.1088/0967-3334/33/9/1419.  
[20] Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230–236. https://doi.org/10.1109/TBME.1985.325532.  
[21] Orphanidou, C., Bonnici, T., Charlton, P., Clifton, D., Vallance, D., & Tarassenko, L. (2015). Signal-quality indices for the electrocardiogram and photoplethysmogram: Derivation and applications to wireless monitoring. *IEEE Journal of Biomedical and Health Informatics*, 19(3), 832–838. https://doi.org/10.1109/JBHI.2014.2338351.  
[22] Satija, U., Ramkumar, B., & Manikandan, M. S. (2018). A review of signal quality evaluation technologies for electrocardiogram signals. *IEEE Reviews in Biomedical Engineering*, 11, 36–52. https://doi.org/10.1109/RBME.2018.2810957.  
"""

paths = [
    "/Volumes/KIOXIA/Projects/d13b-ecg-sqi/manuscript/final_manuscript.md",
    "/Volumes/KIOXIA/Projects/d13b-ecg-sqi/first_draft.md",
    "/Users/mehmetkuntalp/.gemini/antigravity/brain/fd0d5598-c574-4cf6-a79a-861d3e51441d/Final_Manuscript.md"
]

for p in paths:
    with open(p, "w") as f:
        f.write(md_content)
    print(f"Synced markdown to: {p}")
