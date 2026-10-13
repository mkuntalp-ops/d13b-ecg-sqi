# Response to Reviewer 2

**Manuscript:** Zero-shot and task-specific models for ECG usability screening: patient-independent performance, cross-dataset transfer, and computational cost

We thank the reviewer for a careful reading. The review groups its concerns into open clinical questions, methodological assumptions and missing external data. We answer each point below, state what was changed, and give the location in the revised manuscript. Where a concern could be tested with data available to us, we ran the analysis; where it could not, we say so.

While preparing this revision we also re-audited the code and the stored results. That audit found one error that affected results and several descriptive errors. They are listed in the final section, together with what changed.

---

## 1. Unanswered critical questions and clinical limits

### 1.1 Limited patient diversity (15 subjects; Class 3 in only 7)

**Reviewer.** BUT QDB contains 15 subjects and only 7 have unusable (Class 3) recordings, which limits generalization despite subject-independent validation.

**Response.** We agree that 15 subjects cannot be enlarged within BUT QDB. We addressed the concern in two ways.

1. *Second external cohort.* We added the PhysioNet/Computing in Cardiology Challenge 2017 training set [22] (Section 2.5, Section 4.7, Tables 8 and 8b): 8,512 single-lead handheld recordings (AliveCor), 278 of them (3.3%) labelled "too noisy to classify". All models were trained on BUT QDB only and applied frozen. With the pre-specified recording-level maximum, the AUROCs were 0.773–0.887 (rule heuristic 0.887 [0.867–0.905]; LightGBM 0.880; 2D CNN 0.850; 1D CNN 0.773). The ranking differs from BUT QDB: the SQI models and the heuristic, not the 1D CNN, ranked best. This is now stated in the abstract and the conclusion.
2. *Subject-level heterogeneity.* Table 1f now reports the sensitivity for each of the seven subjects with Class 3 windows. The pooled 1D CNN sensitivity of 76.8% ranges from 14% (subject 100) to 100% across subjects, and no model detected more than 53% of subject 111's unusable windows. Limitation 1 was extended accordingly.

### 1.2 Single-lead input vs 12-lead reference labels (CinC 2011)

**Reviewer.** Models score lead II, while CinC 2011 labels judge the whole 12-lead recording. A recording with a clean lead II but noisy other leads is unavoidably misclassified.

**Response.** We quantified this mismatch.

- *All 12 leads* (Section 2.2, Table 2e): we scored every lead of the 565 CinC records used in the transfer analyses and aggregated per record. Averaging the 12 lead scores raised AUROC for every trained model except logistic regression. Over the five random subsets, the 2D CNN rose from 0.781 (lead II) to 0.852, LightGBM from 0.750 to 0.877, the 1D CNN from 0.707 to 0.801 and the multimodal net from 0.668 to 0.841. The maximum over leads helped the rule heuristic (0.683 to 0.860) and d1-3B (0.636 to 0.800 on the reported subset). Part of the CinC 2011 transfer gap is therefore the label mismatch. Because these aggregation rules were chosen after the main analysis, we present them as a sensitivity analysis.
- *A single-lead label.* CinC 2017 (Section 4.7) labels the same single lead that is scored, so it is free of this mismatch.

### 1.3 Prevalence set by quota sampling (28.4%)

**Reviewer.** The 28.4% noise share is set by quotas and does not reflect continuous ambulatory monitoring.

**Response.** We agree, and we now report results at natural prevalences.

- *Natural prevalence in BUT QDB* (Section 2.1): among all 32,956 annotated 10-s windows of the 18 records, 16.0% are Class 3. However, 4,734 of the 5,285 Class 3 windows come from one record (105001). Without it the share is 2.8%, and the median record has none.
- *Re-weighted metrics* (Table 1e): at 2.8%, the CNNs keep an average precision of 0.81–0.83 and a PPV of 0.71–0.77 at τ = 0.50, with 2.4–3.4 false rejections per hour of usable signal. The SQI models fall to an AP of 0.55–0.73 (15–31 false rejections per hour), and zero-shot d1-3B to 0.35.
- *External natural prevalence* (Table 8): on CinC 2017 (3.3%) the AP was 0.06–0.23 for every model, and at τ = 0.50 no model combined useful sensitivity with useful precision (for example, CNNs flagged 89–92% at a PPV of about 0.08).

Limitation 2 was rewritten.

---

## 2. Logical leaps and methodological assumptions

### 2.1 Feature meaning changes between datasets

**Reviewer.** The failure of the SQI features is not only a scaling issue: the high-frequency noise ratio reverses (AUROC 0.777 in BUT QDB vs 0.329 in CinC 2011).

**Response.** We agree and have extended the analysis.

- Table 2g gives the univariate AUROC of every SQI in BUT QDB, CinC 2011 and CinC 2017. The high-frequency noise, powerline and zero-crossing ratios and the flatline fraction change direction between BUT QDB and CinC 2011. bSQI keeps its direction but loses most of its strength there (0.030 vs 0.305), and becomes strongly informative again in CinC 2017 (0.124). This is consistent with the SQI models transferring well to CinC 2017 and poorly to CinC 2011.
- We also identified a concrete mechanism for one feature (Section 3.1, Limitation 9). The flatline fraction counts consecutive equal samples. BUT QDB windows are resampled from 1,000 Hz and take continuous values (mean 6.7 × 10^-5). CinC 2011 lead II is used at its native 500 Hz with 5 µV quantization steps, so even acceptable CinC records have a median flatline fraction of 0.28. Much of the CinC "flatline" signal is therefore quantization, not lead disconnection.

### 2.2 Zero-shot d1-3B: capacity or missing adaptation?

**Reviewer.** d1-3B was neither fine-tuned nor tested few-shot, so it is unclear whether it lags because of capacity or lack of domain adaptation.

**Response.** We tested this directly with linear probes on frozen d1-3B representations (Section 4.6, Table 7b). During the primary forward pass we captured two 2,048-d representations: the hidden state at the answer position, and the mean of the image tokens. We then trained an L2 logistic regression on each with the same subject folds as Table 1.

| d1-3B representation | BUT QDB AUROC [95% CI] | CinC 2011 reported subset | CinC 2011 random subsets | CinC 2017 subset |
|---|---|---|---|---|
| Zero-shot (primary configuration) | 0.852 [0.717–0.944] | 0.636 | 0.583–0.723 | 0.739 |
| Probe on image tokens | 0.979 [0.941–0.996] | 0.777 | 0.756–0.907 | 0.910 |
| Probe on decision state | 0.974 [0.930–0.997] | 0.475 | 0.411–0.482 | 0.883 |

The frozen image representation already separates usable from unusable windows about as well as the CNNs in-domain (0.979 vs 0.989). It also transfers better than every supervised model to both external datasets. The gap is therefore one of adaptation and read-out, not of representational capacity. The decision-state probe also encodes the SQI text and inherits its reversal on CinC 2011.

We also report three prompt configurations (Section 4.6, Table 7: primary, image only, reworded question). The prompt changed the operating point and the BUT QDB AUROC (0.852–0.913), but not the lag behind the CNNs. Fine-tuning and few-shot prompting were not run; Limitation 4 says so.

### 2.3 Latency assumptions (cached inputs; Matplotlib rendering)

**Reviewer.** The 1D CNN latency (1.47 ms) was measured on pre-resampled cached signals and excludes acquisition and preprocessing. The 2D CNN rendering cost (60.2 ms) depends on Python/Matplotlib.

**Response.** We added an extended profile (Section 4.4, Table 4b; three runs of 25 windows, run alone), with three components:

- *Raw-to-decision:* starting from the native 1,000 Hz BUT QDB window in memory, through unit conversion, resampling to 500 Hz and the forward pass. Conversion and resampling add about 0.1 ms, so the 1D CNN takes 1.41 ms on the CPU and 2.90 ms on MPS from the native signal.
- *CPU vs MPS:* forward passes of the 1D CNN, 2D CNN and multimodal net on both devices. The 1D CNN is faster on the CPU (1.30 ms) than on MPS (2.80 ms). The 2D CNN and the multimodal net are faster on MPS (2.92 and 3.17 ms) than on the CPU (5.11 and 4.39 ms).
- *Rendering:* a direct raster renderer (PIL; same canvas, scaling and grid) against Matplotlib. The raster renderer draws a strip in 0.63 ms, against 58.4 ms for Matplotlib, so the rendering bottleneck belongs to the implementation rather than to image-based input as such. Only the timing was compared; the image models were trained on Matplotlib strips and were not re-evaluated on raster strips.

Acquisition time on a device was not measured and is stated as a limitation (Limitation 6).

We also corrected the hardware description: the latencies were measured on an Apple M4 Mac mini (10-core CPU, 10-core GPU, 16 GB), not on an M2 with 24 GB as stated before.

---

## 3. External data needed to complete the study

### 3.1 Large multi-lead datasets with lead-wise labels

**Reviewer.** Multi-lead datasets from hundreds of patients with lead-wise labels are needed to establish clinical reliability.

**Response.** We agree that this is the decisive external test, and we could address it only in part.

- CinC 2017 adds 8,512 recordings from a different device at natural prevalence (Section 4.7), but it is single-lead.
- The 12-lead analysis of CinC 2011 (Table 2e) uses record-level labels.
- We considered PTB-XL, which annotates noise per lead. However, its annotations mark the presence of baseline drift, static or burst noise and electrode problems in recordings that were still interpreted, not unusability. Only a 501-record subset was available to us locally. We therefore did not present it as a usability benchmark.

A multi-lead dataset with lead-wise usability labels remains future work (Limitation 3).

### 3.2 Prospectively validated thresholds

**Reviewer.** The Target 1 and Target 2 operating points were chosen retrospectively on the test sets.

**Response.** We added two analyses in which the thresholds never see the evaluated data (Section 3.5; Tables 1c and 2f).

1. *Nested selection on BUT QDB.* Within each outer fold, the 12 training subjects were split into four inner folds by subject. Every trainable model was retrained on 9 subjects and scored on the other 3. Thresholds were chosen on these inner out-of-fold scores and applied to the held-out subjects. Results (Table 1c):
   - *CNNs:* the nested thresholds reproduced the post-hoc operating points. For the 1D CNN at Target 1, sensitivity was 97.3% at 3.5% FPR (post hoc: 96.0% at 3.5%). For the 2D CNN it was 94.6% at 4.7%.
   - *SQI models:* the post-hoc points were optimistic. LightGBM's Target 1 sensitivity fell from 80.5% to 70.4%. Logistic regression exceeded the FPR limit on the held-out subjects (6.7%). Under Target 2, these models reached 80% sensitivity only at higher FPR (LightGBM 92.9% at 9.5%).
   - *Other models:* the 2D CNN, the multimodal net, the heuristic and d1-3B did not reach 80% sensitivity under nested Target 2 (69.0–78.1%).
2. *Source-selected thresholds for transfer.* Thresholds chosen on the pooled BUT QDB out-of-fold scores were applied frozen to CinC 2011 and CinC 2017. The operating points did not carry over:
   - *2D CNN, Target 1:* 52.0% sensitivity at 10.7% false rejection on CinC 2011, and 96.4% at 50.4% on CinC 2017.
   - *1D CNN, Target 1:* 70.7% at 28.0% on CinC 2011, and 99.6% at 54.5% on CinC 2017.
   - Only the rule heuristic kept false rejection near 5% on CinC 2017, at 43.5–52.9% sensitivity.

Operating points must therefore be re-established on each target population; the conclusion now says so.

The post-hoc operating points of Tables 1b and 2b are kept for reference and are labelled as such. Limitations 7 and 11 now refer to the nested results. A prospective study remains necessary before any threshold is used clinically.

---

## 4. Corrections found during the re-audit

1. **Rendering error (affected results).** The strip renderer inferred the unit from the amplitude and divided any signal exceeding 20 by 1,000. Millivolt signals with large artefacts were therefore drawn as flat lines: 71 CinC 2011 records (all unacceptable; 22 in the reported subset) and 22 NSTDB windows at −6 dB. BUT QDB strips were unaffected (verified pixel-identical). After the fix we recomputed every CinC and NSTDB result that uses strips (2D CNN, multimodal net, rule heuristic, d1-3B).
   - On the reported CinC subset, AUROCs changed by at most 0.001.
   - Across training seeds, one 2D CNN default-threshold sensitivity changed (seed 3: 40.0% to 20.0%).
   - On NSTDB, the earlier "rise at −6 dB" of the image models disappeared (2D CNN 0.115 to 0.012), and the corresponding sentence was removed.
2. **Hardware.** The latency platform is an Apple M4 Mac mini with 16 GB, not an M2 with 24 GB.
3. **Sampling description.** Quotas were applied per subject (up to 50 usable and 60 unusable windows, evenly spaced in time), and 17 of the 18 records contributed windows. Resampling is Fourier-domain (scipy.signal.resample), not polyphase.
4. **Training details.** The following were missing and are now given (Section 3.2):
   - AdamW with learning rate and weight decay of 10^-3;
   - class-weighted loss;
   - batch size 32, 25 epochs and cosine annealing;
   - no early stopping;
   - class-balanced LR and LightGBM;
   - fixed image normalization (mean 0.95, SD 0.15).
5. **Rendering description.** The strips are scaled per strip in amplitude, not at a fixed 10 mm/mV gain.
6. **d1-3B input.** The d1-3B prompt contained a text summary of 10 of the 13 features. An earlier version had stated that d1-3B received no SQIs.
7. **Recording duration.** The 99.39 hours refers to the quality-annotated part of the 478.7 hours of recording.

All tables are generated from the result files by `evaluation/build_manuscript_tables.py` and copied verbatim into the manuscript by `evaluation/sync_manuscript_tables.py`.
