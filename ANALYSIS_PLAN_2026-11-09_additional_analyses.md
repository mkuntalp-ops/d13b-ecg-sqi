# Analysis plan: additional analyses (written 2026-11-09, before any model score on PTB-XL was produced)

Requested by the author on 2026-11-09 ("ek analizleri de yapalım"), after the proposal to test the frozen probe on an untouched dataset, measure the effect of quality gating on a downstream task, and explain the saturation of the multimodal network.

What had been looked at when this plan was written: the PTB-XL metadata columns and the value counts of the four noise columns. No PTB-XL signal had been processed by any model of this study other than the 501-record pilot named below. Plan changes after this date are listed at the end of this file, with their date and reason.

## A. Frozen external test on PTB-XL

**Question.** Does the d1-3B image-token probe, fitted on BUT QDB only and unchanged, rank PTB-XL records with annotated global noise above records with no noise annotation, and does it do so better than the 2D CNN?

**Why PTB-XL, and its limit.** It was not used to develop any model here. Its noise columns mark the presence of noise in recordings that were still interpreted, not unusability. The outcome is therefore a proxy for the study target. Results are described as discrimination of annotated noise.

**Exposure.** A 501-record pilot from an early stage of the project exists (`/Volumes/KIOXIA/Datasets/ptb-xl/records500`, `ecg_images/ptbxl_pilot`). These records are excluded. A separate manuscript by the author uses PTB-XL features for a different task; no model or threshold from it is used here.

**Unit and sampling.** One record per patient. PTB-XL v1.0.3 metadata, 21,799 records. Random sampling, seed 42, from the metadata only, of 300 records per stratum (900 records, 10,800 lead windows). Sampling is limited to this number because d1-3B needs about 1 s per window. A patient enters at most once across strata.

**Strata.**
- `global`: `static_noise` or `burst_noise` contains "alles" or "noisy recording" (case-insensitive).
- `partial`: any entry in `static_noise`, `burst_noise` or `electrodes_problems`, and not `global`.
- `clean`: no entry in `static_noise`, `burst_noise`, `baseline_drift` or `electrodes_problems`.
- Records with `baseline_drift` only are excluded.

**Input.** The 500 Hz signals, each of the 12 leads processed with the pipeline of the CinC 2011 12-lead analysis (same units, renderer, SQI extractor, prompt).

**Models, frozen.** All fitted on the 988 BUT QDB windows with the existing code and seed 42, no refitting on or tuning for PTB-XL: logistic regression, LightGBM, MLP, 1D CNN, 2D CNN, multimodal net, heuristic, zero-shot d1-3B, d1-3B image probe, HuBERT-ECG probe (each lead in the lead II slot). TabPFN is not included.

**Recording score.** Maximum and mean over the 12 leads (lead II is reported descriptively).

**Primary outcome and test.** `global` versus `clean`, AUROC. Two-sided DeLong tests of d1-3B image probe versus 2D CNN, under the maximum and under the mean (same rule for both models), Holm over these two. The hypothesis is supported only if the probe's AUROC is higher with p_Holm < 0.05 under both rules. Any other pattern is reported as not supported.

**Secondary.** (1) Probe versus each of the other nine models, both rules, primary outcome, Holm over the 18 tests. (2) `partial` versus `clean` for all models: AUROC with 95% bootstrap CI, no tests. (3) AUROC without records having a constant lead (descriptive).

**Reporting.** The result is reported whichever way it falls, including in the abstract if the probe is not supported. Sampled prevalence is artificial, so no PPV and no threshold metrics are interpreted.

## B. Effect of quality gating on rhythm classification (CinC 2017)

**Question.** If recordings are rejected by a quality score, does the macro-F1 of a separately trained rhythm classifier improve on the recordings that remain?

**Data.** The 8,234 non-noisy recordings among the 8,512 analysed. "Too noisy" recordings have no rhythm label and cannot enter this analysis, so the main effect of a gate, removing them, is not measured. CinC 2017 has no patient identifiers, so independence of recordings cannot be verified.

**Downstream model.** A 3-class (N, A, O) 1D CNN on the existing 10-s windows, fixed architecture, not tuned, 5-fold CV stratified by rhythm, folds by recording, seed 42. Recording probability is the mean over its windows. Out-of-fold predictions only.

**Quality scores.** The recording-maximum scores of the primary CinC 2017 analysis (models fitted on BUT QDB only). Quality models do not see rhythm labels.

**Endpoint.** Macro-F1 over N, A, O on retained recordings after rejecting the 10% with the highest score (primary); 5% and 20% (secondary). Rejecting a fixed fraction avoids choosing a threshold.

**Comparison.** Change in macro-F1 relative to no rejection, against random rejection of the same fraction (1,000 repetitions). 95% bootstrap CI over recordings (2,000 resamples) for each model's change minus the mean random change. The d1-3B probe minus 2D CNN, and minus HuBERT-ECG probe, is reported with CI, without a test. Gating may also remove recordings with hard rhythms rather than poor signals; this analysis cannot separate the two.

## C. Saturation of the multimodal net on CinC 2011

Descriptive diagnosis, with no change to any primary result. The multimodal net gives the same score (1.0) for all 998 CinC 2011 records. Report the standardised inputs on CinC 2011 relative to BUT QDB, and rescore CinC 2011 with (i) the flatline input replaced by the quantization-insensitive measure and (ii) the image-only and SQI-only streams, to identify which input causes it.

## Stopping rule

No model, threshold, sampling or outcome definition will be changed after the PTB-XL or gating results are seen. Any further analysis prompted by those results will be labelled as exploratory.

## Changes after this plan was written

No change to the sampling, models, endpoints or tests. One code correction, made before any PTB-XL analysis output existed: the reproduction check at the start of `--score-others` failed because the CinC 2011 lead II table has no `lead_idx` column; the column was added for that check (2026-11-09, after the d1-3B scoring and before the other models were scored). The `--analyse` step was run once. Log of the failed attempt: `results/v2_2026-11/ptbxl_frozen_score_others_failed_missing_lead_idx.log`.
