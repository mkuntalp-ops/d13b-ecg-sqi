# TRIPOD+AI checklist (v2 draft, 2026-11-08; updated 2026-11-11 after the code audit and the reframing, and 2026-11-12 after the independent scientific review)

**Checklist source.** Collins GS et al., TRIPOD+AI statement, BMJ 2024;385:e078378 (doi:10.1136/bmj-2023-078378), Table 2: 27 items with sub-items. The item topics below are short paraphrases; consult the published checklist for the full wording.

**Study type.** Development (D) and evaluation (E): model development with patient-independent internal validation, and external evaluation on three datasets.

**Locations.** Section numbers refer to the accompanying main manuscript; Tables S1–S47, Supplementary Methods S-M1 to S-M6 and Supplementary Results S-R1 to S-R7 refer to the accompanying supplementary material.

**Status codes.**
- Reported: the item is fully addressed.
- Partly: the item is addressed in part.
- Not reported: the reason is given.
- N/A: not applicable.

| Item | Topic (paraphrase) | Status | Where / note |
|---|---|---|---|
| 1 | Title identifies development/evaluation, population, outcome | Partly | The title names the finding (in-domain performance did not identify the models that ranked best externally), the task (ECG usability models) and the design (benchmark of eleven approaches with subject-grouped development and heterogeneous external datasets). It does not use the words "development" and "evaluation". |
| 2 | Abstract per TRIPOD+AI for Abstracts | Partly | Structured abstract (250 words), checked against the 13 items of TRIPOD+AI for Abstracts (Table 3 of the statement). Reported: 4 (data sources: BUT QDB, CinC 2011, CinC 2017), 7 (model types; subject-grouped cross-validation; models applied unchanged), 8 (AUROC, false rejection rate, sensitivity; calibration is not named), 12 (interpretation). Partly: 1 (title, as item 1 of this checklist), 3 (objective stated; the words development and evaluation are not used), 6 (outcome named as usability / noisy, not defined), 9 (windows, subjects, records and the CinC 2017 prevalence are given; outcome counts for BUT QDB and CinC 2011 are not), 11 (AUROCs without confidence intervals, for lack of space; intervals are in Tables 1, 3 and S28). Not reported: 2 (no background sentence), 5 (eligibility and setting), 13 (the study was not registered; stated in Section 2.6, not in the abstract). Not applicable: 10 (no single final model). |
| 3a | Healthcare context, rationale, existing models | Reported | Section 1 |
| 3b | Target population, intended purpose, intended users | Reported | Section 1: automatic gate before downstream ECG analysis; output used by developers and operators of analysis pipelines to discard a segment or request a new recording; not a replacement for clinical interpretation. |
| 3c | Known health inequalities | Not reported | The background does not discuss known inequalities in ECG signal quality (for example by skin, body habitus or device population). Section 4.1 states only that known inequalities in signal acquisition could not be examined. This item is separate from our subgroup analysis, which was not performed (Section 4.1; item 14). |
| 4 | Development, evaluation or both | Reported | Section 1 (contributions); Section 2.4 |
| 5a | Data sources, rationale, representativeness | Reported | Section 2.1, Figure 1; representativeness in Section 4.1 (limitations) |
| 5b | Dates of data collection | Not reported | Dataset versions and references are given; collection dates are not. |
| 6a | Setting, number and location of centres | Partly | Section 2.1 gives the recording settings (free-living wearable, mobile phone, handheld device). Centres are not detailed. |
| 6b | Eligibility criteria | Reported | Section 2.1: quota sample over all annotated intervals, then restriction to the official annotation intervals (the quota was applied before the restriction; Table S43); recordings shorter than 10 s excluded |
| 6c | Treatments | N/A | Signal-quality task |
| 7 | Data preprocessing and quality checks | Reported | Sections 2.1 and 2.2: resampling, unit harmonisation, and the deterministic handling of edge cases. The extractor sets non-finite values to 0, clips spectral ratios and moments (counts given), and assigns constant windows flatline and saturation fractions of 1. Supplementary Methods S-M1. |
| 8a | Outcome definition and assessment | Reported | Section 2.1: Class 3 vs 1–2, 12-lead acceptability and "too noisy" labels; recording maximum |
| 8b | Outcome assessors' qualifications | Partly | BUT QDB uses a consensus of three experts (Smital et al. 2020 and the dataset descriptor, Smital et al. 2026; cited in Section 2.1 of the main manuscript). Qualifications are not restated; the CinC labels follow the challenge papers. |
| 8c | Blinding of outcome assessment | Not reported | Not known for the public labels. |
| 9a | How predictors were chosen | Reported | Section 2.2 (SQIs from the literature) |
| 9b | Predictor definitions and measurement | Reported | Section 2.2; Supplementary Methods S-M1 |
| 9c | Assessors of subjective predictors | N/A | Predictors are computed automatically. |
| 10 | Study size and its justification | Partly | Fixed by the public datasets; there is no sample-size calculation. Power is discussed in Section 4.1. |
| 11 | Missing data | Reported | Section 2.2 (edge cases): non-finite SQIs set to 0 by the extractor, clipping counts, constant windows, short recordings excluded; Supplementary Methods S-M1 |
| 12a | Data use and partitioning | Reported | Section 2.4 (subject-grouped folds, single-source transfer, LODO, official partition) |
| 12b | Predictor handling | Reported | Sections 2.2 and 2.3 (standardisation, z-scoring, rendering); Section 2.4 for the three input safeguards of the harmonised pipeline (constant-input rule, quantization-insensitive flatline input, clipping of standardised inputs), a results-informed post hoc sensitivity analysis |
| 12c | Model type, building, tuning, internal validation | Reported | Section 2.3. Task-specific networks and tabular models: fixed hyperparameters, no tuning. TabPFN: package defaults, checkpoint stated. Linear probes: penalty C chosen by inner subject- or recording-grouped cross-validation within the training data. Rule heuristic: no training. Section 2.4; Supplementary Methods S-M2 to S-M4. |
| 12d | Heterogeneity across clusters | Partly | Subject-level cluster bootstrap (Section 2.5), per-subject sensitivity (Table S12) and per-subject AUROC (Table S41, post hoc). There is no formal heterogeneity model. |
| 12e | Performance measures and model comparisons | Reported | Section 2.5 (AUROC, AP, Brier, calibration intercept and slope, sensitivity / false reject rate; DeLong and cluster bootstrap with Holm; bootstrap p-values bounded below by their resolution; secondary-test families stated, including the post hoc CinC 2011 family without constant-lead records, Table S40) |
| 12f | Model updating in evaluation | Reported | Section 2.4 (d): local thresholds and Platt scaling; LODO retraining |
| 12g | How evaluation predictions were calculated | Reported | Sections 2.1 and 2.4 (models frozen; recording maximum) |
| 13 | Class imbalance methods and recalibration | Reported | Section 2.3 (class weights); Section 2.4 (projection of precision metrics to other prevalences, reported as conditional projections; recalibration) |
| 14 | Fairness approaches | Not reported | Section 4.1 (limitations: demographics and fairness) |
| 15 | Model output and threshold selection | Reported | Section 2.4 (operating points: post hoc, nested, transferred, local); test-set isolation stated per analysis type (Section 2.3) |
| 16 | Differences between development and evaluation data | Reported | Sections 2.1 and 3.2 (label definitions, feature shift; Table S6; recording length and number of windows on CinC 2017, Table S42; input incompatibilities between datasets, examined with the harmonised pipeline, Tables S46 and S47) |
| 17 | Ethics approval and consent | Reported | Ethics approval and consent section: secondary analysis of public de-identified data; BUT QDB collected under IRB protocol EC:EK:05b/2018; ethical approval not required |
| 18a | Funding | Reported | Funding section (no specific grant) |
| 18b | Conflicts of interest | Reported | Conflicts of interest section (none declared) |
| 18c | Protocol availability | Partly | A dated written analysis plan is kept with the code (Section 2.6). There is no protocol for the original analyses, and post hoc decisions are listed (Section 2.6), including the analyses added after an independent code audit (item 8 there) the change of framing after all results were known (item 9 there) and the analyses added after an independent scientific review, including the harmonised pipeline, whose safeguards were chosen with knowledge of the earlier external results (item 10 there). |
| 18d | Registration | Reported | Not registered (stated in Section 2.6) |
| 18e | Data availability | Reported | Data and code availability (PhysioNet) |
| 18f | Code availability | Reported | Public repository https://github.com/mkuntalp-ops/d13b-ecg-sqi (MIT licence) with code, saved predictions and table/figure scripts (Data availability statement) |
| 19 | Patient and public involvement | Reported | None (Ethics section) |
| 20a | Participant flow and outcome counts | Reported | Figure 1 (flow diagram); Section 2.1 gives the numeric flow for each dataset: recordings, exclusions, windows, units and outcome counts. LODO training sets are in Section 2.4. |
| 20b | Participant characteristics | Partly | Dataset-level characteristics only; demographics not analysed (Section 4.1). |
| 20c | Evaluation vs development distributions | Reported | Outcome prevalences (Section 2.1); predictor distributions by dataset and class, median [IQR] (Table S29); univariate predictor–outcome associations by dataset (Table S6) |
| 21 | Participants and outcome events in each analysis | Reported | Development and external sets: Section 2.1, Figure 1 and Tables S28, S29 and S32 (units and positives per set). LODO training sets: Section 2.4. Official partition: Section 2.4 and Table S24. Local recalibration: n per draw with fallback counts (Tables 4 and S22). |
| 22 | Full model details for third-party use | Partly | Code, hyperparameters, seeds, architectures and the heuristic formula are given (Supplementary Methods). Weights of the models trained here are not released with this version; the foundation-model weights come from their providers under their licences. |
| 23a | Performance with confidence intervals | Reported | AUROC and AP with 95% confidence intervals for all 11 models in every primary evaluation set (Table S28). Decision-level metrics at τ = 0.5 (sensitivity, false reject rate, PPV, NPV) and calibration intercept and slope with 95% confidence intervals (Table S32). LODO and official-partition AUROC confidence intervals are in Table 2 and Table S24. Exploratory analyses without intervals are stated as such (Table S34). Original and harmonised AUROCs with paired intervals for the difference, without multiplicity adjustment (Table S46). |
| 23b | Heterogeneity across clusters | Reported | Tables S12 and S41 (subject level); Tables S9 and S21 (seed variation); Table S42 (CinC 2017 by number of windows per recording); Table S47 (seed range of the refitted networks under the harmonised pipeline) |
| 24 | Results of model updating | Reported | Table 4, Table S22; LODO (Table 2, Table S21) |
| 25 | Overall interpretation | Partly | Section 4. Fairness is not discussed beyond the limitations (Section 4.1). |
| 26 | Limitations | Reported | Section 4.1 |
| 27a | Handling of poor or unavailable input at implementation | Partly | The model's task is to flag poor input. Section 4 recommends an explicit constant-signal check, because several models missed constant leads (Table S26) and the ranking of the two leading models on CinC 2011 depends on such records (Table S40). Recordings shorter than one window were excluded. |
| 27b | User interaction and expertise | Partly | Section 1 states who uses the output and how; user interaction in a deployed workflow was not studied (Section 4.1). |
| 27c | Next steps for applicability | Reported | Sections 4 and 5 (local calibration with enough positives; multi-lead input) |

**Items that could be strengthened if the target journal requires it.**
- 3c: discuss known inequalities in ECG signal quality.
- 14: fairness analysis needs data that the public datasets do not provide.
- 22: release the weights of the models trained here, after licence review.
