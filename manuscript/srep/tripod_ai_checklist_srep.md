# TRIPOD+AI checklist (Scientific Reports version, 2026-11-13)

**Checklist source.** Collins GS et al., TRIPOD+AI statement, BMJ 2024;385:e078378 (doi:10.1136/bmj-2023-078378), Table 2: 27 items with sub-items. The item topics below are short paraphrases; consult the published checklist for the full wording.

**Study type.** Development (D) and evaluation (E): model development with patient-independent internal validation, and external evaluation on three datasets.

**Locations.** Section names refer to the accompanying main manuscript (Scientific Reports version, unnumbered sections); Supplementary Tables S1–S48, the Supplementary Methods, Supplementary Results and Supplementary Note refer to the accompanying Supplementary Information.

**Status codes.**
- Reported: the item is fully addressed.
- Partly: the item is addressed in part.
- Not reported: the reason is given.
- N/A: not applicable.

| Item | Topic (paraphrase) | Status | Where / note |
|---|---|---|---|
| 1 | Title identifies development/evaluation, population, outcome | Partly | The title names the finding (in-domain performance did not identify the models that ranked best externally), the task (ECG usability models) and the design (benchmark of eleven approaches with subject-grouped development and heterogeneous external datasets). It does not use the words "development" and "evaluation". |
| 2 | Abstract per TRIPOD+AI for Abstracts | Partly | Unstructured abstract (199 words; the journal requires an unstructured abstract of at most 200 words), checked against the 13 items of TRIPOD+AI for Abstracts (Table 3 of the statement). Reported: 2 (healthcare context and rationale, first two sentences), 7 (model types; subject-grouped cross-validation; models applied unchanged), 9 (windows, people and external recordings), 12 (interpretation). Partly: 1 (title, as item 1 of this checklist), 3 (objective stated; the words development and evaluation are not used), 4 (development and external data described, datasets not named), 6 (outcome named as usability, not defined), 8 (AUROC named; calibration not named), 11 (AUROCs without confidence intervals; intervals are in Supplementary Table S28). Not reported: 5 (eligibility and setting), 13 (the study was not registered; stated in Methods). Not applicable: 10 (no single final model). |
| 3a | Healthcare context, rationale, existing models | Reported | Introduction |
| 3b | Target population, intended purpose, intended users | Reported | Introduction: automatic gate before downstream ECG analysis; output used by developers and operators of analysis pipelines to discard a segment or request a new recording; not a replacement for clinical interpretation. |
| 3c | Known health inequalities | Not reported | The background does not discuss known inequalities in ECG signal quality (for example by skin, body habitus or device population). Discussion (limitations paragraph) states only that known inequalities in signal acquisition could not be examined. This item is separate from our subgroup analysis, which was not performed (Discussion (limitations paragraph); item 14). |
| 4 | Development, evaluation or both | Reported | Introduction (contributions); Methods (Evaluation protocols) |
| 5a | Data sources, rationale, representativeness | Reported | Methods (Datasets), Figure 1; representativeness in Discussion (limitations paragraph) (limitations) |
| 5b | Dates of data collection | Not reported | Dataset versions and references are given; collection dates are not. |
| 6a | Setting, number and location of centres | Partly | Methods (Datasets) gives the recording settings (free-living wearable, mobile phone, handheld device). Centres are not detailed. |
| 6b | Eligibility criteria | Reported | Methods (Datasets): quota sample over all annotated intervals, then restriction to the official annotation intervals (the quota was applied before the restriction; Supplementary Table S43); recordings shorter than 10 s excluded |
| 6c | Treatments | N/A | Signal-quality task |
| 7 | Data preprocessing and quality checks | Reported | Methods (Datasets) and Methods (Inputs): resampling, unit harmonisation, and the deterministic handling of edge cases. The extractor sets non-finite values to 0, clips spectral ratios and moments (counts given), and assigns constant windows flatline and saturation fractions of 1. Supplementary Methods (Signal quality indices). |
| 8a | Outcome definition and assessment | Reported | Methods (Datasets): Class 3 vs 1–2, 12-lead acceptability and "too noisy" labels; recording maximum |
| 8b | Outcome assessors' qualifications | Partly | BUT QDB uses a consensus of three experts (Smital et al. 2020 and the dataset descriptor, Smital et al. 2026; cited in Methods (Datasets) of the main manuscript). Qualifications are not restated; the CinC labels follow the challenge papers. |
| 8c | Blinding of outcome assessment | Not reported | Not known for the public labels. |
| 9a | How predictors were chosen | Reported | Methods (Inputs) (SQIs from the literature) |
| 9b | Predictor definitions and measurement | Reported | Methods (Inputs); Supplementary Methods (Signal quality indices) |
| 9c | Assessors of subjective predictors | N/A | Predictors are computed automatically. |
| 10 | Study size and its justification | Partly | Fixed by the public datasets; there is no sample-size calculation. Power is discussed in Discussion (limitations paragraph). |
| 11 | Missing data | Reported | Methods (Inputs) (edge cases): non-finite SQIs set to 0 by the extractor, clipping counts, constant windows, short recordings excluded; Supplementary Methods (Signal quality indices) |
| 12a | Data use and partitioning | Reported | Methods (Evaluation protocols) (subject-grouped folds, single-source transfer, LODO, official partition) |
| 12b | Predictor handling | Reported | Methods (Inputs) and Methods (Models) (standardisation, z-scoring, rendering); Methods (Evaluation protocols) for the three input safeguards of the harmonised pipeline (constant-input rule, quantization-insensitive flatline input, clipping of standardised inputs), a results-informed post hoc sensitivity analysis |
| 12c | Model type, building, tuning, internal validation | Reported | Methods (Models). Task-specific networks and tabular models: fixed hyperparameters, no tuning. TabPFN: package defaults, checkpoint stated. Linear probes: penalty C chosen by inner subject- or recording-grouped cross-validation within the training data. Rule heuristic: no training. Methods (Evaluation protocols); Supplementary Methods (Network architectures, Rule heuristic, Zero-shot d1-3B input). |
| 12d | Heterogeneity across clusters | Partly | Subject-level cluster bootstrap (Methods (Statistical analysis)), per-subject sensitivity (Supplementary Table S12) and per-subject AUROC (Supplementary Table S41, post hoc). There is no formal heterogeneity model. |
| 12e | Performance measures and model comparisons | Reported | Methods (Statistical analysis) (AUROC, AP, Brier, calibration intercept and slope, sensitivity / false reject rate; DeLong and cluster bootstrap with Holm; bootstrap p-values bounded below by their resolution; secondary-test families stated, including the post hoc CinC 2011 family without constant-lead records, Supplementary Table S40) |
| 12f | Model updating in evaluation | Reported | Methods (Evaluation protocols) (d): local thresholds and Platt scaling; LODO retraining |
| 12g | How evaluation predictions were calculated | Reported | Methods (Datasets) and Methods (Evaluation protocols) (models frozen; recording maximum) |
| 13 | Class imbalance methods and recalibration | Reported | Methods (Models) (class weights); Methods (Evaluation protocols) (projection of precision metrics to other prevalences, reported as conditional projections; recalibration) |
| 14 | Fairness approaches | Not reported | Discussion (limitations paragraph: demographics and fairness) |
| 15 | Model output and threshold selection | Reported | Methods (Evaluation protocols) (operating points: post hoc, nested, transferred, local); test-set isolation stated per analysis type (Methods (Models)) |
| 16 | Differences between development and evaluation data | Reported | Methods (Datasets) and Results (label definitions, feature shift; Supplementary Table S6; recording length and number of windows on CinC 2017, Supplementary Table S42; input incompatibilities between datasets, examined with the harmonised pipeline, Supplementary Tables S46 and S47) |
| 17 | Ethics approval and consent | Reported | Ethics approval and consent section: secondary analysis of public de-identified data; BUT QDB collected under IRB protocol EC:EK:05b/2018; ethical approval not required |
| 18a | Funding | Reported | Funding section (no specific grant) |
| 18b | Conflicts of interest | Reported | Conflicts of interest section (none declared) |
| 18c | Protocol availability | Partly | A dated written analysis plan is kept with the code (Methods (Deviations from the analysis plan)). There is no protocol for the original analyses, and post hoc decisions are listed (Methods (Deviations from the analysis plan)), including the analyses added after an independent code audit (item 8 there) the change of framing after all results were known (item 9 there) and the analyses added after an independent scientific review, including the harmonised pipeline, whose safeguards were chosen with knowledge of the earlier external results (item 10 there). |
| 18d | Registration | Reported | Not registered (stated in Methods (Deviations from the analysis plan)) |
| 18e | Data availability | Reported | Data and code availability (PhysioNet) |
| 18f | Code availability | Reported | Public repository https://github.com/mkuntalp-ops/d13b-ecg-sqi (MIT licence) with code, saved predictions and table/figure scripts; analysis version: commit a21e70b on branch v2-revision-2026-11 (Data availability statement) |
| 19 | Patient and public involvement | Reported | None (Ethics section) |
| 20a | Participant flow and outcome counts | Reported | Figure 1 (flow diagram); Methods (Datasets) gives the numeric flow for each dataset: recordings, exclusions, windows, units and outcome counts. LODO training sets are in Methods (Evaluation protocols). |
| 20b | Participant characteristics | Partly | Sex and age of the 15 BUT QDB development subjects (Methods (Datasets)); dataset-level characteristics of the external sets; no demographic subgroup analysis (Discussion (limitations paragraph)). |
| 20c | Evaluation vs development distributions | Reported | Outcome prevalences (Methods (Datasets)); predictor distributions by dataset and class, median [IQR] (Supplementary Table S29); univariate predictor–outcome associations by dataset (Supplementary Table S6) |
| 21 | Participants and outcome events in each analysis | Reported | Development and external sets: Methods (Datasets), Figure 1 and Supplementary Tables S28, S29 and S32 (units and positives per set). LODO training sets: Methods (Evaluation protocols). Official partition: Methods (Evaluation protocols) and Supplementary Table S24. Local recalibration: n per draw with fallback counts (Supplementary Tables S48 and S22). |
| 22 | Full model details for third-party use | Partly | Code, hyperparameters, seeds, architectures and the heuristic formula are given (Supplementary Methods). Weights of the models trained here are not released with this version; the foundation-model weights come from their providers under their licences. |
| 23a | Performance with confidence intervals | Reported | AUROC and AP with 95% confidence intervals for all 11 models in every primary evaluation set (Supplementary Table S28). Decision-level metrics at τ = 0.5 (sensitivity, false reject rate, PPV, NPV) and calibration intercept and slope with 95% confidence intervals (Supplementary Table S32). LODO and official-partition AUROC confidence intervals are in Table 3 and Supplementary Table S24. Exploratory analyses without intervals are stated as such (Supplementary Table S34). Original and harmonised AUROCs with paired intervals for the difference, without multiplicity adjustment (Supplementary Table S46). |
| 23b | Heterogeneity across clusters | Reported | Supplementary Tables S12 and S41 (subject level); Supplementary Tables S9 and S21 (seed variation); Supplementary Table S42 (CinC 2017 by number of windows per recording); Supplementary Table S47 (seed range of the refitted networks under the harmonised pipeline) |
| 24 | Results of model updating | Reported | Supplementary Tables S48 and S22; LODO (Table 3, Supplementary Table S21) |
| 25 | Overall interpretation | Partly | Discussion. Fairness is not discussed beyond the limitations (Discussion (limitations paragraph)). |
| 26 | Limitations | Reported | Discussion (limitations paragraph) |
| 27a | Handling of poor or unavailable input at implementation | Partly | The model's task is to flag poor input. Discussion recommends an explicit constant-signal check, because several models missed constant leads (Supplementary Table S26) and the ranking of the two leading models on CinC 2011 depends on such records (Supplementary Table S40). Recordings shorter than one window were excluded. |
| 27b | User interaction and expertise | Partly | Introduction states who uses the output and how; user interaction in a deployed workflow was not studied (Discussion (limitations paragraph)). |
| 27c | Next steps for applicability | Reported | Discussion and Discussion (final paragraph) (local calibration with enough positives; multi-lead input) |

**Items that could be strengthened if the target journal requires it.**
- 3c: discuss known inequalities in ECG signal quality.
- 14: a subgroup (fairness) analysis was not performed: BUT QDB reports sex and age, but its unusable windows come from 7 subjects, too few for subgroup estimates, and the external datasets were not analysed by demographic group (Discussion, limitations paragraph).
- 22: release the weights of the models trained here, after licence review.
