# TRIPOD+AI checklist (draft, 2026-10-26)

**Checklist source.** Collins GS et al., TRIPOD+AI statement, BMJ 2024;385:e078378 (doi:10.1136/bmj-2023-078378), Table 2: 27 items with sub-items. The item topics below are short paraphrases; consult the published checklist for the full wording.

**Study type.** Development (D) and evaluation (E): model development with patient-independent internal validation, and external evaluation on three datasets.

**Locations.** Section numbers refer to the accompanying main manuscript; Tables S1–S33 and Supplementary Methods S-M1 to S-M4 refer to the accompanying supplementary material.

**Status codes.**
- Reported: the item is fully addressed.
- Partly: the item is addressed in part.
- Not reported: the reason is given.
- N/A: not applicable.

| Item | Topic (paraphrase) | Status | Where / note |
|---|---|---|---|
| 1 | Title identifies development/evaluation, population, outcome | Partly | The title names the task (ECG usability screening), the benchmark design, the model families and the main finding. It does not use the words "development" and "evaluation". |
| 2 | Abstract per TRIPOD+AI for Abstracts | Partly | Structured abstract (250 words). Not checked item by item against the 13-item abstract checklist. |
| 3a | Healthcare context, rationale, existing models | Reported | Section 1 |
| 3b | Target population, intended purpose, intended users | Partly | Section 1 states the purpose (quality gate before automated analysis). Intended users and the place in the care pathway are not stated explicitly. |
| 3c | Known health inequalities | Not reported | The background does not discuss known inequalities in ECG signal quality (for example by skin, body habitus or device population). This item is separate from our subgroup analysis, which was not performed (Limitation 11; item 14). |
| 4 | Development, evaluation or both | Reported | Section 1 (contributions); Section 2.4 |
| 5a | Data sources, rationale, representativeness | Reported | Section 2.1; representativeness in Limitations 1 and 3 |
| 5b | Dates of data collection | Not reported | Dataset versions and references are given; collection dates are not. |
| 6a | Setting, number and location of centres | Partly | Section 2.1 gives the recording settings (free-living wearable, mobile phone, handheld device). Centres are not detailed. |
| 6b | Eligibility criteria | Reported | Section 2.1: windows inside annotations, quotas, recordings shorter than 10 s excluded |
| 6c | Treatments | N/A | Signal-quality task |
| 7 | Data preprocessing and quality checks | Reported | Sections 2.1 and 2.2: resampling, unit harmonisation, and the deterministic handling of edge cases. The extractor sets non-finite values to 0, clips spectral ratios and moments (counts given), and assigns constant windows flatline and saturation fractions of 1. Supplementary Methods S-M1. |
| 8a | Outcome definition and assessment | Reported | Section 2.1: Class 3 vs 1–2, 12-lead acceptability and "too noisy" labels; recording maximum |
| 8b | Outcome assessors' qualifications | Partly | BUT QDB uses a consensus of three experts [10, 20]. Qualifications are not restated; the CinC labels follow the challenge papers. |
| 8c | Blinding of outcome assessment | Not reported | Not known for the public labels. |
| 9a | How predictors were chosen | Reported | Section 2.2 (SQIs from the literature) |
| 9b | Predictor definitions and measurement | Reported | Section 2.2; Supplementary Methods S-M1 |
| 9c | Assessors of subjective predictors | N/A | Predictors are computed automatically. |
| 10 | Study size and its justification | Partly | Fixed by the public datasets; there is no sample-size calculation. Power is discussed in Limitation 1. |
| 11 | Missing data | Reported | Section 2.2 (edge cases): non-finite SQIs set to 0 by the extractor, clipping counts, constant windows, short recordings excluded; Supplementary Methods S-M1 |
| 12a | Data use and partitioning | Reported | Section 2.4 (subject-grouped folds, single-source transfer, LODO, official split) |
| 12b | Predictor handling | Reported | Sections 2.2 and 2.3 (standardisation, z-scoring, rendering) |
| 12c | Model type, building, tuning, internal validation | Reported | Section 2.3. Task-specific networks and tabular models: fixed hyperparameters, no tuning. TabPFN: package defaults, checkpoint stated. Linear probes: penalty C chosen by inner subject- or recording-grouped cross-validation within the training data. Rule heuristic: no training. Section 2.4; Supplementary Methods S-M2 to S-M4. |
| 12d | Heterogeneity across clusters | Partly | Subject-level cluster bootstrap (Section 2.5) and per-subject sensitivity (Table S13). There is no formal heterogeneity model. |
| 12e | Performance measures and model comparisons | Reported | Section 2.5 (AUROC, AP, Brier, sensitivity / false reject rate; DeLong and cluster bootstrap with Holm; secondary-test families stated) |
| 12f | Model updating in evaluation | Reported | Section 2.4 (d): local thresholds and Platt scaling; LODO retraining |
| 12g | How evaluation predictions were calculated | Reported | Sections 2.1 and 2.4 (models frozen; recording maximum) |
| 13 | Class imbalance methods and recalibration | Reported | Section 2.3 (class weights); Section 2.4 (prevalence re-weighting, recalibration) |
| 14 | Fairness approaches | Not reported | Limitation 11 |
| 15 | Model output and threshold selection | Reported | Section 2.4 (operating points: post hoc, nested, transferred, local); test-set isolation stated per analysis type (Section 2.3) |
| 16 | Differences between development and evaluation data | Reported | Sections 2.1 and 3.2 (label definitions, feature shift; Table S6) |
| 17 | Ethics approval and consent | Reported | Ethics approval and consent section: secondary analysis of public de-identified data; BUT QDB collected under IRB protocol EC:EK:05b/2018; ethical approval not required |
| 18a | Funding | Reported | Funding section (no specific grant) |
| 18b | Conflicts of interest | Reported | Conflicts of interest section (none declared) |
| 18c | Protocol availability | Partly | The dated analysis plan for the revision analyses is in the repository (Section 2.5). There is no protocol for the original analyses (stated). |
| 18d | Registration | Reported | Not registered (stated in Section 2.5) |
| 18e | Data availability | Reported | Data and code availability (PhysioNet) |
| 18f | Code availability | Reported | Public repository https://github.com/mkuntalp-ops/d13b-ecg-sqi (MIT licence) with code, saved predictions and table/figure scripts (Data availability statement) |
| 19 | Patient and public involvement | Reported | None (Ethics section) |
| 20a | Participant flow and outcome counts | Reported | Section 2.1 gives the numeric flow for each dataset: recordings, exclusions, windows, units and outcome counts. LODO training sets are in Section 2.4. No diagram (optional in the guideline). |
| 20b | Participant characteristics | Partly | Dataset-level characteristics only; demographics not analysed (Limitation 11). |
| 20c | Evaluation vs development distributions | Reported | Outcome prevalences (Section 2.1); predictor distributions by dataset and class, median [IQR] (Table S31); univariate predictor–outcome associations by dataset (Table S6) |
| 21 | Participants and outcome events in each analysis | Reported | Development and external sets: Section 2.1 and Tables S30 and S31 (units and positives per set). LODO training sets: Section 2.4. Official-protocol analyses: Table S26. Local recalibration: n per draw with fallback counts (Tables 4 and S24). |
| 22 | Full model details for third-party use | Partly | Code, hyperparameters, seeds, architectures and the heuristic formula are given (Supplementary Methods). Trained model weights are not released; the foundation-model weights come from their providers under their licences. |
| 23a | Performance with confidence intervals | Partly | AUROC and AP with 95% confidence intervals are reported for all 11 models in every primary evaluation set (Table S30). LODO and official-protocol AUROC confidence intervals are in Table 2 and Table S26. Table S29 also gives AUROC and AP confidence intervals for seven models on CinC 2017. Decision-level metrics at τ = 0.5 (sensitivity, false reject rate, PPV, NPV) are reported as point estimates only; no confidence intervals were reported for them. |
| 23b | Heterogeneity across clusters | Reported | Table S13 (subject level); Tables S17 and S10 (subset and seed variation) |
| 24 | Results of model updating | Reported | Table 4, Table S24; LODO (Table 2, Table S23) |
| 25 | Overall interpretation | Partly | Section 4. Fairness is not discussed beyond Limitation 11. |
| 26 | Limitations | Reported | Section 4.1 |
| 27a | Handling of poor or unavailable input at implementation | Partly | The model's task is to flag poor input. Section 4 recommends an explicit constant-signal check, because several models missed constant leads (Table S28). Recordings shorter than one window were excluded. |
| 27b | User interaction and expertise | Not reported | Not specified for a deployment. |
| 27c | Next steps for applicability | Reported | Sections 4 and 5 (local calibration with enough positives; multi-lead input) |

**Items that could be strengthened if the target journal requires it.**
- 3b: state the intended users.
- 20a: an optional flow diagram could improve readability; the numeric flow is already reported.
- 27b: describe user interaction.
