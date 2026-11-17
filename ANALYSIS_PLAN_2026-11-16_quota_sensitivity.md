# Analysis plan: sensitivity analysis for the BUT QDB quota of subject 105 (written 2026-11-16, before any score on the added windows was produced)

Status: APPROVED by the author on 2026-11-16 ("Hepsine onay": the plan, the cohort rule and the limited scope), before anything was run. The author asked for the plan on 2026-11-16 ("Duyarlılık analizi planla") after a reviewer comment that the cohort construction under-represents one subject.

What had been looked at when this plan was written: every result of the primary analysis (analysis version a21e70b), and the per-subject candidate counts in `results/v2_2026-11/butqdb_quota_candidates_by_subject.csv` (Supplementary Table S43). No model of this study has scored any of the 53 windows this plan adds. Plan changes after this date will be listed at the end of this file, with date and reason.

## Problem

The cohort quota (up to 50 usable and up to 60 unusable windows per subject, evenly spaced in time) was drawn over all annotated intervals. The 59 sampled windows outside the official intervals were removed afterwards, all unusable and all from record 105001. Subject 105 therefore has 1 unusable window in the 988-window cohort, although 54 Class 3 windows of that subject lie inside the official intervals. For the other six subjects with Class 3 windows the two procedures give the same count (Table S43).

## Question

Do the in-domain results and the external results of the BUT-only models change when subject 105 is represented by the windows that a quota drawn inside the official intervals would have selected?

This is a post hoc sensitivity analysis added in response to review. The primary analysis and the primary cohort stay as they are.

## Sensitivity cohort

- Candidates: the non-overlapping 10-s windows inside the official intervals (`butqdb_official_split.official_mask`), built with `data.extract_butqdb_1000_cohort.build_candidate_windows`.
- Sampling: the existing function `sample_stratified_cohort`, unchanged, applied to these candidates.
- Expected result, to be checked and reported, not assumed: 1,041 windows, 291 unusable (28.0%), 15 subjects; identical to the primary cohort except for 53 added Class 3 windows of subject 105. Every window of the primary cohort that is not in the sensitivity cohort, and every added window, will be listed. If the difference is anything other than these 53 windows, the run stops and the difference is reported before any model is fitted.
- Folds: each subject keeps its fold of the primary analysis (subject 105: fold 3). `GroupKFold` is not re-run, because its assignment depends on group sizes.
- Signals, images and SQIs of the 53 windows are produced with the code used for the primary cohort (same resampling, renderer, SQI extractor). Existing cache files are not rewritten; their SHA-256 is compared before and after.

## Models

All 11 approaches, existing code, same hyperparameters, seed 42, no tuning.

- Refitted on the sensitivity cohort: logistic regression, LightGBM, MLP, TabPFN, 1D CNN, 2D CNN, multimodal net, d1-3B image-token probe, HuBERT-ECG probe (penalty chosen by the same inner subject-grouped cross-validation).
- Not fitted: the rule heuristic and zero-shot d1-3B. They only score the 53 added windows. Their external results cannot change and are reported as identical by construction.
- The logistic-regression replacement of the heuristic is refitted as in the primary analysis.

## Outcomes

In-domain (BUT QDB, 5-fold subject-grouped cross-validation, 1,041 windows):
1. AUROC of each model with subject-level cluster-bootstrap 95% interval (2,000 resamples), next to the primary value.
2. Paired comparison of each model with the best one (cluster bootstrap, Holm), same procedure as the primary analysis: which models are not significantly different from the best.
3. Sensitivity at τ = 0.50 for subject 105 (54 windows instead of 1) and the false reject range, as in Table S12.

External (models fitted on all 1,041 windows, applied unchanged):
4. AUROC on CinC 2011 lead II (998 records) and CinC 2017 (8,512 records), with the difference from the primary model of the same type on the same records (DeLong for paired scores, 95% interval). These differences are descriptive; no multiplicity adjustment and no claim of equivalence.
5. Ranks of the 11 models and the Spearman correlations between in-domain and external AUROC, next to the primary values (0.61, −0.02, 0.19).

## What will be reported as changed or unchanged

Fixed now, before results. Four statements of the manuscript are checked:
- (a) the set of models not significantly different from the in-domain best;
- (b) the model with the highest AUROC on CinC 2011 lead II and on CinC 2017;
- (c) whether the d1-3B probe is significantly above the 2D CNN on CinC 2011 lead II (DeLong, as in the primary analysis; reported unadjusted, since the primary Holm family is not re-run);
- (d) the sign and approximate size of the three Spearman correlations.

Each is reported as it comes out, including changes that weaken the manuscript's statements. No model, threshold or text of the primary analysis is altered on the basis of these results. If a statement changes, the manuscript says so in Results and Limitations; the choice of primary cohort is not switched.

## Not repeated, and why

Leave-one-dataset-out training, the 12-lead analysis, PTB-XL, NSTDB, threshold transfer and local recalibration, calibration, seed repeats, the harmonised pipeline, the gating analysis and latency are not re-run. They would multiply the post hoc analyses without addressing the reviewer's point more directly. The manuscript will state that the sensitivity analysis covers discrimination only. PTB-XL is not touched (final-evaluation exposure unchanged).

## Records

- Output directory: `results/v2_2026-11_quota_sensitivity/` via `ECG_RESULTS_DIR`. Nothing in `results/v2_2026-11/` is written or overwritten.
- New script `experiments/quota_sensitivity_setup.py` (cohort, checks, fold carry-over) and a queue script listing the stages; existing stage scripts are reused unchanged where they read the manifest from `ECG_RESULTS_DIR`. Any stage script that needs a change is listed here before the run.
- Saved: manifest, out-of-fold predictions, external predictions, logs, git commit and dirty state at run time.
- A unit test for the cohort rule (53 added windows, folds carried over).

## Cost

Estimated from the logs of 5–6 November: a few hours on this machine (network training, d1-3B and HuBERT-ECG inference on 53 windows, TabPFN in its own environment). No paid service. No data leaves the machine.

## Consequences if run

- One new supplementary table and a short paragraph in Methods, Results and Limitations; item added to the deviations list (post hoc, after review).
- The analysis version changes (new commit); a new export round follows.

## Approval needed before running

1. This plan as written, or with changes.
2. The cohort rule (re-draw inside the official intervals) rather than an alternative.
3. The limited scope (discrimination only).

## Implementation details fixed before the run (2026-11-16, after reading the stage scripts; no result seen)

- `experiments/quota_sensitivity.py` (new) holds the stages that cannot be taken from an existing script unchanged:
  - `setup`: cohort, checks, extraction of the added windows with `data.extract_butqdb_1000_cohort.process_single_window`. As a check of the code path, the SQIs of the 988 primary windows are recomputed from the cached signals and compared with the primary manifest; the primary rows themselves are copied unchanged.
  - `d1`: zero-shot d1-3B and its two representations for the added windows only, with the classes of `d1_linear_probe.py` (`RealD1DecisionModel`, `Capture`). `d1_linear_probe.extract()` is not used because it expects v1 zero-shot scores for every cached window. One forward pass gives the score and the representation (the primary BUT QDB scores came from a separate pass of the same configuration).
  - `hubert`: embeddings of the added windows (`hubert_ecg_probe.embed`), then the cross-validated and the BUT-only probe with `hubert_ecg_probe.probe()`. The leave-one-dataset-out part of `hubert_ecg_probe.analyse()` is left out (not in scope).
  - `external`: `external_validation.score_windows()` only.
  - `analyse`: the comparison tables of this plan.
- `experiments/quota_sensitivity_tabpfn.py` (new, TabPFN environment): cross-validation and BUT-only external scoring with the functions of `tabpfn_runner.py`; NSTDB and leave-one-dataset-out are left out.
- Run unchanged with `ECG_RESULTS_DIR=results/v2_2026-11_quota_sensitivity`: `run_supervised_cohort_cv.py`, `compute_consolidated_metrics.py`, `d1_linear_probe.py --analysis-only`, `d1_cinc2017_full.py --rescore`, `heuristic_refit.py`, `evaluation/paired_comparisons.py`, `evaluation/model_cis.py`, `evaluation/paired_vs_best.py`. These also write by-products that are not part of this plan (decision-state probe, CinC 2017 balanced subset); they are not reported.
- New cached feature sets get the name `BUT_QDB_quota_sens_2026-11-16`; `feature_cache.save_set` refuses to overwrite.
- Check of the new code: the `hubert` and TabPFN stages are first run on the primary 988-window manifest into a scratch directory and must reproduce the stored primary predictions.
- Whether network training is reproducible run to run on this machine is not recorded. If it is not, differences for the six trained networks mix the cohort effect with run-to-run variation; this will be stated, with the seed ranges of Supplementary Table S9 as the scale.

## Results (added 2026-11-17, after the run of 2026-11-16; all values from `results/v2_2026-11_quota_sensitivity/`)

Cohort, as expected: 1,041 windows, 291 unusable, 15 subjects; 53 windows added (all unusable, all subject 105, record 105001), none removed (`quota_sensitivity_added_windows.csv`, `quota_sensitivity_removed_windows.csv`). The 1,976 cache files of the primary cohort were unchanged (SHA-256).

The four statements fixed above:
- (a) Changed. Primary: logistic regression and the MLP significantly below the 2D CNN (p_Holm = 0.028 each, Holm over all 55 pairs, the criterion used in the manuscript); zero-shot d1-3B 0.051. Sensitivity: only zero-shot d1-3B (0.028); logistic regression 0.702, MLP 0.264. With Holm over the ten comparisons with the best only, the primary analysis has three models below the best (logistic regression, MLP, zero-shot d1-3B) and the sensitivity analysis one (zero-shot d1-3B). The 2D CNN is the best in both (0.985, 0.993).
- (b) CinC 2011 lead II: unchanged (d1-3B image-token probe, 0.824 and 0.828). CinC 2017: changed, from the HuBERT-ECG probe (0.906; sensitivity 0.880, rank 4) to the d1-3B image-token probe (0.894; sensitivity 0.908). In the sensitivity analysis LightGBM and the rule heuristic are not significantly different from the leader with Holm over all pairs (0.255, 0.368); the HuBERT-ECG probe is (0.003).
- (c) Unchanged: d1-3B probe above the 2D CNN on CinC 2011, difference 0.042 (95% CI 0.020–0.065), unadjusted p = 0.0002, p_Holm = 0.004 (primary: 0.037, p_Holm = 0.003).
- (d) BUT QDB vs CinC 2011: 0.61 → 0.57. BUT QDB vs CinC 2017: −0.02 → 0.14. CinC 2011 vs CinC 2017: 0.19 → 0.45.

Largest single change: HuBERT-ECG probe on CinC 2011, 0.562 → 0.759 (Δ 0.197, 95% CI 0.159–0.235). All values per model: `quota_sensitivity_auroc.csv`; manuscript table: Table 28 of `evaluation/build_manuscript_tables.py` (Table S48 of the full-length version, Supplementary Table S49 of the Scientific Reports version).

Checks: `logs/reproduction_control_2026-11-16.txt` (re-running the primary cohort reproduces the stored predictions of the seven models of the cross-validation script with difference 0; the new HuBERT-ECG and TabPFN stages reproduce the stored primary predictions to 1.1e-16).

## Plan changes and corrections (2026-11-17)

- Script name: the plan named `experiments/quota_sensitivity_setup.py`; the script is `experiments/quota_sensitivity.py` (stages `setup`, `d1`, `hubert`, `external`, `analyse`), as listed under "Implementation details".
- Outcome 2 says "Holm" without naming the family. The manuscript's in-domain statements use Holm over all 55 pairs, so the manuscript reports that criterion for the sensitivity analysis too; both families are given under (a) above. A first summary to the author on 2026-11-16 used the family of ten comparisons with the best; the conclusion for (a) is the same, the conclusion for the CinC 2017 comparisons differs (LightGBM and the heuristic are not significantly below the d1-3B probe with Holm over all pairs).
- Not in the manuscript because no saved output supports them: the amplitude of the added windows and the penalty chosen for the HuBERT-ECG probe. The by-products of the unchanged stage scripts (decision-state probe, CinC 2017 balanced subset, logistic-regression replacement of the heuristic) were not examined and are not reported.
- The manuscript was updated on 2026-11-17 at the author's request ("Evet, makaleye işle"): Methods, Results, Discussion, limitations, deviations list, data availability and the new supplementary table. The primary cohort and results are unchanged.
