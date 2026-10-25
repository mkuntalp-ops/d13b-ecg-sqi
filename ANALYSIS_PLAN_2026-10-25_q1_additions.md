# Analysis plan: Q1 additions (written 2026-10-25, before the analyses below were run)

## Purpose

This plan specifies four analyses before they are run, and two more that depend on feasibility checks. It is committed to git before any of them is executed, so that the specification can be dated.

Already known when this plan was written:
- every single-source result;
- the LODO results (seed 42);
- the TabPFN, HuBERT-ECG probe and heuristic-refit results;
- the paired comparisons;
- the d1-3B results on the 556-recording CinC 2017 subset.

The analyses below were chosen because each one addresses a specific weakness of the current claims. None is intended to improve a reported score.

Fixed for all analyses:
- Positive class: unusable.
- The external test data are never used to choose hyperparameters, features or thresholds, except in A3, where that use is the object of study and the evaluation is always on held-out target records.

## A1. Seed repetition of the neural models (single-source and LODO)

**Question.** Are the LODO effects of the neural models (multi-source minus BUT-only training) reproducible across training seeds?

**Design.**
- *Models and seeds.* Seeds 1, 2, 3 and 4 are added to the existing seed 42 for the four seed-dependent models: SQI MLP, 1D CNN, 2D CNN and multimodal net. The seed is set through the module-level `RANDOM_SEED`, as in `experiments/cinc_draws_by_seed.py`.
- *Models not repeated:*
  - Logistic regression and LightGBM use a fixed `random_state` in the main analysis and are not seed-varied.
  - TabPFN and the two linear probes are not repeated.
  - The probes are deterministic given the frozen features.
- *Training per seed:*
  - BUT-only training on the full 1,047-window cohort, scored on CinC 2011 (565 records, lead II) and on all CinC 2017 windows (8,512 recordings, recording maximum);
  - the three LODO splits of `experiments/lodo_multisource.py`.
- *Data choices.* The CinC 2017 training subset (all noisy recordings plus 1,000 non-noisy, sampling seed 42) is held fixed; only the training seed varies. Hyperparameters are unchanged.

**Outcomes.**
- AUROC per seed, model and test set.
- Seed-matched Δ = AUROC(multi-source) − AUROC(BUT-only) on CinC 2011 and CinC 2017.
- On the BUT QDB test, the BUT-only reference is the 5-fold out-of-fold AUROC of the same seed, where available (Table 5 seeds). Otherwise only the LODO AUROC is reported.

**Decision rule.** A neural-model LODO effect is called *consistent* if two conditions hold:
- Δ has the same sign in all five seeds;
- the median Δ satisfies |median Δ| ≥ 0.02.

Statements in the manuscript about multi-source effects for these models are restricted to consistent effects. Seed-42 p-values are kept but labelled as single-seed.

## A2. Zero-shot d1-3B and the d1-3B image-token probe on all of CinC 2017

**Question.** Does the claim "the d1-3B image probe was best or not significantly different from the best in every evaluation in which it was run" hold on the full CinC 2017 set, where it has not been run?

**Design.**
- *Data.* The windows not in the existing subset are added: 23,985 − 1,376 = 22,609 windows. Each runs one forward pass of the primary zero-shot configuration, unchanged, with the same forward hooks as `experiments/d1_linear_probe.py`.
- *Probe.* The existing probe is fitted on all BUT QDB image-token features (deterministic) and applied to every window. The recording score is the maximum over windows.
- *Paired comparisons.* Zero-shot d1-3B and the d1-3B probe are added to the CinC 2017 (8,512 recordings) family in `evaluation/paired_comparisons.py`. With 11 models, that family becomes 55 comparisons (Holm).
- *LODO.* The d1-3B probe LODO is re-run with the same CinC 2017 training subset as the other models (all noisy recordings plus 1,000 non-noisy); the current version uses the balanced 556-recording subset. The CinC 2017 test then becomes all 8,512 recordings. The re-run replaces the current d1-3B probe LODO rows.

**Decision rule.** If the d1-3B probe is significantly worse than the best model on the full CinC 2017 set (Holm-adjusted p < 0.05), the following are removed or rewritten:
- the "never significantly behind the best" statement in the abstract, results and conclusions;
- the corresponding foundation-model statements.

The result is reported either way.

## A3. Local recalibration with a small labelled target sample

**Question.** The manuscript concludes that operating points need local calibration. How many labelled target recordings does that take?

**Design.**
- *Target sets:* CinC 2011 (565 records) and CinC 2017 (8,512 recordings).
- *Models:* every model with BUT-only scores on the full set. These are the eight models of the main analysis, TabPFN, the HuBERT-ECG probe and the d1-3B probe; on CinC 2017, d1-3B is included only after A2.
- *Sampling:* for n ∈ {25, 50, 100, 200}, n target records are drawn by simple random sampling (not stratified), 500 repetitions, seed 42. Evaluation is always on the records not drawn.
- *Threshold methods:*
  - (a) **transferred:** the Target 1 threshold chosen on pooled BUT QDB out-of-fold scores (Table 2f); no target labels used.
  - (b) **local:** the Target 1 rule (maximum sensitivity at FPR ≤ 5%) applied to the n drawn records. If the sample lacks either class, method (a) is used, and the number of such draws is reported.
- *Calibration:* Platt scaling (logistic regression on the logit of the score) fitted on the n drawn records, compared with the uncalibrated BUT-only score. Draws with one class are skipped and counted.

**Outcomes.**
- Achieved FPR and sensitivity on the held-out records: median and 2.5–97.5 percentiles over repetitions.
- The proportion of repetitions with achieved FPR ≤ 7.5% (target 5% plus 2.5 percentage points).
- Brier score before and after Platt scaling.

**Reporting rule.** This is descriptive; no hypothesis test is performed. The analysis is reported for all models and all n; no model or n is selected for the main text based on the result. The main text shows every model at n = 100, and the supplement shows every n.

## A4. HuBERT-ECG preprocessing check

**Question.** Does our HuBERT-ECG input construction follow the authors' published preprocessing?

**Design.** The preprocessing in the authors' public code repository is read. Our construction is:
- band-pass filter 0.05–47 Hz;
- scaling to [−1, 1];
- decimation by 5;
- lead II slot.

**Decision rule.** If the authors' preprocessing differs from ours, the probe is re-run with theirs, which becomes the primary HuBERT-ECG result; ours is kept as a sensitivity analysis. Either way, the manuscript states which preprocessing was used and how it was verified.

## F1 (feasibility first). Official BUT QDB evaluation protocol

The protocol of the BUT QDB data descriptor (Smital et al., Sci Data 2026) is read first. It is run only if three conditions hold:
- it can be implemented on the PhysioNet data already available;
- it needs no new download;
- the models need no change.

If it is run, it is a sensitivity analysis of the main models with fixed hyperparameters, and the primary protocol is not changed. The design is added to this file, as a dated amendment, before it is run.

## F2 (feasibility first). HuBERT-ECG fine-tuning

**Question.** Does fine-tuning help beyond the linear probe?

**Feasibility.** This is tested only if one fold trains in under one hour on the M4.

**Fixed configuration (no tuning):**
- the last four transformer layers and a linear head are unfrozen;
- AdamW with learning rate 10^-5 for the transformer and 10^-3 for the head;
- 10 epochs, batch size 16, class-weighted loss, seed 42;
- the same five folds, then BUT-only transfer to CinC 2011 and CinC 2017.

**Decision rule.** If infeasible, it is reported as not done.

## Unchanged

The following are not changed:
- the primary endpoint (AUROC; positive class unusable);
- the folds;
- the external datasets;
- the Holm families of the existing comparisons (except that the CinC 2017 family gains the two d1-3B models in A2);
- the reporting of every earlier result.

No further model families are added.
