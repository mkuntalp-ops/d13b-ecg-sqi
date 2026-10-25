# Supplementary Material

<!-- Template filled by evaluation/assemble_q1_manuscript.py (writes manuscript/supplement_q1.md). -->

Every table is generated from saved outputs by evaluation/build_manuscript_tables.py (results/manuscript_tables_2026-10-12.md), except Table S22, which is read from results/heuristic_refit_summary.csv. Positive class: unusable.

## Table S1. Patient-independent 5-fold cross-validation on BUT QDB (1,047 windows, 15 subjects), models of the main analysis. 95% CIs: subject-level cluster bootstrap.

<!-- source: Table 1 -->
{{Table 1}}

## Table S2. Nested versus post-hoc threshold selection on BUT QDB.

<!-- source: Table 1c -->
{{Table 1c}}

## Table S3. BUT QDB precision metrics re-weighted to the natural prevalence of Class 3 (16.0%; 2.8% without record 105001).

<!-- source: Table 1e -->
{{Table 1e}}

## Table S4. Thresholds chosen on pooled BUT QDB out-of-fold scores and applied unchanged to the external datasets.

<!-- source: Table 2f -->
{{Table 2f}}

## Table S5. CinC 2011 scored on all 12 leads (aggregation rules chosen after the main analysis).

<!-- source: Table 2e -->
{{Table 2e}}

## Table S6. Univariate AUROC of each SQI across datasets (values below 0.5: higher feature value associated with usable).

<!-- source: Table 2g -->
{{Table 2g}}

## Table S7. NSTDB: paired change in usable probability relative to the clean partner window (mean ± SD).

<!-- source: Table 3b -->
{{Table 3b}}

## Table S8. Extended latency profile (Apple M4).

<!-- source: Table 4b -->
{{Table 4b}}

## Table S9. Seed stability of the 1D CNN, 2D CNN and multimodal net.

<!-- source: Table 5 -->
{{Table 5}}

## Table S10. Training-seed and CinC-subset variation of the supervised models.

<!-- source: Table 6 -->
{{Table 6}}

## Table S11. Post-hoc ROC operating points on BUT QDB (exploratory; thresholds chosen on the evaluated data).

<!-- source: Table 1b -->
{{Table 1b}}

## Table S12. Sensitivity analysis with Class 2 relabelled as unusable (BUT QDB).

<!-- source: Table 1d -->
{{Table 1d}}

## Table S13. Subject-level sensitivity on BUT QDB.

<!-- source: Table 1f -->
{{Table 1f}}

## Table S14. CinC 2011 Set A, reported 150-record subset (first 75 acceptable and 75 unacceptable records).

<!-- source: Table 2 -->
{{Table 2}}

## Table S15. Post-hoc ROC operating points on the CinC 2011 reported subset (exploratory).

<!-- source: Table 2b -->
{{Table 2b}}

## Table S16. Feature-set and scaler sensitivity of the tabular models under transfer.

<!-- source: Table 2c -->
{{Table 2c}}

## Table S17. Transfer to five random balanced CinC 2011 subsets.

<!-- source: Table 2d -->
{{Table 2d}}

## Table S18. NSTDB: usable probability by nominal SNR.

<!-- source: Table 3 -->
{{Table 3}}

## Table S19. Prompt sensitivity of zero-shot d1-3B.

<!-- source: Table 7 -->
{{Table 7}}

## Table S20. Linear probes on frozen d1-3B representations.

<!-- source: Table 7b -->
{{Table 7b}}

## Table S21. CinC 2017 balanced subset (278 noisy + 278 non-noisy recordings) including zero-shot d1-3B.

<!-- source: Table 8b -->
{{Table 8b}}

## Table S22. Rule heuristic with hand-set versus refitted coefficients (same seven inputs; refit by standardised L2 logistic regression inside the BUT QDB folds, transferred like the other models).

{{CSV:heuristic_refit}}

## Table S23. Seed repetition (seeds 42 and 1–4) of BUT-only training and leave-one-dataset-out training for the seed-dependent networks. Consistent: same sign of Δ in all five seeds and |median Δ| ≥ 0.02 (pre-specified).

{{Table 10b}}

## Table S24. Local recalibration on n = 25–200 randomly drawn labelled target recordings (500 draws each), evaluated on the remaining recordings. Prevalence-only Brier score: CinC 2011 0.227, CinC 2017 0.032.

{{Table 11b}}

## Table S25. HuBERT-ECG: preprocessing sensitivity and partial fine-tuning (AUROC). Fine-tuning was not run in LODO.

{{Table 12}}

## Table S26. BUT QDB official protocol (Smital et al. 2026): sensitivity analyses. F1a: pooled out-of-fold AUROC without the 59 windows from the excluded stretch of record 105001 (evaluation only). F1b: retrained on the official training records, tested on the official test records (window-level binary task). 95% CI: subject-level cluster bootstrap; p_Holm: paired comparison with the best model.

{{Table 13}}

## Table S27. Latency of the foundation-model probes and TabPFN (Apple M4; same 25 windows and three runs as Table 4).

{{Table 4c}}
