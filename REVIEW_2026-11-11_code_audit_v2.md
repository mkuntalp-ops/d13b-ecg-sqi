# Independent code audit: d13b-ecg-sqi, v2 analysis

Date: 2026-11-11. Scope: `results/v2_2026-11`, uncommitted working tree on top of commit 55095ed. Review round 1.
Auditor worked read-only; no project file was changed. Audit scripts: `audit1.py`, `audit2.py` (auditor scratchpad, not in the repository).

## 1. Executive summary

- **Overall:** no train/test leakage, no label or alignment error, and no metric miscalculation was found. Every stored AUROC that was recomputed matches to machine precision.
- **Most consequential risk:** the abstract's statement that the d1-3B probe was best on CinC 2011 (0.824 vs 0.787, adjusted p = 0.003) depends on 73 flat lead-II records. Without them the difference is not significant.
- **Other risks:** in-domain inference rests on very few positive subjects, the CinC 2017 primary aggregation is confounded with recording length, the cohort description does not match how the cohort was built, and the v2 results are not tied to a committed code version.
- **Trust:** the point estimates can be trusted as computed. Rankings and "not significantly below best" statements should be read with the limits below.

## 2. Evidence coverage

**Read in full:** cohort builder, `v2_setup`, feature cache, SQI extractor, all model classes, CV runner, metrics, CIs, paired tests, decision/calibration, nested thresholds, external validation, d1 probe, HuBERT probe, TabPFN runner, both external extraction scripts, revision plan, v2 main text (abstract, methods, limitations).

**Run (read-only):**

| Check | Outcome |
|---|---|
| Fold and window integrity on the 988-window manifest (duplicate ids, overlap, subject in two folds) | PASSED |
| AUROC recomputation by rank statistic, 11 models, BUT QDB / CinC 2011 / CinC 2017 | PASSED (max difference 2e-16) |
| Label agreement between prediction files, source tables and CinC 2011 RECORDS lists | PASSED (0 mismatches) |
| Own DeLong implementation vs stored probe vs 2D CNN test on 998 records | PASSED (p = 0.00014 both) |
| Abstract numbers vs result files | PASSED |
| Inner GroupKFold folds of the probes contain both classes | PASSED |
| Unit tests (`python -B -m unittest discover -s tests`, `ECG_RESULTS_DIR=results/v2_2026-11`) | 27 run, 27 PASSED; working tree unchanged (81 status lines before and after) |
| Installed package versions vs `requirements.txt` | Match |

**NOT_RUN / not audited:** LODO, seed repeats, latency, NSTDB, gating, local recalibration, d1-3B inference code, table builders, supplement tables. Nothing was retrained and no features were re-extracted from raw signals.

**Independence caveat:** the auditor read the Coding Agent's revision log and saved project notes; conclusions rest on code and artifacts, not on those notes. This is a separate review of code and saved outputs, not a replication.

## 3. Findings

### HIGH

**CODE-001**
- Severity: HIGH
- Category: Statistics / Manuscript consistency
- Location: abstract and section 3.2 of `manuscript/v2/main_v2.md`; `results/v2_2026-11/cinc2011_constant_lead_check.csv`; `paired_comparisons.csv`
- Evidence: 73 CinC 2011 records have a constant lead II, all unacceptable. Excluding them (925 records, 152 positive), d1 image probe vs 2D CNN is 0.740 vs 0.716; difference 0.024, 95% CI −0.002 to 0.050, unadjusted DeLong p = 0.069. Probe vs HuBERT-ECG probe (0.684) and vs LightGBM (0.665) remain significant unadjusted (p < 0.001).
- Problem: the text says the probe "remained highest" without a test; the abstract reports only the 998-record p-value.
- Scientific impact: the main positive finding for the probe on CinC 2011 is carried by records that any flat-line check detects.
- Recommended action: run the paired tests on the 925 records as a stated sensitivity family (Holm-adjusted) and report them beside the primary result; reword the abstract.
- Acceptance criteria: a saved comparison file for the 925-record set and abstract wording consistent with it.
- Verification status: Verified

### MEDIUM

**CODE-002**
- Severity: MEDIUM
- Category: Manuscript consistency
- Location: section 2.1; `data/extract_butqdb_1000_cohort.py`; `experiments/v2_setup.py`
- Evidence: the quota was drawn over all annotated intervals, then 59 windows were removed. Record 105001 has 54 Class 3 candidate windows inside the official interval (4,734 over all intervals); the cohort contains 1.
- Problem: the text describes a quota sample "restricted to these intervals".
- Scientific impact: subject 105 contributes one positive, and fold 3 has one positive window.
- Recommended action: correct the description, or resample inside the intervals (a new experiment needing researcher approval and a rerun of every model).
- Acceptance criteria: text matches the code path, or a new manifest with its setup report.
- Verification status: Verified

**CODE-003**
- Severity: MEDIUM
- Category: Statistics
- Location: `evaluation/paired_comparisons.py` (`cluster_bootstrap_paired`), `evaluation/model_cis.py`
- Evidence: positives come from 7 of 15 subjects; three subjects (111, 114, 122) hold 180 of 238. Per-subject positives: 100: 22, 105: 1, 111: 60, 113: 4, 114: 60, 122: 60, 124: 31. Fold 1 gives AUROC 1.000 for all 11 models. Four BUT QDB comparisons report p = 0 exactly (floor is 1/2000).
- Problem: a percentile bootstrap over 15 clusters is unreliable, and "none significantly below best" is weak evidence. Partly disclosed in limitations.
- Scientific impact: in-domain intervals and non-significance statements are less certain than their presentation suggests.
- Recommended action: report p as < 0.0005, state the effective number of informative subjects next to the claim, and add per-subject AUROCs for the mixed-class subjects.
- Acceptance criteria: revised table and wording.
- Verification status: Strongly supported

**CODE-004**
- Severity: MEDIUM
- Category: Evaluation
- Location: `experiments/external_validation.py` (recording maximum); `experiments/cinc2017_extraction.py` (up to 3 windows)
- Evidence: 110 of 278 noisy recordings have one window, against 490 of 8,234 others. Window count alone gives AUROC 0.698. On recordings with exactly 3 windows (7,561; 143 noisy), max-aggregated AUROCs rise: HuBERT probe 0.906 → 0.953, rule heuristic 0.887 → 0.951, LightGBM 0.891 → 0.923, d1 probe 0.894 → 0.924, 2D CNN 0.850 → 0.884. Under the mean over all recordings, HuBERT vs LightGBM is 0.941 vs 0.939 (DeLong p = 0.72).
- Problem: the maximum grows with window count, so it penalises short, mostly noisy recordings; the length effect is not disclosed.
- Scientific impact: absolute CinC 2017 AUROCs are biased downward and model order changes with the rule.
- Recommended action: disclose the length effect and report a length-stratified or fixed-window sensitivity table.
- Acceptance criteria: the table generated by script from saved window scores.
- Verification status: Verified

**CODE-005**
- Severity: MEDIUM
- Category: Reproducibility
- Location: working tree; `results/v2_2026-11/`
- Evidence: 41 tracked files modified and about 40 untracked, including `results/v2_2026-11`, `tests/test_v2.py` and the v2 scripts. No output records a code hash. Manifests hold absolute `/Volumes/KIOXIA/...` paths. Training ran on MPS.
- Problem: the reported v2 numbers cannot be tied to a code version; section 2.6 points readers to version history that does not contain v2.
- Scientific impact: a third party cannot reproduce v2 from the repository as it stands.
- Recommended action: commit (with researcher authorisation), stamp runs with the commit, use relative paths.
- Acceptance criteria: a commit whose tree regenerates at least one deterministic table.
- Verification status: Verified

**CODE-006**
- Severity: MEDIUM
- Category: Leakage / Manuscript consistency
- Location: `models/rule_heuristic_baseline.py`; `heuristic_refit_summary.csv`; abstract
- Evidence: the hand-set heuristic scores 0.971 in-domain; its fold-fitted replacement scores 0.905. The hand-set version stays in the paired families and in the abstract's "eight approaches reached 0.938–0.985".
- Problem: an approach whose coefficients may have been set on BUT QDB is counted among the in-domain results in the abstract.
- Scientific impact: overstates the number of leakage-free approaches in the top cluster.
- Recommended action: count seven leakage-free approaches in the abstract, or flag the eighth there.
- Acceptance criteria: abstract and Table 9 footnote agree.
- Verification status: Verified

**CODE-007**
- Severity: MEDIUM
- Category: Other (selection)
- Location: `experiments/d1_linear_probe.py`; `d1_probe_summary.csv`
- Evidence: two d1-3B representations were probed; on CinC 2011 image tokens give 0.824 and the decision state 0.422. Both were already scored on CinC 2011 records in v1 (image 0.76–0.91, state 0.41–0.48 on the 150-record draws).
- Problem: no dated record was found that the image representation was named primary before those results.
- Scientific impact: the headline representation may be outcome-selected on the external set.
- Recommended action: cite such a record, or state in section 2.6 that the choice was outcome-informed.
- Acceptance criteria: the record, or the added sentence.
- Verification status: Potential risk

### LOW

| ID | Category | Issue | Status |
|---|---|---|---|
| CODE-008 | Implementation | `run_supervised_cohort_cv.py` computes inner-validation thresholds that are never saved or used (fold 4's inner set has 1 positive); the docstring says they are saved. `SimpleImputer` is unreachable after `nan_to_num`. No numeric effect (0 NaN features). | Verified |
| CODE-009 | Other | `external_window_predictions.csv` still holds 6,780 old 12-lead rows sharing segment ids with lead II; a missed `dataset` filter would corrupt results. | Verified |
| CODE-010 | Reproducibility | Hyperparameters are unchanged since the first tracked commit (5509bf7, 2026-10-12); earlier tuning cannot be checked. | Unverifiable |
| CODE-011 | Leakage | `ptbxl_frozen_sample.csv` has no patient id, so "one record per patient" cannot be confirmed from the artifact. HuBERT-ECG pre-training overlap with PTB-XL is not resolved, which matters where it beats the probe. | Unverifiable |

## 4. Leakage and evaluation audit

- **Splits:** subject-disjoint, non-overlapping, folds unchanged from v1.
- **Preprocessing and tuning:** scalers, probe penalties and thresholds are fitted on training data only. Probe inner folds all contain both classes. External sets are not used for fitting.
- **Metrics:** positive class, record aggregation and denominators are consistent across files.

## 5. Reproducibility

- **Reproducible now:** all summary statistics from saved predictions; unit tests; installed versions match `requirements.txt`.
- **Unverified:** retraining results (MPS), feature extraction, d1-3B scores, TabPFN environment.

## 6. Manuscript consistency

Abstract numbers agree with the result files. Disagreements are CODE-001 (untested "remained highest"), CODE-002 (cohort description) and CODE-006 (count of eight).

## 7. Handoff to the Coding Agent

1. CODE-001: add the 925-record paired family from saved predictions; revise abstract and section 3.2. No retraining.
2. CODE-004: length-stratified CinC 2017 table from saved window scores; add to limitations.
3. CODE-003, CODE-006, CODE-007: wording and table changes only.
4. CODE-002: correct the text; resampling only if the researcher approves, since it requires rerunning every model.
5. CODE-005: commit after the above, with authorisation; then stamp outputs. Depends on 1–4.
6. LOW items as convenient. Rerun `evaluation/verify_v2_prose.py` and the unit tests after edits.

Evidence required for closure: the saved output file or diff named in each finding's acceptance criteria. A statement that a fix was made does not close a finding.

## 8. Review status

**MAJOR REVISION REQUIRED.** The work needed is reporting and added tests on saved outputs; no finding requires retraining unless the cohort is resampled.
