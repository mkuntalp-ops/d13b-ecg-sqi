# Re-audit and second review (2026-10-13)

Inputs: the second reviewer's report (blind spots: small cohort, 12-lead vs single-lead labels, quota prevalence,
feature shift, zero-shot capacity vs adaptation, latency assumptions, missing external data, retrospective thresholds).
Point-by-point answers: `manuscript/response_to_reviewer_2.md`.

## 1. Errors found in the re-audit

| Finding | Effect | Fix | Evidence |
|---|---|---|---|
| Renderer guessed the unit (`|x| > 20` divided by 1,000) | 71 CinC 2011 lead-II records (all unacceptable; 22 in the reported subset) and 22 NSTDB −6 dB windows drawn as flat lines | Unit guess removed; callers pass mV (float32 kept so BUT strips are pixel-identical); CinC and NSTDB strips re-rendered; every affected result recomputed | `data/ecg_rendering.py`, `experiments/rerender_images.py`, `experiments/rerun_after_render_fix.sh`, `results/rerender_images_report.csv` |
| Result changes from the fix | CinC reported-subset AUROC changed by ≤ 0.001; 2D CNN seed-3 CinC sensitivity 40.0% → 20.0%; NSTDB −6 dB 2D CNN 0.115 → 0.012, multimodal 0.053 → 0.006, d1 0.762 → 0.699 (the "rise at −6 dB" was an artefact) | Tables 2–7 regenerated; text updated | `results/_backup_pre_render_fix_2026-10-13/` |
| Latency hardware misreported | Measurements ran on an Apple M4 Mac mini (10-core CPU, 10-core GPU, 16 GB), not an M2 with 24 GB | Manuscript, Table 4 labels and Figure 4 note corrected | `system_profiler` on the profiling machine |
| Sampling description | Quotas are per subject (≤ 50 usable, ≤ 60 unusable, evenly spaced); 17 of 18 records contribute (100002 none); resampling is FFT-based, not polyphase | Section 2.1 rewritten | `data/extract_butqdb_1000_cohort.py` |
| Training details missing | AdamW 1e-3/1e-3, pos_weight, batch 32, 25 epochs, cosine schedule, no early stopping; balanced LR/LightGBM; fixed image normalisation (0.95/0.15) | Section 3.2 | `models/*.py` |
| Rendering described as 10 mm/mV | Strips are scaled per strip (1st–99th percentile, ≥ 1 mV) | Section 3.2, README | `data/ecg_rendering.py` |
| Flatline SQI semantics | CinC 2011 lead II keeps 5 µV quantisation steps (native 500 Hz); acceptable records have median flatline 0.28 vs 7e-5 in resampled BUT QDB | Section 3.1, 4.2, Limitation 9 | check in this session |
| Duration wording | 99.39 h is the quality-annotated part of 478.7 h of recording | Section 2.1 | BUT QDB headers and annotations |
| Inner thresholds in `run_supervised_cohort_cv.py` | Computed but never saved (docstring said saved) | Noted; superseded by `experiments/nested_thresholds.py` | code |
| d1 tie sensitivity | d1 scores are coarse; stored copies differing by 5e-7 move AUROC by ≤ 0.004 | Probe table reads the Table 2d file | `experiments/d1_linear_probe.py` |

A scripted edit during this revision deleted Sections 2–4 of the manuscript; it was rebuilt from git HEAD by
re-applying the day's edits, and the untouched prefix and suffix were verified identical to the damaged file.

## 2. Analyses added for the second review

| Reviewer point | Analysis | Result (headline) | Script / output |
|---|---|---|---|
| Small cohort; diversity | CinC 2017 external cohort (8,512 recordings, 3.3% noisy); per-subject sensitivity | CinC 2017 AUROC 0.773–0.887 (heuristic best, 1D CNN worst); 1D CNN sensitivity 14–100% across subjects | `experiments/cinc2017_extraction.py`, `experiments/external_validation.py`, `experiments/prevalence_subjects_features.py`; Tables 8, 8b, 1f |
| 12-lead vs lead II | All 12 leads of the 565 CinC records, max/mean aggregation | Mean over leads: 2D CNN 0.781 → 0.852, LightGBM 0.750 → 0.877 (random subsets) | `experiments/cinc2011_twelve_lead_extraction.py`; Table 2e |
| Quota prevalence | Natural prevalence of the candidate pool (16.0%; 2.8% without record 105001); re-weighted AP/PPV | CNN PPV 0.71–0.77 at 2.8%; d1 0.28 | Table 1e |
| Feature shift | Univariate AUROC of every SQI in BUT, CinC 2011, CinC 2017 | HF, powerline, ZCR, flatline reverse in CinC 2011; bSQI strong in CinC 2017 | Table 2g |
| d1 capacity vs adaptation | Linear probes on frozen d1 decision state and image tokens | Image-token probe 0.979 (BUT), 0.777 / 0.756–0.907 (CinC 2011), 0.910 (CinC 2017 subset) | `experiments/d1_linear_probe.py`; Table 7b |
| Latency assumptions | Raw-to-decision, CPU vs MPS, raster renderer | 1D CNN 1.41 ms CPU from native signal; raster 0.63 ms vs Matplotlib 58.4 ms | `experiments/latency_extended.py`; Table 4b |
| Retrospective thresholds | Nested selection on BUT QDB; BUT-selected thresholds applied to CinC 2011/2017 | CNNs keep operating points under nesting; SQI models optimistic; thresholds do not transfer | `experiments/nested_thresholds.py`; Tables 1c, 2f |

## 3. Still open

- A large multi-lead dataset with lead-wise usability labels (PTB-XL noise labels mark noise presence, not unusability; only 501 records are local).
- Fine-tuning or few-shot prompting of d1-3B.
- Accuracy of the image models on fast raster strips (only timing was measured).
- Prospective threshold validation.
- Heuristic coefficient provenance; NSTDB SNR definition not reproduced.
