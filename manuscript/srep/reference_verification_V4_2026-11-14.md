# Reference verification, V4 (2026-11-14)

Supersedes nothing: the V3 report (`reference_verification_V3_2026-11-13.md`) and its Crossref output stay as the V3
record. V4 extends the **claim-support** check, which V3 had done only for the six entries the review singled out, to
the priority groups named in the follow-up review: motivation and research gap, dataset descriptions, methods sources,
external-evaluation and comparison claims. This is a check by the coding assistant, not an independent reviewer
assessment.

## Method

For each entry, the sentence(s) citing it in `main_srep_template.md` were extracted. Each was compared with the source
at the level stated in the "Evidence" column:

- **abstract**: the PubMed abstract;
- **full text**: the open-access full text (Europe PMC), searched for the specific claim;
- **record**: the data provider's page;
- **title**: title and venue only.

Retrieved on 2026-11-14.

## Results: claim support

| Key | Claim in the manuscript (short) | Evidence | Verdict |
| --- | --- | --- | --- |
| butqdb | 15 subjects, 9 women and 6 men, aged 21–83 | record (PhysioNet BUT QDB 1.0.0); median age from the shipped subject-info.csv (checked by `verify_v2_prose.py`) | Supported |
| butqdb, smital2026 | IRB EC:EK:05b/2018; informed written consent | record ("Informed written consent was obtained"); full text of Smital 2026 (IRB protocol, informed consent) | Supported |
| smital2026 | official evaluation protocol and train/test partition | abstract ("standardized evaluation protocol") | Supported |
| smital2020 → smital2026 | "Three experts annotated the recordings" | Smital 2026 full text: three independent experts. Smital 2020 is an SNR-based method paper that defines the three quality levels, but it does not describe the BUT QDB annotation | **Citation corrected in V4**: the annotation is now cited to Smital 2026, and the class definitions to Smital 2020 |
| smital2020 | three levels (full waveform, QRS only, unusable) | abstract | Supported |
| clifford2017 | 8,528 recordings, 9–61 s, label classes | abstract (Europe PMC); the 300-Hz rate comes from the record headers | Supported |
| silva2011, cinc2011protocol | 12-lead 10-s acceptability labels | primary record (V3) | Supported. The counts 998/225/773 are counted from the data files |
| ptbxl | ethics approval PTB-2020-1 | full text ("approved the publication of the anonymous data … (PTB-2020-1)") | Supported |
| nstdb | electrode-motion noise, records 118/119, SNR +24 to −6 dB | record (V3) | Supported |
| vancalster | targets 0 and 1; slope < 1 = too extreme; intercept < 0 = overestimation; calibration-in-the-large irrelevant at internal validation; ≥200 events and ≥200 non-events suggested | full text (each statement found) | Supported |
| moeyersons | cross-dataset evaluation reported | abstract (trained on one study, tested on two others) | Supported |
| moeyersons | "multi-source development, often recommended for robustness" | abstract (training on all three datasets improved results) | One study does not show "often recommended". **Reworded in V4**: "which improved results in an earlier cross-dataset study" |
| guo2022, subbaswamy | under dataset shift, "ranking, calibration and operating points" change | Guo 2022 abstract: AUROC drop, calibration error, false positives and false negatives; Subbaswamy & Saria: review of dataset shift (record) | "Ranking" is not shown directly. **Reworded in V4**: "discrimination, calibration and operating points" |
| heartbeit, stmem, ecgfounder, ecgfm | "offer frozen representations that can be read out with a linear probe" | ECG-FM abstract (linear probing); HeartBEiT abstract and full text (pre-training and fine-tuning; no linear probe or frozen readout found) | Too specific for HeartBEiT. **Reworded in V4**: "pre-trained representations that can be fine-tuned or, frozen, read out with a linear probe" |
| tabpfn | "remove most of the tuning burden" | full text: a single forward pass outperforms baselines tuned for 4 h | Supported |
| cmaecg | ECG combined with text for quality assessment | abstract (ECG signals with textual instructions, pre-trained LLM) | Supported |
| behar | SQIs used to reduce false arrhythmia alarms | abstract | Supported |
| liu2022, huang2024, herraiz | learned from raw signals, spectrograms or images | abstracts: wavelet scattering + LSTM; deep SVDD on ECG; CNN on wavelet scalograms | Supported (as a group) |
| morgado | variant developed on Challenge 2011 data | abstract | Supported |
| shahriari, liu2021, zhao2018, zhu2020, hayn, johannesen, jekova, chen2012 | SQI/learned variants; Challenge 2011 focus issue; single-lead and portable | title | Consistent with the title. Not checked further |
| satija, clifford2012, orphanidou, li2008 | quality gate as the first pipeline step; hand-crafted SQIs and fusion | title (SQI review and method papers) | Consistent with the title. Not checked further |
| merl, kmerl, ecgchat, benchecg | zero-shot classification; arbitrary-lead inputs; report generation; foundation-model benchmark | title (K-MERL, ECG-Chat and BenchECG records also in V3) | Consistent with the title. Not checked further |
| ribeiro, hannun | 12-lead diagnosis; single-lead ambulatory rhythm classification | title | Consistent with the title |
| delong, holm, tripod, albert1984, pantompkins | methods sources | V3 record check; Pan–Tompkins implementation difference disclosed in V3 | Supported as method sources |
| drew, siamquality, hu2025, stmem, ecgchat | see V3 report | | |
| hubert, tabpfn3, d13b | model releases (software, not evidence of performance) | V3 | Cited only as the releases that were used |

## Retraction and correction status

- Unchanged from V3: no Crossref retraction notice.
- Correction notices exist for TRIPOD+AI, Ribeiro 2020 and Hannun 2019; the cited content is unaffected.
- Only Crossref update notices were screened. There was no screening in Retraction Watch or the publisher pages, so "no retraction" means "no retraction notice registered in Crossref on the check date".

## Preprints

As in V3, no peer-reviewed version of either preprint was found on 2026-11-13. The manuscript now says this as "searches on 13 November 2026 found no peer-reviewed version" instead of "have no peer-reviewed version". **Model releases (V4 check, `results/v2_2026-11/model_provenance_V4_2026-11-14.csv`).**
- The SHA-256 of each local weight file was computed (`shasum -a 256`) and compared with the Git LFS object id that the Hugging Face API (`/api/models/<repo>/paths-info/<revision>`) returns at the cited revision. Sizes were compared the same way.
- Both match:
  - HuBERT-ECG `model.safetensors`: 05bc1b13…a1ae, 372,518,672 bytes, revision 8ffa1ad…;
  - TabPFN-3 `tabpfn-v3-classifier-v3_default.ckpt`: d0d865d5…3988, 212,804,803 bytes, revision 24a16a8….
- The local download metadata records the same revisions.
- The experiment code loads these files through default paths that an environment variable can override. No run log records an override, but the logs do not prove its absence.

## Not done

- The entries marked "title" were not checked against abstracts or full texts.
- Full texts of the IEEE and Physiol. Meas. papers were not read.
