# Reference candidates from the external revision guide — existence check (2026-11-06)

Source of the candidates: the external "EKG Makalesi Geliştirme Projesi" guide (LLM-written). Each record below was
looked up in PubMed (E-utilities via the PubMed connector), Crossref (api.crossref.org) or the arXiv API.

"Exists" means the bibliographic record was found and matches authors/title/venue. It does NOT mean the paper supports
the sentence it will be attached to; that check ("claim support") is done separately when the reference is placed in
the text, against the abstract or full text. Nothing here enters the bibliography until both are done.

| # | Reference (as verified) | Identifier | Exists | Guide error / note | Claim support |
|---|---|---|---|---|---|
| 1 | Clifford GD, Moody GB 2012 Signal quality in cardiorespiratory monitoring. Physiol. Meas. 33 E01 | doi:10.1088/0967-3334/33/9/E01 | yes (Crossref) | Guide gave no title; editorial of the 2012 special issue | not checked |
| 2 | Hayn D et al 2012 QRS detection based ECG quality assessment. Physiol. Meas. 33 1449 | PMID 22902864; doi:10.1088/0967-3334/33/9/1449 | yes | — | not checked |
| 3 | Johannesen L, Galeotti L 2012 Automatic ECG quality scoring methodology: mimicking human annotators. Physiol. Meas. 33 1479 | PMID 22902927; doi:10.1088/0967-3334/33/9/1479 | yes | — | not checked |
| 4 | Jekova I et al 2012 Threshold-based system for noise detection in multilead ECG recordings. Physiol. Meas. 33 1463 | PMID 22902891; doi:10.1088/0967-3334/33/9/1463 | yes | — | not checked |
| 5 | Chen Y, Yang H 2012 Self-organized neural network for the quality control of 12-lead ECG signals. Physiol. Meas. 33 1399 | PMID 22902675; doi:10.1088/0967-3334/33/9/1399 | yes | — | not checked |
| 6 | Zhu et al 2020 Quality evaluation of signals collected by portable ECG devices using dimensionality reduction and flexible model selection. Physiol. Meas. 41 105001 | PMID 32947264; doi:10.1088/1361-6579/abba0b | yes | Exists (the article number coincides with BUT QDB record 105001) | not checked |
| 7 | Ding C et al 2024 SiamQuality: a ConvNet-based foundation model for photoplethysmography signals. Physiol. Meas. 45 085004 | doi:10.1088/1361-6579/ad6747 | yes (Crossref) | **PPG model, not ECG**; cite only as a signal-quality foundation-model precedent in another modality | not checked |
| 8 | Nguyen HD, Phan 2025 CMA-ECG: cross-modal attention for enhanced ECG quality assessment and denoising. Physiol. Meas. 46 115001 | doi:10.1088/1361-6579/ae15e4 | yes (Crossref) | Full author list to be taken from the record | not checked |
| 9 | Behar J et al 2013 ECG signal quality during arrhythmia and its application to false alarm reduction. IEEE TBME 60 1660 | PMID 23335659; doi:10.1109/TBME.2013.2240452 | yes | — | not checked |
| 10 | Liu C et al 2018 Dynamic ECG signal quality evaluation based on the generalized bSQI index. IEEE Access 6 41892–41902 | doi:10.1109/access.2018.2860056 | yes (Crossref) | — | not checked |
| 11 | Zhao Z, Zhang Y 2018 SQI quality evaluation mechanism of single-lead ECG signal based on simple heuristic fusion and fuzzy comprehensive evaluation. Front. Physiol. 9 727 | PMID 29962962; doi:10.3389/fphys.2018.00727 | yes | — | not checked |
| 12 | Kuzilek J et al 2014 Independent component analysis and decision trees for ECG holter recording de-noising. PLoS ONE 9 e98450 | PMID 24905359; doi:10.1371/journal.pone.0098450 | yes | **De-noising paper, not SQI fusion**; guide placed it under classic SQI. Use only if the text is about artefact removal | not checked |
| 13 | Morgado E et al 2015 Quality estimation of the electrocardiogram using cross-correlation among leads. Biomed. Eng. Online 14 59 | PMID 26091857; doi:10.1186/s12938-015-0053-1 | yes | **Article number is 59, guide wrote 53** | not checked |
| 14 | Shahriari Y et al 2018 Electrocardiogram signal quality assessment based on structural image similarity metric. IEEE TBME 65 745 | PMID 28644794; doi:10.1109/TBME.2017.2717876 | yes | — | not checked |
| 15 | Liu et al 2021 ECG quality assessment based on hand-crafted statistics and deep-learned S-transform spectrogram features. Comput. Methods Programs Biomed. 208 106269 | doi:10.1016/j.cmpb.2021.106269 | yes (Crossref) | Title from Crossref (truncated there; complete from the record) | not checked |
| 16 | Mondal et al 2024 Fast CNN-based electrocardiogram signal quality assessment using Fourier magnitude spectrum … IEEE Sens. Lett. 8 1–4 | doi:10.1109/lsens.2024.3371479 | probable | Guide gave no title; this is the matching Mondal 2024 Sensors Letters paper — confirm with the authors' list | not checked |
| 17 | Mondal et al 2025 Automatic ECG signal quality assessment using convolutional neural networks … Biomed. Signal Process. Control 108 107876 | doi:10.1016/j.bspc.2025.107876 | yes (Crossref) | — | not checked |
| 18 | Vaid A et al 2023 A foundational vision transformer improves diagnostic performance for electrocardiograms. npj Digit. Med. 6 108 | PMID 37280346; doi:10.1038/s41746-023-00840-9 | yes | — | not checked |
| 19 | Na Y et al 2024 Guiding masked representation learning to capture spatio-temporal relationship of electrocardiogram (ST-MEM) | arXiv:2402.09450 | yes (arXiv) | **First author is Na, guide wrote "Park et al 2024"**; venue (ICLR 2024) to be confirmed | not checked |
| 20 | Li J et al 2025 An electrocardiogram foundation model built on over 10 million recordings with external evaluation across multiple domains (ECGFounder). NEJM AI 2 | doi:10.1056/aioa2401033 | yes (Crossref) | Crossref also lists 10.1056/aix2500913 (same title, no author: likely a related item) — cite aioa2401033 | not checked |
| 21 | McKeen K et al 2025 ECG-FM: an open electrocardiogram foundation model. JAMIA Open 8 ooaf122 | PMID 41113504; doi:10.1093/jamiaopen/ooaf122 | yes | Already cited in v1 | — |
| 22 | Liu C et al 2024 Zero-shot ECG classification with multimodal learning and test-time clinical knowledge enhancement (MERL) | arXiv:2403.06659 | yes (arXiv) | ICML 2024 venue to be confirmed | not checked |
| 23 | Liu R et al 2024 Teach multimodal LLMs to comprehend electrocardiographic images (PULSE, ECGBench) | arXiv:2410.19008 | yes (arXiv) | — | not checked |
| 24 | Liu C et al 2025 Knowledge-enhanced multimodal ECG representation learning with arbitrary-lead inputs (K-MERL) | arXiv:2502.17900 | yes (arXiv) | **Guide wrote "Li et al"; first author is Che Liu.** EMNLP Findings venue to be confirmed | not checked |
| 25 | Zhao Y et al 2024 ECG-Chat: a large ECG-language model for cardiac disease diagnosis | arXiv:2408.08849 | yes (arXiv) | ICME 2025 venue to be confirmed | not checked |
| 26 | Van Calster B et al 2019 Calibration: the Achilles heel of predictive analytics. BMC Med. 17 230 | PMID 31842878; doi:10.1186/s12916-019-1466-7 | yes | Full text read (PMC6912996). The 200 + 200 minimum is suggested for a *precise flexible calibration curve* and is attributed to earlier work, not to this paper; it does not support the guide's use (instability of small-sample threshold re-setting). Supports: definitions and targets of calibration intercept (0; negative = overestimation) and slope (1; < 1 = too extreme); intercept and slope are defensible in small data; calibration-in-the-large is irrelevant at internal validation, relevant externally | checked for: intercept/slope definitions; CITL only external |
| 27 | Ribeiro AH et al 2020 Automatic diagnosis of the 12-lead ECG using a deep neural network. Nat. Commun. 11 1760 | PMID 32273514; doi:10.1038/s41467-020-15432-4 | yes | — | not checked |
| 28 | Wagner P et al 2020 PTB-XL, a large publicly available electrocardiography dataset. Sci. Data 7 154 | PMID 32451379; doi:10.1038/s41597-020-0495-6 | yes | Not used in this study's data; cite only if the text needs it | not checked |
| 29 | Hannun AY et al 2019 Cardiologist-level arrhythmia detection and classification in ambulatory ECGs using a deep neural network. Nat. Med. 25 65 | PMID 30617320; doi:10.1038/s41591-018-0268-3 | yes | — | not checked |
| 30 | Drew BJ et al 2014 Insights into the problem of alarm fatigue with physiologic monitor devices: a comprehensive observational study … PLoS ONE 9 e110274 | doi:10.1371/journal.pone.0110274 | yes (Crossref) | Guide gave no details | not checked |

Summary: 30 candidates checked; 29 records found (1 probable: #16). Guide errors found: wrong article number (#13),
wrong first author (#19, #24), mischaracterised content (#7 PPG not ECG, #12 de-noising not SQI fusion).
Conference venues of arXiv papers (#19, #22, #24, #25) are not yet confirmed; cite the arXiv record until they are.
The full scite verification was not available (paid plan required).


## Claim-support check (2026-11-08)

Each abstract was read (PubMed efetch, Crossref or arXiv API). "Supported claim" is the only statement the reference is used for
in the v2 text. A reference without a readable abstract is not cited.

| # | Supported claim (from the abstract) | Decision |
|---|---|---|
| 1 | Editorial, no abstract | Not cited |
| 2 | Quality measures from basic signal properties and QRS-detection robustness, CinC 2011 | Cited |
| 3 | Two-step: reject macroscopic errors (absent signal, saturation), then quantify noise | Cited |
| 4 | Threshold tests per noise type in 12-lead ECG | Cited |
| 5 | 12-lead quality control via VCG, recurrence analysis and self-organizing map | Cited |
| 6 | Portable-device ECG quality with feature reduction and model selection | Cited |
| 7 | PPG self-supervised foundation model for low-quality signals (not ECG) | Cited, as PPG precedent only |
| 8 | Multimodal framework combining ECG and text for denoising and quality assessment | Cited |
| 9 | SQIs + SVM across arrhythmias, false-alarm reduction | Cited |
| 10 | No abstract retrievable | Not cited |
| 11 | Heuristic SQI fusion + fuzzy evaluation, single lead | Cited |
| 12 | De-noising, off-topic | Not cited |
| 13 | 12-lead quality via cross-covariance among leads | Cited |
| 14 | Image-based quality via structural similarity, mimics clinician annotation | Cited |
| 15 | Hand-crafted statistics + deep S-transform features, multi-lead | Cited |
| 16, 17 | No abstract retrievable | Not cited |
| 18 | Vision transformer pre-trained on 8.5 million ECGs (HeartBEiT) | Cited |
| 19 | Self-supervised ECG representation learning (ST-MEM) | Cited, arXiv record |
| 20 | General-purpose ECG foundation model (ECGFounder) | Cited |
| 21 | ECG-FM (already in v1) | Cited |
| 22 | Zero-shot ECG classification with multimodal learning (MERL) | Cited, arXiv record |
| 23 | Multimodal LLMs for ECG images (PULSE) | Cited, arXiv record |
| 24 | Knowledge-enhanced multimodal ECG representation, arbitrary leads | Cited, arXiv record |
| 25 | ECG-language model (ECG-Chat) | Cited, arXiv record |
| 26 | Calibration needs assessing at external validation; calibration curves need large samples. The 200+200 figure is NOT used | Cited |
| 27 | Deep network on 12-lead ECG | Cited |
| 28 | Dataset not used in this study | Not cited |
| 29 | Deep network on single-lead ambulatory ECG | Cited |
| 30 | ICU monitor alarm burden, alarm fatigue | Cited |
