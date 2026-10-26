# Author information and declarations

**1. Author.** Mehmet Kuntalp (sole author). Affiliation: Department of Electrical and Electronics Engineering, Dokuz Eylül University, İzmir, Turkey. Corresponding author: Mehmet Kuntalp. Email: mehmet.kuntalp@deu.edu.tr. Postal address: Tınaztepe Campus, Buca, 35160 İzmir, Turkey. ORCID: 0000-0002-3381-9026.

**2. Funding and sponsor roles.** This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

**3. Financial/non-financial conflicts.** The author declares no competing interests.

**4. Ethics and consent.** This study is a secondary analysis of publicly available, de-identified ECG datasets, used under the terms of their public releases: BUT QDB (collected under IRB protocol EC:EK:05b/2018 of Brno University of Technology), the PhysioNet/Computing in Cardiology Challenge 2011 and 2017 datasets, and the MIT-BIH Noise Stress Test Database. No new data were collected from human participants, and the author had no access to identifiable information. Ethical approval was not required for this secondary analysis of public de-identified data.

**5. Author contributions.** Mehmet Kuntalp (sole author): Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review & editing. Acknowledgements: The author thanks the investigators and institutions that created and made available the BUT QDB, the PhysioNet/Computing in Cardiology Challenge 2011 and 2017 datasets and the MIT-BIH Noise Stress Test Database, and the developers of the openly released models used in this study.

**6. Data availability.** The source ECG data are publicly available from PhysioNet: BUT QDB (Nemcova et al 2020), the PhysioNet/Computing in Cardiology Challenge 2011 (Silva et al 2011) and 2017 (Clifford et al 2017) datasets, and the MIT-BIH Noise Stress Test Database (Moody et al 1984).

The repository at https://github.com/mkuntalp-ops/d13b-ecg-sqi (MIT licence) contains:
- the analysis code;
- the dated analysis plan;
- the cohort and fold manifests;
- the saved predictions of every model;
- the scripts that regenerate every table and figure from those predictions.

The following are not redistributed:
- raw ECG waveforms;
- the weights of LiquidAI/d1-3B, HuBERT-ECG and TabPFN, which are available from their providers under their own licences (HuBERT-ECG: CC BY-NC 4.0).

**7. AI disclosure.** Generative-AI tools were used in three distinct roles.

- *Antigravity (Google):* used to write an initial version of the analysis code and of the manuscript draft.
- *Claude Code (Anthropic):* used to audit and correct that code, including independent recomputation of the reported metrics and re-execution checks; to write the later analysis, table-generation, figure-generation and verification scripts; and to revise and edit the manuscript text.
- *Perplexity:* used as an LLM-as-a-judge. It reviewed successive versions of the manuscript, Supplementary Material and reporting checklist and commented on methodological, statistical and reporting problems, numerical consistency and reference accuracy.

Each critique was checked against the stored study outputs before any change was accepted, and suggestions not supported by the data were not adopted.

The author defined the research question, scientific objectives, datasets, evaluation protocol and interpretation framework; executed all code locally; reviewed the implementation, analytic outputs, reported results and references; and retains full responsibility for the study design, data handling, methods, analyses, results, interpretations, originality and integrity of this manuscript.

**8. Related submission.** A separate manuscript by the same author, PMEA-107866 ("Teacher–Student Distillation of TabPFN for Retrospective Two-Lead ECG Features"), is under consideration at Physiological Measurement. The two manuscripts share no datasets, analyses or results; both use TabPFN, in different roles (see cover letter).

**9. Reporting guideline.** TRIPOD+AI; the completed checklist is provided as a supplementary file.
