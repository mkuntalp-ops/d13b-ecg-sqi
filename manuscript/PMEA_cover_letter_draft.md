Dear Editor,

Please consider our manuscript, "In-domain rankings did not identify the best-transferring model in ECG usability screening: a patient-independent, multi-dataset benchmark of task-specific, tabular and foundation models", as a Research paper in Physiological Measurement.

**Purpose.** Signal-quality assessment is the first step of most automated ECG analysis pipelines. Key methodological work on ECG signal quality indices, including detector-agreement bSQI and SQI fusion, was published in this journal. The study asks a practical question: does a model's ranking under patient-independent in-domain evaluation tell a practitioner which model will work on another device and label definition?

**Design.**
- Eleven approaches were developed on BUT QDB with subject-grouped cross-validation and evaluated without change on the CinC 2011 and CinC 2017 datasets. The approaches are:
  - signal-quality-index classifiers, including a tabular foundation model;
  - raw-signal and image convolutional networks and a multimodal network;
  - a rule-based heuristic;
  - a zero-shot vision-language model;
  - linear probes on two frozen foundation models.
- A subset of seven was additionally evaluated under paired electrode-motion noise stress using NSTDB.
- The analyses include:
  - paired AUROC comparisons with multiplicity control;
  - leave-one-dataset-out training repeated over five seeds;
  - operating points re-set on small labelled target samples;
  - sensitivity analyses under the official BUT QDB protocol.

**Findings.**
- Ten of eleven approaches did not differ significantly from the best in-domain model, yet their external rankings disagreed.
- Multi-source training helped some architectures and harmed others.
- At realistic prevalence, thresholds re-set on 100–200 labelled recordings controlled false rejections but left sensitivity poorly determined.
- Several trained models scored completely flat leads as usable.

**Practical message.** ECG quality models need external evaluation on data whose labels match the intended use, local calibration with enough positive examples, and an explicit check for constant signals.

**Supporting material.** The later analyses were specified in a dated analysis plan before they were run, and post hoc analyses are labelled as such. The supplementary material provides:
- full feature definitions, architectures and the heuristic formula;
- confidence intervals for all models;
- predictor distributions by dataset;
- a completed TRIPOD+AI checklist.

**Related submission.** For transparency, I note that a separate manuscript, "Teacher–Student Distillation of TabPFN for Retrospective Two-Lead ECG Features: Procedure-Matched Evaluation and Cohort-Composition Sensitivity" (PMEA-107866), is currently under consideration at Physiological Measurement.
- *That study:* it examines procedure-matched knowledge distillation and label-smoothing controls for retrospective ECG classification, primarily using PTB-XL-derived lead I/II features.
- *The present submission:* it addresses a different research question, ECG usability screening and cross-dataset transfer. It uses BUT QDB, the CinC 2011 and 2017 datasets, and NSTDB noise-stress experiments.
- *Overlap:* both studies use TabPFN, but in different roles. It is the teacher model for distillation in the earlier manuscript and one of eleven compared approaches here. The manuscripts share no datasets, analyses or results, and they have distinct prediction targets, principal analyses and conclusions.

**Data, code and declarations.**
- The analysis code, saved predictions, the analysis plan and the scripts that generate every table and figure are available at https://github.com/mkuntalp-ops/d13b-ecg-sqi.
- Generative-AI assistance is disclosed in the AI-use statement of the manuscript.
- The author approves this submission. The manuscript is original, has not been published previously and is not under consideration elsewhere.
- Funding, competing interests and ethics are disclosed in the manuscript.

Sincerely,

Mehmet Kuntalp
Department of Electrical and Electronics Engineering, Dokuz Eylül University, İzmir, Turkey
mehmet.kuntalp@deu.edu.tr
