# Cover letter

Mehmet Kuntalp
Department of Electrical and Electronics Engineering, Dokuz Eylül University
Tınaztepe Campus, Buca, 35160 İzmir, Turkey
mehmet.kuntalp@deu.edu.tr · ORCID 0000-0002-3381-9026

21 November 2026

The Editors
*Scientific Reports*

Dear Editors,

Please consider the enclosed Article, "In a retrospective benchmark, in-domain performance did not identify the electrocardiogram quality models that ranked best on heterogeneous external datasets", for publication in *Scientific Reports*.

Automatic quality checks decide which electrocardiogram segments reach downstream analysis, in wearable monitoring as well as in clinical pipelines. Such models are usually chosen by their performance on data like those used to develop them. We compared eleven approaches, from signal-quality-index classifiers and task-specific neural networks to a zero-shot vision-language model and linear probes on two frozen foundation models, under one subject-grouped protocol, and applied them unchanged to external datasets with different acquisition, labels and evaluation units. In-domain, none of the 28 pairwise comparisons among the eight approaches with the highest AUROCs was significant after Holm adjustment, although one of them, a hand-set rule, may not be independent of the development data; externally their rankings disagreed, and the retrospective agreement between in-domain and external rank was moderate on one dataset and not detected on the other. Restoring 53 development windows that the cohort quota had omitted changed the leader on one external dataset, and we report this sensitivity analysis alongside the primary result. We show that a small number of dataset artefacts, such as flat leads and recording length, moved the external rankings, that a harmonised input pipeline removed most of the spread between models on CinC 2011 without restoring the rank agreement, and that operating points chosen on development data rarely met their false-reject target externally.

We believe the work suits *Scientific Reports* because its main contribution is a technically sound, reproducible evaluation rather than a new model, and its retrospective conclusion, that in-domain benchmarks did not identify the externally best-ranked quality models in these settings, is relevant to anyone developing or deploying biosignal quality checks. All datasets are public, and the code, saved predictions and the scripts that generate every table and figure are publicly available in the project repository (see the Code availability statement). Decisions taken after results were seen are listed in the Methods, and the analyses removed from the main text for length are reported unchanged in the Supplementary Information.

The manuscript has not been published and is not under consideration elsewhere. The author has had no prior discussions with a *Scientific Reports* Editor about this work.

Suggested reviewers are provided in the submission system.

Thank you for considering this submission.

Yours sincerely,

Mehmet Kuntalp
