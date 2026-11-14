# Reference verification, V6 addendum (2026-11-14)

Follow-up to review finding PAPER-015. The V3 and V4 reports stay as records of their rounds; the V3 report now
carries a dated correction note at the sentence concerned. This is a check by the coding assistant, not an
independent reviewer assessment.

## PAPER-015: the ECGFounder correction notice

**What was wrong in the V3 report.** It said that the earlier review's mention of corrections to references 26 and 50
"refers to" the TRIPOD+AI, Ribeiro and Hannun notices. That explanation was a guess and it was wrong: ECGFounder
(Li et al., NEJM AI 2025, doi:10.1056/AIoa2401033) has its own correction, which the V3 check missed.

**Why it was missed.** The V3 check read update notices from Crossref only. Crossref registers the erratum as an
ordinary journal article with no "update-to" link to the original (checked again on 2026-11-14 for both DOIs), so a
Crossref-only screen cannot find it.

**The record.**
- Erratum: "An Electrocardiogram Foundation Model Built on Over 10 Million Recordings", NEJM AI 2(10), AIx2500913,
  doi:10.1056/AIx2500913, PubMed 42932615, publication type "Published Erratum".
- Europe PMC links it as "Erratum for" PubMed 40771651 (the cited article), and lists "Erratum in" on the cited article.

**Content of the correction: NOT READ.** PubMed and Crossref carry no abstract for the erratum, and the publisher page
(ai.nejm.org) returned a bot check (HTTP 403) that was not bypassed. What the erratum changes is therefore unknown here.

**Effect on the citation.** The manuscript cites ECGFounder twice, both times in a list:
1. as one of several "ECG-specific models [that] offer pre-trained representations that can be fine-tuned or, frozen,
   read out with a linear probe";
2. as one of the "other ECG foundation models [that] were not evaluated".

Neither sentence uses a number, a performance result or a method detail from the paper; they rely only on the paper
describing a pre-trained ECG foundation model, which its title and abstract state (PubMed 40771651: "designed to be
both an effective out-of-the-box solution and easily fine-tunable"). An erratum, which does not withdraw the article,
is unlikely to affect that. This is an inference from how the paper is used, not a reading of the erratum; the author
should open the notice once before submission.

## Second-source screen of all references (new in V6)

Because the Crossref-only screen had this gap, every DOI in `references_srep.md` was also looked up in Europe PMC, whose
records carry PubMed's erratum and retraction links (`reference_europepmc_notices_V6_2026-11-14.csv`):

- 37 of the 42 DOIs are indexed. Notices found: TRIPOD+AI (erratum, BMJ 385:q902), ECGFounder (erratum, above),
  Ribeiro 2020 (erratum, Nat Commun 11:2227), Hannun 2019 (erratum, Nat Med 25:530). No retraction and no expression
  of concern.
- 5 DOIs are not indexed there (BenchECG, the BUT QDB PhysioNet record, Albert & Anderson 1984, K-MERL, ECG-Chat) and
  10 entries have no DOI; for these the screen says nothing.
- TRIPOD+AI, Ribeiro and Hannun: as in V3, the cited content is the reporting guideline as a whole and the general
  capability of deep networks for ECG diagnosis, which the notices do not withdraw. Their contents were not re-read in V6.

"No retraction" therefore means: none registered in Crossref or PubMed/Europe PMC for the indexed entries on 2026-11-14.

## Model weights (PAPER-008)

Unchanged from V4: the local weight SHA-256 values equal the Hugging Face LFS object ids at the cited revisions
(`model_provenance_V4_2026-11-14.csv`). The reviewer could not open the Hub API; the command to repeat the check is

    curl -s -X POST https://huggingface.co/api/models/<repo>/paths-info/<revision> -d "paths=<file>"

for `Edoardo-Coppola/hubert-ecg-base` / `8ffa1ad3593366ca45f5351d75317d4ff8f14465` / `model.safetensors` and
`Prior-Labs/tabpfn_3` / `24a16a89d245878b846555110985634aa2e656d7` / `tabpfn-v3-classifier-v3_default.ckpt`; the
field `lfs.oid` is the SHA-256.
