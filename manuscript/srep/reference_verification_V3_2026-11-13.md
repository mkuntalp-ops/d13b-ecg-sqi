# Reference verification, V3 (2026-11-13)

Follow-up to the third review (PAPER-008 and the references left UNVERIFIED in its section 7). Scope: every entry of
`references_srep.md` (52 entries after V3; the v2 list has the same keys). This is a check by the coding assistant, not an
independent reviewer assessment.

## Method

1. **Bibliographic record.** Each entry with a DOI was resolved with Crossref (DataCite for the PhysioNet DOI) by
   `evaluation/crossref_reference_check.py`, which compares title (similarity ≥ 0.85), first author, year, volume
   and first page with the entry and lists update notices (corrections, retractions).
   Machine-readable output: `reference_crossref_check_V3_2026-11-13.csv`.
2. **Entries without a DOI** were checked against the primary record named below.
3. **Claim support** was checked for the entries the review singled out (Drew, SiamQuality, Pan–Tompkins, Hu 2025,
   ST-MEM, ECG-Chat), using the PubMed abstract or the primary record against the sentence that cites the entry.

## Results

**Bibliographic records.**
- Of the 42 entries with a DOI, 38 matched on all fields.
- Four were flagged and explained:
  - Li 2008: Crossref "issued" is the online date (10 Dec 2007); the print issue is Physiol. Meas. 29(1), 2008, as cited.
  - Subbaswamy & Saria: Crossref has no print metadata; PubMed 31742354 gives Biostatistics 21(2):345–352, 2020, as cited.
  - Van Calster: Crossref lists the STRATOS topic group as first "author"; the article authors are as cited.
  - Smital 2026: Crossref adds volume 13, article 1567. **Corrected in V3.**
- No retraction notices. Three entries have correction notices, and the cited content is unaffected: TRIPOD+AI (bmj.q902), Ribeiro 2020 (author correction 10.1038/s41467-020-16172-1) and Hannun 2019 (publisher correction 10.1038/s41591-019-0359-9). The review's mention of corrections to references 26 and 50 refers to these.
  - **Correction added 2026-11-14 (V6, review PAPER-015):** the sentence above was wrong. ECGFounder (Li et al., NEJM AI 2025) has its own erratum (doi:10.1056/AIx2500913), which this Crossref-only check missed. See `reference_verification_V6_2026-11-14.md`.

**Entries without a DOI (10), plus the new albert1984.**

| Key | Primary record | Status |
| --- | --- | --- |
| silva2011 | Paper PDF on cinc.org, footer "Computing in Cardiology 2011;38:273−276" | Verified |
| nstdb | PhysioNet NSTDB page: Computers in Cardiology 1984;11:381–384 | Verified |
| delong | PubMed 3203132: Biometrics 44(3):837–845, 1988 | Verified |
| holm | Scand. J. Stat. 6(2):65–70, 1979 (JSTOR 10.2307/4615733); confirmed by several bibliographic records, not by the JSTOR page itself (it did not load) | Verified (secondary records) |
| merl | PMLR v235 page liu24bg: authors, title, pp. 31949–31963 | Verified |
| stmem | ICLR 2024 proceedings page (proceedings.iclr.cc); the srep URL now points to the proceedings | Verified, accepted conference paper |
| cinc2011protocol | moody-challenge.physionet.org/2011 (challenge protocol); the PhysioNet challenge-2011 page describes the same protocol | Verified (web resource) |
| d13b | Hugging Face LiquidAI/d1-3B, revision 051bcc4… (checked in an earlier round) | Verified (model release) |
| hubert | Hugging Face Edoardo-Coppola/hubert-ecg-base, revision 8ffa1ad3593366ca45f5351d75317d4ff8f14465, matching the local download metadata | Verified (model release), **new in V3** |
| tabpfn3 | Hugging Face Prior-Labs/tabpfn_3, revision 24a16a89d245878b846555110985634aa2e656d7, which contains tabpfn-v3-classifier-v3_default.ckpt and matches the local download metadata; tabpfn package 9.0.0 in the TabPFN environment | Verified (model release), **new in V3** |
| albert1984 | Crossref 10.1093/biomet/71.1.1: Biometrika 71(1):1–10 | Verified, **new in V3** |

**Preprints (PAPER-008).**
- The searches on 2026-11-13 found no peer-reviewed version of either preprint:
  - HuBERT-ECG: medRxiv 10.1101/2024.11.14.24317328;
  - TabPFN-3: arXiv 2605.13986, revised 28 May 2026.
- At the author's decision, both were removed from the bibliography. The keys `hubert` and `tabpfn3` now cite the exact model releases that were used.
- The TabPFN Nature paper (`tabpfn`) is cited only for TabPFN in general, never as documentation of TabPFN-3.
- Not finding a final version does not prove that none exists.

**Claim support.**
- **Drew 2014 (PubMed 25338067):** 2,558,760 alarms in 461 ICU patients over 31 days, and 88.8% of the annotated arrhythmia alarms were false. This supports "monitors in intensive care generate a large alarm burden".
- **SiamQuality (PubMed 39048103):** the paper describes a self-supervised foundation model built to "handle low-quality issue" in PPG. The wording was sharpened to "proposed to cope with low-quality photoplethysmography".
- **Hu 2025 (PubMed 41374526):** a Transformer on single-lead ECG for quality assessment. This supports "learned from raw signals".
- **Pan–Tompkins:** the code (`features/qrs_detectors.py`) is a simplified re-implementation. It keeps the band-pass filtering, differentiation, squaring and moving-window integration, but uses a single global threshold instead of the original adaptive thresholds and search-back. **The Methods text now says so in V3.**
- **ST-MEM:** an accepted ICLR 2024 paper, which supports "ECG-specific models".
- **ECG-Chat:** Crossref resolves 10.1109/ICME59968.2025.11209476 to the ICME 2025 paper (MATCH), so it is a proceedings paper, not a preprint.

## Not done

- The other entries were checked for their bibliographic record only, not for claim support.
- No comprehensive retraction screening beyond the Crossref update notices.
