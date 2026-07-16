# Reference and citation verification

Audit date: 17 July 2026. Scope: all 44 entries printed in the main-paper
bibliography, not merely all entries stored in `references.bib`.

## Procedure

1. Extract the cited keys from the compiled LaTeX auxiliary file.
2. Resolve every available DOI through Crossref and compare title, authors,
   year, venue, volume, issue and pages.
3. Verify books and proceedings without an original Crossref DOI against
   publisher or library catalogues.
4. Verify the Dry Bean dataset DOI and licence against UCI documentation.
5. Read every sentence containing a citation and check that the cited work
   supports the claim made in that sentence.

The machine-readable Crossref output is `docs/reference_audit.json`, generated
by `scripts/audit_references.py`.

## Corrections made during the audit

- Hartigan (1978): corrected DOI from `10.1214/aos/1176344076` (an unrelated
  kernel-density article) to `10.1214/aos/1176344071`.
- Sherman (1993): corrected DOI from `10.2307/2951777` (an unrelated
  game-theory article) to `10.2307/2951780`.
- Scott and Symons (1971): corrected DOI to `10.2307/2529003`.
- Added verified DOI metadata for Dempster et al. (1977), Celeux and Govaert
  (1992), Banfield and Raftery (1993), Louis (1982), Efron (1979), White
  (1982), McLachlan and Peel (2000), McLachlan and Krishnan (2008), Pal and
  Heumann (2026), and the UCI dataset.
- Changed Andrews (1994) from an article record to a Handbook chapter.
- Reworded the Hartigan--Bock and Ganesalingam--Hui discussion so that the
  prose does not claim more than those papers establish.

## Verification classes

- **Registry exact:** DOI resolves to the cited work and the substantive
  bibliographic fields agree. This covers all journal articles carrying a DOI.
- **Original-edition verified:** an electronic DOI may describe a later
  digitization, while the cited year is the original print edition. This
  applies to Federer (1969), Silverman (1986), Evans and Gariepy (2015), and
  similar monographs.
- **Library/publisher verified:** Titterington et al. (1985), Keribin (2000),
  Huber (1967), Efron and Tibshirani (1993), and Wand and Jones (1995) were
  checked against publisher, proceedings, or library records because an
  original Crossref journal record is unavailable or inappropriate.
- **DataCite/UCI verified:** `10.24432/C50S4B` identifies the 2020 Dry Bean
  dataset; UCI specifies CC BY 4.0 for repository datasets.

## Semantic citation audit

The citations are used for their documented subjects: mixture modelling and
EM; classification likelihood and CEM; order selection; partition validation;
moving-boundary k-means theory; nonsmooth root-n and cube-root asymptotics;
misspecified M-estimation; empirical processes; coarea and kernel estimation;
bootstrap inference; and the Dry Bean data provenance. No citation is used as
evidence for a theorem proved in this manuscript, and no cited source is
represented as providing the proposed feasible surface-curvature estimator.

