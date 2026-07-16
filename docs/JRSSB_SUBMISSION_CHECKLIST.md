# JRSS B submission-readiness checklist

This checklist records items that must be resolved before submission. It is
not part of the manuscript.

## Scientific package

- [x] Explicit estimand under possible misspecification.
- [x] Consistency, local expansion, asymptotic normality, bootstrap validity,
  feasible covariance consistency, Wald and delta-method results.
- [x] Active-boundary curvature theorem and detailed technical lemmas.
- [x] Finite-sample stabilization with asymptotic-inactivity result.
- [x] Primary simulations, large-sample extension, bandwidth sensitivity,
  boundary-estimator convergence, failures, and target uncertainty.
- [x] Two real-data illustrations with local and unrestricted bootstrap
  diagnostics.
- [x] Complete summarized results in the supplement.

## Files for editorial submission

- [x] Main manuscript PDF.
- [x] Supplementary-material PDF.
- [x] Bibliography and source files.
- [x] Replace placeholder author list and add affiliations, corresponding
  author, and contribution statement.
- [ ] Add ORCID identifiers if the authors wish to display them.
- [x] Add funding and acknowledgements based on the previous manuscript.
- [ ] Create a citable, versioned code archive and insert its DOI.
- [ ] Confirm that the anonymized archive contains no author or HPC account
  identifiers.
- [ ] Prepare a concise cover letter explaining the general moving-boundary
  contribution, rather than presenting the work only as a CEM application.
- [x] Convert the source to the downloaded OUP/RSS submission template.
- [ ] Flatten `\input` files if requested by the submission system.
- [ ] Recheck the current Oxford Academic JRSS B instructions immediately
  before upload, including article type, file naming, declarations, and any
  word or page guidance.

## Reproducibility archive

- [x] Deterministic seeds and simulation configuration.
- [x] Raw and processed simulation outputs retained locally.
- [x] Raw and processed application bootstrap outputs retained locally.
- [x] Scripts generating manuscript summaries and supplementary tables.
- [x] Data provenance and licence recorded for Dry Bean.
- [x] Add one command (`scripts/build_submission.sh`) that regenerates figures,
  supplementary tables, journal sources, and both PDFs from archived results.
- [ ] Freeze the Python environment and record package versions and HPC
  software modules.
- [ ] Run the reproduction workflow from a clean checkout.
