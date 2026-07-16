# Research contract and non-negotiable checks

## Target

The primary target is JRSS Series B.  This is an aspiration, not a guarantee.
Every claim must remain useful and correct under a fallback submission to a
strong specialist methodology journal.

## Separation from the earlier paper

- The parent project is read-only historical evidence.
- New theory is drafted in `paper/main.tex` and supporting documents here.
- New numerical results must be generated from scripts in this project.
- Legacy targets, tables, and figures may be used for debugging but may not be
  silently copied into the new manuscript.

## Mathematical checks

- Distinguish the data-generating law, fitted mixture model, and CML target.
- Integrate only over active decision faces.
- State identifiability and nondegeneracy assumptions explicitly.
- Connect any algorithmic solution to the estimator used in the theorems.
- Prove consistency of every estimated covariance component.
- Use the established convention throughout: $p$ is data dimension, $d_j$ is
  component-$j$ parameter dimension, and $d$ is total parameter dimension.
- Typeset all vectors and matrices in bold in the manuscript, supplement,
  equations, and captions. Scalars, scalar-valued functions, dimensions, and
  hypothesis labels remain unbolded.

## Computational checks

- A single distribution adapter defines sampling, log density, parameters,
  scores, and Hessians.
- Numerical derivatives are checked against independently evaluated densities.
- Hard EM is requested explicitly and tested.
- Oracle procedures are diagnostic only and never the primary practical method.
- Random seeds, software versions, failures, and run manifests are recorded.

## Reporting checks

- Monte Carlo uncertainty accompanies simulation summaries.
- Negative or unstable results are retained and explained.
- Real-data conclusions are substantive, not merely demonstrations that code
  runs.
- All final tables and figures are reproducible from public code and data.
