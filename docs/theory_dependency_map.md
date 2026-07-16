# Theory dependency map

This document will be filled before theorem prose is added to the manuscript.
The intended dependency order is:

1. parameter space, label convention, component regularity -- drafted;
2. active classification regions and population CML target -- drafted;
3. existence and identification -- drafted;
4. uniform convergence and consistency -- drafted;
5. active-boundary geometry and population second-order expansion -- drafted;
6. local empirical-process expansion and root-n rate -- drafted;
7. asymptotic linearity and normality -- drafted;
8. feasible fixed-region, score, and boundary-curvature estimators -- drafted;
9. covariance consistency and studentized inference -- drafted;
10. transfer from approximate algorithmic maximizers -- initial proposition
    drafted; computationally verifiable conditions remain to be developed.

No theorem will be promoted to the main manuscript until its assumptions,
inputs, outputs, and downstream dependants are recorded here.

## Current dependency details

### Population curvature theorem

- Inputs: local unconstrained parameterization; continuous data density;
  twice parameter-differentiable component scores; regular active faces;
  negligible multiple-tie junctions in Hausdorff measure; integrable volume
  and surface derivatives.
- Output: `H = -I_c + B`, where `B` integrates only over active faces.
- Used by: local quadratic expansion, root-n rate, asymptotic linearity,
  feasible covariance estimation.
- Outstanding audit: provide a full appendix lemma formalizing the partition
  of unity/distributional differentiation argument.

### Consistency theorem

- Inputs: compact identifiable parameter space, continuity, integrable
  envelope, separated population maximizer, approximate global maximization.
- Output: convergence to the CML functional of `P_0`.
- Used by: localization for all first-order results.
- Outstanding audit: formulate a separate local algorithmic result that does
  not pretend a generic CEM stationary point is a global approximate maximizer.

### Local asymptotic expansion

- Inputs: boundary margin, Lipschitz derivative envelope, negligible
  first-order mass at multiple-face junctions, and positive curvature.
- Output: a locally asymptotically quadratic empirical criterion and root-n
  asymptotic linearity.
- Used by: normality, bandwidth plug-in condition, Wald inference.
- Outstanding audit: expand the empirical-process proof in the appendix with
  an explicit entropy bound for the localized max-score class.

### Feasible boundary estimator

- Inputs: bounded compactly supported kernel, `h -> 0`, `nh -> infinity`,
  parameter plug-in error `o_p(h)`, surface-moment continuity, and negligible
  junction neighbourhoods.
- Output: consistent, grid-free estimate of the active-boundary curvature.
- Used by: sandwich covariance and Wald inference.
- Outstanding audit: derive bias and variance expansions, a data-driven
  bandwidth rule, and an optional cross-fitted version for sensitivity checks.

### Boundary-estimator precision -- drafted

- Oracle bias: second order in the bandwidth for a symmetric second-order
  kernel.
- Oracle variance: order `(nh)^{-1}` with an explicit surface covariance.
- Oracle CLT: available under undersmoothing `nh^5 -> 0`, or after leading-bias
  centering.
- Plug-in rate: includes smoothing bias, stochastic error, parameter
  substitution, and multiple-junction contributions.
- Outstanding audit: turn the theoretical rate into a stable automatic
  bandwidth selector and verify it numerically.

### Bootstrap and transformed inference -- drafted

- Nonparametric bootstrap validity is stated conditionally on consistent basin
  selection and negligible bootstrap optimization error.
- Wald inference and a multivariate delta-method corollary cover smooth
  scientific transformations.
- Outstanding audit: design a bootstrap initialization protocol whose
  assumptions are measurable in the simulation study.

### Technical supplement -- drafted

- Smooth positive-part approximation plus coarea formula for active-face
  curvature.
- Partition-of-unity assembly over active strata.
- Explicit local bracketing-entropy bound for the max-score remainder class.
- Oracle boundary-estimator moment expansion and uniform plug-in lemma.
