# Implementation status

## Validated foundation

- One unconstrained coordinate vector contains reference-category mixture
  logits and component-specific unconstrained parameters.
- Positive-definite matrices use log-Cholesky coordinates.
- Normal and Student-t log densities use SciPy's reference implementations.
- The skew-normal density is
  `2 phi_p(x; xi, Omega) Phi(alpha^T Omega^{-1/2}(x-xi))` with the symmetric
  inverse square root. A direct test matches FMVMM 3.0 to numerical precision.
- Classification scores, hard assignments, active pairs, pairwise contrast
  gradients, fixed-class scores, and fixed-class Hessians share the same model
  object and parameterization.
- The boundary estimator implements the sample kernel tube formula from the
  paper and never substitutes the fitted mixture density for `P_0`.
- Sandwich assembly uses `A_hat = I_c_hat - B_hat`.
- FMVMM fits are rejected unless `em_type="hard"`; internal result-array
  lengths and fitted weights are validated before adaptation.
- All family-to-component permutations are fitted and selected by BIC.
  Canonical reordering uses family identity and fitted location only.
- Vectorized finite differences are checked against the retained
  observation-wise reference path. In a 250-observation two-dimensional
  Normal/skew-normal check they were approximately 182 times faster, with a
  maximum gradient difference below `8e-10`.
- Pair-specific pilot bandwidths use the natural dimensionless log-odds scale
  and return global contrast spread, effective-support diagnostics, and
  explicit low-count warnings. A rejected global-scale rule is preserved only
  as an optional diagnostic because it severely oversmoothed the local pilot.
- Bootstrap refitting conditions on the selected family assignment by default,
  as required by the local asymptotic theorem. Optional assignment reselection
  is retained as a separate model-selection stability diagnostic; it is not
  summarized as though it were the same Gaussian covariance target.

## Validation suite

The current suite tests distribution round trips, exact skew-normal agreement,
rejection of the old logistic approximation, contrast-gradient antisymmetry,
objective identity, kernel normalization, a known one-dimensional boundary
integral, positive semidefiniteness of the boundary estimate, and the complete
numerical sandwich pipeline.

## Controlled convergence experiment

For `g(x; theta)=theta-x`, standard-normal data, and `theta=0`, the exact
boundary curvature is `phi(0)`. Across 500 replications, RMSE decreased from
0.0452 at `n=250` to 0.00960 at `n=16000`; absolute bias at the largest sample
size was 0.00077. Raw and summarized results are stored under
`results/processed/boundary_convergence*.csv`.

## End-to-end local diagnostic

A ten-replication, two-dimensional Normal/Student-t diagnostic at `n=500`
completed without fitting or curvature failures. For most coordinates the
boundary-corrected standard errors moved toward the empirical standard
deviations and the local-bootstrap estimates. This run is diagnostic only and
is far too small for coverage claims.

Fresh k-means bootstrap fits produced distant local solutions even when the
family assignment was held fixed. The inferential bootstrap now uses local CML
optimization initialized at the original estimate. Of 200 attempted local
bootstrap fits, 156 remained successful within the trust region and 44 were
retained as explicit failures. Improving this optimizer and lowering that
failure rate is required before the formal simulation.

## Deliberate limitations of version 0.4.0

- Finite differences are the auditable reference implementation, not the
  intended high-performance simulation backend.
- The pilot bandwidth is not yet an estimated MSE-optimal selector.
- The positive-curvature diagnostic does not silently regularize a matrix that
  fails the theoretical definiteness condition.
- Simulation and real-data scripts must not be built until analytic or
  automatic derivatives agree with this reference implementation.
- FMVMM 3 returned no completed Normal/Student-t candidate in the univariate
  smoke configuration. The validated pilot therefore begins at observation
  dimension `p=2`; univariate support is not claimed.
- The current local bootstrap trust box is a diagnostic safeguard, not a final
  optimization algorithm. Its radius requires sensitivity analysis.
