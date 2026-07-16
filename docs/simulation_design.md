# Confirmatory simulation design

The design is frozen before confirmatory results are generated. Pilot runs in
`results/processed/end_to_end_*` are diagnostics and cannot be pooled with the
confirmatory study.

## Questions and estimands

1. Does the hard estimator concentrate around the population CML target?
2. Does the proposed standard error estimate empirical Monte Carlo variability
   more accurately than fixed-classification inference?
3. Do nominal 95% Wald intervals achieve coverage of the CML target?
4. Are conclusions stable for bandwidth multipliers 0.75, 1, and 1.5?
5. Does the method remain usable beyond two bivariate components?

The primary estimand is the population CML functional, not the generating
ordinary-mixture parameter. Independent reference-sample fits quantify target
Monte Carlo error.

## Confirmatory scenarios

### Core: 24,000 fits

- Families: Normal/Student-t and Normal/skew-normal.
- Observation dimensions: `p=2,5`.
- Euclidean location separations: 6.0, 4.25, and 3.0 along a unit diagonal
  direction, labeled separated, moderate, and strong overlap. For `p>=5`, the
  strong-overlap separation is 3.75 because the Normal--t population
  classification criterion collapses a component at 3.0. The discarded
  value 1.75 produced component collapse outside the regular unique-target
  regime addressed by the theory.
- Balanced weights.
- Sample sizes: 500 and 2,000.
- 1,000 replications per cell.

### Imbalance: 4,000 fits

- Both two-component family pairs.
- `p=2`, moderate overlap, weights `(0.2, 0.8)`.
- Sample sizes 500 and 2,000; 1,000 replications per cell.

### Multiple active faces: 2,000 fits

- Three components: Normal, Student-t, and skew-normal.
- Triangular location geometry with weights `(0.3, 0.4, 0.3)`.
- `p=2,5`, sample size 1,000, and 1,000 replications per cell.

### Dimension stress: 1,000 fits

- Both two-component family pairs.
- `p=10`, moderate overlap, sample size 1,000.
- 500 replications per cell.

Total primary fits: 31,000. Every fit evaluates three bandwidth multipliers
without refitting.

## Target approximation

Each scenario uses 10 independent reference samples of size 100,000. The two
`p=10` targets use five reference samples initially and are extended if any
coordinate's target MCSE is non-negligible relative to the empirical standard
deviation at `n=1,000`.

## Primary summaries

For every unconstrained coordinate and scientifically interpretable
transformation:

- bias and RMSE relative to the CML target;
- empirical SD;
- mean and median estimated SE;
- SE/empirical-SD ratio;
- 95% coverage and mean interval width;
- Monte Carlo standard errors for bias, SD, and coverage;
- fit, curvature, bandwidth-support, and numerical failure rates.

The primary bandwidth multiplier is 1.0. The other multipliers are sensitivity
analyses. No scenario or coordinate is removed because results are unfavorable.
The base bandwidth is `1.06 n^{-1/5}` on the dimensionless log-posterior-odds
scale. Global contrast spread is diagnostic only; it is not a bandwidth
multiplier.

## Comparators

- Fixed-classification sandwich (`B=0`).
- Active-boundary corrected sandwich.
- Eigenvalue-stabilized active-boundary sandwich as the primary finite-sample
  implementation, with raw correction failures and shrinkage factors reported.
- Empirical Monte Carlo SD as the principal variability benchmark.
- Local nonparametric bootstrap only in selected diagnostic cells after its
  trust-region failure rate is acceptably low.
- Fresh global refits are a model/basin-selection stability analysis, not a
  substitute for the local bootstrap covariance target.
