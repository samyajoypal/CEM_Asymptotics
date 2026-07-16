# Confirmatory target audit (2026-07-13)

All 18 scenarios completed and all requested reference fits returned without an
exception.  This computational success is not sufficient for target validity.
The independent-reference estimates reveal distinct fitted basins in several
difficult scenarios.

The prespecified target gate is:

1. no failed reference fits;
2. maximum coordinate MCSE at most 0.05;
3. maximum coordinate range across reference fits at most 0.5; and
4. no visually or numerically distinct family-assignment basin.

The following scenarios fail that gate:

| scenario | maximum MCSE | maximum range | diagnosis |
|---|---:|---:|---|
| `nt_p2_moderate_imbalanced` | 0.943 | 10.088 | one collapsed/alternative basin |
| `nt_p2_strong_balanced` | 1.368 | 10.610 | two collapsed/alternative fits |
| `nt_p5_strong_balanced` | 0.275 | 3.083 | family-assignment basin switch |
| `nsn_p2_strong_balanced` | 0.132 | 1.357 | weakly identified skewness coordinates |
| `nsn_p5_strong_balanced` | 0.096 | 1.028 | weakly identified skewness coordinates |

The remaining 13 scenarios pass these numerical thresholds, although formal
family-assignment diagnostics must still be retained in the simulation output.

## Decision

Do not average the incompatible basins and do not launch the full Monte Carlo
study against those averages.  The implementation must first match the compact,
interior parameter space assumed by the theory (in particular, weights bounded
away from zero and bounded tail/skewness coordinates), use multiple starts on
the same reference sample, and record the objective gap and selected basin.
Targets must then be recomputed and this gate rerun.  This is a methodological
issue, not an HPC failure.

## Root-cause correction

Comparison with the old paper showed that its successful "strong" case had
Euclidean location separation 4.24, whereas the first JRSS B design used 1.75.
The latter was an extreme component-collapse experiment rather than a
comparable overlap setting. The primary design now uses separations 6.0, 4.25,
and 3.0, retains non-oracle assignment selection, and treats compact-set
admissibility as a reported diagnostic rather than rejecting a fit after
estimation.
