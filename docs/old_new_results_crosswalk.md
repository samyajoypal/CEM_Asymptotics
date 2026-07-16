# Result crosswalk: previous JSTA study and current JRSS B study

Audit date: 17 July 2026.

The previous main manuscript contained four simulation figures and six tables;
its supplement contained 48 table inputs and 64 separate parameter plots. The
current confirmatory archive contains 18 scenarios, 7,623 summary rows, and
1,089 distinct scenario--sample-size--coordinate combinations.

| Result type in previous version | Current counterpart | Audit outcome |
|---|---|---|
| Bias relative to the numerical CML target | Main simulation profiles; complete coordinate bias atlas; machine-readable summary | Retained with redesigned targets and many more scenarios |
| RMSE/consistency over sample size | Main profiles and boundary-convergence figure; complete coordinate RMSE atlas | Retained and expanded |
| Empirical SD versus corrected SE | Main calibration figures; complete coordinate SE/SD atlas | Retained and expanded to naive/raw/stabilized inference |
| 95% interval coverage | Main selected table and calibration figures; complete coordinate coverage atlas | Retained and expanded |
| Asymptotic normality | Previously inferred mainly from consistency and SE plots | Strengthened: replication-level Gaussian quantiles, KS distances, scaled RMSE and scaled bias for all 1,089 coordinates |
| Normal--Student-t experiments | Normal--Student-t experiments in dimensions 2, 5 and 10, multiple overlaps and imbalance | Retained and expanded |
| Normal--skew-normal experiments | Corrected skew-normal implementation in dimensions 2, 5 and 10 | Retained and corrected |
| Two-component designs | Two-component designs plus three-component heterogeneous stress designs | Retained and expanded |
| Boundary-corrected versus naive inference | Naive, raw correction, stabilized correction, bandwidth sensitivity | Retained and expanded |
| Parameter-level tables | Complete coordinate CSV plus scenario table and coordinate atlases | Retained in a more compact representation |
| Breast Cancer Wisconsin bootstrap example | Replaced by Dermason--Seker regular example and Cali--Barbunya stress example, each with local and unrestricted bootstrap diagnostics | Superseded, not duplicated |
| Failed-fit handling | Explicit failure and stabilization accounting, with failed seeds retained | Expanded |
| Bandwidth analysis | Prespecified multipliers 0.75, 1 and 1.5 | New |
| Direct boundary-estimator convergence | Bias, SD and RMSE through n=16,000 | New |
| Optimization-basin uncertainty | Unrestricted-bootstrap distance diagnostics | New |

## Conclusion

No diagnostic category from the previous simulation study is absent after the
coordinate atlases were added. The only old empirical object not repeated is
the Breast Cancer dataset itself. It is intentionally superseded by two Dry
Bean analyses selected under the new screening protocol; the underlying
real-data result type--analytic SE versus bootstrap variability--is retained
and strengthened. The old 64 single-parameter plots are not copied verbatim,
because they concern obsolete results and parameterization. Their new-result
counterparts are the four 18-panel coordinate atlases in the supplement.

