"""Build compact scenario-level paper tables from coordinate-level summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    failures = pd.read_csv(args.failures)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = summary[
        summary.method.eq("naive")
        | (summary.method.eq("corrected_stabilized")
           & summary.bandwidth_multiplier.eq(1.0))
    ].copy()
    scenario = selected.groupby(
        ["scenario", "n", "method"], dropna=False
    ).agg(
        usable_replications=("replications", "min"),
        median_absolute_bias=("bias", lambda values: values.abs().median()),
        median_se_to_sd=("se_to_empirical_sd", "median"),
        median_coverage=("coverage_95", "median"),
        mean_coverage=("coverage_95", "mean"),
        median_interval_width=("mean_interval_width_95", "median"),
        stabilization_frequency=("stabilization_frequency", "median"),
    ).reset_index()
    scenario.to_csv(args.output_dir / "scenario_performance.csv", index=False)

    fit_failures = failures[failures.status.eq("fit_or_naive_failed")]
    fit_counts = fit_failures.groupby(["scenario", "n"]).replication.nunique()
    planned = summary[summary.method.eq("naive")].groupby(
        ["scenario", "n"]
    ).replications.min().add(fit_counts, fill_value=0)
    failure_table = pd.DataFrame({
        "fit_failures": fit_counts,
        "planned_replications": planned,
    }).fillna(0).reset_index()
    failure_table["fit_failure_rate"] = (
        failure_table.fit_failures / failure_table.planned_replications
    )
    failure_table.to_csv(args.output_dir / "fit_failure_rates.csv", index=False)

    raw = failures[failures.status.eq("corrected_raw_failed")]
    raw_counts = raw.groupby(["scenario", "n"]).replication.nunique()
    raw_table = raw_counts.rename("replications_with_any_raw_failure").reset_index()
    raw_table.to_csv(args.output_dir / "raw_curvature_failures.csv", index=False)

    print(scenario.to_string(index=False))
    print("\nFit failures\n", failure_table.to_string(index=False))


if __name__ == "__main__":
    main()
