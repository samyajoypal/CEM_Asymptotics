"""Summarize confirmatory chunks with Monte Carlo uncertainty."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    successful = data[data.status.eq("success")].copy()
    group_columns = [
        "scenario", "n", "method", "bandwidth_multiplier", "coordinate", "coordinate_name"
    ]

    def summarize(group: pd.DataFrame) -> pd.Series:
        count = len(group)
        empirical_sd = group.estimate.std(ddof=1)
        coverage = group.coverage_95.mean()
        result = {
                "replications": count,
                "bias": group.error.mean(),
                "bias_mcse": group.error.std(ddof=1) / np.sqrt(count),
                "rmse": np.sqrt(group.squared_error.mean()),
                "empirical_sd": empirical_sd,
                "mean_estimated_se": group.estimated_se.mean(),
                "median_estimated_se": group.estimated_se.median(),
                "se_to_empirical_sd": group.estimated_se.mean() / empirical_sd,
                "coverage_95": coverage,
                "coverage_mcse": np.sqrt(coverage * (1 - coverage) / count),
                "mean_interval_width_95": group.interval_width_95.mean(),
                "minimum_curvature_eigenvalue": group.curvature_min_eigenvalue.min(),
                "mean_fit_seconds": group.fit_seconds.mean(),
            }
        if "boundary_shrinkage_factor" in group:
            shrinkage = pd.to_numeric(group.boundary_shrinkage_factor, errors="coerce")
            result["mean_boundary_shrinkage_factor"] = shrinkage.mean()
            result["stabilization_frequency"] = (shrinkage < 1 - 1e-12).mean()
        if "minimum_kernel_support" in group:
            result["median_minimum_kernel_support"] = pd.to_numeric(
                group.minimum_kernel_support, errors="coerce"
            ).median()
        if "fit_selected_admissible" in group:
            admissible = group.fit_selected_admissible.astype(str).str.lower().map(
                {"true": 1.0, "false": 0.0}
            )
            result["selected_admissible_rate"] = admissible.mean()
        return pd.Series(result)

    summary = successful.groupby(group_columns, dropna=False).apply(
        summarize, include_groups=False
    ).reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    failure_path = args.output.with_name(f"{args.output.stem}_failures.csv")
    data[~data.status.eq("success")].to_csv(failure_path, index=False)
    print(summary.head(30).to_string(index=False))
    print(f"\nSaved {args.output}\nSaved {failure_path}")


if __name__ == "__main__":
    main()
