"""Small diagnostic study before committing to the full simulation design."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cem_inference import (
    EllipticalParameters,
    MixtureSpecification,
    NormalAdapter,
    StudentTAdapter,
    StudentTParameters,
    approximate_cml_target,
    bootstrap_coordinates,
    compute_inference,
    fit_fmvmm_hard,
)


def default_specification() -> MixtureSpecification:
    return MixtureSpecification(
        adapters=(NormalAdapter(2), StudentTAdapter(2)),
        weights=np.array([0.5, 0.5]),
        component_parameters=(
            EllipticalParameters(
                np.array([-1.5, -1.5]), np.array([[1.0, 0.2], [0.2, 1.0]])
            ),
            StudentTParameters(
                np.array([1.5, 1.5]), np.array([[1.1, 0.15], [0.15, 1.1]]), 6.0
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--bootstrap-replications", type=int, default=50)
    parser.add_argument("--target-size", type=int, default=20_000)
    parser.add_argument("--target-replications", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/processed/end_to_end_pilot")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specification = default_specification()
    fit_kwargs = {"max_iter": 120, "tol": 1e-5, "n_jobs": 1}
    target = approximate_cml_target(
        specification,
        reference_size=args.target_size,
        replications=args.target_replications,
        seed=args.seed,
        fit_kwargs=fit_kwargs,
    )
    pd.DataFrame(target.estimates).to_csv(
        args.output_dir / "target_estimates.csv", index=False
    )
    pd.DataFrame(
        {
            "coordinate": np.arange(len(target.mean)),
            "target": target.mean,
            "target_mcse": target.monte_carlo_standard_error,
        }
    ).to_csv(args.output_dir / "target_summary.csv", index=False)

    rows = []
    rngs = np.random.SeedSequence(args.seed + 1).spawn(args.replications)
    for replication, child_seed in enumerate(rngs):
        observations, _ = specification.sample(
            args.sample_size, np.random.default_rng(child_seed)
        )
        try:
            fit = fit_fmvmm_hard(
                observations, list(specification.adapters), **fit_kwargs
            )
            inference = compute_inference(observations, fit)
            bootstrap, bootstrap_failures = bootstrap_coordinates(
                observations,
                fit,
                replications=args.bootstrap_replications,
                seed=args.seed + 10_000 + replication,
                fit_kwargs=fit_kwargs,
                reselect_assignments=False,
            )
            bootstrap_se = (
                np.std(bootstrap, axis=0, ddof=1)
                if len(bootstrap) >= 2
                else np.full(fit.model.parameter_dimension, np.nan)
            )
            for coordinate, estimate in enumerate(fit.coordinates):
                target_value = target.mean[coordinate]
                rows.append(
                    {
                        "replication": replication,
                        "status": "success",
                        "coordinate": coordinate,
                        "estimate": estimate,
                        "target": target_value,
                        "error": estimate - target_value,
                        "naive_se": inference.naive_standard_errors[coordinate],
                        "corrected_se": inference.corrected_standard_errors[coordinate],
                        "bootstrap_se": bootstrap_se[coordinate],
                        "naive_cover": abs(estimate - target_value)
                        <= 1.96 * inference.naive_standard_errors[coordinate],
                        "corrected_cover": abs(estimate - target_value)
                        <= 1.96 * inference.corrected_standard_errors[coordinate],
                        "bootstrap_cover": abs(estimate - target_value)
                        <= 1.96 * bootstrap_se[coordinate],
                        "corrected_min_eigenvalue": inference.corrected_curvature_min_eigenvalue,
                        "bootstrap_successes": len(bootstrap),
                        "bootstrap_failures": len(bootstrap_failures),
                        "bandwidth_warnings": " | ".join(inference.warnings),
                    }
                )
        except Exception as error:
            rows.append(
                {
                    "replication": replication,
                    "status": "failed",
                    "error_message": f"{type(error).__name__}: {error}",
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "replication_results.csv", index=False)
    successful = result[result.status == "success"].copy()
    if successful.empty:
        print(result[["replication", "error_message"]].to_string(index=False))
        raise RuntimeError("Every pilot replication failed.")
    summary = successful.groupby("coordinate").agg(
        bias=("error", "mean"),
        empirical_sd=("estimate", "std"),
        mean_naive_se=("naive_se", "mean"),
        mean_corrected_se=("corrected_se", "mean"),
        mean_bootstrap_se=("bootstrap_se", "mean"),
        naive_coverage=("naive_cover", "mean"),
        corrected_coverage=("corrected_cover", "mean"),
        bootstrap_coverage=("bootstrap_cover", "mean"),
    )
    summary.to_csv(args.output_dir / "summary.csv")
    print(summary.to_string())
    print("\nStatus counts:")
    print(result.groupby("status").replication.nunique().to_string())
    print(f"\nSaved {args.output_dir}")


if __name__ == "__main__":
    main()
