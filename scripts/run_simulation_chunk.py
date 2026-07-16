"""Run one resumable confirmatory Monte Carlo chunk."""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from cem_inference import (
    estimate_boundary_curvature,
    fit_fmvmm_hard,
    sandwich_covariance,
    select_pairwise_bandwidths,
    stabilize_boundary_curvature,
)
from cem_inference.scenarios import get_scenario


def stable_scenario_seed(identifier: str) -> int:
    return zlib.crc32(identifier.encode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--bandwidth-multipliers", nargs="+", type=float, default=[0.75, 1.0, 1.5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target_frame = pd.read_csv(args.target).sort_values("coordinate")
    target = target_frame.target.to_numpy(float)
    target_mcse = target_frame.target_mcse.to_numpy(float)
    specification = get_scenario(args.scenario)
    rows = []

    for replication in range(args.start, args.start + args.replications):
        replication_seed = np.random.SeedSequence(
            [args.seed, stable_scenario_seed(args.scenario), args.sample_size, replication]
        )
        observations, _ = specification.sample(
            args.sample_size, np.random.default_rng(replication_seed)
        )
        started = time.perf_counter()
        try:
            fit = fit_fmvmm_hard(
                observations,
                list(specification.adapters),
                max_iter=160,
                tol=1e-6,
                n_jobs=1,
            )
            if len(target) != fit.model.parameter_dimension:
                raise ValueError(
                    f"Target dimension {len(target)} != fitted dimension "
                    f"{fit.model.parameter_dimension}."
                )
            model, coordinates = fit.model, fit.coordinates
            fit_metadata = {
                "fit_initialization": fit.initialization,
                "fit_classification_objective": fit.classification_objective,
                "fit_objective_gap": fit.objective_gap,
                "fit_admissible_candidates": fit.admissible_candidates,
                "fit_total_candidates": fit.total_candidates,
                "fit_selected_admissible": fit.selected_admissible,
                "fit_minimum_weight": float(np.min(fit.weights)),
            }
            classified_scores = model.classified_score_matrix(observations, coordinates)
            fixed_information = model.fixed_classification_information(
                observations, coordinates
            )
            zeros = np.zeros_like(fixed_information)
            naive_covariance, naive_curvature, _ = sandwich_covariance(
                fixed_information, zeros, classified_scores
            )
            naive_se = np.sqrt(np.maximum(np.diag(naive_covariance), 0) / args.sample_size)
            fit_seconds = time.perf_counter() - started
            for coordinate, estimate in enumerate(coordinates):
                error = estimate - target[coordinate]
                rows.append(
                    {
                        "scenario": args.scenario,
                        "n": args.sample_size,
                        "replication": replication,
                        "status": "success",
                        "method": "naive",
                        "bandwidth_multiplier": np.nan,
                        "coordinate": coordinate,
                        "coordinate_name": model.coordinate_names[coordinate],
                        "estimate": estimate,
                        "target": target[coordinate],
                        "target_mcse": target_mcse[coordinate],
                        "error": error,
                        "squared_error": error**2,
                        "estimated_se": naive_se[coordinate],
                        "coverage_95": abs(error) <= 1.96 * naive_se[coordinate],
                        "interval_width_95": 2 * 1.96 * naive_se[coordinate],
                        "curvature_min_eigenvalue": float(
                            np.min(np.linalg.eigvalsh(naive_curvature))
                        ),
                        "fit_seconds": fit_seconds,
                        **fit_metadata,
                    }
                )

            for multiplier in args.bandwidth_multipliers:
                try:
                    bandwidths, diagnostics = select_pairwise_bandwidths(
                        model,
                        observations,
                        coordinates,
                        multiplier=1.06 * multiplier,
                    )
                    boundary = estimate_boundary_curvature(
                        model, observations, coordinates, bandwidth=bandwidths
                    )
                    support_counts = [
                        diagnostic.observations_in_kernel_support
                        for diagnostic in diagnostics
                    ]
                    diagnostic_json = json.dumps(
                        [diagnostic.__dict__ for diagnostic in diagnostics]
                    )
                    boundary_versions = []
                    stabilized, shrinkage_factor, generalized_maximum = (
                        stabilize_boundary_curvature(
                            fixed_information, boundary, args.sample_size
                        )
                    )
                    boundary_versions.append(
                        ("corrected_stabilized", stabilized, shrinkage_factor, generalized_maximum)
                    )
                    boundary_versions.append(
                        ("corrected_raw", boundary, 1.0, generalized_maximum)
                    )
                    for method_name, boundary_version, shrinkage, relative_maximum in boundary_versions:
                        try:
                            covariance, curvature, _ = sandwich_covariance(
                                fixed_information, boundary_version, classified_scores
                            )
                        except Exception as error:
                            rows.append(
                                {
                                    "scenario": args.scenario,
                                    "n": args.sample_size,
                                    "replication": replication,
                                    "status": f"{method_name}_failed",
                                    "method": method_name,
                                    "bandwidth_multiplier": multiplier,
                                    "error_message": f"{type(error).__name__}: {error}",
                                    "boundary_shrinkage_factor": shrinkage,
                                    "raw_generalized_boundary_eigenvalue": relative_maximum,
                                    "fit_seconds": fit_seconds,
                                    **fit_metadata,
                                }
                            )
                            continue
                        standard_errors = np.sqrt(
                            np.maximum(np.diag(covariance), 0) / args.sample_size
                        )
                        for coordinate, estimate in enumerate(coordinates):
                            error = estimate - target[coordinate]
                            rows.append({
                                "scenario": args.scenario,
                                "n": args.sample_size,
                                "replication": replication,
                                "status": "success",
                                "method": method_name,
                                "bandwidth_multiplier": multiplier,
                                "coordinate": coordinate,
                                "coordinate_name": model.coordinate_names[coordinate],
                                "estimate": estimate,
                                "target": target[coordinate],
                                "target_mcse": target_mcse[coordinate],
                                "error": error,
                                "squared_error": error**2,
                                "estimated_se": standard_errors[coordinate],
                                "coverage_95": abs(error) <= 1.96 * standard_errors[coordinate],
                                "interval_width_95": 2 * 1.96 * standard_errors[coordinate],
                                "curvature_min_eigenvalue": float(
                                    np.min(np.linalg.eigvalsh(curvature))
                                ),
                                "minimum_kernel_support": min(support_counts),
                                "bandwidth_diagnostics": diagnostic_json,
                                "boundary_shrinkage_factor": shrinkage,
                                "raw_generalized_boundary_eigenvalue": relative_maximum,
                                "fit_seconds": fit_seconds,
                                **fit_metadata,
                            })
                except Exception as error:
                    rows.append(
                        {
                            "scenario": args.scenario,
                            "n": args.sample_size,
                            "replication": replication,
                            "status": "boundary_estimation_failed",
                            "method": "boundary",
                            "bandwidth_multiplier": multiplier,
                            "error_message": f"{type(error).__name__}: {error}",
                            "fit_seconds": fit_seconds,
                            **fit_metadata,
                        }
                    )
        except Exception as error:
            rows.append(
                {
                    "scenario": args.scenario,
                    "n": args.sample_size,
                    "replication": replication,
                    "status": "fit_or_naive_failed",
                    "method": "fit",
                    "error_message": f"{type(error).__name__}: {error}",
                    "fit_seconds": time.perf_counter() - started,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    print(frame.groupby(["status", "method"], dropna=False).replication.nunique())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
