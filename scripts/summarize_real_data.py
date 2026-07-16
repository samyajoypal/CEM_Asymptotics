"""Summarize analytic and bootstrap inference for the selected applications."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from cem_inference import (
    NormalAdapter, StudentTAdapter, estimate_boundary_curvature, fit_fmvmm_hard,
    sandwich_covariance, select_pairwise_bandwidths, stabilize_boundary_curvature,
)
from run_real_data_screen import load_dataset


def matrix(frame: pd.DataFrame, dimension: int):
    success = frame[frame.status.eq("success") & frame.coordinate.notna()].copy()
    wide = success.pivot(index="replication", columns="coordinate", values="estimate")
    return wide.reindex(columns=range(dimension))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset in ("dry_bean_dermason_seker", "dry_bean_cali_barbunya"):
        raw, labels = load_dataset(dataset)
        observations = PCA(n_components=2, svd_solver="full").fit_transform(
            StandardScaler().fit_transform(raw)
        )
        fit = fit_fmvmm_hard(
            observations, [NormalAdapter(2), StudentTAdapter(2)],
            max_iter=160, tol=1e-6,
        )
        model, coordinates = fit.model, fit.coordinates
        scores = model.classified_score_matrix(observations, coordinates)
        fixed = model.fixed_classification_information(observations, coordinates)
        bandwidths, diagnostics = select_pairwise_bandwidths(
            model, observations, coordinates, multiplier=1.06
        )
        boundary = estimate_boundary_curvature(
            model, observations, coordinates, bandwidth=bandwidths
        )
        naive, _, _ = sandwich_covariance(fixed, np.zeros_like(fixed), scores)
        stabilized_boundary, shrinkage, raw_relative = stabilize_boundary_curvature(
            fixed, boundary, len(observations)
        )
        stabilized, _, _ = sandwich_covariance(fixed, stabilized_boundary, scores)
        naive_se = np.sqrt(np.maximum(np.diag(naive), 0) / len(observations))
        stabilized_se = np.sqrt(np.maximum(np.diag(stabilized), 0) / len(observations))

        frames = {}
        for mode in ("local", "global"):
            files = sorted((args.raw_root / dataset).glob(f"{mode}_*.csv"))
            frames[mode] = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        local = matrix(frames["local"], model.parameter_dimension)
        global_ = matrix(frames["global"], model.parameter_dimension)
        local_sd = local.std(ddof=1).to_numpy()
        global_sd = global_.std(ddof=1).to_numpy()
        parameter = pd.DataFrame({
            "coordinate": np.arange(model.parameter_dimension),
            "coordinate_name": model.coordinate_names,
            "estimate": coordinates,
            "naive_se": naive_se,
            "stabilized_se": stabilized_se,
            "local_bootstrap_sd": local_sd,
            "global_bootstrap_sd": global_sd,
            "naive_to_local_bootstrap": naive_se / local_sd,
            "stabilized_to_local_bootstrap": stabilized_se / local_sd,
        })
        parameter.to_csv(args.output_dir / f"{dataset}_parameters.csv", index=False)

        global_distance = np.max(np.abs(global_.to_numpy() - coordinates), axis=1)
        global_rows = frames["global"].drop_duplicates("replication")
        summaries.append({
            "dataset": dataset, "n": len(observations),
            "ari_external_labels": adjusted_rand_score(
                labels, model.assignments(observations, coordinates)
            ),
            "minimum_weight": fit.weights.min(),
            "kernel_support": diagnostics[0].observations_in_kernel_support,
            "median_se_inflation": np.median(stabilized_se / naive_se),
            "maximum_se_inflation": np.max(stabilized_se / naive_se),
            "boundary_shrinkage_factor": shrinkage,
            "raw_generalized_boundary_eigenvalue": raw_relative,
            "local_successes": len(local),
            "local_failures": 500 - len(local),
            "global_successes": len(global_),
            "global_failures": 200 - len(global_),
            "global_local_basin_fraction": np.mean(global_distance <= 0.75),
            "global_admissible_fraction": global_rows.fit_selected_admissible.astype(
                str
            ).str.lower().eq("true").mean(),
            "median_naive_to_local_bootstrap": np.nanmedian(naive_se / local_sd),
            "median_stabilized_to_local_bootstrap": np.nanmedian(
                stabilized_se / local_sd
            ),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "application_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
