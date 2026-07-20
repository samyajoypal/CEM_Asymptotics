"""Build delta-method and local-bootstrap summaries on interpretable scales."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cem_inference import (
    NormalAdapter, StudentTAdapter, estimate_boundary_curvature, fit_fmvmm_hard,
    sandwich_covariance, select_pairwise_bandwidths, stabilize_boundary_curvature,
)
from run_real_data_screen import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def quantities(model, coordinates):
    unpacked = model.unpack(np.asarray(coordinates, dtype=float))
    first, second = unpacked.component_parameters
    return np.array([
        unpacked.weights[0],
        np.linalg.norm(first.location - second.location),
        second.degrees_of_freedom,
    ])


def jacobian(function, coordinates, step=1e-5):
    result = np.empty((3, len(coordinates)))
    for column in range(len(coordinates)):
        increment = np.zeros_like(coordinates)
        increment[column] = step
        result[:, column] = (
            function(coordinates + increment) - function(coordinates - increment)
        ) / (2 * step)
    return result


def main():
    dataset = "dry_bean_dermason_seker"
    raw, _ = load_dataset(dataset)
    observations = PCA(n_components=2, svd_solver="full").fit_transform(
        StandardScaler().fit_transform(raw)
    )
    fit = fit_fmvmm_hard(
        observations, [NormalAdapter(2), StudentTAdapter(2)], max_iter=160, tol=1e-6
    )
    model, coordinates = fit.model, fit.coordinates
    scores = model.classified_score_matrix(observations, coordinates)
    fixed = model.fixed_classification_information(observations, coordinates)
    bandwidths, _ = select_pairwise_bandwidths(
        model, observations, coordinates, multiplier=1.06
    )
    boundary = estimate_boundary_curvature(
        model, observations, coordinates, bandwidth=bandwidths
    )
    naive, _, _ = sandwich_covariance(fixed, np.zeros_like(fixed), scores)
    stabilized_boundary, _, _ = stabilize_boundary_curvature(
        fixed, boundary, len(observations)
    )
    corrected, _, _ = sandwich_covariance(fixed, stabilized_boundary, scores)

    transform = lambda value: quantities(model, value)
    gradient = jacobian(transform, coordinates)
    naive_se = np.sqrt(np.diag(gradient @ naive @ gradient.T) / len(observations))
    corrected_se = np.sqrt(
        np.diag(gradient @ corrected @ gradient.T) / len(observations)
    )

    files = sorted((ROOT / "results/raw/real_data" / dataset).glob("local_*.csv"))
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    success = frame[frame.status.eq("success") & frame.coordinate.notna()]
    wide = success.pivot(index="replication", columns="coordinate", values="estimate")
    wide = wide.reindex(columns=range(model.parameter_dimension))
    bootstrap = np.vstack([transform(row) for row in wide.to_numpy()])
    bootstrap_sd = bootstrap.std(axis=0, ddof=1)
    estimate = transform(coordinates)

    result = pd.DataFrame({
        "quantity": ["Mixing weight pi_1", "Location distance", "Student-t df nu"],
        "estimate": estimate,
        "naive_se": naive_se,
        "corrected_se": corrected_se,
        "local_bootstrap_sd": bootstrap_sd,
        "corrected_ci_lower": estimate - 1.96 * corrected_se,
        "corrected_ci_upper": estimate + 1.96 * corrected_se,
    })
    output = ROOT / "results/processed/real_data/final/dry_bean_dermason_seker_interpretable.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
