"""Grid-free active-boundary curvature estimation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .model import ClassificationLikelihoodModel


def epanechnikov(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 0.75 * np.maximum(1.0 - values**2, 0.0)


def estimate_boundary_curvature(
    model: ClassificationLikelihoodModel,
    observations: np.ndarray,
    coordinates: np.ndarray,
    bandwidth: float | dict[tuple[int, int], float],
    kernel: Callable[[np.ndarray], np.ndarray] = epanechnikov,
    derivative_step: float = 1e-5,
) -> np.ndarray:
    """Estimate the active-face matrix B using kernel tubes around g_jl=0."""
    observations = np.atleast_2d(np.asarray(observations, dtype=float))
    scores = model.component_scores(observations, coordinates)
    result = np.zeros((model.parameter_dimension, model.parameter_dimension))
    for first, second in model.component_pairs:
        pair = (first, second)
        pair_bandwidth = bandwidth[pair] if isinstance(bandwidth, dict) else bandwidth
        if pair_bandwidth <= 0:
            raise ValueError(f"bandwidth for pair {pair} must be positive")
        contrasts = scores[:, first] - scores[:, second]
        weights = kernel(contrasts / pair_bandwidth)
        active = model.active_pair_mask(scores, first, second)
        relevant = active & (weights != 0)
        if not np.any(relevant):
            continue
        gradients = model.pairwise_contrast_gradients(
            observations[relevant],
            coordinates,
            first,
            second,
            step=derivative_step,
        )
        result += np.einsum(
            "n,ni,nj->ij", weights[relevant], gradients, gradients, optimize=True
        ) / pair_bandwidth
    result /= len(observations)
    return 0.5 * (result + result.T)


def sandwich_covariance(
    fixed_information: np.ndarray,
    boundary_curvature: np.ndarray,
    classified_scores: np.ndarray,
    *,
    require_positive_definite: bool = True,
    eigenvalue_tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (V, A, J), where covariance(theta_hat) is V / n."""
    fixed_information = np.asarray(fixed_information, dtype=float)
    boundary_curvature = np.asarray(boundary_curvature, dtype=float)
    classified_scores = np.asarray(classified_scores, dtype=float)
    score_covariance = classified_scores.T @ classified_scores / len(classified_scores)
    positive_curvature = fixed_information - boundary_curvature
    positive_curvature = 0.5 * (positive_curvature + positive_curvature.T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(positive_curvature)))
    if require_positive_definite and minimum_eigenvalue <= eigenvalue_tolerance:
        raise np.linalg.LinAlgError(
            "Estimated positive curvature is not positive definite: "
            f"minimum eigenvalue={minimum_eigenvalue:.6g}."
        )
    inverse = (
        np.linalg.inv(positive_curvature)
        if minimum_eigenvalue > eigenvalue_tolerance
        else np.linalg.pinv(positive_curvature, hermitian=True)
    )
    covariance = inverse @ score_covariance @ inverse.T
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, positive_curvature, score_covariance


def stabilize_boundary_curvature(
    fixed_information: np.ndarray,
    boundary_curvature: np.ndarray,
    sample_size: int,
    *,
    relative_floor: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Shrink B only enough to keep I_c-B uniformly positive.

    The constraint is lambda_max(I_c^{-1/2} B I_c^{-1/2}) <= 1-epsilon_n,
    with epsilon_n=n^{-1/4} by default. The returned tuple is
    (stabilized_B, shrinkage_factor, raw_generalized_max_eigenvalue).
    """
    fixed = 0.5 * (
        np.asarray(fixed_information, dtype=float)
        + np.asarray(fixed_information, dtype=float).T
    )
    boundary = 0.5 * (
        np.asarray(boundary_curvature, dtype=float)
        + np.asarray(boundary_curvature, dtype=float).T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(fixed)
    if np.min(eigenvalues) <= 0:
        raise np.linalg.LinAlgError(
            "Fixed-classification information must be positive definite before "
            "boundary stabilization."
        )
    inverse_sqrt = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    relative_boundary = inverse_sqrt @ boundary @ inverse_sqrt
    relative_boundary = 0.5 * (relative_boundary + relative_boundary.T)
    maximum = float(np.max(np.linalg.eigvalsh(relative_boundary)))
    epsilon = sample_size ** -0.25 if relative_floor is None else relative_floor
    if not 0 < epsilon < 1:
        raise ValueError("relative_floor must lie strictly between zero and one")
    factor = 1.0 if maximum <= 1 - epsilon else (1 - epsilon) / maximum
    return factor * boundary, float(factor), maximum
