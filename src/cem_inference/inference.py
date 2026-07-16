"""End-to-end feasible inference and bootstrap diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bandwidth import (
    PairBandwidthDiagnostic,
    bandwidth_warnings,
    select_pairwise_bandwidths,
)
from .boundary import estimate_boundary_curvature, sandwich_covariance
from .fmvmm_adapter import AdaptedFMVMMFit, fit_fmvmm_hard
from .optimization import optimize_local_cml


@dataclass(frozen=True)
class InferenceResult:
    naive_covariance: np.ndarray
    corrected_covariance: np.ndarray
    naive_standard_errors: np.ndarray
    corrected_standard_errors: np.ndarray
    fixed_information: np.ndarray
    boundary_curvature: np.ndarray
    score_covariance: np.ndarray
    naive_curvature_min_eigenvalue: float
    corrected_curvature_min_eigenvalue: float
    bandwidths: dict[tuple[int, int], float]
    bandwidth_diagnostics: tuple[PairBandwidthDiagnostic, ...]
    warnings: tuple[str, ...]


def compute_inference(
    observations: np.ndarray,
    fit: AdaptedFMVMMFit,
    *,
    bandwidth_multiplier: float = 1.06,
    score_step: float = 1e-5,
    hessian_step: float = 2e-4,
) -> InferenceResult:
    observations = np.asarray(observations, dtype=float)
    model, coordinates = fit.model, fit.coordinates
    scores = model.classified_score_matrix(observations, coordinates, step=score_step)
    fixed = model.fixed_classification_information(
        observations, coordinates, step=hessian_step
    )
    bandwidths, diagnostics = select_pairwise_bandwidths(
        model,
        observations,
        coordinates,
        multiplier=bandwidth_multiplier,
    )
    boundary = estimate_boundary_curvature(
        model,
        observations,
        coordinates,
        bandwidth=bandwidths,
        derivative_step=score_step,
    )
    zeros = np.zeros_like(boundary)
    naive, naive_curvature, score_covariance = sandwich_covariance(
        fixed, zeros, scores
    )
    corrected, corrected_curvature, corrected_score_covariance = sandwich_covariance(
        fixed, boundary, scores
    )
    np.testing.assert_allclose(score_covariance, corrected_score_covariance)
    sample_size = len(observations)
    return InferenceResult(
        naive_covariance=naive,
        corrected_covariance=corrected,
        naive_standard_errors=np.sqrt(np.maximum(np.diag(naive), 0) / sample_size),
        corrected_standard_errors=np.sqrt(
            np.maximum(np.diag(corrected), 0) / sample_size
        ),
        fixed_information=fixed,
        boundary_curvature=boundary,
        score_covariance=score_covariance,
        naive_curvature_min_eigenvalue=float(np.min(np.linalg.eigvalsh(naive_curvature))),
        corrected_curvature_min_eigenvalue=float(
            np.min(np.linalg.eigvalsh(corrected_curvature))
        ),
        bandwidths=bandwidths,
        bandwidth_diagnostics=tuple(diagnostics),
        warnings=tuple(bandwidth_warnings(diagnostics)),
    )


def bootstrap_coordinates(
    observations: np.ndarray,
    fit: AdaptedFMVMMFit,
    *,
    replications: int,
    seed: int,
    fit_kwargs: dict | None = None,
    reselect_assignments: bool = False,
    local_trust_radius: float | None = 0.75,
) -> tuple[np.ndarray, list[str]]:
    """Refit bootstrap samples and retain every failure message.

    By default this conditions on the fitted family assignment, matching the
    local bootstrap theorem. Setting ``reselect_assignments=True`` measures the
    additional, generally non-Gaussian family-assignment selection variability.
    """
    observations = np.asarray(observations, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    failures = []
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    for replication in range(replications):
        indices = rng.integers(0, len(observations), size=len(observations))
        try:
            bootstrap_sample = observations[indices]
            if local_trust_radius is not None:
                local = optimize_local_cml(
                    bootstrap_sample,
                    fit.model,
                    fit.coordinates,
                    trust_radius=local_trust_radius,
                )
                if not local.success:
                    raise RuntimeError(
                        "Local bootstrap optimization failed or reached its trust "
                        f"boundary: {local.message}"
                    )
                estimates.append(local.coordinates)
            else:
                bootstrap_fit = fit_fmvmm_hard(
                    bootstrap_sample,
                    list(fit.model.adapters),
                    assignment_permutations=reselect_assignments,
                    **fit_kwargs,
                )
                estimates.append(bootstrap_fit.coordinates)
        except Exception as error:  # failure is part of the reported diagnostic
            failures.append(f"replication={replication}: {type(error).__name__}: {error}")
    if not estimates:
        return np.empty((0, fit.model.parameter_dimension)), failures
    return np.vstack(estimates), failures
