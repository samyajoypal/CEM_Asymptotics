"""CML-target approximation with explicit reference-sample uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fmvmm_adapter import fit_fmvmm_hard
from .sampling import MixtureSpecification


@dataclass(frozen=True)
class CMLTargetStudy:
    estimates: np.ndarray
    objectives: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    monte_carlo_standard_error: np.ndarray
    reference_size: int
    replications: int
    failures: tuple[str, ...]
    fit_diagnostics: tuple[dict, ...]


def approximate_cml_target(
    specification: MixtureSpecification,
    *,
    reference_size: int,
    replications: int,
    seed: int,
    fit_kwargs: dict | None = None,
) -> CMLTargetStudy:
    """Fit independent large reference samples and quantify target uncertainty."""
    if reference_size <= 0 or replications < 2:
        raise ValueError("reference_size must be positive and replications at least two")
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    child_seeds = np.random.SeedSequence(seed).spawn(replications)
    estimates, objectives, failures, diagnostics = [], [], [], []
    for replication, child_seed in enumerate(child_seeds):
        observations, _ = specification.sample(
            reference_size, np.random.default_rng(child_seed)
        )
        try:
            fit = fit_fmvmm_hard(
                observations, list(specification.adapters), **fit_kwargs
            )
            estimates.append(fit.coordinates)
            objectives.append(fit.model.objective(observations, fit.coordinates))
            diagnostics.append({
                "initialization": fit.initialization,
                "objective_gap": fit.objective_gap,
                "admissible_candidates": fit.admissible_candidates,
                "total_candidates": fit.total_candidates,
                "selected_admissible": fit.selected_admissible,
                "minimum_weight": float(np.min(fit.weights)),
            })
        except Exception as error:
            failures.append(f"replication={replication}: {type(error).__name__}: {error}")
    if len(estimates) < 2:
        raise RuntimeError(
            f"Only {len(estimates)} target fits succeeded; failures={failures}"
        )
    estimate_matrix = np.vstack(estimates)
    covariance = np.cov(estimate_matrix, rowvar=False, ddof=1)
    return CMLTargetStudy(
        estimates=estimate_matrix,
        objectives=np.asarray(objectives),
        mean=np.mean(estimate_matrix, axis=0),
        covariance=covariance,
        monte_carlo_standard_error=np.sqrt(np.diag(covariance) / len(estimates)),
        reference_size=reference_size,
        replications=replications,
        failures=tuple(failures),
        fit_diagnostics=tuple(diagnostics),
    )
