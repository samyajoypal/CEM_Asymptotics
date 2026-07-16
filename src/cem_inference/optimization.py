"""Local empirical-CML optimization used to preserve the fitted basin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .model import ClassificationLikelihoodModel


@dataclass(frozen=True)
class LocalOptimizationResult:
    coordinates: np.ndarray
    objective: float
    initial_objective: float
    success: bool
    hit_trust_boundary: bool
    message: str
    iterations: int


def optimize_local_cml(
    observations: np.ndarray,
    model: ClassificationLikelihoodModel,
    initial_coordinates: np.ndarray,
    *,
    trust_radius: float = 0.75,
    max_iter: int = 250,
    tolerance: float = 1e-8,
) -> LocalOptimizationResult:
    """Maximize empirical CML inside a coordinate-wise local trust box."""
    if trust_radius <= 0:
        raise ValueError("trust_radius must be positive")
    observations = np.asarray(observations, dtype=float)
    initial = np.asarray(initial_coordinates, dtype=float)
    lower = initial - trust_radius
    upper = initial + trust_radius

    def negative_objective(coordinates: np.ndarray) -> float:
        value = model.objective(observations, coordinates)
        return -value if np.isfinite(value) else 1e300

    initial_objective = -negative_objective(initial)
    result = minimize(
        negative_objective,
        initial,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"maxiter": max_iter, "ftol": tolerance, "gtol": tolerance},
    )
    distance = np.abs(np.asarray(result.x) - initial)
    hit_boundary = bool(np.any(distance >= 0.98 * trust_radius))
    success = bool(
        result.success
        and np.isfinite(result.fun)
        and -float(result.fun) >= initial_objective - 1e-8
        and not hit_boundary
    )
    return LocalOptimizationResult(
        coordinates=np.asarray(result.x, dtype=float),
        objective=-float(result.fun),
        initial_objective=initial_objective,
        success=success,
        hit_trust_boundary=hit_boundary,
        message=str(result.message),
        iterations=int(getattr(result, "nit", -1)),
    )
