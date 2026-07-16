"""Pairwise bandwidth selection and diagnostics for boundary tubes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ClassificationLikelihoodModel


@dataclass(frozen=True)
class PairBandwidthDiagnostic:
    pair: tuple[int, int]
    bandwidth: float
    robust_scale: float
    active_observations: int
    observations_in_kernel_support: int
    effective_fraction: float


def robust_scale(values: np.ndarray, minimum: float = 1e-3) -> float:
    values = np.asarray(values, dtype=float)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr_scale = (q75 - q25) / 1.349
    standard = np.std(values, ddof=1) if len(values) > 1 else 0.0
    candidates = [value for value in (iqr_scale, standard) if np.isfinite(value) and value > 0]
    return max(min(candidates) if candidates else minimum, minimum)


def select_pairwise_bandwidths(
    model: ClassificationLikelihoodModel,
    observations: np.ndarray,
    coordinates: np.ndarray,
    *,
    multiplier: float = 1.06,
    rate_exponent: float = -0.2,
    minimum: float = 1e-3,
    scale_by_global_contrast: bool = False,
) -> tuple[dict[tuple[int, int], float], list[PairBandwidthDiagnostic]]:
    observations = np.atleast_2d(np.asarray(observations, dtype=float))
    scores = model.component_scores(observations, coordinates)
    bandwidths: dict[tuple[int, int], float] = {}
    diagnostics: list[PairBandwidthDiagnostic] = []
    for pair in model.component_pairs:
        first, second = pair
        active = model.active_pair_mask(scores, first, second)
        contrasts = scores[:, first] - scores[:, second]
        active_contrasts = contrasts[active]
        scale = robust_scale(active_contrasts, minimum=minimum)
        bandwidth_scale = scale if scale_by_global_contrast else 1.0
        bandwidth = max(
            multiplier * bandwidth_scale * len(observations) ** rate_exponent,
            minimum,
        )
        in_support = active & (np.abs(contrasts) <= bandwidth)
        count = int(np.sum(in_support))
        bandwidths[pair] = bandwidth
        diagnostics.append(
            PairBandwidthDiagnostic(
                pair=pair,
                bandwidth=bandwidth,
                robust_scale=scale,
                active_observations=int(np.sum(active)),
                observations_in_kernel_support=count,
                effective_fraction=count / len(observations),
            )
        )
    return bandwidths, diagnostics


def bandwidth_warnings(
    diagnostics: list[PairBandwidthDiagnostic], minimum_effective_count: int = 30
) -> list[str]:
    """Return explicit reliability warnings; never silently widen a bandwidth."""
    messages = []
    for diagnostic in diagnostics:
        if diagnostic.observations_in_kernel_support < minimum_effective_count:
            messages.append(
                f"Pair {diagnostic.pair} has only "
                f"{diagnostic.observations_in_kernel_support} observations in the "
                f"kernel support (recommended minimum {minimum_effective_count})."
            )
        if diagnostic.active_observations == 0:
            messages.append(f"Pair {diagnostic.pair} is never active in the fitted partition.")
    return messages
