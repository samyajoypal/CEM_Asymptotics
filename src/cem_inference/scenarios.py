"""Prespecified simulation scenarios for the JRSS B study."""

from __future__ import annotations

import numpy as np

from .distributions import (
    EllipticalParameters,
    NormalAdapter,
    SkewNormalAdapter,
    SkewNormalParameters,
    StudentTAdapter,
    StudentTParameters,
)
from .sampling import MixtureSpecification


# Euclidean separation between component location parameters.  The former
# paper used 11.31, 5.66, and 4.24 in p=2.  These values retain a genuinely
# overlapping regime without moving the primary inferential study into the
# component-collapse regime discovered by the target audit.
OVERLAP_DISTANCE = {"separated": 6.0, "moderate": 4.25, "strong": 3.0}


def ar1_covariance(dimension: int, rho: float = 0.25, scale: float = 1.0) -> np.ndarray:
    indices = np.arange(dimension)
    return scale * rho ** np.abs(indices[:, None] - indices[None, :])


def two_component_scenario(
    family: str,
    dimension: int,
    overlap: str,
    weights: tuple[float, float] = (0.5, 0.5),
) -> MixtureSpecification:
    if overlap not in OVERLAP_DISTANCE:
        raise ValueError(f"Unknown overlap level: {overlap}")
    direction = np.ones(dimension) / np.sqrt(dimension)
    distance = OVERLAP_DISTANCE[overlap]
    # For p >= 5 the Normal--t classification criterion collapses a component
    # at distance 3 even in 100,000-observation references.  The primary
    # regular-inference design therefore uses the prespecified nearby value
    # 3.75; collapse remains an explicitly reported diagnostic outcome.
    if overlap == "strong" and dimension >= 5:
        distance = 3.75
    half_shift = 0.5 * distance * direction
    normal = NormalAdapter(dimension)
    normal_parameters = EllipticalParameters(
        -half_shift, ar1_covariance(dimension, rho=0.25)
    )
    if family == "normal_t":
        second_adapter = StudentTAdapter(dimension)
        second_parameters = StudentTParameters(
            half_shift, ar1_covariance(dimension, rho=0.2, scale=1.1), 6.0
        )
    elif family == "normal_skewnormal":
        second_adapter = SkewNormalAdapter(dimension)
        shape = np.zeros(dimension)
        shape[0] = 3.0
        if dimension > 1:
            shape[1] = -2.0
        second_parameters = SkewNormalParameters(
            half_shift, ar1_covariance(dimension, rho=-0.15), shape
        )
    else:
        raise ValueError(f"Unknown family: {family}")
    return MixtureSpecification(
        adapters=(normal, second_adapter),
        weights=np.asarray(weights, dtype=float),
        component_parameters=(normal_parameters, second_parameters),
    )


def three_component_scenario(dimension: int) -> MixtureSpecification:
    if dimension < 2:
        raise ValueError("The three-component geometry requires dimension at least two.")
    radius = 2.3
    locations = np.zeros((3, dimension))
    locations[0, :2] = [-radius, -radius / np.sqrt(3)]
    locations[1, :2] = [radius, -radius / np.sqrt(3)]
    locations[2, :2] = [0.0, 2 * radius / np.sqrt(3)]
    shape = np.zeros(dimension)
    shape[:2] = [2.5, -1.5]
    return MixtureSpecification(
        adapters=(NormalAdapter(dimension), StudentTAdapter(dimension), SkewNormalAdapter(dimension)),
        weights=np.array([0.3, 0.4, 0.3]),
        component_parameters=(
            EllipticalParameters(locations[0], ar1_covariance(dimension, 0.2)),
            StudentTParameters(
                locations[1], ar1_covariance(dimension, 0.15, 1.1), 7.0
            ),
            SkewNormalParameters(
                locations[2], ar1_covariance(dimension, -0.1), shape
            ),
        ),
    )


def scenario_registry() -> dict[str, MixtureSpecification]:
    scenarios: dict[str, MixtureSpecification] = {}
    abbreviations = {"normal_t": "nt", "normal_skewnormal": "nsn"}
    for family, abbreviation in abbreviations.items():
        for dimension in (2, 5):
            for overlap in OVERLAP_DISTANCE:
                scenarios[f"{abbreviation}_p{dimension}_{overlap}_balanced"] = (
                    two_component_scenario(family, dimension, overlap)
                )
        scenarios[f"{abbreviation}_p2_moderate_imbalanced"] = two_component_scenario(
            family, 2, "moderate", weights=(0.2, 0.8)
        )
        scenarios[f"{abbreviation}_p10_moderate_balanced"] = two_component_scenario(
            family, 10, "moderate"
        )
    scenarios["ntsn_p2_moderate_balanced"] = three_component_scenario(2)
    scenarios["ntsn_p5_moderate_balanced"] = three_component_scenario(5)
    return scenarios


def get_scenario(identifier: str) -> MixtureSpecification:
    registry = scenario_registry()
    if identifier not in registry:
        raise KeyError(f"Unknown scenario {identifier!r}; choices={sorted(registry)}")
    return registry[identifier]
