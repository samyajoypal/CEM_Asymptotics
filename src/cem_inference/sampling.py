"""Reproducible sampling from heterogeneous finite mixtures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributions import DistributionAdapter


@dataclass(frozen=True)
class MixtureSpecification:
    adapters: tuple[DistributionAdapter, ...]
    weights: np.ndarray
    component_parameters: tuple

    def __post_init__(self):
        if len(self.adapters) != len(self.component_parameters):
            raise ValueError("One parameter object is required for every adapter.")
        weights = np.asarray(self.weights, dtype=float)
        if weights.shape != (len(self.adapters),) or np.any(weights <= 0):
            raise ValueError("weights must be positive with one entry per adapter")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("weights must sum to one")
        if len({adapter.dimension for adapter in self.adapters}) != 1:
            raise ValueError("All adapters must use the same observation dimension.")

    @property
    def observation_dimension(self) -> int:
        return self.adapters[0].dimension

    def sample(
        self, size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.choice(len(self.adapters), size=size, p=self.weights)
        observations = np.empty((size, self.observation_dimension), dtype=float)
        for component, (adapter, parameters) in enumerate(
            zip(self.adapters, self.component_parameters)
        ):
            selected = np.flatnonzero(labels == component)
            if len(selected):
                observations[selected] = adapter.sample(rng, len(selected), parameters)
        return observations, labels
