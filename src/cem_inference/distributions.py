"""Exact component-density adapters with shared unconstrained coordinates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import multivariate_normal, multivariate_t

from .linalg import (
    covariance_to_unconstrained,
    packed_cholesky_size,
    symmetric_inverse_sqrt,
    unconstrained_to_covariance,
)


class DistributionAdapter(ABC):
    """Interface joining fitting parameters to the paper's density definition."""

    name: str

    def __init__(self, dimension: int):
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)

    @property
    @abstractmethod
    def parameter_size(self) -> int: ...

    @abstractmethod
    def pack(self, parameters: Any) -> np.ndarray: ...

    @abstractmethod
    def unpack(self, coordinates: np.ndarray) -> Any: ...

    @abstractmethod
    def logpdf(self, observations: np.ndarray, parameters: Any) -> np.ndarray: ...

    @abstractmethod
    def sample(self, rng: np.random.Generator, size: int, parameters: Any) -> np.ndarray: ...

    def validate_observations(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=float)
        observations = np.atleast_2d(observations)
        if observations.shape[1] != self.dimension:
            raise ValueError(
                f"Expected observations with {self.dimension} columns, "
                f"got {observations.shape}."
            )
        return observations


@dataclass(frozen=True)
class EllipticalParameters:
    location: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class StudentTParameters:
    location: np.ndarray
    scale: np.ndarray
    degrees_of_freedom: float


@dataclass(frozen=True)
class SkewNormalParameters:
    location: np.ndarray
    covariance: np.ndarray
    shape: np.ndarray


class NormalAdapter(DistributionAdapter):
    name = "normal"

    @property
    def parameter_size(self) -> int:
        return self.dimension + packed_cholesky_size(self.dimension)

    def pack(self, parameters: EllipticalParameters | tuple) -> np.ndarray:
        if not isinstance(parameters, EllipticalParameters):
            parameters = EllipticalParameters(*parameters)
        return np.concatenate(
            [np.asarray(parameters.location, dtype=float),
             covariance_to_unconstrained(parameters.covariance)]
        )

    def unpack(self, coordinates: np.ndarray) -> EllipticalParameters:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (self.parameter_size,):
            raise ValueError("Normal coordinate vector has the wrong size.")
        return EllipticalParameters(
            coordinates[: self.dimension].copy(),
            unconstrained_to_covariance(coordinates[self.dimension :], self.dimension),
        )

    def logpdf(self, observations: np.ndarray, parameters: EllipticalParameters) -> np.ndarray:
        observations = self.validate_observations(observations)
        return np.asarray(
            multivariate_normal.logpdf(
                observations, mean=parameters.location, cov=parameters.covariance
            ),
            dtype=float,
        ).reshape(-1)

    def sample(
        self, rng: np.random.Generator, size: int, parameters: EllipticalParameters
    ) -> np.ndarray:
        return np.atleast_2d(
            rng.multivariate_normal(parameters.location, parameters.covariance, size=size)
        ).reshape(size, self.dimension)


class StudentTAdapter(DistributionAdapter):
    name = "student_t"

    def __init__(self, dimension: int, minimum_df: float = 2.0):
        super().__init__(dimension)
        self.minimum_df = float(minimum_df)

    @property
    def parameter_size(self) -> int:
        return self.dimension + packed_cholesky_size(self.dimension) + 1

    def pack(self, parameters: StudentTParameters | tuple) -> np.ndarray:
        if not isinstance(parameters, StudentTParameters):
            parameters = StudentTParameters(*parameters)
        excess = float(parameters.degrees_of_freedom) - self.minimum_df
        if excess <= 0:
            raise ValueError("Degrees of freedom must exceed minimum_df.")
        return np.concatenate(
            [np.asarray(parameters.location, dtype=float),
             covariance_to_unconstrained(parameters.scale),
             np.asarray([np.log(excess)])]
        )

    def unpack(self, coordinates: np.ndarray) -> StudentTParameters:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (self.parameter_size,):
            raise ValueError("Student-t coordinate vector has the wrong size.")
        covariance_end = self.dimension + packed_cholesky_size(self.dimension)
        return StudentTParameters(
            coordinates[: self.dimension].copy(),
            unconstrained_to_covariance(
                coordinates[self.dimension : covariance_end], self.dimension
            ),
            self.minimum_df + float(np.exp(coordinates[-1])),
        )

    def logpdf(self, observations: np.ndarray, parameters: StudentTParameters) -> np.ndarray:
        observations = self.validate_observations(observations)
        return np.asarray(
            multivariate_t.logpdf(
                observations,
                loc=parameters.location,
                shape=parameters.scale,
                df=parameters.degrees_of_freedom,
            ),
            dtype=float,
        ).reshape(-1)

    def sample(
        self, rng: np.random.Generator, size: int, parameters: StudentTParameters
    ) -> np.ndarray:
        gaussian = rng.multivariate_normal(
            np.zeros(self.dimension), parameters.scale, size=size
        )
        scaling = np.sqrt(
            rng.chisquare(parameters.degrees_of_freedom, size=size)
            / parameters.degrees_of_freedom
        )
        return parameters.location + gaussian / scaling[:, None]


class SkewNormalAdapter(DistributionAdapter):
    """Azzalini-type multivariate skew normal used by FMVMM.

    The density is 2 phi_p(x; xi, Omega)
    Phi{alpha^T Omega^{-1/2}(x-xi)}, with the symmetric inverse square root.
    """

    name = "skew_normal"

    @property
    def parameter_size(self) -> int:
        return 2 * self.dimension + packed_cholesky_size(self.dimension)

    def pack(self, parameters: SkewNormalParameters | tuple) -> np.ndarray:
        if not isinstance(parameters, SkewNormalParameters):
            parameters = SkewNormalParameters(*parameters)
        return np.concatenate(
            [np.asarray(parameters.location, dtype=float),
             covariance_to_unconstrained(parameters.covariance),
             np.asarray(parameters.shape, dtype=float)]
        )

    def unpack(self, coordinates: np.ndarray) -> SkewNormalParameters:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (self.parameter_size,):
            raise ValueError("Skew-normal coordinate vector has the wrong size.")
        covariance_end = self.dimension + packed_cholesky_size(self.dimension)
        return SkewNormalParameters(
            coordinates[: self.dimension].copy(),
            unconstrained_to_covariance(
                coordinates[self.dimension : covariance_end], self.dimension
            ),
            coordinates[covariance_end:].copy(),
        )

    def logpdf(self, observations: np.ndarray, parameters: SkewNormalParameters) -> np.ndarray:
        observations = self.validate_observations(observations)
        inverse_sqrt = symmetric_inverse_sqrt(parameters.covariance)
        standardized = (observations - parameters.location) @ inverse_sqrt
        skew_argument = standardized @ parameters.shape
        normal_part = multivariate_normal.logpdf(
            observations, mean=parameters.location, cov=parameters.covariance
        )
        return np.log(2.0) + np.asarray(normal_part) + log_ndtr(skew_argument)

    def sample(
        self, rng: np.random.Generator, size: int, parameters: SkewNormalParameters
    ) -> np.ndarray:
        covariance = np.asarray(parameters.covariance, dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance_sqrt = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
        delta = covariance_sqrt @ parameters.shape
        delta /= np.sqrt(1.0 + parameters.shape @ parameters.shape)
        augmented_covariance = np.block(
            [
                [np.ones((1, 1)), delta[None, :]],
                [delta[:, None], covariance],
            ]
        )
        latent = rng.multivariate_normal(
            np.zeros(self.dimension + 1), augmented_covariance, size=size
        )
        signs = np.where(latent[:, :1] >= 0, 1.0, -1.0)
        return parameters.location + signs * latent[:, 1:]
