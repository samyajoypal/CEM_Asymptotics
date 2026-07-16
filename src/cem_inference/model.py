"""Classification-likelihood model matching the notation in the paper."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .distributions import DistributionAdapter
from .linalg import softmax_reference, weights_to_reference_logits
from .numerical import gradient, hessian


@dataclass(frozen=True)
class UnpackedModel:
    weights: np.ndarray
    component_parameters: tuple


class ClassificationLikelihoodModel:
    def __init__(self, adapters: list[DistributionAdapter]):
        if len(adapters) < 2:
            raise ValueError("At least two component adapters are required.")
        dimensions = {adapter.dimension for adapter in adapters}
        if len(dimensions) != 1:
            raise ValueError("All adapters must have the same observation dimension.")
        self.adapters = tuple(adapters)
        self.observation_dimension = adapters[0].dimension
        self.n_components = len(adapters)
        self.parameter_dimension = self.n_components - 1 + sum(
            adapter.parameter_size for adapter in adapters
        )

    @property
    def component_slices(self) -> tuple[slice, ...]:
        slices = []
        start = self.n_components - 1
        for adapter in self.adapters:
            end = start + adapter.parameter_size
            slices.append(slice(start, end))
            start = end
        return tuple(slices)

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        names = [f"weight_logit_{index + 1}_vs_{self.n_components}"
                 for index in range(self.n_components - 1)]
        for component, adapter in enumerate(self.adapters, start=1):
            names.extend(
                f"component_{component}_{adapter.name}_coordinate_{index}"
                for index in range(adapter.parameter_size)
            )
        return tuple(names)

    def pack(self, weights: np.ndarray, component_parameters: list | tuple) -> np.ndarray:
        if len(component_parameters) != self.n_components:
            raise ValueError("One parameter object is required per component.")
        parts = [weights_to_reference_logits(weights)]
        parts.extend(
            adapter.pack(parameters)
            for adapter, parameters in zip(self.adapters, component_parameters)
        )
        return np.concatenate(parts)

    def unpack(self, coordinates: np.ndarray) -> UnpackedModel:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (self.parameter_dimension,):
            raise ValueError(
                f"Expected {self.parameter_dimension} coordinates, got {coordinates.shape}."
            )
        weight_end = self.n_components - 1
        weights = softmax_reference(coordinates[:weight_end])
        parameters = []
        start = weight_end
        for adapter in self.adapters:
            end = start + adapter.parameter_size
            parameters.append(adapter.unpack(coordinates[start:end]))
            start = end
        return UnpackedModel(weights, tuple(parameters))

    def component_scores(self, observations: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        unpacked = self.unpack(coordinates)
        observations = np.atleast_2d(np.asarray(observations, dtype=float))
        columns = [
            np.log(unpacked.weights[index])
            + adapter.logpdf(observations, unpacked.component_parameters[index])
            for index, adapter in enumerate(self.adapters)
        ]
        return np.column_stack(columns)

    def objective(self, observations: np.ndarray, coordinates: np.ndarray) -> float:
        return float(np.mean(np.max(self.component_scores(observations, coordinates), axis=1)))

    def assignments(self, observations: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        return np.argmax(self.component_scores(observations, coordinates), axis=1)

    def active_pair_mask(
        self, scores: np.ndarray, first: int, second: int
    ) -> np.ndarray:
        if self.n_components == 2:
            return np.ones(scores.shape[0], dtype=bool)
        remaining = [i for i in range(self.n_components) if i not in (first, second)]
        return np.minimum(scores[:, first], scores[:, second]) > np.max(
            scores[:, remaining], axis=1
        )

    def pairwise_contrast_gradients(
        self,
        observations: np.ndarray,
        coordinates: np.ndarray,
        first: int,
        second: int,
        step: float = 1e-5,
    ) -> np.ndarray:
        observations = np.atleast_2d(np.asarray(observations, dtype=float))
        gradients = np.empty((len(observations), self.parameter_dimension))
        for column in range(self.parameter_dimension):
            increment = np.zeros(self.parameter_dimension)
            increment[column] = step
            plus = self.component_scores(observations, coordinates + increment)
            minus = self.component_scores(observations, coordinates - increment)
            gradients[:, column] = (
                (plus[:, first] - plus[:, second])
                - (minus[:, first] - minus[:, second])
            ) / (2 * step)
        return gradients

    def pairwise_contrast_gradients_reference(
        self,
        observations: np.ndarray,
        coordinates: np.ndarray,
        first: int,
        second: int,
        step: float = 1e-5,
    ) -> np.ndarray:
        """Slow observation-wise implementation retained as a validation oracle."""
        observations = np.atleast_2d(np.asarray(observations, dtype=float))
        gradients = np.empty((len(observations), self.parameter_dimension))
        for row, observation in enumerate(observations):
            def contrast(local_coordinates: np.ndarray) -> float:
                scores = self.component_scores(observation[None, :], local_coordinates)
                return float(scores[0, first] - scores[0, second])
            gradients[row] = gradient(contrast, coordinates, step=step)
        return gradients

    def classified_score_matrix(
        self, observations: np.ndarray, coordinates: np.ndarray, step: float = 1e-5
    ) -> np.ndarray:
        observations = np.atleast_2d(np.asarray(observations, dtype=float))
        winners = self.assignments(observations, coordinates)
        result = np.empty((len(observations), self.parameter_dimension))
        rows = np.arange(len(observations))
        for column in range(self.parameter_dimension):
            increment = np.zeros(self.parameter_dimension)
            increment[column] = step
            plus = self.component_scores(observations, coordinates + increment)
            minus = self.component_scores(observations, coordinates - increment)
            result[:, column] = (
                plus[rows, winners] - minus[rows, winners]
            ) / (2 * step)
        return result

    def classified_score_matrix_reference(
        self, observations: np.ndarray, coordinates: np.ndarray, step: float = 1e-5
    ) -> np.ndarray:
        observations = np.atleast_2d(np.asarray(observations, dtype=float))
        winners = self.assignments(observations, coordinates)
        result = np.empty((len(observations), self.parameter_dimension))
        for row, (observation, winner) in enumerate(zip(observations, winners)):
            def winning_score(local_coordinates: np.ndarray) -> float:
                return float(self.component_scores(
                    observation[None, :], local_coordinates
                )[0, winner])
            result[row] = gradient(winning_score, coordinates, step=step)
        return result

    def fixed_classification_information(
        self, observations: np.ndarray, coordinates: np.ndarray, step: float = 2e-4
    ) -> np.ndarray:
        observations = np.atleast_2d(np.asarray(observations, dtype=float))
        winners = self.assignments(observations, coordinates)
        unpacked = self.unpack(coordinates)
        result = np.zeros((self.parameter_dimension, self.parameter_dimension))

        reference_weights = unpacked.weights[:-1]
        weight_information = np.diag(reference_weights) - np.outer(
            reference_weights, reference_weights
        )
        weight_size = self.n_components - 1
        result[:weight_size, :weight_size] = weight_information

        for component, (adapter, parameter_slice) in enumerate(
            zip(self.adapters, self.component_slices)
        ):
            selected = observations[winners == component]
            if len(selected) == 0:
                continue
            local_start = coordinates[parameter_slice]

            def mean_component_logpdf(local_coordinates: np.ndarray) -> float:
                parameters = adapter.unpack(local_coordinates)
                return float(np.mean(adapter.logpdf(selected, parameters)))

            block = -len(selected) / len(observations) * hessian(
                mean_component_logpdf, local_start, step=step
            )
            result[parameter_slice, parameter_slice] = block
        return 0.5 * (result + result.T)

    def fixed_classification_information_reference(
        self, observations: np.ndarray, coordinates: np.ndarray, step: float = 2e-4
    ) -> np.ndarray:
        """Full-coordinate finite difference retained to validate block structure."""
        observations = np.atleast_2d(np.asarray(observations, dtype=float))
        winners = self.assignments(observations, coordinates)
        rows = np.arange(len(observations))

        def mean_fixed_classification_score(local_coordinates: np.ndarray) -> float:
            scores = self.component_scores(observations, local_coordinates)
            return float(np.mean(scores[rows, winners]))

        result = -hessian(mean_fixed_classification_score, coordinates, step=step)
        return 0.5 * (result + result.T)

    @property
    def component_pairs(self):
        return tuple(combinations(range(self.n_components), 2))
