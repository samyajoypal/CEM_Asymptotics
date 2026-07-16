"""Linear-algebra parameterizations used by all component adapters."""

from __future__ import annotations

import numpy as np


def packed_cholesky_size(dimension: int) -> int:
    return dimension * (dimension + 1) // 2


def covariance_to_unconstrained(covariance: np.ndarray) -> np.ndarray:
    """Pack a positive-definite covariance using log-Cholesky coordinates."""
    covariance = np.asarray(np.real_if_close(covariance), dtype=float)
    chol = np.linalg.cholesky(covariance)
    values: list[float] = []
    for row in range(chol.shape[0]):
        for column in range(row + 1):
            value = chol[row, column]
            values.append(float(np.log(value)) if row == column else float(value))
    return np.asarray(values)


def unconstrained_to_covariance(values: np.ndarray, dimension: int) -> np.ndarray:
    """Reconstruct a positive-definite covariance from log-Cholesky coordinates."""
    values = np.asarray(values, dtype=float)
    expected = packed_cholesky_size(dimension)
    if values.shape != (expected,):
        raise ValueError(f"Expected {expected} covariance coordinates, got {values.shape}.")
    chol = np.zeros((dimension, dimension), dtype=float)
    index = 0
    for row in range(dimension):
        for column in range(row + 1):
            value = values[index]
            chol[row, column] = np.exp(value) if row == column else value
            index += 1
    return chol @ chol.T


def symmetric_inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric inverse square root of a positive-definite matrix."""
    matrix = np.asarray(np.real_if_close(matrix), dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.min(eigenvalues) <= 0:
        raise ValueError("Matrix must be positive definite.")
    return (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T


def softmax_reference(logits: np.ndarray) -> np.ndarray:
    """Map k-1 reference-category logits to k strictly positive weights."""
    logits = np.asarray(logits, dtype=float)
    augmented = np.concatenate([logits, np.zeros(1)])
    augmented -= np.max(augmented)
    weights = np.exp(augmented)
    return weights / np.sum(weights)


def weights_to_reference_logits(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) < 2 or np.any(weights <= 0):
        raise ValueError("Weights must be a positive one-dimensional vector.")
    weights = weights / np.sum(weights)
    return np.log(weights[:-1]) - np.log(weights[-1])
