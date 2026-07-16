"""Auditable finite-difference derivatives for theory validation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def gradient(function: Callable[[np.ndarray], float], point: np.ndarray, step: float = 1e-5) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    result = np.empty_like(point)
    for index in range(point.size):
        increment = np.zeros_like(point)
        increment[index] = step
        result[index] = (function(point + increment) - function(point - increment)) / (2 * step)
    return result


def hessian(function: Callable[[np.ndarray], float], point: np.ndarray, step: float = 2e-4) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    size = point.size
    result = np.empty((size, size), dtype=float)
    center = function(point)
    for row in range(size):
        row_step = np.zeros(size)
        row_step[row] = step
        result[row, row] = (
            function(point + row_step) - 2 * center + function(point - row_step)
        ) / step**2
        for column in range(row):
            column_step = np.zeros(size)
            column_step[column] = step
            value = (
                function(point + row_step + column_step)
                - function(point + row_step - column_step)
                - function(point - row_step + column_step)
                + function(point - row_step - column_step)
            ) / (4 * step**2)
            result[row, column] = result[column, row] = value
    return result
