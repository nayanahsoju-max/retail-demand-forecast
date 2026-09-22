"""Forecast accuracy metrics used consistently across all models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Calculate root mean squared error."""
    observed, forecasts = _validated_arrays(actual, predicted)
    return float(np.sqrt(np.mean(np.square(observed - forecasts))))


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Calculate mean absolute error."""
    observed, forecasts = _validated_arrays(actual, predicted)
    return float(np.mean(np.abs(observed - forecasts)))


def wape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Calculate weighted absolute percentage error as a percentage."""
    observed, forecasts = _validated_arrays(actual, predicted)
    denominator = np.sum(np.abs(observed))
    return float("nan") if denominator == 0 else float(100 * np.sum(np.abs(observed - forecasts)) / denominator)


def calculate_metrics(actual: ArrayLike, predicted: ArrayLike) -> dict[str, float]:
    """Return the standard, model-agnostic accuracy metric set."""
    return {"rmse": rmse(actual, predicted), "mae": mae(actual, predicted), "wape": wape(actual, predicted)}


def _validated_arrays(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert arrays to finite one-dimensional numeric arrays of equal length."""
    observed = np.asarray(actual, dtype=float).reshape(-1)
    forecasts = np.asarray(predicted, dtype=float).reshape(-1)
    if len(observed) == 0 or len(observed) != len(forecasts):
        raise ValueError("Actual and predicted arrays must be non-empty and equally sized")
    if not np.isfinite(observed).all() or not np.isfinite(forecasts).all():
        raise ValueError("Metrics require finite values")
    return observed, forecasts
