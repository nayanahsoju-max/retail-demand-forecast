"""Leakage-safe feature engineering for daily retail time series."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


DEFAULT_SERIES_COLUMNS = ("store_nbr", "family")


def add_calendar_features(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Add deterministic calendar fields known at forecast creation time."""
    result = frame.copy()
    dates = pd.to_datetime(result[date_column])
    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_month"] = dates.dt.day
    result["month"] = dates.dt.month
    result["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    result["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    return result


def add_lag_features(
    frame: pd.DataFrame,
    lags: Sequence[int],
    target_column: str = "sales",
    series_columns: Sequence[str] = DEFAULT_SERIES_COLUMNS,
) -> pd.DataFrame:
    """Add prior target values per series; no row can see its present or future target."""
    result = _sort_series(frame, series_columns)
    grouped = result.groupby(list(series_columns), observed=True)[target_column]
    for lag in lags:
        if lag < 1:
            raise ValueError("Lag periods must be positive")
        result[f"{target_column}_lag_{lag}"] = grouped.shift(lag)
    return result


def add_rolling_features(
    frame: pd.DataFrame,
    windows: Sequence[int],
    target_column: str = "sales",
    series_columns: Sequence[str] = DEFAULT_SERIES_COLUMNS,
) -> pd.DataFrame:
    """Add trailing mean and standard deviation based only on observations before a row."""
    result = _sort_series(frame, series_columns)
    grouped = result.groupby(list(series_columns), observed=True)[target_column]
    prior_values = grouped.shift(1)
    group_keys = [result[column] for column in series_columns]
    for window in windows:
        if window < 1:
            raise ValueError("Rolling windows must be positive")
        rolled = prior_values.groupby(group_keys, observed=True).rolling(window, min_periods=window)
        result[f"{target_column}_rolling_mean_{window}"] = rolled.mean().reset_index(level=list(range(len(series_columns))), drop=True)
        result[f"{target_column}_rolling_std_{window}"] = rolled.std().reset_index(level=list(range(len(series_columns))), drop=True)
    return result


def build_features(
    frame: pd.DataFrame,
    lags: Sequence[int] = (1, 7, 14, 28),
    windows: Sequence[int] = (7, 14, 28),
    target_column: str = "sales",
    series_columns: Sequence[str] = DEFAULT_SERIES_COLUMNS,
) -> pd.DataFrame:
    """Construct calendar, lag, and rolling features in an explicitly leakage-safe order."""
    result = add_calendar_features(frame)
    result = add_lag_features(result, lags, target_column, series_columns)
    return add_rolling_features(result, windows, target_column, series_columns)


def _sort_series(frame: pd.DataFrame, series_columns: Sequence[str]) -> pd.DataFrame:
    """Validate keys and return a stable chronological copy for grouped transforms."""
    required = set(series_columns).union({"date"})
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing series columns: {sorted(missing)}")
    return frame.copy().assign(date=lambda item: pd.to_datetime(item["date"])).sort_values(
        [*series_columns, "date"]
    ).reset_index(drop=True)
