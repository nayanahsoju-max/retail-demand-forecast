"""Custom rolling-origin validation for ordered retail time series."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .metrics import calculate_metrics


@dataclass(frozen=True)
class BacktestWindow:
    """One chronological train/test partition identified by its forecast origin."""

    window_id: int
    train_indices: pd.Index
    test_indices: pd.Index
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class ForecastModel(Protocol):
    """Minimal model contract required by the rolling backtest runner."""

    name: str

    def fit(self, train: pd.DataFrame) -> ForecastModel:
        """Fit the model on a training frame."""

    def predict(self, future: pd.DataFrame) -> pd.Series:
        """Predict target values aligned with the future frame."""


class RollingWindowSplitter:
    """Generate expanding-window, fixed-horizon splits using unique calendar dates."""

    def __init__(self, initial_train_days: int, horizon_days: int, step_days: int, max_windows: int | None = None) -> None:
        if min(initial_train_days, horizon_days, step_days) < 1:
            raise ValueError("All rolling-window periods must be positive")
        self.initial_train_days = initial_train_days
        self.horizon_days = horizon_days
        self.step_days = step_days
        self.max_windows = max_windows

    def split(self, frame: pd.DataFrame, date_column: str = "date") -> Iterator[BacktestWindow]:
        """Yield no-lookahead train/test windows ordered by the supplied date column."""
        if date_column not in frame:
            raise ValueError(f"Missing date column: {date_column}")
        dates = pd.to_datetime(frame[date_column])
        unique_dates = pd.Index(sorted(dates.dropna().unique()))
        start = self.initial_train_days
        count = 0
        while start + self.horizon_days <= len(unique_dates):
            if self.max_windows is not None and count >= self.max_windows:
                break
            train_dates = unique_dates[:start]
            test_dates = unique_dates[start : start + self.horizon_days]
            yield BacktestWindow(
                window_id=count,
                train_indices=frame.index[dates.isin(train_dates)],
                test_indices=frame.index[dates.isin(test_dates)],
                train_end=pd.Timestamp(train_dates[-1]),
                test_start=pd.Timestamp(test_dates[0]),
                test_end=pd.Timestamp(test_dates[-1]),
            )
            start += self.step_days
            count += 1


class BacktestRunner:
    """Fit a model repeatedly and return prediction-level and aggregate results."""

    def __init__(self, splitter: RollingWindowSplitter, target_column: str = "sales") -> None:
        self.splitter = splitter
        self.target_column = target_column

    def run(self, model: ForecastModel, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run model through every split and return predictions and window metrics."""
        prediction_rows: list[pd.DataFrame] = []
        metric_rows: list[dict[str, object]] = []
        for window in self.splitter.split(frame):
            train, test = frame.loc[window.train_indices], frame.loc[window.test_indices]
            fitted = model.fit(train)
            predicted = pd.Series(fitted.predict(test), index=test.index, dtype=float)
            if len(predicted) != len(test):
                raise ValueError("Model prediction count must equal the test frame length")
            metrics = calculate_metrics(test[self.target_column], predicted)
            metric_rows.append({"model": model.name, "window_id": window.window_id, **metrics,
                                "train_end": window.train_end, "test_end": window.test_end})
            prediction_rows.append(pd.DataFrame({"date": test["date"], "actual": test[self.target_column],
                                                 "prediction": predicted, "model": model.name,
                                                 "window_id": window.window_id}))
        return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(metric_rows)
