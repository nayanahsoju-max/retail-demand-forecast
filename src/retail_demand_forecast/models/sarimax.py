"""SARIMAX baseline implementation for a single Favorita sales series."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper


class SARIMAXForecaster:
    """Univariate SARIMAX forecaster implementing the shared backtest model contract."""

    name = "sarimax"

    def __init__(
        self,
        order: Sequence[int] = (1, 1, 1),
        seasonal_order: Sequence[int] = (0, 1, 1, 7),
        target_column: str = "sales",
        date_column: str = "date",
    ) -> None:
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.target_column = target_column
        self.date_column = date_column
        self._result: SARIMAXResultsWrapper | None = None

    def fit(self, train: pd.DataFrame) -> "SARIMAXForecaster":
        """Fit the statistical model to chronologically ordered historical sales."""
        self._validate_train(train)
        ordered = train.sort_values(self.date_column)
        target = ordered.set_index(pd.to_datetime(ordered[self.date_column]))[self.target_column].astype(float)
        model = SARIMAX(
            target,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False)
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        """Forecast one value per future row, preserving its original row index."""
        if self._result is None:
            raise RuntimeError("Call fit before predict")
        if future.empty:
            return pd.Series(dtype=float, index=future.index, name="prediction")
        forecast = self._result.get_forecast(steps=len(future)).predicted_mean.clip(lower=0)
        return pd.Series(forecast.to_numpy(), index=future.index, name="prediction")

    def _validate_train(self, train: pd.DataFrame) -> None:
        """Check required fields and prohibit accidental multi-series aggregation."""
        required = {self.date_column, self.target_column}
        missing = required.difference(train.columns)
        if missing:
            raise ValueError(f"SARIMAX training frame missing columns: {sorted(missing)}")
        series_keys = [key for key in ("store_nbr", "family") if key in train]
        if series_keys and len(train[series_keys].drop_duplicates()) != 1:
            raise ValueError("SARIMAXForecaster accepts one store/family series per fit")
