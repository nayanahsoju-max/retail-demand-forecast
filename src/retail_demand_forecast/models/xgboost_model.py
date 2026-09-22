"""Gradient-boosted retail forecaster with recursive leakage-safe features."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from xgboost import XGBRegressor

from retail_demand_forecast.features.engineering import build_features


class XGBoostForecaster:
    """XGBoost regressor using calendar, lag, and rolling features for one sales series."""

    name = "xgboost"

    def __init__(
        self,
        lags: Sequence[int] = (1, 7, 14, 28),
        windows: Sequence[int] = (7, 14, 28),
        target_column: str = "sales",
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.lags, self.windows = tuple(lags), tuple(windows)
        self.target_column = target_column
        self.model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            objective="reg:squarederror", n_jobs=1, random_state=random_state,
        )
        self._history: pd.DataFrame | None = None
        self._feature_columns: list[str] = []

    def fit(self, train: pd.DataFrame) -> "XGBoostForecaster":
        """Build causal training features and fit a gradient-boosted regression model."""
        self._validate_frame(train)
        self._history = train.sort_values("date").copy()
        featured = build_features(self._history, self.lags, self.windows, self.target_column)
        excluded = {"date", self.target_column, "store_nbr", "family"}
        self._feature_columns = [
            column for column in featured.select_dtypes(include="number").columns if column not in excluded
        ]
        fit_frame = featured.dropna(subset=self._feature_columns)
        if fit_frame.empty:
            raise ValueError("Training data is shorter than the configured lag/rolling windows")
        self.model.fit(fit_frame[self._feature_columns], fit_frame[self.target_column])
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        """Recursively forecast future rows without accessing their observed target values."""
        if self._history is None:
            raise RuntimeError("Call fit before predict")
        self._validate_frame(future, require_target=False)
        history = self._history.copy()
        predictions: list[float] = []
        indices: list[object] = []
        for index, row in future.sort_values("date").iterrows():
            next_row = row.copy()
            next_row[self.target_column] = float("nan")
            candidate = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
            featured = build_features(candidate, self.lags, self.windows, self.target_column)
            values = featured.loc[[len(featured) - 1], self._feature_columns]
            if values.isna().any(axis=None):
                raise ValueError("Insufficient history to compute XGBoost prediction features")
            prediction = max(0.0, float(self.model.predict(values)[0]))
            next_row[self.target_column] = prediction
            history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
            predictions.append(prediction)
            indices.append(index)
        return pd.Series(predictions, index=indices, name="prediction").reindex(future.index)

    def _validate_frame(self, frame: pd.DataFrame, require_target: bool = True) -> None:
        """Validate a single store/family time-series frame."""
        required = {"date", "store_nbr", "family"}
        if require_target:
            required.add(self.target_column)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"XGBoost frame missing columns: {sorted(missing)}")
        if len(frame[["store_nbr", "family"]].drop_duplicates()) != 1:
            raise ValueError("XGBoostForecaster accepts one store/family series per fit")
