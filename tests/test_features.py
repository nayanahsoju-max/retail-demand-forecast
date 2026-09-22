"""Tests that engineered time-series features remain causal."""

import pandas as pd

from retail_demand_forecast.features.engineering import build_features


def _sales_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=6, freq="D"),
        "store_nbr": [1] * 6,
        "family": ["GROCERY"] * 6,
        "sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })


def test_lags_do_not_leak_future_values() -> None:
    """Changing a future target cannot alter earlier lag or rolling features."""
    original = build_features(_sales_frame(), lags=(1, 2), windows=(2,))
    changed = _sales_frame()
    changed.loc[5, "sales"] = 9_999.0
    mutated = build_features(changed, lags=(1, 2), windows=(2,))
    feature_columns = [column for column in original if column.startswith("sales_")]
    pd.testing.assert_frame_equal(original.loc[:4, feature_columns], mutated.loc[:4, feature_columns])
    assert original.loc[3, "sales_lag_1"] == 30.0
    assert original.loc[3, "sales_rolling_mean_2"] == 25.0


def test_calendar_features_are_available_for_all_dates() -> None:
    """Calendar transforms add forecast-known fields without dropping rows."""
    features = build_features(_sales_frame(), lags=(), windows=())
    assert {"day_of_week", "month", "is_weekend"}.issubset(features.columns)
    assert len(features) == 6
