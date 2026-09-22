"""Tests for rolling-origin validation and standard metrics."""

import pandas as pd
import pytest

from retail_demand_forecast.backtest.metrics import calculate_metrics
from retail_demand_forecast.backtest.rolling import RollingWindowSplitter


def test_rolling_windows_are_ordered_and_expanding() -> None:
    """Every test period must occur after its full training history."""
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10), "sales": range(10)})
    windows = list(RollingWindowSplitter(4, 2, 2).split(frame))
    assert len(windows) == 3
    assert windows[0].train_end < windows[0].test_start
    assert len(windows[1].train_indices) == 6
    assert windows[-1].test_end == pd.Timestamp("2020-01-10")


def test_standard_metrics_have_expected_values() -> None:
    """Metrics should use familiar exact values for a small forecast."""
    metrics = calculate_metrics([10, 20], [8, 26])
    assert metrics["mae"] == 4.0
    assert metrics["rmse"] == pytest.approx(4.47213595)
    assert metrics["wape"] == pytest.approx(26.6666667)
