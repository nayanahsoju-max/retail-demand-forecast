"""Tests for dashboard database reads and chart data transformations."""

import pandas as pd

from dashboard.data_access import prepare_chart_data, read_database
from retail_demand_forecast.db.repository import ForecastRepository, create_database


def test_dashboard_reads_database_and_prepares_model_lines(tmp_path) -> None:
    """Persisted values should become actual plus one line per forecast model."""
    repository = ForecastRepository(create_database(f"sqlite:///{tmp_path / 'dashboard.db'}"))
    predictions = pd.DataFrame({"date": ["2020-01-01", "2020-01-01"], "model": ["sarimax", "xgboost"],
                                "actual": [10.0, 10.0], "prediction": [9.0, 11.0], "window_id": [0, 0]})
    metrics = pd.DataFrame({"model": ["sarimax"], "window_id": [0], "rmse": [1.0], "mae": [1.0],
                            "wape": [10.0], "train_end": ["2019-12-31"], "test_end": ["2020-01-01"]})
    repository.save_predictions(predictions, 1, "GROCERY")
    repository.save_backtest_metrics(metrics, 1, "GROCERY")
    stored_predictions, stored_metrics = read_database(repository, 1, "GROCERY")
    chart = prepare_chart_data(stored_predictions)
    assert len(stored_metrics) == 1
    assert set(chart["series"]) == {"Actual", "sarimax", "xgboost"}
    assert chart.loc[chart["series"] == "Actual", "sales"].iloc[0] == 10.0
