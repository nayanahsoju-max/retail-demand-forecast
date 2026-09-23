"""Tests for SQLite prediction and metric persistence."""

import pandas as pd

from retail_demand_forecast.db.repository import ForecastRepository, create_database


def test_repository_persists_and_updates_backtest_outputs(tmp_path) -> None:
    """SQLite storage should be queryable and idempotent per model/window key."""
    repository = ForecastRepository(create_database(f"sqlite:///{tmp_path / 'forecast.db'}"))
    predictions = pd.DataFrame({"date": ["2020-01-01"], "model": ["xgboost"], "actual": [10.0],
                                "prediction": [9.0], "window_id": [0]})
    metrics = pd.DataFrame({"model": ["xgboost"], "window_id": [0], "rmse": [1.0], "mae": [1.0],
                            "wape": [10.0], "train_end": ["2019-12-31"], "test_end": ["2020-01-01"]})
    assert repository.save_predictions(predictions, 1, "GROCERY") == 1
    assert repository.save_backtest_metrics(metrics, 1, "GROCERY") == 1
    predictions.loc[0, "prediction"] = 8.0
    repository.save_predictions(predictions, 1, "GROCERY")
    stored = repository.get_predictions(1, "GROCERY", "xgboost")
    assert len(stored) == 1
    assert stored[0].prediction == 8.0
    assert repository.get_backtest_metrics(1, "GROCERY")[0].wape == 10.0
