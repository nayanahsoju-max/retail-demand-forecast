"""Tests for recursive XGBoost forecasting."""

import pandas as pd

from retail_demand_forecast.backtest.rolling import BacktestRunner, RollingWindowSplitter
from retail_demand_forecast.models.xgboost_model import XGBoostForecaster


def test_xgboost_runs_recursively_through_backtest() -> None:
    """Predictions should align to test dates without using their target values."""
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=40), "store_nbr": 1,
        "family": "GROCERY", "sales": [20 + (index % 7) for index in range(40)],
        "onpromotion": [index % 2 for index in range(40)],
    })
    model = XGBoostForecaster(lags=(1, 7), windows=(7,), n_estimators=10, max_depth=2)
    runner = BacktestRunner(RollingWindowSplitter(initial_train_days=25, horizon_days=5, step_days=5))
    predictions, metrics = runner.run(model, frame)
    assert len(predictions) == 15
    assert predictions["prediction"].notna().all()
    assert (metrics["model"] == "xgboost").all()


def test_xgboost_prediction_ignores_future_sales() -> None:
    """Mutating held-out actuals must not change recursive predictions."""
    train = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=20), "store_nbr": 1, "family": "GROCERY",
        "sales": range(20),
    })
    future = pd.DataFrame({
        "date": pd.date_range("2020-01-21", periods=2), "store_nbr": 1, "family": "GROCERY",
        "sales": [99.0, 999.0],
    })
    model = XGBoostForecaster(lags=(1, 3), windows=(3,), n_estimators=5, max_depth=2).fit(train)
    expected = model.predict(future)
    future["sales"] = -1.0
    pd.testing.assert_series_equal(expected, model.predict(future))
