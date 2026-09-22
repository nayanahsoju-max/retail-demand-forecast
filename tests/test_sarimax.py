"""Integration tests for SARIMAX and rolling-origin evaluation."""

import pandas as pd

from retail_demand_forecast.backtest.rolling import BacktestRunner, RollingWindowSplitter
from retail_demand_forecast.models.sarimax import SARIMAXForecaster


def test_sarimax_runs_through_backtest() -> None:
    """The SARIMAX wrapper produces one non-negative forecast per validation row."""
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=30),
        "store_nbr": 1,
        "family": "GROCERY",
        "sales": [20 + (index % 7) for index in range(30)],
    })
    model = SARIMAXForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    runner = BacktestRunner(RollingWindowSplitter(initial_train_days=20, horizon_days=5, step_days=5))
    predictions, metrics = runner.run(model, frame)
    assert len(predictions) == 10
    assert predictions["prediction"].ge(0).all()
    assert set(metrics) >= {"model", "rmse", "mae", "wape"}
