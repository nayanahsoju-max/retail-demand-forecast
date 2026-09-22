"""Integration tests for the LSTM forecasting wrapper."""

import pandas as pd

from retail_demand_forecast.backtest.rolling import BacktestRunner, RollingWindowSplitter
from retail_demand_forecast.models.lstm import LSTMForecaster


def test_lstm_runs_through_backtest() -> None:
    """LSTM forecasts should be aligned, finite, and non-negative."""
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=30), "store_nbr": 1,
        "family": "GROCERY", "sales": [10 + (index % 5) for index in range(30)],
    })
    model = LSTMForecaster(lookback=5, hidden_size=4, epochs=1, learning_rate=0.01)
    runner = BacktestRunner(RollingWindowSplitter(initial_train_days=20, horizon_days=5, step_days=5))
    predictions, metrics = runner.run(model, frame)
    assert len(predictions) == 10
    assert predictions["prediction"].notna().all()
    assert predictions["prediction"].ge(0).all()
    assert (metrics["model"] == "lstm").all()
