"""Tests for forecast result HTTP endpoints."""

import pandas as pd

from retail_demand_forecast.api.app import create_app
from retail_demand_forecast.db.repository import ForecastRepository, create_database


def test_api_returns_persisted_predictions_and_metrics(tmp_path) -> None:
    """HTTP responses should expose the repository's persisted result history."""
    database_url = f"sqlite:///{tmp_path / 'api.db'}"
    repository = ForecastRepository(create_database(database_url))
    repository.save_predictions(pd.DataFrame({"date": ["2020-01-01"], "model": ["sarimax"],
                                               "actual": [10.0], "prediction": [9.0], "window_id": [0]}), 1, "GROCERY")
    repository.save_backtest_metrics(pd.DataFrame({"model": ["sarimax"], "window_id": [0], "rmse": [1.0],
                                                    "mae": [1.0], "wape": [10.0], "train_end": ["2019-12-31"],
                                                    "test_end": ["2020-01-01"]}), 1, "GROCERY")
    app = create_app(database_url)
    handlers = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
    assert handlers["/health"]().status == "ok"
    predictions = handlers["/predictions"](store_nbr=1, family="GROCERY", model=None)
    assert predictions[0].prediction == 9.0
    backtests = handlers["/backtests"](store_nbr=1, family="GROCERY")
    assert backtests[0].wape == 10.0
