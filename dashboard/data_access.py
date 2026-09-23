"""Data retrieval and chart preparation used by the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from retail_demand_forecast.db.repository import ForecastRepository


def read_database(
    repository: ForecastRepository, store_nbr: int, family: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one series' prediction and backtest records from the configured database."""
    predictions = [
        {"date": row.date, "actual": row.actual, "prediction": row.prediction, "model": row.model,
         "window_id": row.window_id}
        for row in repository.get_predictions(store_nbr, family)
    ]
    metrics = [
        {"model": row.model, "window_id": row.window_id, "rmse": row.rmse, "mae": row.mae,
         "wape": row.wape, "train_end": row.train_end, "test_end": row.test_end}
        for row in repository.get_backtest_metrics(store_nbr, family)
    ]
    return pd.DataFrame(predictions), pd.DataFrame(metrics)


def read_api(base_url: str, store_nbr: int, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prediction and metric JSON from a running Retail Demand Forecast API."""
    params = {"store_nbr": store_nbr, "family": family}
    predictions = _get_json(f"{base_url.rstrip('/')}/predictions", params)
    metrics = _get_json(f"{base_url.rstrip('/')}/backtests", params)
    return pd.DataFrame(predictions), pd.DataFrame(metrics)


def prepare_chart_data(predictions: pd.DataFrame) -> pd.DataFrame:
    """Convert model-level prediction rows into Plotly-ready actual/predicted series."""
    if predictions.empty:
        return pd.DataFrame(columns=["date", "series", "sales"])
    source = predictions.copy()
    source["date"] = pd.to_datetime(source["date"])
    actual = source.groupby("date", as_index=False)["actual"].mean().rename(columns={"actual": "sales"})
    actual["series"] = "Actual"
    forecast = source.pivot_table(index="date", columns="model", values="prediction", aggfunc="mean").reset_index()
    forecast = forecast.melt(id_vars="date", var_name="series", value_name="sales")
    return pd.concat([actual[["date", "series", "sales"]], forecast], ignore_index=True).sort_values("date")


def _get_json(url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Issue a bounded API request and return its decoded JSON list."""
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload: list[dict[str, Any]] = response.json()
    return payload
