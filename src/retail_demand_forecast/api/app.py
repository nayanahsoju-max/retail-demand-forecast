"""FastAPI application exposing stored demand forecast results."""

from __future__ import annotations

import os

from fastapi import FastAPI, Query

from retail_demand_forecast.db.repository import ForecastRepository, create_database

from .schemas import BacktestMetricResponse, HealthResponse, PredictionResponse


def create_app(database_url: str | None = None) -> FastAPI:
    """Create an API application connected to the configured forecast-result database."""
    url = database_url or os.getenv("DATABASE_URL", "sqlite:///retail_forecast.db")
    repository = ForecastRepository(create_database(url))
    app = FastAPI(title="Retail Demand Forecast API", version="0.1.0")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        """Report API liveness."""
        return HealthResponse(status="ok")

    @app.get("/predictions", response_model=list[PredictionResponse], tags=["forecasts"])
    def get_predictions(
        store_nbr: int = Query(ge=1),
        family: str = Query(min_length=1),
        model: str | None = Query(default=None),
    ) -> list[PredictionResponse]:
        """Return persisted forecasts for a store/family, optionally for one model."""
        records = repository.get_predictions(store_nbr, family, model)
        return [
            PredictionResponse(
                id=record.id, model=record.model, store_nbr=record.store_nbr, family=record.family,
                date=record.date, actual=record.actual, prediction=record.prediction, window_id=record.window_id,
            )
            for record in records
        ]

    @app.get("/backtests", response_model=list[BacktestMetricResponse], tags=["backtests"])
    def get_backtests(
        store_nbr: int = Query(ge=1), family: str = Query(min_length=1)
    ) -> list[BacktestMetricResponse]:
        """Return rolling-window metric history for a store/family series."""
        records = repository.get_backtest_metrics(store_nbr, family)
        return [
            BacktestMetricResponse(
                id=record.id, model=record.model, store_nbr=record.store_nbr, family=record.family,
                window_id=record.window_id, rmse=record.rmse, mae=record.mae, wape=record.wape,
                train_end=record.train_end, test_end=record.test_end,
            )
            for record in records
        ]

    return app


app = create_app()
