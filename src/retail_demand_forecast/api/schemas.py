"""Pydantic response models for the forecasting HTTP API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """API liveness response."""

    status: str


class PredictionResponse(BaseModel):
    """One persisted daily forecast and optional observed value."""

    id: int
    model: str
    store_nbr: int
    family: str
    date: datetime
    actual: float | None
    prediction: float
    window_id: int | None


class BacktestMetricResponse(BaseModel):
    """Metrics for a model's individual rolling backtest window."""

    id: int
    model: str
    store_nbr: int
    family: str
    window_id: int
    rmse: float
    mae: float
    wape: float
    train_end: datetime
    test_end: datetime
