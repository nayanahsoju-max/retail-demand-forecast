"""SQLAlchemy ORM entities for forecast outputs."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class shared by every persistence entity."""


class PredictionRecord(Base):
    """A single observed/forecasted daily sales point from a backtest."""

    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("model", "store_nbr", "family", "date", "window_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    store_nbr: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    family: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction: Mapped[float] = mapped_column(Float, nullable=False)
    window_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class BacktestMetricRecord(Base):
    """Accuracy metrics for one model and rolling-origin validation window."""

    __tablename__ = "backtest_metrics"
    __table_args__ = (UniqueConstraint("model", "store_nbr", "family", "window_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    store_nbr: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    family: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    window_id: Mapped[int] = mapped_column(Integer, nullable=False)
    rmse: Mapped[float] = mapped_column(Float, nullable=False)
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    wape: Mapped[float] = mapped_column(Float, nullable=False)
    train_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    test_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
