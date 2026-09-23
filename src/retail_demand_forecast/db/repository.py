"""Database creation and repository operations for forecasting outputs."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from .models import BacktestMetricRecord, Base, PredictionRecord

LOGGER = logging.getLogger(__name__)


def create_database(url: str) -> Engine:
    """Create all forecast tables and return an engine for SQLite or PostgreSQL."""
    options = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=options)
    Base.metadata.create_all(engine)
    return engine


class ForecastRepository:
    """Transactional persistence and retrieval of predictions and backtest results."""

    def __init__(self, engine: Engine) -> None:
        self._sessions = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Provide a committing session and roll it back if an operation fails."""
        session = self._sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_predictions(self, frame: pd.DataFrame, store_nbr: int, family: str) -> int:
        """Insert prediction rows, replacing an existing model/window series if repeated."""
        required = {"date", "model", "prediction", "window_id"}
        self._require_columns(frame, required)
        records = [
            PredictionRecord(
                model=str(row.model), store_nbr=store_nbr, family=family,
                date=pd.Timestamp(row.date).to_pydatetime(),
                actual=self._optional_float(getattr(row, "actual", None)),
                prediction=float(row.prediction), window_id=int(row.window_id),
            )
            for row in frame.itertuples(index=False)
        ]
        with self.session_scope() as session:
            for record in records:
                existing = session.scalar(select(PredictionRecord).where(
                    PredictionRecord.model == record.model, PredictionRecord.store_nbr == store_nbr,
                    PredictionRecord.family == family, PredictionRecord.date == record.date,
                    PredictionRecord.window_id == record.window_id,
                ))
                if existing is None:
                    session.add(record)
                else:
                    existing.actual, existing.prediction = record.actual, record.prediction
        LOGGER.info("Persisted %d prediction rows for store=%s family=%s", len(records), store_nbr, family)
        return len(records)

    def save_backtest_metrics(self, frame: pd.DataFrame, store_nbr: int, family: str) -> int:
        """Insert or update rolling-window metric rows from a backtest result frame."""
        required = {"model", "window_id", "rmse", "mae", "wape", "train_end", "test_end"}
        self._require_columns(frame, required)
        records = [
            BacktestMetricRecord(
                model=str(row.model), store_nbr=store_nbr, family=family, window_id=int(row.window_id),
                rmse=float(row.rmse), mae=float(row.mae), wape=float(row.wape),
                train_end=pd.Timestamp(row.train_end).to_pydatetime(), test_end=pd.Timestamp(row.test_end).to_pydatetime(),
            )
            for row in frame.itertuples(index=False)
        ]
        with self.session_scope() as session:
            for record in records:
                existing = session.scalar(select(BacktestMetricRecord).where(
                    BacktestMetricRecord.model == record.model, BacktestMetricRecord.store_nbr == store_nbr,
                    BacktestMetricRecord.family == family, BacktestMetricRecord.window_id == record.window_id,
                ))
                if existing is None:
                    session.add(record)
                else:
                    existing.rmse, existing.mae, existing.wape = record.rmse, record.mae, record.wape
                    existing.train_end, existing.test_end = record.train_end, record.test_end
        LOGGER.info("Persisted %d metric rows for store=%s family=%s", len(records), store_nbr, family)
        return len(records)

    def get_predictions(self, store_nbr: int, family: str, model: str | None = None) -> list[PredictionRecord]:
        """Fetch forecast records ordered by date, optionally restricted to a model."""
        statement = select(PredictionRecord).where(
            PredictionRecord.store_nbr == store_nbr, PredictionRecord.family == family
        )
        if model:
            statement = statement.where(PredictionRecord.model == model)
        with self.session_scope() as session:
            return list(session.scalars(statement.order_by(PredictionRecord.date)).all())

    def get_backtest_metrics(self, store_nbr: int, family: str) -> list[BacktestMetricRecord]:
        """Fetch model/window metric history for one retail series."""
        statement = select(BacktestMetricRecord).where(
            BacktestMetricRecord.store_nbr == store_nbr, BacktestMetricRecord.family == family
        ).order_by(BacktestMetricRecord.model, BacktestMetricRecord.window_id)
        with self.session_scope() as session:
            return list(session.scalars(statement).all())

    @staticmethod
    def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Result frame missing columns: {sorted(missing)}")

    @staticmethod
    def _optional_float(value: object) -> float | None:
        return None if value is None or pd.isna(value) else float(value)
