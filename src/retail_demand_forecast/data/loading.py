"""Loading and cleaning routines for Favorita source files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
REQUIRED_TRAIN_COLUMNS = {"date", "store_nbr", "family", "sales", "onpromotion"}


def load_sales(
    data_dir: str | Path,
    store_nbr: int | None = None,
    family: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Load, clean, and optionally filter Favorita's ``train.csv`` efficiently."""
    path = Path(data_dir) / "train.csv"
    filters: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, parse_dates=["date"]):
        if store_nbr is not None:
            chunk = chunk.loc[chunk["store_nbr"] == store_nbr]
        if family is not None:
            chunk = chunk.loc[chunk["family"] == family]
        if start_date is not None:
            chunk = chunk.loc[chunk["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            chunk = chunk.loc[chunk["date"] <= pd.Timestamp(end_date)]
        if not chunk.empty:
            filters.append(chunk)
    frame = pd.concat(filters, ignore_index=True) if filters else pd.DataFrame()
    return clean_sales(frame)


def clean_sales(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize raw daily sales rows without imputing target values."""
    missing = REQUIRED_TRAIN_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Sales data is missing required columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["sales"] = pd.to_numeric(result["sales"], errors="coerce").clip(lower=0)
    result["onpromotion"] = pd.to_numeric(result["onpromotion"], errors="coerce").fillna(0).astype(int)
    result = result.dropna(subset=["date", "store_nbr", "family", "sales"])
    keys = ["date", "store_nbr", "family"]
    if result.duplicated(keys).any():
        LOGGER.warning("Aggregating duplicate sales rows by date, store, and family")
        result = result.groupby(keys, as_index=False).agg(sales=("sales", "sum"), onpromotion=("onpromotion", "max"))
    return result.sort_values(keys).reset_index(drop=True)


def load_dimensions(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load Favorita dimension tables with parsed dates where applicable."""
    root = Path(data_dir)
    return {
        "stores": pd.read_csv(root / "stores.csv"),
        "holidays": pd.read_csv(root / "holidays_events.csv", parse_dates=["date"]),
        "oil": pd.read_csv(root / "oil.csv", parse_dates=["date"]),
        "transactions": pd.read_csv(root / "transactions.csv", parse_dates=["date"]),
    }
