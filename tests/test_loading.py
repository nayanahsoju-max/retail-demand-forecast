"""Tests for Favorita data ingestion."""

import pandas as pd
import pytest

from retail_demand_forecast.data.loading import clean_sales


def test_clean_sales_normalizes_invalid_values_and_duplicates() -> None:
    """Cleaning should clip sales and aggregate same-series duplicate days."""
    raw = pd.DataFrame({
        "date": ["2017-01-01", "2017-01-01", "bad-date"],
        "store_nbr": [1, 1, 1], "family": ["FOODS", "FOODS", "FOODS"],
        "sales": [-2, 3, 4], "onpromotion": [None, 2, 0],
    })
    cleaned = clean_sales(raw)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "sales"] == 3
    assert cleaned.loc[0, "onpromotion"] == 2


def test_clean_sales_requires_source_columns() -> None:
    """Missing target columns are rejected early."""
    with pytest.raises(ValueError, match="sales"):
        clean_sales(pd.DataFrame({"date": []}))
