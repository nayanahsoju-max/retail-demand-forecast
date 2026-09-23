"""Tests for the all-model comparison workflow."""

from pathlib import Path

import pandas as pd

from retail_demand_forecast.db.repository import ForecastRepository, create_database
from scripts.compare_models import run_comparison


class LastValueModel:
    """Tiny deterministic model used to test workflow plumbing."""

    name = "last_value"

    def fit(self, train: pd.DataFrame) -> "LastValueModel":
        """Store the last observed target."""
        self.value = float(train["sales"].iloc[-1])
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        """Repeat the stored value over the requested rows."""
        return pd.Series(self.value, index=future.index)


def test_comparison_saves_prediction_and_metric_artifacts(tmp_path: Path) -> None:
    """The workflow emits appendable CSV artifacts for downstream persistence."""
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=8), "sales": range(8)})
    predictions, metrics = run_comparison(
        frame, tmp_path,
        {"initial_train_days": 4, "horizon_days": 2, "step_days": 2},
        {"last_value": LastValueModel},
        database_url=f"sqlite:///{tmp_path / 'comparison.db'}", store_nbr=1, family="GROCERY",
    )
    assert len(predictions) == 4
    assert len(metrics) == 2
    assert (tmp_path / "backtest_predictions.csv").exists()
    assert (tmp_path / "backtest_metrics.csv").exists()
    repository = ForecastRepository(create_database(f"sqlite:///{tmp_path / 'comparison.db'}"))
    assert len(repository.get_predictions(1, "GROCERY")) == 4
    assert len(repository.get_backtest_metrics(1, "GROCERY")) == 2
