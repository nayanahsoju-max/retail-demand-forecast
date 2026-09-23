"""Run configured rolling backtests for all forecasting model families."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from retail_demand_forecast.backtest.rolling import (
    BacktestRunner,
    ForecastModel,
    RollingWindowSplitter,
)
from retail_demand_forecast.data.loading import load_sales
from retail_demand_forecast.db.repository import ForecastRepository, create_database
from retail_demand_forecast.models import LSTMForecaster, SARIMAXForecaster, XGBoostForecaster

LOGGER = logging.getLogger(__name__)


def run_comparison(
    frame: pd.DataFrame,
    output_dir: str | Path,
    backtest_config: Mapping[str, Any],
    model_factories: Mapping[str, Callable[[], ForecastModel]] | None = None,
    database_url: str | None = None,
    store_nbr: int | None = None,
    family: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest each model, save CSV artifacts, and optionally persist database results."""
    splitter = RollingWindowSplitter(
        initial_train_days=int(backtest_config["initial_train_days"]),
        horizon_days=int(backtest_config["horizon_days"]),
        step_days=int(backtest_config["step_days"]),
        max_windows=backtest_config.get("max_windows"),
    )
    factories = model_factories or _default_factories(backtest_config.get("models", {}))
    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    for name, factory in factories.items():
        LOGGER.info("Running backtest for %s", name)
        prediction_frame, metric_frame = BacktestRunner(splitter).run(factory(), frame)
        predictions.append(prediction_frame)
        metrics.append(metric_frame)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_metrics = pd.concat(metrics, ignore_index=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    all_predictions.to_csv(destination / "backtest_predictions.csv", index=False)
    all_metrics.to_csv(destination / "backtest_metrics.csv", index=False)
    if database_url is not None:
        if store_nbr is None or family is None:
            raise ValueError("Database persistence requires both store_nbr and family")
        repository = ForecastRepository(create_database(database_url))
        repository.save_predictions(all_predictions, store_nbr, family)
        repository.save_backtest_metrics(all_metrics, store_nbr, family)
    LOGGER.info("Saved %d predictions and %d metric rows to %s", len(all_predictions), len(all_metrics), destination)
    return all_predictions, all_metrics


def _default_factories(config: Mapping[str, Any]) -> dict[str, Callable[[], ForecastModel]]:
    """Build delayed constructors from the YAML model configuration."""
    sarimax = config.get("sarimax", {})
    xgboost = config.get("xgboost", {})
    lstm = config.get("lstm", {})
    return {
        "sarimax": lambda: SARIMAXForecaster(**sarimax),
        "xgboost": lambda: XGBoostForecaster(**xgboost),
        "lstm": lambda: LSTMForecaster(**lstm),
    }


def main() -> None:
    """Parse CLI arguments, load a requested series, and persist its model comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=int, help="Favorita store number")
    parser.add_argument("--family", required=True, help="Favorita product family")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    parser.add_argument("--data-dir", default=None, help="Override configured data directory")
    parser.add_argument("--output-dir", default=None, help="Override configured artifact directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_dir = args.data_dir or config["paths"]["data_dir"]
    output_dir = args.output_dir or config["paths"]["artifact_dir"]
    sales = load_sales(data_dir, store_nbr=args.store, family=args.family)
    if sales.empty:
        raise ValueError("No sales rows match the requested store and family")
    backtest_config = {**config["backtest"], "models": config["models"]}
    run_comparison(
        sales, output_dir, backtest_config, database_url=config["database"]["url"],
        store_nbr=args.store, family=args.family,
    )


if __name__ == "__main__":
    main()
