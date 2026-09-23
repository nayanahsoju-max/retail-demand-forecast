# Retail Demand Forecast

An end-to-end retail demand forecasting system built around Kaggle's
Corporacion Favorita Store Sales dataset. It combines leakage-safe feature
engineering, statistical and machine-learning forecasts, rolling-origin
validation, persistence, and serving/dashboard layers.

## Problem and approach

The target is daily unit sales for a Favorita store and product family.
Historical sales are joined with calendar, promotion, holiday, oil, and
weather signals when available. Three model families share the same
`fit`/`predict` contract:

- **SARIMAX** — interpretable seasonal statistical baseline.
- **XGBoost** — calendar, lag, and trailing rolling-statistic features with
  recursive multi-step prediction.
- **LSTM** — scaled lookback sequences with recursive PyTorch inference.

Validation uses expanding rolling windows. Every lag and rolling statistic is
computed only from observations strictly before the prediction date; tests
explicitly check that changing a future target cannot change earlier features.
Metrics are RMSE, MAE, and WAPE.

## Architecture

```text
Favorita CSVs + external APIs
          |
     data loading/cleaning -----> feature engineering
          |                              |
          +-------- rolling backtest <---+---- SARIMAX / XGBoost / LSTM
                              |
                  CSV artifacts + SQLAlchemy
                              |
             FastAPI (/predictions, /backtests)
                              |
                    Streamlit + Plotly dashboard
```

The package lives under `src/retail_demand_forecast`: `data`, `features`,
`models`, `backtest`, `db`, and `api`. `scripts/compare_models.py` orchestrates
the full comparison. `dashboard/` contains the interactive UI, and
`docker-compose.yml` runs the API, dashboard, and PostgreSQL services.

## Setup

The repository assumes the supplied `retail-forecast` conda environment and
the Favorita CSVs in `data/` already exist.

```powershell
conda activate retail-forecast
$env:PYTHONPATH = "src"
```

For a fresh environment, install the project requirements with
`python -m pip install -r requirements.txt`; do not commit the dataset or
generated artifacts.

Run the test suite:

```powershell
pytest -q
```

## Run a model comparison

The command below filters one store/family, runs the configured four rolling
windows, writes `backtest_predictions.csv` and `backtest_metrics.csv` under
`artifacts/`, and persists the same results to the configured SQLite database.

```powershell
$env:PYTHONPATH = "src"
python scripts/compare_models.py --store 1 --family "GROCERY I"
```

Configuration is in [config.yaml](config.yaml), including date/window sizes,
feature periods, model hyperparameters, artifact location, and database URL.
External holiday/weather responses are cached under `data/cache/`.

## Real backtest results

### Store 1 / GROCERY I

Four 16-day rolling windows were run against the real training CSV.

| Model | Mean RMSE | Mean MAE | Mean WAPE |
|---|---:|---:|---:|
| SARIMAX | **213.305** | 176.638 | 9.859% |
| XGBoost | 214.101 | **166.436** | **9.342%** |
| LSTM | 493.094 | 363.629 | 20.305% |

SARIMAX had the lowest RMSE; XGBoost had the lowest MAE and WAPE.

### Five-series sample

Three high-volume Grocery I series (stores 44, 45, and 3) and two low-volume
series (store 36 Hardware and store 6 Home Appliances) were evaluated over the
same four windows. The table below pools all 320 forecast points per model.

| Model | RMSE | MAE | WAPE |
|---|---:|---:|---:|
| SARIMAX | **1129.045** | **571.760** | **13.123%** |
| XGBoost | 1589.523 | 606.953 | 13.931% |
| LSTM | 1471.273 | 848.636 | 19.478% |

WAPE is highly sensitive for the low-volume series because their actual-sales
denominators are very small. Raw per-window CSVs are generated under
`artifacts/` when comparisons are run.

## Serve the API

```powershell
$env:PYTHONPATH = "src"
uvicorn retail_demand_forecast.api.app:app --reload
```

Interactive docs: <http://127.0.0.1:8000/docs>

- `GET /health`
- `GET /predictions?store_nbr=1&family=GROCERY%20I&model=xgboost`
- `GET /backtests?store_nbr=1&family=GROCERY%20I`

## Run the dashboard

```powershell
$env:PYTHONPATH = "src"
streamlit run dashboard/app.py
```

Open <http://localhost:8501>. The sidebar can read from the local database or
a running API, then plots actuals against each model and summarizes metrics.

## Docker deployment

Docker Compose starts PostgreSQL, FastAPI, and Streamlit:

```powershell
docker compose up --build
```

The API is at <http://localhost:8000/docs> and the dashboard is at
<http://localhost:8501>. PostgreSQL data is stored in the `postgres_data`
volume. The image sets `PYTHONPATH=/app/src:/app` so both package and
top-level dashboard imports resolve.

## CI and development

GitHub Actions runs Ruff and pytest on every push and pull request. Public
functions use type hints and docstrings; runtime paths use logging rather than
print statements. Generated CSVs, caches, local SQLite files, and the Kaggle
dataset are excluded from version control.
