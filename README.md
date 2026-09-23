# Retail Demand Forecast

An end-to-end retail demand forecasting system built on Kaggle's Corporacion
Favorita Store Sales dataset. It covers leakage-safe feature engineering,
statistical and machine learning models, rolling-origin backtesting,
persistence, and a serving/dashboard layer on top.

## Problem and approach

The goal is predicting daily unit sales for a given Favorita store and
product family. Historical sales data is joined with calendar, promotion,
holiday, oil price, and weather signals where available. Three models are
implemented, all sharing the same `fit`/`predict` interface so they can be
swapped in and out of the same backtest:

- **SARIMAX**: a seasonal statistical baseline, useful mainly for its
  interpretability.
- **XGBoost**: uses calendar features plus lag and rolling-statistic
  features, with recursive multi-step prediction.
- **LSTM**: a small PyTorch model trained on scaled lookback windows, also
  predicting recursively.

Validation is done with expanding rolling windows rather than a single
train/test split. Every lag and rolling feature only looks at data strictly
before the date being predicted; there are tests that specifically check
that changing a future value can't leak into an earlier prediction. Models
are compared using RMSE, MAE, and WAPE.

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

The main package is under `src/retail_demand_forecast`, split into `data`,
`features`, `models`, `backtest`, `db`, and `api` modules. `scripts/compare_models.py`
runs the full pipeline end to end. `dashboard/` has the Streamlit UI, and
`docker-compose.yml` spins up the API, dashboard, and PostgreSQL together.

## Setup

This assumes the `retail-forecast` conda environment is already set up and
the Favorita CSVs are sitting in `data/`.

```bash
conda activate retail-forecast
set PYTHONPATH=src
```

If you're setting this up fresh, install everything with
`python -m pip install -r requirements.txt`. Don't commit the dataset or
anything generated under `artifacts/`.

One thing to watch for: the Streamlit dashboard imports from a top-level
`dashboard` module, not just from `retail_demand_forecast`. So when running
the dashboard specifically, `PYTHONPATH` needs both the repo root and `src/`:

```bash
set PYTHONPATH=src;.
streamlit run dashboard/app.py
```

The API and the comparison script only need `src/`:

```bash
set PYTHONPATH=src
uvicorn retail_demand_forecast.api.app:app --reload
```

To run the tests:

```bash
pytest -q
```

## Running a comparison

This runs all three models for one store/family, across four 16-day rolling
windows, and saves the results both as CSVs under `artifacts/` and into the
configured SQLite database.

```bash
set PYTHONPATH=src
python scripts/compare_models.py --store 1 --family "GROCERY I"
```

Most of the settings (window sizes, feature lags, model hyperparameters,
artifact paths, database URL) live in `config.yaml`. Weather and holiday API
responses get cached under `data/cache/` so repeated runs don't hammer those
APIs.

## Results

### Store 1, GROCERY I

Four 16-day rolling windows, run against the real training data:

| Model | Mean RMSE | Mean MAE | Mean WAPE |
|---|---:|---:|---:|
| SARIMAX | **213.305** | 176.638 | 9.859% |
| XGBoost | 214.101 | **166.436** | **9.342%** |
| LSTM | 493.094 | 363.629 | 20.305% |

SARIMAX edges out XGBoost on RMSE, but XGBoost wins on MAE and WAPE. LSTM
lags behind both, likely because a single-series lookback model like this
needs more training data than one store/family combination provides.

### Broader sample, five series

Three high-volume Grocery I series (stores 44, 45, 3) and two low-volume
series (store 36 Hardware, store 6 Home Appliances), same four rolling
windows. Numbers below pool all 320 forecasts per model:

| Model | RMSE | MAE | WAPE |
|---|---:|---:|---:|
| SARIMAX | **1129.045** | **571.760** | **13.123%** |
| XGBoost | 1589.523 | 606.953 | 13.931% |
| LSTM | 1471.273 | 848.636 | 19.478% |

Worth noting: WAPE gets unstable on the low-volume series, since it's
sensitive to how close actual sales are to zero. A store selling one or two
hardware items a day can have a technically huge WAPE even when the raw
error is tiny. Per-window CSVs for all of this get generated under
`artifacts/` whenever you run a comparison yourself.

## Running the API

```bash
set PYTHONPATH=src
uvicorn retail_demand_forecast.api.app:app --reload
```

Docs are at <http://127.0.0.1:8000/docs>. Main endpoints:

- `GET /health`
- `GET /predictions?store_nbr=1&family=GROCERY%20I&model=xgboost`
- `GET /backtests?store_nbr=1&family=GROCERY%20I`

## Running the dashboard

```bash
set PYTHONPATH=src;.
streamlit run dashboard/app.py
```

Then open <http://localhost:8501>. You can point it at either the local
database or a running instance of the API, and it'll plot actuals against
each model's predictions along with the summary metrics.

## Docker

`docker compose up --build` starts PostgreSQL, the FastAPI service, and the
Streamlit dashboard together. The API ends up at
<http://localhost:8000/docs>, the dashboard at <http://localhost:8501>.
Postgres data persists in the `postgres_data` volume. The image sets
`PYTHONPATH=/app/src:/app` so both the package and the top-level dashboard
module resolve correctly inside the container.

## CI

GitHub Actions runs Ruff and pytest on every push and pull request. Public
functions have type hints and docstrings, and the code uses logging instead
of print statements for anything that runs at runtime. The Kaggle dataset,
generated CSVs, caches, and local SQLite files are all excluded from version
control.
