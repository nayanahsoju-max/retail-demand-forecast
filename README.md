# Retail Demand Forecast

Production-oriented forecasting for Kaggle's Corporación Favorita Store Sales dataset.

## Project status

This package compares SARIMAX, XGBoost, and LSTM models using rolling-origin
backtests, persists outputs, and exposes them through FastAPI and Streamlit.

## Layout

`src/retail_demand_forecast` contains data, feature, model, backtest, API, and
database modules. `dashboard` contains the Streamlit UI and `tests` contains
unit tests.
