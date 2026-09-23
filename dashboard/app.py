"""Interactive Streamlit dashboard for comparing persisted retail forecasts."""

from __future__ import annotations

import os

import plotly.express as px
import streamlit as st

from retail_demand_forecast.db.repository import ForecastRepository, create_database

from dashboard.data_access import prepare_chart_data, read_api, read_database


def main() -> None:
    """Render source controls, forecast chart, and model accuracy comparison."""
    st.set_page_config(page_title="Retail Demand Forecast", layout="wide")
    st.title("Retail Demand Forecast Comparison")
    source = st.sidebar.radio("Data source", ["Database", "API"])
    database_url = os.getenv("DATABASE_URL", "sqlite:///retail_forecast.db")
    api_url = st.sidebar.text_input("API URL", os.getenv("FORECAST_API_URL", "http://127.0.0.1:8000"))
    store_nbr = st.sidebar.number_input("Store number", min_value=1, value=1, step=1)
    family = st.sidebar.text_input("Product family", value="GROCERY I")
    if not family.strip():
        st.info("Enter a product family to load results.")
        return
    try:
        if source == "Database":
            repository = ForecastRepository(create_database(database_url))
            predictions, metrics = read_database(repository, int(store_nbr), family.strip())
        else:
            predictions, metrics = read_api(api_url, int(store_nbr), family.strip())
    except Exception as error:  # noqa: BLE001 - dashboard should render data errors instead of crashing
        st.error(f"Unable to load forecast results: {error}")
        return
    if predictions.empty:
        st.info("No persisted forecasts found. Run scripts/compare_models.py for this series first.")
        return
    chart_data = prepare_chart_data(predictions)
    st.subheader(f"Store {int(store_nbr)} — {family.strip()}")
    st.plotly_chart(
        px.line(chart_data, x="date", y="sales", color="series", title="Forecast versus actual sales"),
        use_container_width=True,
    )
    if metrics.empty:
        st.info("No persisted backtest metrics found for this series.")
        return
    summary = metrics.groupby("model", as_index=False)[["rmse", "mae", "wape"]].mean().sort_values("wape")
    left, right = st.columns(2)
    with left:
        st.subheader("Mean rolling-window metrics")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    with right:
        st.subheader("WAPE by model")
        st.plotly_chart(px.bar(summary, x="model", y="wape", color="model"), use_container_width=True)


if __name__ == "__main__":
    main()
