"""Cached external holiday and weather enrichment clients."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger(__name__)
RETRY = retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)


class ExternalDataFetcher:
    """Retrieve external time series and persist raw responses in a local cache."""

    def __init__(self, cache_dir: str | Path = "data/cache", timeout_seconds: int = 20) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds

    def fetch_holidays(self, year: int, country_code: str = "EC") -> pd.DataFrame:
        """Return national public holidays for a year, cached by country and year."""
        cache_key = f"holidays_{country_code}_{year}.json"
        payload = self._get_json(
            cache_key, f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
        )
        rows = [
            {"date": item["date"], "holiday_name": item["name"], "is_holiday": 1}
            for item in payload
            if item.get("global", True)
        ]
        return pd.DataFrame(rows, columns=["date", "holiday_name", "is_holiday"]).assign(
            date=lambda frame: pd.to_datetime(frame["date"])
        )

    def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str | date,
        end_date: str | date,
    ) -> pd.DataFrame:
        """Return daily weather from Open-Meteo's historical archive for a location."""
        start, end = str(start_date), str(end_date)
        cache_key = f"weather_{latitude:.3f}_{longitude:.3f}_{start}_{end}.json"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_mean,precipitation_sum,rain_sum",
            "timezone": "America/Guayaquil",
        }
        payload = self._get_json(cache_key, "https://archive-api.open-meteo.com/v1/archive", params)
        daily = payload.get("daily", {})
        return pd.DataFrame(
            {
                "date": pd.to_datetime(daily.get("time", [])),
                "temperature_mean": daily.get("temperature_2m_mean", []),
                "precipitation_sum": daily.get("precipitation_sum", []),
                "rain_sum": daily.get("rain_sum", []),
            }
        )

    def _get_json(self, cache_key: str, url: str, params: dict[str, Any] | None = None) -> Any:
        """Read a cached response, or request and atomically cache a JSON response."""
        path = self.cache_dir / cache_key
        if path.exists():
            LOGGER.info("Using cached external data: %s", path)
            return json.loads(path.read_text(encoding="utf-8"))
        payload = self._request_json(url, params)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        LOGGER.info("Cached external data: %s", path)
        return payload

    @RETRY
    def _request_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Perform a retried HTTP GET and validate the response."""
        response = requests.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()
