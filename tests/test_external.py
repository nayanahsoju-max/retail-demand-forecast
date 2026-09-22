"""Tests for cached external enrichments."""

from pathlib import Path

from retail_demand_forecast.data.external import ExternalDataFetcher


def test_holidays_are_cached_without_a_second_request(monkeypatch, tmp_path: Path) -> None:
    """A previously fetched response should avoid network access."""
    calls = 0

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):  # type: ignore[no-untyped-def]
            return [{"date": "2020-01-01", "name": "New Year", "global": True}]

    def fake_get(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr("requests.get", fake_get)
    fetcher = ExternalDataFetcher(tmp_path)
    first = fetcher.fetch_holidays(2020)
    second = fetcher.fetch_holidays(2020)
    assert calls == 1
    assert first.equals(second)
    assert list(tmp_path.glob("holidays_EC_2020.json"))


def test_weather_payload_is_normalized(monkeypatch, tmp_path: Path) -> None:
    """Open-Meteo daily fields should map to forecast feature names."""
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):  # type: ignore[no-untyped-def]
            return {"daily": {"time": ["2020-01-01"], "temperature_2m_mean": [23.0],
                              "precipitation_sum": [1.2], "rain_sum": [1.0]}}

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    weather = ExternalDataFetcher(tmp_path).fetch_weather(-0.18, -78.47, "2020-01-01", "2020-01-01")
    assert weather.loc[0, "temperature_mean"] == 23.0
    assert weather.loc[0, "date"].strftime("%Y-%m-%d") == "2020-01-01"
