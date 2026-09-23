"""Static checks for container deployment configuration."""

from pathlib import Path

import yaml


def test_compose_defines_api_and_postgres_services() -> None:
    """Compose should connect the API to a healthy PostgreSQL service."""
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    assert {"api", "db"}.issubset(compose["services"])
    assert "DATABASE_URL" in compose["services"]["api"]["environment"]
    assert compose["services"]["api"]["depends_on"]["db"]["condition"] == "service_healthy"


def test_dockerfile_runs_the_fastapi_application() -> None:
    """The API image should expose the service port and launch Uvicorn."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "EXPOSE 8000" in dockerfile
    assert "retail_demand_forecast.api.app:app" in dockerfile
