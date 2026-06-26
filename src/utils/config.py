"""
config.py

Centralized configuration loader for the SmartStock Intelligence Platform.

"""

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a .env file at the project root, if present.
# This must run before the Settings dataclass below reads any environment
# variables, so it is called at import time.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)


@dataclass(frozen=True)
class Settings:
    """
    Immutable settings object holding all environment-driven configuration.

    Attributes:
        finnhub_api_key: API key for Finnhub (company profile/fundamentals data).
        newsapi_api_key: API key for NewsAPI.org (news articles).
        twelvedata_api_key: API key for Twelve Data (historical stock price data).
        data_raw_dir: Absolute path to the data/raw/ directory.
        request_timeout_seconds: Default timeout for outbound HTTP requests.
        postgres_host: PostgreSQL server host.
        postgres_port: PostgreSQL server port.
        postgres_db: PostgreSQL database name.
        postgres_user: PostgreSQL username.
        postgres_password: PostgreSQL password.
    """

    finnhub_api_key: str
    newsapi_api_key: str
    twelvedata_api_key: str
    data_raw_dir: Path
    request_timeout_seconds: int = 30
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "smartstock"
    postgres_user: str = "postgres"
    postgres_password: str = ""


def _get_required_env(key: str) -> str:
    """
    Read an environment variable, returning an empty string if missing
    rather than raising immediately.

    Ingestion scripts are responsible for checking that required keys are
    non-empty before making API calls, so they can produce a clear,
    actionable error message (see fetch_stock_data.py for the pattern).

    Args:
        key: The environment variable name to read.

    Returns:
        The variable's value, or an empty string if not set.
    """
    return os.getenv(key, "").strip()


def load_settings() -> Settings:
    """
    Build and return a Settings object from the current environment.

    Returns:
        A populated, immutable Settings instance.
    """
    return Settings(
        finnhub_api_key=_get_required_env("FINNHUB_API_KEY"),
        newsapi_api_key=_get_required_env("NEWSAPI_API_KEY"),
        twelvedata_api_key=_get_required_env("TWELVEDATA_API_KEY"),
        data_raw_dir=PROJECT_ROOT / "data" / "raw",
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        postgres_db=os.getenv("POSTGRES_DB", "smartstock"),
        postgres_user=os.getenv("POSTGRES_USER", "postgres"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
    )


# Singleton settings instance, imported by other modules as:
#   from src.utils.config import settings
settings = load_settings()
