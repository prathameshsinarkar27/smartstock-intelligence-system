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
        tracked_symbols_path: Absolute path to config/tracked_symbols.txt,
            the pipeline's default symbol list.
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
    tracked_symbols_path: Path = PROJECT_ROOT / "config" / "tracked_symbols.txt"


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
        tracked_symbols_path=PROJECT_ROOT / "config" / "tracked_symbols.txt",
    )


# Singleton settings instance, imported by other modules as:
#   from src.utils.config import settings
settings = load_settings()


class TrackedSymbolsError(Exception):
    """Raised when config/tracked_symbols.txt is missing, unreadable, or empty."""


def load_tracked_symbols(path: Path | None = None) -> list[str]:
    """
    Read the default list of tracked stock symbols from a config file
    (config/tracked_symbols.txt by default).

    Used by src/pipeline/run_pipeline.py as the symbol list, so the
    most common case (`python -m src.pipeline.run_pipeline`) doesn't 
    require typing out every symbol on the command line.

    Args:
        path: Path to the symbols file. Defaults to
            settings.tracked_symbols_path (config/tracked_symbols.txt at
            the project root).

    Returns:
        A list of unique, uppercase, trimmed stock ticker symbols, in the
        order they first appear in the file.

    Raises:
        TrackedSymbolsError: If the file does not exist, cannot be read,
            or contains no symbols after filtering out comments/blank
            lines (an empty or comment-only file is treated the same as a
            missing one, since either case leaves the pipeline with
            nothing to process).
    """
    symbols_path = path or settings.tracked_symbols_path

    if not symbols_path.exists():
        raise TrackedSymbolsError(
            f"Tracked symbols file not found: {symbols_path}. "
            f"Create it (one symbol per line) or pass --symbols explicitly."
        )

    try:
        raw_lines = symbols_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TrackedSymbolsError(f"Could not read tracked symbols file {symbols_path}: {exc}") from exc

    seen: set[str] = set()
    symbols: list[str] = []

    for line in raw_lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        symbol = stripped.upper()

        if symbol in seen:
            continue

        seen.add(symbol)
        symbols.append(symbol)

    if not symbols:
        raise TrackedSymbolsError(
            f"Tracked symbols file {symbols_path} contains no symbols "
            f"(only blank lines and/or comments). Add at least one symbol, "
            f"one per line, or pass --symbols explicitly."
        )

    return symbols
