"""
config.py

Centralized configuration loader for the SmartStock Intelligence Platform.
"""

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables before Settings is initialized.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)


@dataclass(frozen=True)
class Settings:
    """
    Immutable configuration loaded from environment variables.

    API keys, database settings, paths, RAG embedding settings, and
    scheduler configuration are defined here.
    """

    finnhub_api_key: str
    newsapi_api_key: str
    twelvedata_api_key: str
    gemini_api_key: str
    data_raw_dir: Path
    gemini_model: str = "gemini-2.5-flash"
    request_timeout_seconds: int = 30
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "smartstock"
    postgres_user: str = "postgres"
    postgres_password: str = ""
    tracked_symbols_path: Path = PROJECT_ROOT / "config" / "tracked_symbols.txt"
    rag_embed_batch_size: int = 10
    rag_embed_inter_batch_delay_seconds: float = 1.0
    rag_embed_max_retries: int = 5
    scheduler_daily_time: str = "18:00"
    scheduler_weekly_day: str = "sun"
    scheduler_weekly_time: str = "19:00"
    scheduler_timezone: str = "UTC"


def _get_required_env(key: str) -> str:
    """Read an environment variable or return an empty string."""
    return os.getenv(key, "").strip()


def load_settings() -> Settings:
    """Build a Settings object from the current environment."""
    return Settings(
        finnhub_api_key=_get_required_env("FINNHUB_API_KEY"),
        newsapi_api_key=_get_required_env("NEWSAPI_API_KEY"),
        twelvedata_api_key=_get_required_env("TWELVEDATA_API_KEY"),
        gemini_api_key=_get_required_env("GEMINI_API_KEY"),
        data_raw_dir=PROJECT_ROOT / "data" / "raw",
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        postgres_db=os.getenv("POSTGRES_DB", "smartstock"),
        postgres_user=os.getenv("POSTGRES_USER", "postgres"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
        tracked_symbols_path=PROJECT_ROOT / "config" / "tracked_symbols.txt",
        rag_embed_batch_size=int(os.getenv("RAG_EMBED_BATCH_SIZE", "10")),
        rag_embed_inter_batch_delay_seconds=float(os.getenv("RAG_EMBED_INTER_BATCH_DELAY_SECONDS", "1.0")),
        rag_embed_max_retries=int(os.getenv("RAG_EMBED_MAX_RETRIES", "5")),
        scheduler_daily_time=os.getenv("SCHEDULER_DAILY_TIME", "18:00").strip(),
        scheduler_weekly_day=os.getenv("SCHEDULER_WEEKLY_DAY", "sun").strip().lower(),
        scheduler_weekly_time=os.getenv("SCHEDULER_WEEKLY_TIME", "19:00").strip(),
        scheduler_timezone=os.getenv("SCHEDULER_TIMEZONE", "UTC").strip(),
    )


# Shared settings instance.
settings = load_settings()


class TrackedSymbolsError(Exception):
    """Raised when the tracked symbols file is missing, unreadable, or empty."""


def load_tracked_symbols(path: Path | None = None) -> list[str]:
    """
    Load unique, uppercase stock symbols from the tracked symbols file.

    Blank lines and comments are ignored, and duplicate symbols are removed
    while preserving their first occurrence.
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
