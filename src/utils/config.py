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
        gemini_api_key: API key for the Gemini API (Google AI Studio),
            used by the AI Research Assistant (Phase 10).
        gemini_model: Which Gemini model to call. Configurable rather than
            hardcoded since available model names change over time and
            depend on the API key's plan/access — see .env.example for
            the current default and how to override it.
        data_raw_dir: Absolute path to the data/raw/ directory.
        request_timeout_seconds: Default timeout for outbound HTTP requests.
        postgres_host: PostgreSQL server host.
        postgres_port: PostgreSQL server port.
        postgres_db: PostgreSQL database name.
        postgres_user: PostgreSQL username.
        postgres_password: PostgreSQL password.
        tracked_symbols_path: Absolute path to config/tracked_symbols.txt,
            the pipeline's default symbol list.
        rag_embed_batch_size: Max texts per Gemini embedContent call in
            src/rag/embeddings.py. Kept small by default to stay under
            free-tier rate limits; raise it on a paid tier for fewer,
            larger calls.
        rag_embed_inter_batch_delay_seconds: Fixed pause between
            successful embedding batches, to stay under free-tier
            requests-per-minute quotas proactively rather than relying
            solely on 429 retries. 0 disables the pause.
        rag_embed_max_retries: How many times to retry a single
            embedding batch after a 429 RESOURCE_EXHAUSTED response
            before giving up on it.
        scheduler_daily_time: "HH:MM" (24h) time the daily job (data
            pipeline + sentiment scoring + ML predictions using the
            already-trained model) runs, for both
            src/scheduler/run_scheduler.py (Docker) and the Windows Task
            Scheduler wrapper scripts.
        scheduler_weekly_day: Three-letter day name ("mon".."sun") the
            weekly job (model retraining + evaluation + a fresh
            prediction pass) runs.
        scheduler_weekly_time: "HH:MM" (24h) time the weekly job runs.
        scheduler_timezone: IANA timezone name (e.g. "America/New_York")
            the above two times are interpreted in. Defaults to UTC so
            behavior is identical regardless of the host machine's/
            container's local timezone unless explicitly overridden.
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


# Singleton settings instance, imported by other modules as:
#   from src.utils.config import settings
settings = load_settings()


class TrackedSymbolsError(Exception):
    """Raised when config/tracked_symbols.txt is missing, unreadable, or empty."""


def load_tracked_symbols(path: Path | None = None) -> list[str]:
    """
    Read the default list of tracked stock symbols from a config file
    (config/tracked_symbols.txt by default).

    Used by src/pipeline/run_pipeline.py as the symbol list when the
    pipeline is run with no --symbols flag, so the most common case
    (`python -m src.pipeline.run_pipeline`) doesn't require typing out
    every symbol on the command line.

    File format: one symbol per line. Blank lines and lines starting with
    "#" (comments) are ignored. Symbols are trimmed of surrounding
    whitespace and uppercased. Duplicate symbols (after trimming/
    uppercasing) are ignored, keeping only the first occurrence, so the
    file can be organized into commented sector groupings without
    worrying about accidental repeats across groups.

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
