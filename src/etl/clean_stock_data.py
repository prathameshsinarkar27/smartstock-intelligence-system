"""
clean_stock_data.py

Cleans raw CSV data produced by src/ingestion/  before it is
transformed (transform_data.py) and loaded into PostgreSQL (load_to_db.py).

This is responsible ONLY for:
    - Removing duplicate rows
    - Removing rows with missing/invalid required values
    - Fixing data types (strings -> numbers/dates where applicable)

Usage:
    python -m src.etl.clean_stock_data --symbols AAPL MSFT
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaningError(Exception):
    """Raised when a raw CSV file is missing, empty, or cannot be cleaned."""


def _read_raw_csv(path: Path) -> pd.DataFrame:
    """
    Read a raw CSV file produced by the ingestion layer.

    Args:
        path: Path to the raw CSV file.

    Returns:
        A DataFrame with the file's contents.

    Raises:
        DataCleaningError: If the file does not exist or cannot be parsed.
    """
    if not path.exists():
        raise DataCleaningError(f"Raw file not found: {path}")

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataCleaningError(f"Raw file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataCleaningError(f"Raw file could not be parsed: {path}: {exc}") from exc


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw stock price DataFrame (from {SYMBOL}_prices_raw.csv).

    Cleaning steps:
        1. Drop rows with a missing/unparseable date.
        2. Coerce open/high/low/close/volume to numeric, dropping rows
           where any of them fail to parse.
        3. Drop rows where high < low (physically invalid candle).
        4. Drop exact duplicate (symbol, date) rows, keeping the first.

    Args:
        df: Raw price DataFrame with columns:
            symbol, timestamp, date, open, high, low, close, volume.

    Returns:
        A cleaned DataFrame with the same columns, fewer (or equal) rows.
    """
    cleaned = df.copy()
    initial_count = len(cleaned)

    # Step 1: valid date required.
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date"])

    # Step 2: numeric coercion for price/volume columns.
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.dropna(subset=numeric_cols)

    # Step 3: physically invalid candles (high must be >= low).
    cleaned = cleaned[cleaned["high"] >= cleaned["low"]]

    # Step 4: de-duplicate on (symbol, date), keeping the first occurrence.
    cleaned = cleaned.drop_duplicates(subset=["symbol", "date"], keep="first")

    removed_count = initial_count - len(cleaned)
    if removed_count > 0:
        logger.warning("clean_price_data: removed %d invalid/duplicate row(s)", removed_count)

    return cleaned.reset_index(drop=True)


def clean_company_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw company fundamentals DataFrame (from companies_raw.csv).

    Cleaning steps:
        1. Drop rows with a missing symbol (the natural key).
        2. Drop rows with a missing company_name (not useful without one).
        3. Coerce market_cap/pe_ratio/eps to numeric, leaving NaN (rather
           than dropping the row) for any that fail to parse, since a
           company missing one fundamental metric is still useful data.
        4. Drop exact duplicate symbol rows, keeping the first.

    Args:
        df: Raw company DataFrame with columns:
            symbol, company_name, sector, industry, market_cap, pe_ratio,
            eps, country, currency, exchange.

    Returns:
        A cleaned DataFrame with the same columns, fewer (or equal) rows.
    """
    cleaned = df.copy()
    initial_count = len(cleaned)

    # Step 1 & 2: required fields.
    cleaned = cleaned.dropna(subset=["symbol"])
    cleaned = cleaned[cleaned["symbol"].astype(str).str.strip() != ""]
    cleaned = cleaned.dropna(subset=["company_name"])
    cleaned = cleaned[cleaned["company_name"].astype(str).str.strip() != ""]

    # Step 3: numeric coercion, NaN allowed (not dropped) for optional metrics.
    for col in ["market_cap", "pe_ratio", "eps"]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Step 4: de-duplicate on symbol, keeping the first occurrence.
    cleaned = cleaned.drop_duplicates(subset=["symbol"], keep="first")

    removed_count = initial_count - len(cleaned)
    if removed_count > 0:
        logger.warning("clean_company_data: removed %d invalid/duplicate row(s)", removed_count)

    return cleaned.reset_index(drop=True)


def clean_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw news article DataFrame (from {SYMBOL}_news_raw.csv).

    Cleaning steps:
        1. Drop rows with a missing/empty title (an article needs a title
           to be useful for downstream sentiment analysis or display).
        2. Drop rows with a missing/empty url (used as the dedup key here
           and later as the uq_news_articles_company_url constraint key in
           the database).
        3. Coerce published_date to a parseable datetime, leaving NaT
           (rather than dropping the row) if it fails to parse, since an
           article missing a clean date is still useful content.
        4. Drop exact duplicate (symbol, url) rows, keeping the first.

    Args:
        df: Raw news DataFrame with columns:
            symbol, title, content, source, published_date, url.

    Returns:
        A cleaned DataFrame with the same columns, fewer (or equal) rows.
    """
    cleaned = df.copy()
    initial_count = len(cleaned)

    # Step 1: required title.
    cleaned = cleaned.dropna(subset=["title"])
    cleaned = cleaned[cleaned["title"].astype(str).str.strip() != ""]

    # Step 2: required url.
    cleaned = cleaned.dropna(subset=["url"])
    cleaned = cleaned[cleaned["url"].astype(str).str.strip() != ""]

    # Step 3: best-effort date parsing; UTC to avoid tz-naive/tz-aware
    # comparison issues later, NaT allowed.
    cleaned["published_date"] = pd.to_datetime(
        cleaned["published_date"], errors="coerce", utc=True
    )

    # Step 4: de-duplicate on (symbol, url), keeping the first occurrence.
    cleaned = cleaned.drop_duplicates(subset=["symbol", "url"], keep="first")

    removed_count = initial_count - len(cleaned)
    if removed_count > 0:
        logger.warning("clean_news_data: removed %d invalid/duplicate row(s)", removed_count)

    return cleaned.reset_index(drop=True)


def clean_prices_for_symbol(symbol: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Read and clean the raw price CSV for a single symbol.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        raw_dir: Directory containing raw CSVs. Defaults to settings.data_raw_dir.

    Returns:
        A cleaned price DataFrame for the symbol.

    Raises:
        DataCleaningError: If the raw file is missing or unreadable.
    """
    raw_dir = raw_dir or settings.data_raw_dir
    path = raw_dir / f"{symbol.upper()}_prices_raw.csv"
    df = _read_raw_csv(path)
    return clean_price_data(df)


def clean_news_for_symbol(symbol: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Read and clean the raw news CSV for a single symbol.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        raw_dir: Directory containing raw CSVs. Defaults to settings.data_raw_dir.

    Returns:
        A cleaned news DataFrame for the symbol.

    Raises:
        DataCleaningError: If the raw file is missing or unreadable.
    """
    raw_dir = raw_dir or settings.data_raw_dir
    path = raw_dir / f"{symbol.upper()}_news_raw.csv"
    df = _read_raw_csv(path)
    return clean_news_data(df)


def clean_companies(raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Read and clean the combined raw companies CSV.

    Args:
        raw_dir: Directory containing raw CSVs. Defaults to settings.data_raw_dir.

    Returns:
        A cleaned company DataFrame.

    Raises:
        DataCleaningError: If the raw file is missing or unreadable.
    """
    raw_dir = raw_dir or settings.data_raw_dir
    path = raw_dir / "companies_raw.csv"
    df = _read_raw_csv(path)
    return clean_company_data(df)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone script execution."""
    parser = argparse.ArgumentParser(
        description="Clean raw ingestion CSVs (prices, companies, news) for a list of symbols."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="One or more stock ticker symbols, e.g. --symbols AAPL MSFT GOOGL",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for standalone script execution. Cleans price and news data
    per symbol, and the combined company file once, logging row counts.
    This does not write any output files itself — it is intended primarily
    as a manual verification tool; transform_data.py calls the clean_*
    functions directly as part of the full pipeline.
    """
    args = parse_args()

    try:
        companies_df = clean_companies()
        logger.info("Cleaned companies_raw.csv: %d row(s) remain", len(companies_df))
    except DataCleaningError as exc:
        logger.error(str(exc))

    for symbol in args.symbols:
        try:
            prices_df = clean_prices_for_symbol(symbol)
            logger.info("Cleaned %s prices: %d row(s) remain", symbol, len(prices_df))
        except DataCleaningError as exc:
            logger.error(str(exc))

        try:
            news_df = clean_news_for_symbol(symbol)
            logger.info("Cleaned %s news: %d row(s) remain", symbol, len(news_df))
        except DataCleaningError as exc:
            logger.error(str(exc))


if __name__ == "__main__":
    main()
