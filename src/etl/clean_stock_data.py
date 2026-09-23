"""
clean_stock_data.py

Cleans raw CSV data before transformation and loading into PostgreSQL.

Handles duplicate removal, required-field validation, and data type
conversion for price, company, and news data.

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
    Read a raw CSV file.

    Args:
        path: Path to the raw CSV file.

    Returns:
        DataFrame containing the file contents.

    Raises:
        DataCleaningError: If the file is missing or cannot be parsed.
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
    Clean a raw stock price DataFrame.

    Validates dates and numeric fields, removes invalid candles,
    and de-duplicates symbol/date records.

    Args:
        df: Raw price DataFrame.

    Returns:
        Cleaned price DataFrame.
    """
    cleaned = df.copy()
    initial_count = len(cleaned)

    # Require a valid date.
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date"])

    # Convert price and volume fields to numeric.
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.dropna(subset=numeric_cols)

    # Remove physically invalid candles.
    cleaned = cleaned[cleaned["high"] >= cleaned["low"]]

    # Keep the first record for each symbol/date pair.
    cleaned = cleaned.drop_duplicates(subset=["symbol", "date"], keep="first")

    removed_count = initial_count - len(cleaned)
    if removed_count > 0:
        logger.warning("clean_price_data: removed %d invalid/duplicate row(s)", removed_count)

    return cleaned.reset_index(drop=True)


def clean_company_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw company fundamentals DataFrame.

    Validates required fields, converts optional numeric metrics,
    and removes duplicate symbols.

    Args:
        df: Raw company DataFrame.

    Returns:
        Cleaned company DataFrame.
    """
    cleaned = df.copy()
    initial_count = len(cleaned)

    # Validate required fields.
    cleaned = cleaned.dropna(subset=["symbol"])
    cleaned = cleaned[cleaned["symbol"].astype(str).str.strip() != ""]
    cleaned = cleaned.dropna(subset=["company_name"])
    cleaned = cleaned[cleaned["company_name"].astype(str).str.strip() != ""]

    # Convert optional numeric metrics; invalid values become NaN.
    for col in ["market_cap", "pe_ratio", "eps"]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Keep the first record for each symbol.
    cleaned = cleaned.drop_duplicates(subset=["symbol"], keep="first")

    removed_count = initial_count - len(cleaned)
    if removed_count > 0:
        logger.warning("clean_company_data: removed %d invalid/duplicate row(s)", removed_count)

    return cleaned.reset_index(drop=True)


def clean_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw news article DataFrame.

    Validates title and URL fields, parses publication dates,
    and removes duplicate symbol/URL records.

    Args:
        df: Raw news DataFrame.

    Returns:
        Cleaned news DataFrame.
    """
    cleaned = df.copy()
    initial_count = len(cleaned)

    # Validate required title.
    cleaned = cleaned.dropna(subset=["title"])
    cleaned = cleaned[cleaned["title"].astype(str).str.strip() != ""]

    # Validate required URL.
    cleaned = cleaned.dropna(subset=["url"])
    cleaned = cleaned[cleaned["url"].astype(str).str.strip() != ""]

    # Parse dates as UTC; invalid values become NaT.
    cleaned["published_date"] = pd.to_datetime(
        cleaned["published_date"], errors="coerce", utc=True
    )

    # Keep the first record for each symbol/URL pair.
    cleaned = cleaned.drop_duplicates(subset=["symbol", "url"], keep="first")

    removed_count = initial_count - len(cleaned)
    if removed_count > 0:
        logger.warning("clean_news_data: removed %d invalid/duplicate row(s)", removed_count)

    return cleaned.reset_index(drop=True)


def clean_prices_for_symbol(symbol: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Read and clean raw price data for a symbol.

    Args:
        symbol: Stock ticker symbol.
        raw_dir: Directory containing raw CSVs.

    Returns:
        Cleaned price DataFrame.

    Raises:
        DataCleaningError: If the raw file is missing or unreadable.
    """
    raw_dir = raw_dir or settings.data_raw_dir
    path = raw_dir / f"{symbol.upper()}_prices_raw.csv"
    df = _read_raw_csv(path)
    return clean_price_data(df)


def clean_news_for_symbol(symbol: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Read and clean raw news data for a symbol.

    Args:
        symbol: Stock ticker symbol.
        raw_dir: Directory containing raw CSVs.

    Returns:
        Cleaned news DataFrame.

    Raises:
        DataCleaningError: If the raw file is missing or unreadable.
    """
    raw_dir = raw_dir or settings.data_raw_dir
    path = raw_dir / f"{symbol.upper()}_news_raw.csv"
    df = _read_raw_csv(path)
    return clean_news_data(df)


def clean_companies(raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Read and clean the combined company data.

    Args:
        raw_dir: Directory containing raw CSVs.

    Returns:
        Cleaned company DataFrame.

    Raises:
        DataCleaningError: If the raw file is missing or unreadable.
    """
    raw_dir = raw_dir or settings.data_raw_dir
    path = raw_dir / "companies_raw.csv"
    df = _read_raw_csv(path)
    return clean_company_data(df)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
    Run the cleaning process for company, price, and news data.

    This function logs results but does not write output files.
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
