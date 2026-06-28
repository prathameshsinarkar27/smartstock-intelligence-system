"""
transform_data.py

Transforms cleaned DataFrames (from clean_stock_data.py) into column sets
that exactly match the PostgreSQL warehouse schema (database/tables.sql),
and writes the results to data/processed/ as CSV files.

This is responsible ONLY for:
    - Renaming/reordering columns to match target database tables
    - Dropping columns that have no corresponding database column
      (logging a warning when doing so)
    - Attaching the foreign key (company_id) where required, once that ID
      is known (see load_to_db.py, which resolves symbol -> company_id
      and is the actual point at which historical_prices/news_articles
      rows become attachable to a company_id; this module keeps the
      natural key "symbol" rather than reaching into the database itself,
      keeping transform_data.py free of any DB dependency)

Usage:
    python -m src.etl.transform_data --symbols AAPL MSFT
"""

import argparse
from pathlib import Path

import pandas as pd

from src.etl.clean_stock_data import (
    DataCleaningError,
    clean_companies,
    clean_news_for_symbol,
    clean_prices_for_symbol,
)
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns kept for each target table, in the exact order the corresponding
# database table expects them (excluding surrogate keys / foreign keys /
# server-generated timestamps, which load_to_db.py handles).
PRICE_TARGET_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]
COMPANY_TARGET_COLUMNS = ["symbol", "company_name", "sector", "industry", "market_cap", "pe_ratio", "eps"]
NEWS_TARGET_COLUMNS = ["symbol", "title", "content", "source", "published_date", "url"]


def transform_price_data(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape a cleaned price DataFrame to match historical_prices' column
    set.

    The cleaned DataFrame already carries the right column names and types
    from clean_stock_data.py; this function drops the ingestion-only
    "timestamp" column (which has no corresponding database column — see
    docs/02_ARCHITECTURE.md on Twelve Data not providing a unix timestamp)
    and enforces column order.

    Args:
        cleaned_df: Output of clean_price_data() / clean_prices_for_symbol().

    Returns:
        A DataFrame with exactly PRICE_TARGET_COLUMNS, in that order.
    """
    dropped_cols = [col for col in cleaned_df.columns if col not in PRICE_TARGET_COLUMNS]
    if dropped_cols:
        logger.warning(
            "transform_price_data: dropping column(s) not present in historical_prices table: %s",
            dropped_cols,
        )

    return cleaned_df[PRICE_TARGET_COLUMNS].copy()


def transform_company_data(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape a cleaned company DataFrame to match companies' column set.

    The companies table (database/tables.sql) does not have columns for
    country, currency, or exchange, even though Finnhub's profile endpoint
    returns them. Those fields are dropped here, with a warning logged so
    the data loss is visible rather than silent.

    Args:
        cleaned_df: Output of clean_company_data() / clean_companies().

    Returns:
        A DataFrame with exactly COMPANY_TARGET_COLUMNS, in that order.
    """
    dropped_cols = [col for col in cleaned_df.columns if col not in COMPANY_TARGET_COLUMNS]
    if dropped_cols:
        logger.warning(
            "transform_company_data: dropping column(s) not present in companies table: %s",
            dropped_cols,
        )

    return cleaned_df[COMPANY_TARGET_COLUMNS].copy()


def transform_news_data(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape a cleaned news DataFrame to match news_articles' column set.

    All columns produced by clean_news_data() already correspond 1:1 to
    news_articles columns, so this function only enforces column order
    (kept as a named step, rather than skipped, so every data type goes
    through the same reshape-and-validate pattern and any future schema
    drift is caught immediately rather than silently passed through).

    Args:
        cleaned_df: Output of clean_news_data() / clean_news_for_symbol().

    Returns:
        A DataFrame with exactly NEWS_TARGET_COLUMNS, in that order.
    """
    dropped_cols = [col for col in cleaned_df.columns if col not in NEWS_TARGET_COLUMNS]
    if dropped_cols:
        logger.warning(
            "transform_news_data: dropping column(s) not present in news_articles table: %s",
            dropped_cols,
        )

    return cleaned_df[NEWS_TARGET_COLUMNS].copy()


def _write_processed_csv(df: pd.DataFrame, filename: str, output_dir: Path | None = None) -> Path:
    """
    Write a transformed DataFrame to data/processed/ as a CSV file.

    Args:
        df: The transformed DataFrame to write.
        filename: Output filename, e.g. "AAPL_prices_processed.csv".
        output_dir: Directory to write into. Defaults to
            settings.data_raw_dir's sibling "processed" directory.

    Returns:
        The path to the written CSV file.
    """
    if output_dir is None:
        # data_raw_dir is .../data/raw; processed/ is its sibling.
        output_dir = settings.data_raw_dir.parent / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d row(s) to %s", len(df), output_path)
    return output_path


def run_transform_for_symbol(symbol: str, output_dir: Path | None = None) -> dict[str, Path]:
    """
    Clean and transform price and news data for a single symbol, writing
    both outputs to data/processed/.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        output_dir: Directory to write processed CSVs into. Defaults to
            data/processed/.

    Returns:
        A dict mapping "prices" and "news" to the paths of the written
        files. A key is omitted if that data type failed to clean (e.g.
        the raw file was missing).
    """
    written: dict[str, Path] = {}

    try:
        cleaned_prices = clean_prices_for_symbol(symbol)
        transformed_prices = transform_price_data(cleaned_prices)
        written["prices"] = _write_processed_csv(
            transformed_prices, f"{symbol.upper()}_prices_processed.csv", output_dir
        )
    except DataCleaningError as exc:
        logger.error("Skipping price transform for '%s': %s", symbol, exc)

    try:
        cleaned_news = clean_news_for_symbol(symbol)
        transformed_news = transform_news_data(cleaned_news)
        written["news"] = _write_processed_csv(
            transformed_news, f"{symbol.upper()}_news_processed.csv", output_dir
        )
    except DataCleaningError as exc:
        logger.error("Skipping news transform for '%s': %s", symbol, exc)

    return written


def run_transform_companies(output_dir: Path | None = None) -> Path | None:
    """
    Clean and transform the combined company fundamentals file, writing
    the output to data/processed/companies_processed.csv.

    Args:
        output_dir: Directory to write the processed CSV into. Defaults to
            data/processed/.

    Returns:
        The path to the written file, or None if cleaning failed (e.g.
        the raw file was missing).
    """
    try:
        cleaned_companies = clean_companies()
        transformed_companies = transform_company_data(cleaned_companies)
        return _write_processed_csv(transformed_companies, "companies_processed.csv", output_dir)
    except DataCleaningError as exc:
        logger.error("Skipping company transform: %s", exc)
        return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone script execution."""
    parser = argparse.ArgumentParser(
        description="Transform cleaned ingestion data into database-schema-matching CSVs."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="One or more stock ticker symbols, e.g. --symbols AAPL MSFT GOOGL",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for standalone script execution."""
    args = parse_args()

    run_transform_companies()

    for symbol in args.symbols:
        run_transform_for_symbol(symbol)

    logger.info("Transform step complete.")


if __name__ == "__main__":
    main()
