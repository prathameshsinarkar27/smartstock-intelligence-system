"""
transform_data.py

Transforms cleaned DataFrames into CSVs matching the PostgreSQL
warehouse schema.

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

# Target columns in database table order.
PRICE_TARGET_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]
COMPANY_TARGET_COLUMNS = ["symbol", "company_name", "sector", "industry", "market_cap", "pe_ratio", "eps"]
NEWS_TARGET_COLUMNS = ["symbol", "title", "content", "source", "published_date", "url"]


def transform_price_data(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape cleaned price data to match the historical_prices schema.

    Args:
        cleaned_df: Cleaned price DataFrame.

    Returns:
        DataFrame containing PRICE_TARGET_COLUMNS in order.
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
    Reshape cleaned company data to match the companies schema.

    Args:
        cleaned_df: Cleaned company DataFrame.

    Returns:
        DataFrame containing COMPANY_TARGET_COLUMNS in order.
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
    Reshape cleaned news data to match the news_articles schema.

    Args:
        cleaned_df: Cleaned news DataFrame.

    Returns:
        DataFrame containing NEWS_TARGET_COLUMNS in order.
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
    Write a transformed DataFrame to the processed data directory.

    Args:
        df: Transformed DataFrame.
        filename: Output CSV filename.
        output_dir: Optional output directory.

    Returns:
        Path to the written CSV file.
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
    Clean and transform price and news data for a symbol.

    Args:
        symbol: Stock ticker symbol.
        output_dir: Optional output directory.

    Returns:
        Paths of successfully written price and news files.
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
    Clean and transform the company fundamentals data.

    Args:
        output_dir: Optional output directory.

    Returns:
        Path to the written file, or None if cleaning failed.
    """
    try:
        cleaned_companies = clean_companies()
        transformed_companies = transform_company_data(cleaned_companies)
        return _write_processed_csv(transformed_companies, "companies_processed.csv", output_dir)
    except DataCleaningError as exc:
        logger.error("Skipping company transform: %s", exc)
        return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
    """Run the data transformation process."""
    args = parse_args()

    run_transform_companies()

    for symbol in args.symbols:
        run_transform_for_symbol(symbol)

    logger.info("Transform step complete.")


if __name__ == "__main__":
    main()
