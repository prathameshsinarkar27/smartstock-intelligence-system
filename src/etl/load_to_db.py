"""
load_to_db.py

Loads transformed CSVs into the PostgreSQL warehouse.

Usage:
    python -m src.etl.load_to_db --symbols AAPL MSFT
"""

import argparse
from pathlib import Path

import pandas as pd
from psycopg2.extras import execute_values

from src.utils.config import settings
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoadError(Exception):
    """Raised when a processed CSV is missing or cannot be loaded."""


def _processed_dir() -> Path:
    """Return the processed data directory."""
    return settings.data_raw_dir.parent / "processed"


def _read_processed_csv(filename: str) -> pd.DataFrame:
    """
    Read a processed CSV file.

    Args:
        filename: Filename within the processed data directory.

    Returns:
        DataFrame containing the file contents.

    Raises:
        DataLoadError: If the file does not exist.
    """
    path = _processed_dir() / filename
    if not path.exists():
        raise DataLoadError(f"Processed file not found: {path}. Run transform_data.py first.")
    return pd.read_csv(path)


def load_companies(df: pd.DataFrame) -> int:
    """
    Upsert company rows into the companies table.

    Existing symbols are updated so repeated ETL runs refresh
    company records without creating duplicates.

    Args:
        df: Transformed company DataFrame.

    Returns:
        Number of rows upserted.
    """
    if df.empty:
        logger.warning("load_companies: received an empty DataFrame, nothing to load.")
        return 0

    # Replace pandas NaN with None so psycopg2 writes SQL NULL rather than
    # the literal string "nan".
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    rows = [
        (
            r["symbol"],
            r["company_name"],
            r["sector"],
            r["industry"],
            r["market_cap"],
            r["pe_ratio"],
            r["eps"],
        )
        for r in records
    ]

    query = """
        INSERT INTO companies (symbol, company_name, sector, industry, market_cap, pe_ratio, eps)
        VALUES %s
        ON CONFLICT (symbol) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            sector = EXCLUDED.sector,
            industry = EXCLUDED.industry,
            market_cap = EXCLUDED.market_cap,
            pe_ratio = EXCLUDED.pe_ratio,
            eps = EXCLUDED.eps,
            updated_at = now();
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows)

    logger.info("Upserted %d company row(s).", len(rows))
    return len(rows)


def _get_company_id_map(symbols: list[str]) -> dict[str, int]:
    """
    Resolve company IDs for the given symbols.

    Args:
        symbols: Stock ticker symbols to resolve.

    Returns:
        Mapping of uppercase symbols to company IDs.
    """
    upper_symbols = [s.upper() for s in symbols]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol, company_id FROM companies WHERE symbol = ANY(%s);",
                (upper_symbols,),
            )
            rows = cur.fetchall()

    return {symbol: company_id for symbol, company_id in rows}


def load_prices(symbol: str, df: pd.DataFrame) -> int:
    """
    Upsert price rows into historical_prices for a symbol.

    Existing company/date records are updated on repeated loads.

    Args:
        symbol: Stock ticker symbol.
        df: Transformed price DataFrame.

    Returns:
        Number of rows upserted.

    Raises:
        DataLoadError: If the symbol is not present in companies.
    """
    if df.empty:
        logger.warning("load_prices: received an empty DataFrame for '%s', nothing to load.", symbol)
        return 0

    company_id_map = _get_company_id_map([symbol])
    company_id = company_id_map.get(symbol.upper())
    if company_id is None:
        raise DataLoadError(
            f"Cannot load prices for '{symbol}': no matching row in companies table. "
            f"Run load_companies() first."
        )

    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    rows = [
        (company_id, r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"])
        for r in records
    ]

    query = """
        INSERT INTO historical_prices (company_id, date, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (company_id, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows)

    logger.info("Upserted %d price row(s) for '%s'.", len(rows), symbol)
    return len(rows)


def load_news(symbol: str, df: pd.DataFrame) -> int:
    """
    Upsert news articles into news_articles for a symbol.

    Existing company/URL records are updated on repeated loads.

    Args:
        symbol: Stock ticker symbol.
        df: Transformed news DataFrame.

    Returns:
        Number of rows upserted.

    Raises:
        DataLoadError: If the symbol is not present in companies.
    """
    if df.empty:
        logger.warning("load_news: received an empty DataFrame for '%s', nothing to load.", symbol)
        return 0

    company_id_map = _get_company_id_map([symbol])
    company_id = company_id_map.get(symbol.upper())
    if company_id is None:
        raise DataLoadError(
            f"Cannot load news for '{symbol}': no matching row in companies table. "
            f"Run load_companies() first."
        )

    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    rows = [
        (company_id, r["title"], r["content"], r["source"], r["published_date"], r["url"])
        for r in records
    ]

    query = """
        INSERT INTO news_articles (company_id, title, content, source, published_date, url)
        VALUES %s
        ON CONFLICT (company_id, url) DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            source = EXCLUDED.source,
            published_date = EXCLUDED.published_date;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows)

    logger.info("Upserted %d news row(s) for '%s'.", len(rows), symbol)
    return len(rows)


def run_load_companies() -> int:
    """
    Load processed company data into the companies table.

    Returns:
        Number of company rows upserted.

    Raises:
        DataLoadError: If the processed file is missing.
    """
    df = _read_processed_csv("companies_processed.csv")
    return load_companies(df)


def run_load_for_symbol(symbol: str) -> dict[str, int]:
    """
    Load processed price and news data for a symbol.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        Row counts for successfully loaded prices and news.
    """
    results: dict[str, int] = {}

    try:
        prices_df = _read_processed_csv(f"{symbol.upper()}_prices_processed.csv")
        results["prices"] = load_prices(symbol, prices_df)
    except DataLoadError as exc:
        logger.error(str(exc))

    try:
        news_df = _read_processed_csv(f"{symbol.upper()}_news_processed.csv")
        results["news"] = load_news(symbol, news_df)
    except DataLoadError as exc:
        logger.error(str(exc))

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load processed ETL CSVs into the PostgreSQL warehouse.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="One or more stock ticker symbols, e.g. --symbols AAPL MSFT GOOGL",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the database loading process.

    Companies are loaded first because prices and news depend on
    company ID lookups.
    """
    args = parse_args()

    try:
        run_load_companies()
    except DataLoadError as exc:
        logger.error(str(exc))
        logger.error("Cannot proceed without company data loaded. Exiting.")
        return

    for symbol in args.symbols:
        run_load_for_symbol(symbol)

    logger.info("Load step complete.")


if __name__ == "__main__":
    main()
