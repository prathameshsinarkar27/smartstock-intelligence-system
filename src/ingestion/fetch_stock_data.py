"""
fetch_stock_data.py

Fetches daily OHLCV data from Twelve Data for stock symbols and saves
one raw CSV file per symbol.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import requests

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

TWELVEDATA_TIME_SERIES_URL = "https://api.twelvedata.com/time_series"
DEFAULT_OUTPUT_SIZE = 365  # ~1 trading year of daily candles
DEFAULT_INTERVAL = "1day"
RATE_LIMIT_SLEEP_SECONDS = 7.6  # Twelve Data free tier: 8 calls/minute


class StockDataFetchError(Exception):
    """Raised when a stock data API call fails or returns invalid data."""


def _build_time_series_params(symbol: str, output_size: int, interval: str) -> dict[str, Any]:
    """Build query parameters for a Twelve Data time-series request."""
    return {
        "symbol": symbol,
        "interval": interval,
        "outputsize": output_size,
        "apikey": settings.twelvedata_api_key,
    }


def fetch_time_series(symbol: str, output_size: int = DEFAULT_OUTPUT_SIZE,
                       interval: str = DEFAULT_INTERVAL) -> dict[str, Any]:
    """
    Call the Twelve Data /time_series endpoint for a single symbol.
    """
    params = _build_time_series_params(symbol, output_size, interval)

    try:
        response = requests.get(
            TWELVEDATA_TIME_SERIES_URL,
            params=params,
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise StockDataFetchError(f"HTTP request failed for symbol '{symbol}': {exc}") from exc

    payload = response.json()

    if payload.get("status") == "error":
        raise StockDataFetchError(
            f"Twelve Data returned an error for '{symbol}': {payload.get('message', 'unknown error')}"
        )

    return payload


def save_time_series_to_csv(symbol: str, payload: dict[str, Any], output_dir: Path) -> Path:
    """Write Twelve Data time-series data to a CSV file."""
    values = payload.get("values", [])

    if not values:
        raise StockDataFetchError(f"No time series data available for symbol '{symbol}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol.upper()}_prices_raw.csv"

    # Reverse Twelve Data's descending order to chronological order.
    chronological_values = list(reversed(values))

    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["symbol", "timestamp", "date", "open", "high", "low", "close", "volume"])

        for row in chronological_values:
            row_date = row.get("datetime", "")
            writer.writerow([
                symbol.upper(),
                "",  # Twelve Data does not provide a Unix timestamp.
                row_date,
                row.get("open", ""),
                row.get("high", ""),
                row.get("low", ""),
                row.get("close", ""),
                row.get("volume", ""),
            ])

    logger.info("Saved %d rows for '%s' to %s", len(chronological_values), symbol, output_path)
    return output_path


def fetch_and_save_symbols(symbols: list[str], output_size: int = DEFAULT_OUTPUT_SIZE,
                            output_dir: Path | None = None) -> list[Path]:
    """Fetch and save daily price data for a list of symbols."""
    if output_dir is None:
        output_dir = settings.data_raw_dir

    written_paths: list[Path] = []

    for idx, symbol in enumerate(symbols):
        try:
            logger.info("Fetching price data for '%s' (%d/%d)...", symbol, idx + 1, len(symbols))
            payload = fetch_time_series(symbol, output_size=output_size)
            path = save_time_series_to_csv(symbol, payload, output_dir)
            written_paths.append(path)
        except StockDataFetchError as exc:
            logger.error("Skipping '%s': %s", symbol, exc)

        # Respect Twelve Data's free-tier rate limit between requests.
        if idx < len(symbols) - 1:
            time.sleep(RATE_LIMIT_SLEEP_SECONDS)

    return written_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch daily stock price data from Twelve Data.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="One or more stock ticker symbols, e.g. --symbols AAPL MSFT GOOGL",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=DEFAULT_OUTPUT_SIZE,
        help=f"Number of most-recent daily candles to fetch (default: {DEFAULT_OUTPUT_SIZE}, max 5000)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the stock data fetcher."""
    if not settings.twelvedata_api_key:
        logger.error(
            "TWELVEDATA_API_KEY is not set. Add it to your .env file before running this script. "
            "See docs/PHASE_1_SETUP_GUIDE.md for instructions."
        )
        sys.exit(1)

    args = parse_args()
    written_paths = fetch_and_save_symbols(args.symbols, output_size=args.output_size)

    if not written_paths:
        logger.error("No data was successfully fetched for any symbol.")
        sys.exit(1)

    logger.info("Done. Wrote %d file(s) to %s", len(written_paths), settings.data_raw_dir)


if __name__ == "__main__":
    main()
