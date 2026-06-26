"""
fetch_stock_data.py

Pulls daily historical OHLCV (open/high/low/close/volume) price data from the
Twelve Data API for a list of stock symbols, and writes the raw, untouched
response data to data/raw/ as CSV files.

NOTE: This module originally used Finnhub's /stock/candle endpoint, but that
endpoint requires a paid Finnhub plan. It now uses Twelve Data's
/time_series endpoint instead, which supports daily OHLCV data on the free
tier. Finnhub is still used elsewhere for company profile/fundamentals 
data, which remains free on Finnhub.

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
    """
    Build the query parameters for a Twelve Data /time_series request.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        output_size: Number of most-recent candles to request (max 5000 on
            Twelve Data; the free tier comfortably supports the default used
            here).
        interval: Twelve Data interval string (e.g. "1day" for daily candles).

    Returns:
        A dict of query parameters ready to pass to requests.get().
    """
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

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        output_size: Number of most-recent daily candles to request.
        interval: Twelve Data interval string (default "1day").

    Returns:
        The parsed JSON response from Twelve Data, containing a "meta"
        object and a "values" list of dicts with "datetime", "open",
        "high", "low", "close", and "volume" keys (as strings).

    Raises:
        StockDataFetchError: If the HTTP request fails, or Twelve Data
            returns an error payload (Twelve Data reports many errors,
            such as invalid API key or unknown symbol, as HTTP 200 with
            a JSON body like {"status": "error", "message": "..."}
            rather than a non-200 HTTP status, so both cases are checked).
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
    """
    Write a Twelve Data time_series payload to a CSV file in the raw data
    directory.

    Args:
        symbol: Stock ticker symbol, used in the output filename.
        payload: The parsed JSON response from fetch_time_series().
        output_dir: Directory to write the CSV file into.

    Returns:
        The path to the written CSV file.

    Raises:
        StockDataFetchError: If the payload has no candle data to write.
    """
    values = payload.get("values", [])

    if not values:
        raise StockDataFetchError(f"No time series data available for symbol '{symbol}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol.upper()}_prices_raw.csv"

    # Twelve Data returns values in descending date order (most recent
    # first); reverse them so the CSV is chronological, matching the order
    # the previous Finnhub-based implementation produced.
    chronological_values = list(reversed(values))

    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["symbol", "timestamp", "date", "open", "high", "low", "close", "volume"])

        for row in chronological_values:
            row_date = row.get("datetime", "")
            writer.writerow([
                symbol.upper(),
                "",  # No unix timestamp provided by Twelve Data; left blank.
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
    """
    Fetch and save daily price data for a list of symbols, respecting
    Twelve Data's free-tier rate limit by sleeping between requests.

    Args:
        symbols: List of stock ticker symbols to fetch.
        output_size: Number of most-recent daily candles to request per symbol.
        output_dir: Directory to write CSV files into. Defaults to
            settings.data_raw_dir.

    Returns:
        A list of paths to successfully written CSV files. Symbols that
        failed to fetch are logged as errors and skipped, not raised.
    """
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
    """Parse command-line arguments for standalone script execution."""
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
    """Entry point for standalone script execution."""
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
