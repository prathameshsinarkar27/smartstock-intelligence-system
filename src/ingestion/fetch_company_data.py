"""
fetch_company_data.py

Pulls company profile and basic fundamentals data from the Finnhub API for a
list of stock symbols, and writes the raw, untouched response data to
data/raw/ as a single combined CSV file.

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

FINNHUB_PROFILE_URL = "https://finnhub.io/api/v1/stock/profile2"
FINNHUB_METRICS_URL = "https://finnhub.io/api/v1/stock/metric"
RATE_LIMIT_SLEEP_SECONDS = 1.1  # Finnhub free tier: ~60 calls/minute

# Columns written to the output CSV, in order. Maps to fields blueprinted in
# the `companies` table (database/tables.sql, Phase 2) where applicable, plus
# a couple of extra raw fields kept for completeness.
OUTPUT_COLUMNS = [
    "symbol",
    "company_name",
    "sector",
    "industry",
    "market_cap",
    "pe_ratio",
    "eps",
    "country",
    "currency",
    "exchange",
]


class CompanyDataFetchError(Exception):
    """Raised when a company data API call fails or returns invalid data."""


def fetch_company_profile(symbol: str) -> dict[str, Any]:
    """
    Call the Finnhub /stock/profile2 endpoint for a single symbol.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".

    Returns:
        The parsed JSON response from Finnhub containing company profile
        fields such as name, industry/finnhubIndustry, marketCapitalization,
        country, currency, and exchange.

    Raises:
        CompanyDataFetchError: If the HTTP request fails.
    """
    try:
        response = requests.get(
            FINNHUB_PROFILE_URL,
            params={"symbol": symbol, "token": settings.finnhub_api_key},
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CompanyDataFetchError(f"Profile request failed for '{symbol}': {exc}") from exc

    payload = response.json()

    if not payload:
        logger.warning("Finnhub returned an empty profile for '%s'", symbol)

    return payload


def fetch_company_metrics(symbol: str) -> dict[str, Any]:
    """
    Call the Finnhub /stock/metric endpoint for a single symbol to retrieve
    basic fundamentals (P/E ratio, EPS, etc.).

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".

    Returns:
        The "metric" sub-dictionary from Finnhub's response, containing
        keys like "peNormalizedAnnual" and "epsInclExtraItemsAnnual".

    Raises:
        CompanyDataFetchError: If the HTTP request fails.
    """
    try:
        response = requests.get(
            FINNHUB_METRICS_URL,
            params={"symbol": symbol, "metric": "all", "token": settings.finnhub_api_key},
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CompanyDataFetchError(f"Metrics request failed for '{symbol}': {exc}") from exc

    payload = response.json()
    return payload.get("metric", {}) or {}


def _combine_profile_and_metrics(symbol: str, profile: dict[str, Any],
                                  metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Merge a company profile and metrics dict into one flat row matching
    OUTPUT_COLUMNS.

    Args:
        symbol: Stock ticker symbol.
        profile: Output of fetch_company_profile().
        metrics: Output of fetch_company_metrics().

    Returns:
        A dict with keys matching OUTPUT_COLUMNS, ready to write as a CSV row.
    """
    return {
        "symbol": symbol.upper(),
        "company_name": profile.get("name", ""),
        "sector": profile.get("gicsSector", profile.get("finnhubIndustry", "")),
        "industry": profile.get("finnhubIndustry", ""),
        "market_cap": profile.get("marketCapitalization", ""),
        "pe_ratio": metrics.get("peNormalizedAnnual", ""),
        "eps": metrics.get("epsInclExtraItemsAnnual", ""),
        "country": profile.get("country", ""),
        "currency": profile.get("currency", ""),
        "exchange": profile.get("exchange", ""),
    }


def fetch_and_save_companies(symbols: list[str], output_dir: Path | None = None) -> Path:
    """
    Fetch profile + metrics data for a list of symbols and save them as one
    combined CSV file.

    Args:
        symbols: List of stock ticker symbols to fetch.
        output_dir: Directory to write the CSV file into. Defaults to
            settings.data_raw_dir.

    Returns:
        The path to the written CSV file.

    Raises:
        CompanyDataFetchError: If no symbols could be fetched successfully.
    """
    if output_dir is None:
        output_dir = settings.data_raw_dir

    rows: list[dict[str, Any]] = []

    for idx, symbol in enumerate(symbols):
        try:
            logger.info("Fetching company data for '%s' (%d/%d)...", symbol, idx + 1, len(symbols))
            profile = fetch_company_profile(symbol)
            time.sleep(RATE_LIMIT_SLEEP_SECONDS)
            metrics = fetch_company_metrics(symbol)
            rows.append(_combine_profile_and_metrics(symbol, profile, metrics))
        except CompanyDataFetchError as exc:
            logger.error("Skipping '%s': %s", symbol, exc)

        if idx < len(symbols) - 1:
            time.sleep(RATE_LIMIT_SLEEP_SECONDS)

    if not rows:
        raise CompanyDataFetchError("No company data was successfully fetched for any symbol.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "companies_raw.csv"

    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d company record(s) to %s", len(rows), output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone script execution."""
    parser = argparse.ArgumentParser(description="Fetch company profile/fundamentals data from Finnhub.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="One or more stock ticker symbols, e.g. --symbols AAPL MSFT GOOGL",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for standalone script execution."""
    if not settings.finnhub_api_key:
        logger.error(
            "FINNHUB_API_KEY is not set. Add it to your .env file before running this script. "
            "See docs/PHASE_1_SETUP_GUIDE.md for instructions."
        )
        sys.exit(1)

    args = parse_args()

    try:
        output_path = fetch_and_save_companies(args.symbols)
    except CompanyDataFetchError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Done. Wrote company data to %s", output_path)


if __name__ == "__main__":
    main()
