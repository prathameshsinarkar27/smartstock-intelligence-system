"""
fetch_company_data.py

Fetches company profiles and basic fundamentals from Finnhub and saves
the combined raw data to a CSV file.
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

# Output fields mapped to the companies table where applicable.
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
    """Fetch the Finnhub company profile for a symbol."""
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
    """Fetch basic fundamentals from Finnhub for a symbol."""
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
    """Combine profile and metrics data into an output row."""
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
    """Fetch company data for the given symbols and save it as CSV."""
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
    """Run the company data fetcher."""
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
