"""
fetch_news.py

Fetches recent news articles from NewsAPI for stock symbols and saves
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

NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"
DEFAULT_PAGE_SIZE = 50  # Max allowed per page on the free tier
RATE_LIMIT_SLEEP_SECONDS = 1.0

OUTPUT_COLUMNS = [
    "symbol",
    "title",
    "content",
    "source",
    "published_date",
    "url",
]


class NewsFetchError(Exception):
    """Raised when a news API call fails or returns invalid data."""


def fetch_news_for_symbol(symbol: str, page_size: int = DEFAULT_PAGE_SIZE) -> list[dict[str, Any]]:
    """Fetch recent news articles for a stock symbol."""
    try:
        response = requests.get(
            NEWSAPI_EVERYTHING_URL,
            params={
                "q": symbol,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "apiKey": settings.newsapi_api_key,
            },
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise NewsFetchError(f"HTTP request failed for symbol '{symbol}': {exc}") from exc

    payload = response.json()

    if payload.get("status") != "ok":
        raise NewsFetchError(
            f"NewsAPI returned an error for '{symbol}': {payload.get('message', 'unknown error')}"
        )

    return payload.get("articles", [])


def _flatten_article(symbol: str, article: dict[str, Any]) -> dict[str, Any]:
    """Convert a NewsAPI article into an output row."""
    source = article.get("source") or {}
    return {
        "symbol": symbol.upper(),
        "title": article.get("title", "") or "",
        "content": article.get("content", "") or article.get("description", "") or "",
        "source": source.get("name", "") or "",
        "published_date": article.get("publishedAt", "") or "",
        "url": article.get("url", "") or "",
    }


def save_news_to_csv(symbol: str, articles: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write news articles for one symbol to a CSV file."""
    if not articles:
        raise NewsFetchError(f"No articles available for symbol '{symbol}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol.upper()}_news_raw.csv"

    rows = [_flatten_article(symbol, article) for article in articles]

    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d article(s) for '%s' to %s", len(rows), symbol, output_path)
    return output_path


def fetch_and_save_symbols(symbols: list[str], output_dir: Path | None = None) -> list[Path]:
    """Fetch and save news articles for a list of symbols."""
    if output_dir is None:
        output_dir = settings.data_raw_dir

    written_paths: list[Path] = []

    for idx, symbol in enumerate(symbols):
        try:
            logger.info("Fetching news for '%s' (%d/%d)...", symbol, idx + 1, len(symbols))
            articles = fetch_news_for_symbol(symbol)
            path = save_news_to_csv(symbol, articles, output_dir)
            written_paths.append(path)
        except NewsFetchError as exc:
            logger.error("Skipping '%s': %s", symbol, exc)

        if idx < len(symbols) - 1:
            time.sleep(RATE_LIMIT_SLEEP_SECONDS)

    return written_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch news articles from NewsAPI.org.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="One or more stock ticker symbols, e.g. --symbols AAPL MSFT GOOGL",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"Number of articles to fetch per symbol (default: {DEFAULT_PAGE_SIZE})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the news fetcher."""
    if not settings.newsapi_api_key:
        logger.error(
            "NEWSAPI_API_KEY is not set. Add it to your .env file before running this script. "
            "See docs/PHASE_1_SETUP_GUIDE.md for instructions."
        )
        sys.exit(1)

    args = parse_args()
    written_paths = fetch_and_save_symbols(args.symbols)

    if not written_paths:
        logger.error("No news data was successfully fetched for any symbol.")
        sys.exit(1)

    logger.info("Done. Wrote %d file(s) to %s", len(written_paths), settings.data_raw_dir)


if __name__ == "__main__":
    main()
