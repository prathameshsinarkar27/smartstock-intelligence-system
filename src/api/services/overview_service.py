"""
overview_service.py

Builds data for the Market Overview page.
"""

from typing import Any

from src.analytics.kpi_calculator import (
    get_market_overview_kpis,
    get_sector_performance,
    get_top_movers,
)
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

RECENT_NEWS_LIMIT = 8


def _dominant_sentiment_label(
    positive_count: int | None,
    negative_count: int | None,
    neutral_count: int | None,
) -> str | None:
    """
    Return the dominant sentiment label from article counts.

    Returns None when no scored articles exist.
    """
    if positive_count is None and negative_count is None and neutral_count is None:
        return None

    counts = {
        "positive": positive_count or 0,
        "negative": negative_count or 0,
        "neutral": neutral_count or 0,
    }
    return max(counts, key=lambda label: (counts[label], label == "positive", label == "negative"))


def get_recent_news(limit: int = RECENT_NEWS_LIMIT) -> list[dict[str, Any]]:
    """Return the latest market news articles."""
    return get_news(symbol=None, limit=limit)


def get_news(symbol: str | None = None, limit: int = RECENT_NEWS_LIMIT) -> list[dict[str, Any]]:
    """
    Return recent news, optionally filtered by symbol.
    """
    query = """
        SELECT c.symbol, c.company_name, na.title, na.source, na.published_date, na.url
        FROM news_articles na
        JOIN companies c ON c.company_id = na.company_id
        WHERE %(symbol)s::text IS NULL OR c.symbol = %(symbol)s
        ORDER BY na.published_date DESC NULLS LAST
        LIMIT %(limit)s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"symbol": symbol.upper() if symbol else None, "limit": limit})
            rows = cur.fetchall()

    return [
        {
            "symbol": symbol_val,
            "company_name": company_name,
            "title": title,
            "source": source,
            "published_date": published_date,
            "url": url,
        }
        for symbol_val, company_name, title, source, published_date, url in rows
    ]


def get_filtered_companies(
    sector: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return tracked companies with optional sector and search filters.
    """
    query = """
        WITH ranked_prices AS (
            SELECT
                company_id,
                close,
                ROW_NUMBER() OVER (PARTITION BY company_id ORDER BY date DESC) AS rn
            FROM historical_prices
        ),
        latest_two AS (
            SELECT
                company_id,
                MAX(CASE WHEN rn = 1 THEN close END) AS latest_close,
                MAX(CASE WHEN rn = 2 THEN close END) AS previous_close
            FROM ranked_prices
            WHERE rn IN (1, 2)
            GROUP BY company_id
        )
        SELECT
            c.symbol,
            c.company_name,
            c.sector,
            c.industry,
            c.market_cap,
            c.pe_ratio,
            lt.latest_close,
            lt.previous_close,
            css.positive_count,
            css.negative_count,
            css.neutral_count
        FROM companies c
        LEFT JOIN latest_two lt ON lt.company_id = c.company_id
        LEFT JOIN company_sentiment_summary css ON css.company_id = c.company_id
        WHERE
            (%(sector)s::text IS NULL OR c.sector = %(sector)s)
            AND (
                %(search)s::text IS NULL
                OR c.symbol ILIKE %(search_pattern)s
                OR c.company_name ILIKE %(search_pattern)s
            )
        ORDER BY c.symbol;
    """

    search_pattern = f"%{search}%" if search else None

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                {"sector": sector, "search": search, "search_pattern": search_pattern},
            )
            rows = cur.fetchall()

    companies = []
    for (
        symbol_val, company_name, sector_val, industry, market_cap, pe_ratio,
        latest_close, previous_close, positive_count, negative_count, neutral_count,
    ) in rows:
        daily_change_pct = None
        if latest_close is not None and previous_close is not None and previous_close != 0:
            daily_change_pct = float((latest_close - previous_close) / previous_close * 100)

        companies.append({
            "symbol": symbol_val,
            "company_name": company_name,
            "sector": sector_val,
            "industry": industry,
            "market_cap": float(market_cap) if market_cap is not None else None,
            "pe_ratio": float(pe_ratio) if pe_ratio is not None else None,
            "current_price": float(latest_close) if latest_close is not None else None,
            "daily_change_pct": daily_change_pct,
            "sentiment_label": _dominant_sentiment_label(positive_count, negative_count, neutral_count),
        })

    return companies


def get_all_sectors() -> list[str]:
    """Return all distinct tracked company sectors."""
    query = "SELECT DISTINCT sector FROM companies WHERE sector IS NOT NULL ORDER BY sector;"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return [row[0] for row in rows]


def build_overview_page_data(sector: str | None = None, search: str | None = None) -> dict[str, Any]:
    """
    Build all data required by the Market Overview page.
    """
    return {
        "market_kpis": get_market_overview_kpis(),
        "top_movers": get_top_movers(),
        "sector_performance": get_sector_performance(),
        "recent_news": get_recent_news(),
        "companies": get_filtered_companies(sector=sector, search=search),
        "all_sectors": get_all_sectors(),
        "active_sector": sector,
        "active_search": search,
    }
