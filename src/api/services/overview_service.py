"""
overview_service.py

Business logic for the Market Overview page (src/api/routes/overview.py).

This module composes data from src/analytics/kpi_calculator.py and a
lightweight direct query for recent news into the exact shape the
Market Overview template needs.
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
    Reduce a company's sentiment_scores counts (as returned by the
    company_sentiment_summary view) to a single label for the Market
    Overview table's compact Sentiment column.

    Args:
        positive_count: Count of this company's articles scored
            "positive". None if the company has no scored articles
            (company_sentiment_summary has no row for it at all).
        negative_count: Count scored "negative". None under the same
            condition as positive_count.
        neutral_count: Count scored "neutral". None under the same
            condition as positive_count.

    Returns:
        "positive", "negative", or "neutral" — whichever count is
        highest. Ties are broken in that same order (positive beats
        negative beats neutral), matching the intuition that a tied
        positive/negative split is more notable than defaulting to
        neutral. Returns None if the company has no scored articles at
        all (all three counts None, from the LEFT JOIN finding no
        matching view row).
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
    """
    Fetch the most recently published news articles across all tracked
    companies, for the Market Overview page's "Latest Market News"
    section.

    Args:
        limit: Maximum number of articles to return.

    Returns:
        A list of dicts with symbol, company_name, title, source,
        published_date, and url, ordered most-recent-first. Returns an
        empty list if no news has been loaded yet.
    """
    query = """
        SELECT c.symbol, c.company_name, na.title, na.source, na.published_date, na.url
        FROM news_articles na
        JOIN companies c ON c.company_id = na.company_id
        ORDER BY na.published_date DESC NULLS LAST
        LIMIT %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()

    return [
        {
            "symbol": symbol,
            "company_name": company_name,
            "title": title,
            "source": source,
            "published_date": published_date,
            "url": url,
        }
        for symbol, company_name, title, source, published_date, url in rows
    ]


def get_filtered_companies(
    sector: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch all tracked companies with their latest price, optionally
    filtered by sector and/or a case-insensitive search term matched
    against symbol or company name.

    Args:
        sector: If provided, only companies in this exact sector are returned.
        search: If provided, only companies whose symbol or company_name
            contains this term (case-insensitive) are returned.

    Returns:
        A list of dicts with symbol, company_name, sector, industry,
        market_cap, pe_ratio, current_price, daily_change_pct (None if
        fewer than two days of price history exist for that company), and
        sentiment_label (Phase 7) — one of "positive", "negative",
        "neutral" (whichever has the highest scored-article count for
        that company; ties favor positive, then negative, then neutral),
        or None if the company has no scored news articles yet.
        Ordered by symbol. Returns an empty list if no companies match
        (or none are loaded yet).
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
    """
    Fetch the distinct list of sectors currently represented among tracked
    companies, for the Market Overview page's sector filter dropdown.

    Returns:
        A sorted list of distinct, non-null sector names. Empty if no
        companies are loaded yet.
    """
    query = "SELECT DISTINCT sector FROM companies WHERE sector IS NOT NULL ORDER BY sector;"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return [row[0] for row in rows]


def build_overview_page_data(sector: str | None = None, search: str | None = None) -> dict[str, Any]:
    """
    Assemble everything the Market Overview template needs in one call.

    Args:
        sector: Optional sector filter, forwarded to get_filtered_companies().
        search: Optional search term, forwarded to get_filtered_companies().

    Returns:
        A dict with keys: market_kpis, top_movers, sector_performance,
        recent_news, companies, all_sectors, active_sector, active_search.
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
