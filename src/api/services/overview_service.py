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
        market_cap, pe_ratio, current_price, and daily_change_pct (None if
        fewer than two days of price history exist for that company),
        ordered by symbol. Returns an empty list if no companies match (or
        none are loaded yet).
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
            lt.previous_close
        FROM companies c
        LEFT JOIN latest_two lt ON lt.company_id = c.company_id
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
    for symbol, company_name, sector_val, industry, market_cap, pe_ratio, latest_close, previous_close in rows:
        daily_change_pct = None
        if latest_close is not None and previous_close is not None and previous_close != 0:
            daily_change_pct = float((latest_close - previous_close) / previous_close * 100)

        companies.append({
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector_val,
            "industry": industry,
            "market_cap": float(market_cap) if market_cap is not None else None,
            "pe_ratio": float(pe_ratio) if pe_ratio is not None else None,
            "current_price": float(latest_close) if latest_close is not None else None,
            "daily_change_pct": daily_change_pct,
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
