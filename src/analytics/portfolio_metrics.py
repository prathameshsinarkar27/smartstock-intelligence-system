"""
portfolio_metrics.py

Computes Portfolio Analyzer metrics for a single user's holdings:
per-holding market value / cost value / unrealized P&L, a portfolio-wide
summary, and sector concentration — built on top of the `watchlist` table
and `watchlist_overview` view (database/views.sql)

A "holding" here means a watchlist row with shares > 0 (a real position).
"""

from typing import Any

from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_portfolio_holdings(user_name: str) -> list[dict[str, Any]]:
    """
    Fetch every watchlist/portfolio row for a user (both real positions
    and watch-only entries), enriched with sentiment and ML risk where
    available.

    Args:
        user_name: The watchlist owner to fetch rows for.

    Returns:
        A list of dicts, one per watchlist row, each with:
            watchlist_id, symbol, company_name, sector,
            shares, avg_cost_basis, purchased_at,
            latest_close, latest_price_date,
            market_value, cost_value, unrealized_pl, unrealized_pl_pct,
            is_position (bool — True if shares > 0),
            sentiment_score (signed -100..100, or None),
            ml_risk_score (0..1, or None).
        Ordered by market_value descending (real positions first, largest
        first), then by symbol for watch-only rows. Empty list if the
        user has no watchlist entries yet.
    """
    query = """
        SELECT
            wo.watchlist_id,
            wo.symbol,
            wo.company_name,
            wo.sector,
            wo.shares,
            wo.avg_cost_basis,
            wo.purchased_at,
            wo.latest_close,
            wo.latest_price_date,
            wo.market_value,
            wo.cost_value,
            wo.unrealized_pl,
            wo.unrealized_pl_pct,
            css.positive_count,
            css.negative_count,
            css.neutral_count,
            lpred.risk_score
        FROM watchlist_overview wo
        LEFT JOIN company_sentiment_summary css ON css.company_id = wo.company_id
        LEFT JOIN latest_predictions lpred ON lpred.company_id = wo.company_id
        WHERE wo.user_name = %s
        ORDER BY wo.market_value DESC NULLS LAST, wo.symbol ASC;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_name,))
            rows = cur.fetchall()

    holdings: list[dict[str, Any]] = []
    for row in rows:
        (
            watchlist_id, symbol, company_name, sector,
            shares, avg_cost_basis, purchased_at,
            latest_close, latest_price_date,
            market_value, cost_value, unrealized_pl, unrealized_pl_pct,
            positive_count, negative_count, neutral_count, risk_score,
        ) = row

        holdings.append({
            "watchlist_id": watchlist_id,
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "shares": float(shares) if shares is not None else 0.0,
            "avg_cost_basis": float(avg_cost_basis) if avg_cost_basis is not None else None,
            "purchased_at": purchased_at,
            "latest_close": float(latest_close) if latest_close is not None else None,
            "latest_price_date": latest_price_date,
            "market_value": float(market_value) if market_value is not None else None,
            "cost_value": float(cost_value) if cost_value is not None else None,
            "unrealized_pl": float(unrealized_pl) if unrealized_pl is not None else None,
            "unrealized_pl_pct": float(unrealized_pl_pct) if unrealized_pl_pct is not None else None,
            "is_position": bool(shares and shares > 0),
            "sentiment_score": _sentiment_score(positive_count, negative_count, neutral_count),
            "ml_risk_score": float(risk_score) if risk_score is not None else None,
        })

    return holdings


def _sentiment_score(
    positive_count: int | None,
    negative_count: int | None,
    neutral_count: int | None,
) -> float | None:
    """
    Reduce sentiment_scores counts to the same signed -100..100 scale used
    by the Company Detail page's Sentiment Score KPI (Phase 7), so the
    Portfolio page can show a consistent number per holding.

    Returns:
        100 * (positive - negative) / total, rounded to 1 decimal, or
        None if the company has no scored articles at all.
    """
    total = (positive_count or 0) + (negative_count or 0) + (neutral_count or 0)
    if total == 0:
        return None
    return round(((positive_count or 0) - (negative_count or 0)) / total * 100, 1)


def get_portfolio_summary(user_name: str) -> dict[str, Any]:
    """
    Aggregate a user's real positions (shares > 0) into portfolio-wide
    totals for the Portfolio page's KPI row.

    Args:
        user_name: The watchlist owner to summarize.

    Returns:
        A dict with:
            - position_count: number of real holdings (shares > 0).
            - watch_only_count: number of watch-only entries (shares = 0).
            - total_market_value: sum of market_value across positions.
            - total_cost_value: sum of cost_value across positions.
            - total_unrealized_pl: total_market_value - total_cost_value.
            - total_unrealized_pl_pct: total P&L as a % of total cost.
            - avg_sentiment_score: mean sentiment_score across positions
              that have scored news (None if none do).
            - avg_ml_risk_score: mean ml_risk_score across positions that
              have a prediction (None if none do).
        All numeric totals are 0/None-safe if the user has no positions
        yet (watch-only entries or an empty watchlist).
    """
    holdings = get_portfolio_holdings(user_name)
    positions = [h for h in holdings if h["is_position"]]

    total_market_value = sum(h["market_value"] or 0.0 for h in positions)
    total_cost_value = sum(h["cost_value"] or 0.0 for h in positions)
    total_unrealized_pl = total_market_value - total_cost_value

    total_unrealized_pl_pct = (
        round((total_unrealized_pl / total_cost_value) * 100, 2)
        if total_cost_value > 0
        else None
    )

    sentiment_values = [h["sentiment_score"] for h in positions if h["sentiment_score"] is not None]
    risk_values = [h["ml_risk_score"] for h in positions if h["ml_risk_score"] is not None]

    return {
        "position_count": len(positions),
        "watch_only_count": len(holdings) - len(positions),
        "total_market_value": total_market_value,
        "total_cost_value": total_cost_value,
        "total_unrealized_pl": total_unrealized_pl,
        "total_unrealized_pl_pct": total_unrealized_pl_pct,
        "avg_sentiment_score": round(sum(sentiment_values) / len(sentiment_values), 1) if sentiment_values else None,
        "avg_ml_risk_score": round(sum(risk_values) / len(risk_values), 4) if risk_values else None,
    }


def get_sector_concentration(user_name: str) -> list[dict[str, Any]]:
    """
    Break a user's real positions (shares > 0) down by sector, for the
    Portfolio page's sector concentration chart.

    Args:
        user_name: The watchlist owner to analyze.

    Returns:
        A list of dicts with sector, market_value, and pct_of_portfolio,
        sorted by market_value descending. Positions with no known sector
        (company not yet loaded via the pipeline) are grouped under
        "Unknown". Empty list if the user has no real positions.
    """
    holdings = get_portfolio_holdings(user_name)
    positions = [h for h in holdings if h["is_position"]]

    if not positions:
        return []

    total_market_value = sum(h["market_value"] or 0.0 for h in positions)

    sector_totals: dict[str, float] = {}
    for h in positions:
        sector = h["sector"] or "Unknown"
        sector_totals[sector] = sector_totals.get(sector, 0.0) + (h["market_value"] or 0.0)

    breakdown = [
        {
            "sector": sector,
            "market_value": value,
            "pct_of_portfolio": round((value / total_market_value) * 100, 2) if total_market_value > 0 else 0.0,
        }
        for sector, value in sector_totals.items()
    ]

    return sorted(breakdown, key=lambda item: item["market_value"], reverse=True)
