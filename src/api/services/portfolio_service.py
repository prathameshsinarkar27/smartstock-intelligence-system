"""
portfolio_service.py

Business logic for the Portfolio Analyzer page (src/api/routes/portfolio.py).

Follows the same split as overview_service.py / stock_detail_service.py:
read-side aggregation lives in src/analytics/portfolio_metrics.py and is
composed here into the exact shape the portfolio.html template needs.
Write operations (add/update/remove a holding) are simple single-table
upserts against `watchlist`, so — like overview_service.py's direct query
for recent news — they're implemented directly here rather than added to
the analytics module, which is read-only by convention.
"""

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from src.analytics.portfolio_metrics import (
    get_portfolio_holdings,
    get_portfolio_summary,
    get_sector_concentration,
)
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_USER = "default"


class PortfolioInputError(ValueError):
    """Raised when add/update holding form input fails validation."""


def build_portfolio_page_data(user_name: str | None) -> dict[str, Any]:
    """
    Assemble everything the Portfolio Analyzer page needs to render.

    Args:
        user_name: The watchlist owner to display. Falls back to
            DEFAULT_USER if blank/None, so the page always has something
            sensible to show on first visit.

    Returns:
        A dict with user_name, holdings, summary, and sector_breakdown —
        matching src/analytics/portfolio_metrics.py's return shapes.
        holdings/summary/sector_breakdown are all empty/zeroed (not
        errors) if the user has no watchlist entries yet.
    """
    resolved_user = (user_name or "").strip() or DEFAULT_USER

    return {
        "user_name": resolved_user,
        "holdings": get_portfolio_holdings(resolved_user),
        "summary": get_portfolio_summary(resolved_user),
        "sector_breakdown": get_sector_concentration(resolved_user),
    }


def _parse_decimal(raw: str, field_name: str) -> Decimal:
    """Parse a form field into a Decimal, raising PortfolioInputError with a clear message on failure."""
    try:
        return Decimal(raw.strip())
    except (InvalidOperation, AttributeError) as exc:
        raise PortfolioInputError(f"{field_name} must be a number, got {raw!r}.") from exc


def add_or_update_holding(
    user_name: str,
    symbol: str,
    shares: str,
    avg_cost_basis: str,
    purchased_at: str | None,
) -> None:
    """
    Insert a new position or update an existing one (matched on
    user_name + symbol), setting shares and avg_cost_basis. Used by the
    Portfolio page's "Add / Update Holding" form.

    Args:
        user_name: The watchlist owner.
        symbol: Stock ticker symbol (case-insensitive; stored uppercase).
        shares: Number of shares held, as a form string. Must be > 0 —
            use remove_holding() to close a position instead of setting
            shares to 0 here, since shares = 0 requires avg_cost_basis to
            be NULL (see the chk_watchlist_shares_cost_consistency
            constraint in database/tables.sql).
        avg_cost_basis: Average price paid per share, as a form string.
            Must be > 0.
        purchased_at: Optional ISO date string (YYYY-MM-DD). Blank/None
            is stored as NULL.

    Raises:
        PortfolioInputError: If symbol is blank, shares/avg_cost_basis
            aren't valid positive numbers, or purchased_at isn't a valid
            date.
    """
    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol:
        raise PortfolioInputError("Symbol is required.")

    shares_value = _parse_decimal(shares, "Shares")
    if shares_value <= 0:
        raise PortfolioInputError("Shares must be greater than 0. To close a position, remove it instead.")

    cost_value = _parse_decimal(avg_cost_basis, "Average cost basis")
    if cost_value <= 0:
        raise PortfolioInputError("Average cost basis must be greater than 0.")

    parsed_date: date | None = None
    if purchased_at and purchased_at.strip():
        try:
            parsed_date = date.fromisoformat(purchased_at.strip())
        except ValueError as exc:
            raise PortfolioInputError(f"Purchased date must be YYYY-MM-DD, got {purchased_at!r}.") from exc

    query = """
        INSERT INTO watchlist (user_name, symbol, shares, avg_cost_basis, purchased_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_name, symbol)
        DO UPDATE SET
            shares = EXCLUDED.shares,
            avg_cost_basis = EXCLUDED.avg_cost_basis,
            purchased_at = EXCLUDED.purchased_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_name, clean_symbol, shares_value, cost_value, parsed_date))

    logger.info("Upserted holding %s x %s shares for user '%s'.", clean_symbol, shares_value, user_name)


def add_watch_only(user_name: str, symbol: str) -> None:
    """
    Add a symbol to the watchlist with no position (shares = 0), i.e. the
    original Phase 0-11 "just watch this symbol" behavior, still
    available as a lighter-weight alternative to add_or_update_holding().

    Args:
        user_name: The watchlist owner.
        symbol: Stock ticker symbol (case-insensitive; stored uppercase).

    Raises:
        PortfolioInputError: If symbol is blank.
    """
    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol:
        raise PortfolioInputError("Symbol is required.")

    query = """
        INSERT INTO watchlist (user_name, symbol, shares, avg_cost_basis, purchased_at)
        VALUES (%s, %s, 0, NULL, NULL)
        ON CONFLICT (user_name, symbol) DO NOTHING;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_name, clean_symbol))

    logger.info("Added watch-only entry %s for user '%s'.", clean_symbol, user_name)


def remove_holding(user_name: str, symbol: str) -> None:
    """
    Remove a symbol from a user's watchlist/portfolio entirely (whether
    it was a real position or a watch-only entry).

    Args:
        user_name: The watchlist owner.
        symbol: Stock ticker symbol (case-insensitive) to remove.
    """
    clean_symbol = (symbol or "").strip().upper()

    query = "DELETE FROM watchlist WHERE user_name = %s AND symbol = %s;"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_name, clean_symbol))

    logger.info("Removed %s from user '%s' watchlist.", clean_symbol, user_name)
