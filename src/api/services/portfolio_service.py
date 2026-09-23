"""
portfolio_service.py

Business logic for the Portfolio Analyzer page.
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
    """Raised when portfolio form input is invalid."""


def build_portfolio_page_data(user_name: str | None) -> dict[str, Any]:
    """
    Build the data required by the Portfolio Analyzer page.

    Args:
        user_name: Portfolio owner. Defaults to DEFAULT_USER.

    Returns:
        Portfolio user, holdings, summary, and sector breakdown.
    """
    resolved_user = (user_name or "").strip() or DEFAULT_USER

    return {
        "user_name": resolved_user,
        "holdings": get_portfolio_holdings(resolved_user),
        "summary": get_portfolio_summary(resolved_user),
        "sector_breakdown": get_sector_concentration(resolved_user),
    }


def _parse_decimal(raw: str, field_name: str) -> Decimal:
    """Parse a form value as Decimal."""
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
    Add or update a portfolio holding.

    Raises:
        PortfolioInputError: If any input is invalid.
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
    Add a symbol to the watchlist without a position.

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
    Remove a symbol from a user's watchlist or portfolio.
    """
    clean_symbol = (symbol or "").strip().upper()

    query = "DELETE FROM watchlist WHERE user_name = %s AND symbol = %s;"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_name, clean_symbol))

    logger.info("Removed %s from user '%s' watchlist.", clean_symbol, user_name)
