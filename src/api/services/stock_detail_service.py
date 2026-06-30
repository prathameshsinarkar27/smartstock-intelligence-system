"""
stock_detail_service.py

Business logic for the Company Detail page (src/api/routes/stock_detail.py).

Composes data from src/analytics/kpi_calculator.py and a direct price
history query into the shape the Company Detail template (and its Plotly
chart) needs.
"""

from typing import Any

from src.analytics.kpi_calculator import get_company_kpis
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_price_history(symbol: str) -> list[dict[str, Any]]:
    """
    Fetch the full OHLCV price history for a single company, oldest first
    (the order a price chart and a "history table read top-to-bottom"
    both expect).

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".

    Returns:
        A list of dicts with date, open, high, low, close, volume, sorted
        ascending by date. Empty list if the symbol doesn't exist or has
        no price history loaded yet.
    """
    query = """
        SELECT hp.date, hp.open, hp.high, hp.low, hp.close, hp.volume
        FROM historical_prices hp
        JOIN companies c ON c.company_id = hp.company_id
        WHERE c.symbol = %s
        ORDER BY hp.date ASC;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol.upper(),))
            rows = cur.fetchall()

    return [
        {
            "date": row[0],
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": row[5],
        }
        for row in rows
    ]


def build_company_detail_page_data(symbol: str) -> dict[str, Any] | None:
    """
    Assemble everything the Company Detail template needs in one call.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".

    Returns:
        None if the symbol has no row in the companies table (the route
        should respond with a 404 in that case). Otherwise a dict with
        keys: kpis (from get_company_kpis) and price_history (oldest-first
        list, possibly empty if no price data has been loaded yet for an
        otherwise-valid company).
    """
    kpis = get_company_kpis(symbol)
    if kpis is None:
        return None

    return {
        "kpis": kpis,
        "price_history": get_price_history(symbol),
    }
