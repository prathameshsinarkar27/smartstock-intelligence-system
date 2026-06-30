"""
stock_detail.py

Route for the Company Detail page (/stocks/{symbol}): company profile, 
KPI cards, an interactive Plotly price chart, and a historical price table. 
Sections for technical indicators, news & sentiment, ML predictions, 
and AI-generated insights are present.

"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.services.stock_detail_service import build_company_detail_page_data, get_price_history
from src.api.templating import templates
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/stocks/{symbol}", response_class=HTMLResponse)
async def company_detail(request: Request, symbol: str):
    """
    Render the Company Detail page for a single symbol.

    Args:
        request: Injected by FastAPI; required by Jinja2Templates.
        symbol: Stock ticker symbol from the URL path, e.g. "AAPL".

    Returns:
        The rendered stock_detail.html template on success. A simple
        404-styled HTML response (still using the same base layout, via
        the "not_found.html" template) if the symbol has no row in the
        companies table at all — as opposed to a known company with no
        price history yet, which renders normally with empty-state
        sections instead.
    """
    page_data = build_company_detail_page_data(symbol)

    if page_data is None:
        return templates.TemplateResponse(
            request=request,
            name="not_found.html",
            context={"symbol": symbol.upper()},
            status_code=404,
        )

    return templates.TemplateResponse(
        request=request,
        name="stock_detail.html",
        context=page_data,
    )


@router.get("/stocks/{symbol}/chart-data", response_class=JSONResponse)
async def company_chart_data(symbol: str):
    """
    Return this company's price history as JSON, for the page's own
    Plotly.js chart to fetch client-side after the page loads.

    Args:
        symbol: Stock ticker symbol from the URL path, e.g. "AAPL".

    Returns:
        A JSON object with parallel arrays: dates, open, high, low, close,
        volume — the shape Plotly.js candlestick/line traces expect.
        Empty arrays if the symbol has no price history loaded (the
        company itself may still exist).
    """
    price_history = get_price_history(symbol)

    return {
        "dates": [row["date"].isoformat() for row in price_history],
        "open": [row["open"] for row in price_history],
        "high": [row["high"] for row in price_history],
        "low": [row["low"] for row in price_history],
        "close": [row["close"] for row in price_history],
        "volume": [row["volume"] for row in price_history],
    }
