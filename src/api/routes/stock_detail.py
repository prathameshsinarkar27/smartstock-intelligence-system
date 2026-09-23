"""
stock_detail.py

Route for the Company Detail page (/stocks/{symbol}): company profile, 
KPI cards, an interactive Plotly price chart, and a historical price table. 
Sections for technical indicators, news & sentiment, ML predictions, 
and AI-generated insights are present.

"""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.analytics.technical_indicators import compute_all_indicators
from src.api.services.stock_detail_service import (
    ask_about_report,
    build_company_detail_page_data,
    get_price_history,
)
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
        context={**page_data, "rag_result": None},
    )


@router.post("/stocks/{symbol}/ask", response_class=HTMLResponse)
async def ask_about_report_route(request: Request, symbol: str, question: str = Form(...)):
    """
    Handle the "Ask About This Company's Report" panel's form submission
    (Phase 11's RAG system, first made reachable from the dashboard
    itself here — previously only the RAG CLI or the JSON API's
    POST /api/assistant/{symbol}/ask, Phase 13).

    Re-renders the full Company Detail page (same as the GET route)
    rather than redirecting, so the potentially-long answer text and its
    source citations don't need to round-trip through a URL query
    string — the trade-off is that submitting the form is a normal
    (non-AJAX) POST, so the page does a full reload with the answer
    included; acceptable for how infrequently a single question is asked
    compared to how often the page is just viewed.

    Args:
        request: Injected by FastAPI; required by Jinja2Templates.
        symbol: Stock ticker symbol from the URL path, e.g. "AAPL".
        question: The submitted question, from the panel's text field.

    Returns:
        The rendered stock_detail.html template, identical to a normal
        GET, with the additional rag_result context key populated (see
        ask_about_report()'s docstring for its shape) so the panel can
        show the answer, its sources, or an explanatory error message.
        A 404 (via not_found.html) if the symbol itself doesn't exist —
        same behavior as the GET route.
    """
    page_data = build_company_detail_page_data(symbol)

    if page_data is None:
        return templates.TemplateResponse(
            request=request,
            name="not_found.html",
            context={"symbol": symbol.upper()},
            status_code=404,
        )

    rag_result = ask_about_report(symbol, question)

    return templates.TemplateResponse(
        request=request,
        name="stock_detail.html",
        context={**page_data, "rag_result": rag_result},
    )


@router.get("/stocks/{symbol}/chart-data", response_class=JSONResponse)
async def company_chart_data(symbol: str):
    """
    Return this company's price history and technical indicators as JSON,
    for the page's own Plotly.js chart to fetch client-side after the
    page loads.

    Args:
        symbol: Stock ticker symbol from the URL path, e.g. "AAPL".

    Returns:
        A JSON object with:
            - dates, open, high, low, close, volume: parallel OHLCV
              arrays, the shape Plotly.js candlestick/line traces expect.
            - indicators: the full output of
              src/analytics/technical_indicators.py's compute_all_indicators(),
              i.e. sma_20/sma_50/ema_12/ema_26/rsi_14 arrays plus nested
              macd and bollinger dicts — each array the same length as
              dates, with null entries wherever that indicator isn't
              computable yet (not enough price history).
        All arrays are empty (and indicators' nested arrays are empty)
        if the symbol has no price history loaded (the company itself may
        still exist).
    """
    price_history = get_price_history(symbol)
    closes = [row["close"] for row in price_history]
    indicators = compute_all_indicators(closes) if closes else {
        "sma_20": [], "sma_50": [], "ema_12": [], "ema_26": [], "rsi_14": [],
        "macd": {"macd_line": [], "signal_line": [], "histogram": []},
        "bollinger": {"middle_band": [], "upper_band": [], "lower_band": []},
    }

    return {
        "dates": [row["date"].isoformat() for row in price_history],
        "open": [row["open"] for row in price_history],
        "high": [row["high"] for row in price_history],
        "low": [row["low"] for row in price_history],
        "close": [row["close"] for row in price_history],
        "volume": [row["volume"] for row in price_history],
        "indicators": indicators,
    }
