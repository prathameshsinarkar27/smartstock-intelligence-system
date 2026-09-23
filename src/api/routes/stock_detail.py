"""
stock_detail.py

Routes for the Company Detail page, including company information,
price charts, technical indicators, and report Q&A.
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
    """Render the Company Detail page for a stock symbol."""
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
    Answer a report question and re-render the Company Detail page.

    Returns a 404 page if the stock symbol is not found.
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
    Return price history and technical indicators as JSON.

    Returns empty arrays when no price history is available.
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
