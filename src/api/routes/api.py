"""
api.py

JSON API endpoints for stocks, news, sentiment, predictions,
AI assistant, and portfolio data.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from src.analytics.portfolio_metrics import (
    get_portfolio_holdings,
    get_portfolio_summary,
    get_sector_concentration,
)
from src.api.schemas import (
    AssistantAskRequest,
    AssistantAskResponse,
    AssistantInsight,
    CompanyProfile,
    NewsItem,
    PortfolioHoldingRequest,
    PortfolioResponse,
    PredictionResponse,
    PricePoint,
    SentimentResponse,
    SimpleStatus,
    StockDetail,
    StockSummary,
    WatchlistRequest,
)
from src.api.services.overview_service import get_filtered_companies, get_news
from src.api.services.portfolio_service import (
    DEFAULT_USER,
    PortfolioInputError,
    add_or_update_holding,
    add_watch_only,
    remove_holding,
)
from src.api.services.stock_detail_service import (
    get_company_ai_analysis,
    get_company_ml_explanation,
    get_company_ml_prediction,
    get_company_sentiment,
    get_latest_indicator_summary,
    get_price_history,
)
from src.analytics.kpi_calculator import get_company_kpis
from src.genai.llm_utils import LLMConfigError, LLMRequestError
from src.genai.stock_assistant import DISCLAIMER_TEXT, get_company_ai_insight
from src.rag.rag_pipeline import answer_question
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["JSON API"])


def _require_known_symbol(symbol: str) -> str:
    """
    Validate and normalize a stock symbol.

    Returns:
        Uppercase symbol.

    Raises:
        HTTPException: 404 if the symbol is unknown.
    """
    normalized = symbol.strip().upper()
    query = "SELECT 1 FROM companies WHERE symbol = %s;"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (normalized,))
            exists = cur.fetchone() is not None

    if not exists:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol!r}")

    return normalized


# ---------------------------------------------------------------------
# /api/stocks
# ---------------------------------------------------------------------

@router.get("/stocks", response_model=list[StockSummary])
async def list_stocks(
    sector: str | None = Query(default=None, description="Filter to a single exact sector name."),
    search: str | None = Query(default=None, description="Case-insensitive match against symbol or company name."),
) -> list[dict[str, Any]]:
    """List tracked companies with optional sector and search filters."""
    return get_filtered_companies(sector=sector, search=search)


@router.get("/stocks/{symbol}", response_model=StockDetail)
async def get_stock_detail(symbol: str) -> dict[str, Any]:
    """Return company KPIs and latest technical indicators."""
    kpis = get_company_kpis(symbol)
    if kpis is None:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol!r}")

    price_history = get_price_history(symbol)
    return {**kpis, "indicators": get_latest_indicator_summary(price_history)}


@router.get("/stocks/{symbol}/prices", response_model=list[PricePoint])
async def get_stock_prices(symbol: str) -> list[dict[str, Any]]:
    """Return full OHLCV price history for a company."""
    _require_known_symbol(symbol)
    return get_price_history(symbol)


# ---------------------------------------------------------------------
# /api/news
# ---------------------------------------------------------------------

@router.get("/news", response_model=list[NewsItem])
async def list_news(
    symbol: str | None = Query(default=None, description="Restrict to one company's news."),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Return recent news, optionally filtered by company."""
    if symbol is not None:
        _require_known_symbol(symbol)
    return get_news(symbol=symbol, limit=limit)


# ---------------------------------------------------------------------
# /api/company
# ---------------------------------------------------------------------

@router.get("/company/{symbol}", response_model=CompanyProfile)
async def get_company_profile(symbol: str) -> dict[str, Any]:
    """Return company fundamental information."""
    kpis = get_company_kpis(symbol)
    if kpis is None:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol!r}")

    return {
        "symbol": kpis["symbol"],
        "company_name": kpis["company_name"],
        "sector": kpis["sector"],
        "industry": kpis["industry"],
        "market_cap": kpis["market_cap"],
        "pe_ratio": kpis["pe_ratio"],
    }


# ---------------------------------------------------------------------
# /api/sentiment
# ---------------------------------------------------------------------

@router.get("/sentiment/{symbol}", response_model=SentimentResponse)
async def get_sentiment(symbol: str) -> dict[str, Any]:
    """Return sentiment metrics and recent scored articles."""
    normalized = _require_known_symbol(symbol)
    return {"symbol": normalized, **get_company_sentiment(normalized)}


# ---------------------------------------------------------------------
# /api/predict
# ---------------------------------------------------------------------

@router.get("/predict/{symbol}", response_model=PredictionResponse)
async def get_prediction(symbol: str) -> dict[str, Any]:
    """
    Return the latest ML prediction and SHAP explanation.

    Returns null prediction and explanation when no prediction exists.
    """
    normalized = _require_known_symbol(symbol)
    prediction = get_company_ml_prediction(normalized)
    explanation = get_company_ml_explanation(normalized) if prediction is not None else None
    return {"symbol": normalized, "prediction": prediction, "explanation": explanation}


# ---------------------------------------------------------------------
# /api/assistant
# ---------------------------------------------------------------------

@router.get("/assistant/{symbol}", response_model=AssistantInsight)
async def get_assistant_insight(
    symbol: str,
    refresh: bool = Query(default=False, description="Bypass today's cached analysis and regenerate it."),
) -> dict[str, Any]:
    """
    Return the latest AI-generated research summary for a company.

    Cached for the current day unless refresh is requested.
    """
    normalized = _require_known_symbol(symbol)

    try:
        insight = get_company_ai_insight(normalized, force_refresh=refresh)
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=f"AI assistant is not configured: {exc}") from exc
    except LLMRequestError as exc:
        raise HTTPException(status_code=502, detail=f"AI assistant request failed: {exc}") from exc

    if insight is None:
        # Keep a defensive 404 for concurrent data changes.
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol!r}")

    return {**insight, "disclaimer": DISCLAIMER_TEXT}


@router.post("/assistant/{symbol}/ask", response_model=AssistantAskResponse)
async def ask_assistant(symbol: str, body: AssistantAskRequest) -> dict[str, Any]:
    """
    Answer a question using the company's ingested annual reports.
    """
    normalized = _require_known_symbol(symbol)

    try:
        result = answer_question(normalized, body.question, top_k=body.top_k)
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=f"RAG assistant is not configured: {exc}") from exc
    except LLMRequestError as exc:
        raise HTTPException(status_code=502, detail=f"RAG assistant request failed: {exc}") from exc

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No ingested report found for {normalized!r}. Ingest one first: "
            f"python -m src.rag.rag_pipeline ingest --symbol {normalized} --pdf <path>",
        )

    return {
        "symbol": result.symbol,
        "question": result.question,
        "answer": result.answer,
        "sources": result.sources,
    }


# ---------------------------------------------------------------------
# /api/portfolio
# ---------------------------------------------------------------------

@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio(
    user: str = Query(default=DEFAULT_USER, description="Portfolio owner. No login system exists in this project."),
) -> dict[str, Any]:
    """Return portfolio holdings, summary, and sector breakdown."""
    resolved_user = (user or "").strip() or DEFAULT_USER
    return {
        "user_name": resolved_user,
        "holdings": get_portfolio_holdings(resolved_user),
        "summary": get_portfolio_summary(resolved_user),
        "sector_breakdown": get_sector_concentration(resolved_user),
    }


@router.post("/portfolio/holdings", response_model=SimpleStatus, status_code=201)
async def upsert_holding(body: PortfolioHoldingRequest) -> dict[str, str]:
    """Add or update a portfolio holding."""
    try:
        add_or_update_holding(
            user_name=body.user_name,
            symbol=body.symbol,
            shares=str(body.shares),
            avg_cost_basis=str(body.avg_cost_basis),
            purchased_at=body.purchased_at.isoformat() if body.purchased_at else None,
        )
    except PortfolioInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "ok", "detail": f"Saved {body.symbol.upper()} for {body.user_name}."}


@router.post("/portfolio/watchlist", response_model=SimpleStatus, status_code=201)
async def add_watch_only_entry(body: WatchlistRequest) -> dict[str, str]:
    """Add a symbol to the watchlist without a position."""
    try:
        add_watch_only(user_name=body.user_name, symbol=body.symbol)
    except PortfolioInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "ok", "detail": f"Watching {body.symbol.upper()} for {body.user_name}."}


@router.delete("/portfolio/holdings/{symbol}", response_model=SimpleStatus)
async def delete_holding(
    symbol: str,
    user: str = Query(default=DEFAULT_USER, description="Portfolio owner."),
) -> dict[str, str]:
    """Remove a symbol from the user's portfolio or watchlist."""
    resolved_user = (user or "").strip() or DEFAULT_USER
    remove_holding(user_name=resolved_user, symbol=symbol)
    return {"status": "ok", "detail": f"Removed {symbol.upper()} for {resolved_user}."}
