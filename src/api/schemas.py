"""
schemas.py

Pydantic models for Phase 13's JSON API (src/api/routes/api.py).

These are response/request shapes only — they don't duplicate business
logic. Every field here mirrors a key already returned by an existing
service function (src/api/services/*.py, src/analytics/*.py) or RAGAnswer
(src/rag/rag_pipeline.py); this module exists so FastAPI can validate
request bodies and generate accurate OpenAPI docs (visible at /docs) for
API consumers who aren't using the HTML dashboard.

The HTML dashboard's routes (overview.py, stock_detail.py, portfolio.py)
intentionally do NOT use these models — they return TemplateResponse, not
JSON, so typed schemas would add no value there. Only api.py uses them.
"""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# /api/stocks
# ---------------------------------------------------------------------

class StockSummary(BaseModel):
    """One row of GET /api/stocks — mirrors overview_service.get_filtered_companies()."""

    symbol: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    current_price: float | None = None
    daily_change_pct: float | None = None
    sentiment_label: str | None = None


class TechnicalIndicators(BaseModel):
    """Latest technical indicator values — mirrors stock_detail_service.get_latest_indicator_summary()."""

    sma_20: float | None = None
    sma_50: float | None = None
    ema_12: float | None = None
    ema_26: float | None = None
    rsi_14: float | None = None
    rsi_signal: str | None = None
    macd_line: float | None = None
    macd_signal_line: float | None = None
    macd_histogram: float | None = None
    macd_signal: str | None = None
    bollinger_upper: float | None = None
    bollinger_middle: float | None = None
    bollinger_lower: float | None = None


class StockDetail(BaseModel):
    """GET /api/stocks/{symbol} — mirrors kpi_calculator.get_company_kpis() + technical indicators."""

    symbol: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    current_price: float | None = None
    previous_close: float | None = None
    daily_change_pct: float | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    volume: int | None = None
    period_high: float | None = None
    period_low: float | None = None
    period_start_date: date | None = None
    period_end_date: date | None = None
    sentiment_score: float | None = None
    ml_risk_score: float | None = None
    indicators: TechnicalIndicators


class PricePoint(BaseModel):
    """One row of GET /api/stocks/{symbol}/prices — mirrors stock_detail_service.get_price_history()."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


# ---------------------------------------------------------------------
# /api/news
# ---------------------------------------------------------------------

class NewsItem(BaseModel):
    """One row of GET /api/news — mirrors overview_service.get_news()."""

    symbol: str
    company_name: str | None = None
    title: str
    source: str | None = None
    published_date: datetime | date | None = None
    url: str | None = None


# ---------------------------------------------------------------------
# /api/company
# ---------------------------------------------------------------------

class CompanyProfile(BaseModel):
    """GET /api/company/{symbol} — company fundamentals only, a subset of StockDetail."""

    symbol: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None


# ---------------------------------------------------------------------
# /api/sentiment
# ---------------------------------------------------------------------

class SentimentArticle(BaseModel):
    """One scored article — mirrors stock_detail_service.get_company_sentiment()'s "articles" list."""

    title: str
    source: str | None = None
    published_date: datetime | date | None = None
    url: str | None = None
    sentiment: str
    confidence_score: float


class SentimentResponse(BaseModel):
    """GET /api/sentiment/{symbol} — mirrors stock_detail_service.get_company_sentiment()."""

    symbol: str
    positive_count: int
    negative_count: int
    neutral_count: int
    total_scored_articles: int
    avg_confidence_score: float | None = None
    articles: list[SentimentArticle] = Field(default_factory=list)


# ---------------------------------------------------------------------
# /api/predict
# ---------------------------------------------------------------------

class MLPrediction(BaseModel):
    """Mirrors stock_detail_service.get_company_ml_prediction()."""

    prediction_date: date
    trend_prediction: str
    risk_score: float


class SHAPContribution(BaseModel):
    """One feature's contribution — matches src.explainability.shap_analysis's per-feature output shape."""

    feature: str
    contribution: float
    direction: str


class PredictionResponse(BaseModel):
    """GET /api/predict/{symbol} — mirrors stock_detail_service.get_company_ml_prediction()/get_company_ml_explanation()."""

    symbol: str
    prediction: MLPrediction | None = None
    explanation: dict[str, Any] | None = None


# ---------------------------------------------------------------------
# /api/assistant
# ---------------------------------------------------------------------

class AssistantInsight(BaseModel):
    """GET /api/assistant/{symbol} — mirrors src.genai.stock_assistant.get_company_ai_insight()."""

    symbol: str
    outlook: str
    summary: str
    key_considerations: list[str] = Field(default_factory=list)
    generated_at: datetime
    disclaimer: str


class AssistantAskRequest(BaseModel):
    """Request body for POST /api/assistant/{symbol}/ask."""

    question: str = Field(..., min_length=1, max_length=2000, description="A question about the company's ingested annual report(s).")
    top_k: int = Field(default=5, ge=1, le=20, description="How many report excerpts to retrieve as context.")


class AssistantAskSource(BaseModel):
    """One cited source excerpt — mirrors a RAGAnswer.sources entry."""

    source_file: str
    page: int | None = None


class AssistantAskResponse(BaseModel):
    """Response body for POST /api/assistant/{symbol}/ask — mirrors src.rag.rag_pipeline.RAGAnswer."""

    symbol: str
    question: str
    answer: str
    sources: list[AssistantAskSource] = Field(default_factory=list)


# ---------------------------------------------------------------------
# /api/portfolio
# ---------------------------------------------------------------------

class PortfolioHolding(BaseModel):
    """One row of GET /api/portfolio — mirrors portfolio_metrics.get_portfolio_holdings()."""

    watchlist_id: int
    symbol: str
    company_name: str | None = None
    sector: str | None = None
    shares: float
    avg_cost_basis: float | None = None
    purchased_at: date | None = None
    latest_close: float | None = None
    latest_price_date: date | None = None
    market_value: float | None = None
    cost_value: float | None = None
    unrealized_pl: float | None = None
    unrealized_pl_pct: float | None = None
    is_position: bool
    sentiment_score: float | None = None
    ml_risk_score: float | None = None


class PortfolioSummary(BaseModel):
    """Mirrors portfolio_metrics.get_portfolio_summary()."""

    position_count: int
    watch_only_count: int
    total_market_value: float
    total_cost_value: float
    total_unrealized_pl: float
    total_unrealized_pl_pct: float | None = None
    avg_sentiment_score: float | None = None
    avg_ml_risk_score: float | None = None


class SectorConcentration(BaseModel):
    """One row of portfolio_metrics.get_sector_concentration()."""

    sector: str
    market_value: float
    pct_of_portfolio: float


class PortfolioResponse(BaseModel):
    """GET /api/portfolio."""

    user_name: str
    holdings: list[PortfolioHolding] = Field(default_factory=list)
    summary: PortfolioSummary
    sector_breakdown: list[SectorConcentration] = Field(default_factory=list)


class PortfolioHoldingRequest(BaseModel):
    """Request body for POST /api/portfolio/holdings."""

    user_name: str = Field(..., min_length=1, max_length=100)
    symbol: str = Field(..., min_length=1, max_length=20)
    shares: float = Field(..., gt=0, description="Must be > 0 — use DELETE to close a position instead of setting shares to 0.")
    avg_cost_basis: float = Field(..., gt=0)
    purchased_at: date | None = None


class WatchlistRequest(BaseModel):
    """Request body for POST /api/portfolio/watchlist."""

    user_name: str = Field(..., min_length=1, max_length=100)
    symbol: str = Field(..., min_length=1, max_length=20)


class SimpleStatus(BaseModel):
    """Generic acknowledgement response for write endpoints that don't return a resource body."""

    status: str
    detail: str | None = None
