"""
schemas.py

Pydantic models for the JSON API.

Defines request and response schemas used by FastAPI for
validation and OpenAPI documentation.
"""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# /api/stocks
# ---------------------------------------------------------------------

class StockSummary(BaseModel):
    """One stock summary returned by the stocks endpoint."""

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
    """Latest technical indicator values for a stock."""

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
    """Detailed stock information with technical indicators."""

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
    """One OHLCV price record."""

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
    """One news article returned by the news endpoint."""

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
    """Company fundamentals returned by the company endpoint."""

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
    """One scored news article."""

    title: str
    source: str | None = None
    published_date: datetime | date | None = None
    url: str | None = None
    sentiment: str
    confidence_score: float


class SentimentResponse(BaseModel):
    """Sentiment metrics and scored articles for a stock."""

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
    """Latest ML trend and risk prediction."""

    prediction_date: date
    trend_prediction: str
    risk_score: float


class SHAPContribution(BaseModel):
    """One feature's contribution to a model prediction."""

    feature: str
    contribution: float
    direction: str


class PredictionResponse(BaseModel):
    """ML prediction and optional SHAP explanation."""

    symbol: str
    prediction: MLPrediction | None = None
    explanation: dict[str, Any] | None = None


# ---------------------------------------------------------------------
# /api/assistant
# ---------------------------------------------------------------------

class AssistantInsight(BaseModel):
    """AI-generated research insight for a stock."""

    symbol: str
    outlook: str
    summary: str
    key_considerations: list[str] = Field(default_factory=list)
    generated_at: datetime
    disclaimer: str


class AssistantAskRequest(BaseModel):
    """Request body for report-based assistant queries."""

    question: str = Field(..., min_length=1, max_length=2000, description="A question about the company's ingested annual report(s).")
    top_k: int = Field(default=5, ge=1, le=20, description="How many report excerpts to retrieve as context.")


class AssistantAskSource(BaseModel):
    """One cited source from the RAG response."""

    source_file: str
    page: int | None = None


class AssistantAskResponse(BaseModel):
    """Response from the report-based assistant."""

    symbol: str
    question: str
    answer: str
    sources: list[AssistantAskSource] = Field(default_factory=list)


# ---------------------------------------------------------------------
# /api/portfolio
# ---------------------------------------------------------------------

class PortfolioHolding(BaseModel):
    """One portfolio holding or watchlist entry."""

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
    """Aggregated portfolio metrics."""

    position_count: int
    watch_only_count: int
    total_market_value: float
    total_cost_value: float
    total_unrealized_pl: float
    total_unrealized_pl_pct: float | None = None
    avg_sentiment_score: float | None = None
    avg_ml_risk_score: float | None = None


class SectorConcentration(BaseModel):
    """Portfolio allocation for one sector."""

    sector: str
    market_value: float
    pct_of_portfolio: float


class PortfolioResponse(BaseModel):
    """Portfolio holdings, summary, and sector breakdown."""

    user_name: str
    holdings: list[PortfolioHolding] = Field(default_factory=list)
    summary: PortfolioSummary
    sector_breakdown: list[SectorConcentration] = Field(default_factory=list)


class PortfolioHoldingRequest(BaseModel):
    """Request body for adding or updating a portfolio holding."""

    user_name: str = Field(..., min_length=1, max_length=100)
    symbol: str = Field(..., min_length=1, max_length=20)
    shares: float = Field(..., gt=0, description="Must be > 0 — use DELETE to close a position instead of setting shares to 0.")
    avg_cost_basis: float = Field(..., gt=0)
    purchased_at: date | None = None


class WatchlistRequest(BaseModel):
    """Request body for adding a watchlist symbol."""

    user_name: str = Field(..., min_length=1, max_length=100)
    symbol: str = Field(..., min_length=1, max_length=20)


class SimpleStatus(BaseModel):
    """Generic acknowledgement response."""

    status: str
    detail: str | None = None
