"""
stock_assistant.py

Orchestrates the AI Research Assistant (Phase 10): gathers a company's
existing structured data (price/fundamentals from
src.analytics.kpi_calculator, sentiment counts, and the Phase 8/9 ML
prediction + SHAP explanation), builds a prompt from it
(src/genai/prompts.py), and calls Gemini for a structured analysis
(src/genai/llm_utils.py).

"""

from datetime import date, datetime, timezone
from typing import Any

from src.analytics.kpi_calculator import get_company_kpis
from src.explainability.shap_analysis import explain_company_prediction
from src.genai.llm_utils import generate_structured_analysis
from src.genai.prompts import SYSTEM_INSTRUCTION, CompanyAnalysis, build_company_analysis_prompt
from src.ml.train_model import ModelNotTrainedError
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

DISCLAIMER_TEXT = (
    "AI-generated analysis for informational purposes only. Not financial advice."
)

# Cache populated lazily; keyed by (upper-case symbol, calendar date) so
# each company gets at most one Gemini call per day. See module docstring.
_cache: dict[tuple[str, date], dict[str, Any]] = {}


def clear_cache() -> None:
    """Drop all cached AI insights, forcing the next request to regenerate them."""
    _cache.clear()


def _fetch_latest_prediction(symbol: str) -> dict[str, Any] | None:
    """
    Fetch a company's latest trend prediction label (the one piece of
    Phase 8 output not already included in get_company_kpis()'s
    ml_risk_score field).

    Args:
        symbol: Stock ticker symbol.

    Returns:
        A dict with trend_prediction, or None if no prediction exists yet.
    """
    query = """
        SELECT trend_prediction
        FROM latest_predictions
        WHERE symbol = %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol.upper(),))
            row = cur.fetchone()

    if row is None:
        return None

    return {"trend_prediction": row[0]}


def _fetch_sentiment_counts(symbol: str) -> dict[str, int]:
    """
    Fetch a company's scored-article sentiment counts for the prompt's
    news sentiment section.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        A dict with positive_count, negative_count, neutral_count — all
        0 if the company has no scored articles yet.
    """
    query = """
        SELECT positive_count, negative_count, neutral_count
        FROM company_sentiment_summary
        WHERE symbol = %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol.upper(),))
            row = cur.fetchone()

    if row is None:
        return {"positive_count": 0, "negative_count": 0, "neutral_count": 0}

    positive_count, negative_count, neutral_count = row
    return {"positive_count": positive_count, "negative_count": negative_count, "neutral_count": neutral_count}


def _build_context(symbol: str) -> dict[str, Any] | None:
    """
    Gather everything build_company_analysis_prompt() needs for one
    company, from whichever of Phases 1-9's data happens to be available.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        None if the symbol isn't a tracked company at all (no
        companies-table row). Otherwise a context dict — always including
        company profile and price/fundamentals (from get_company_kpis(),
        Phase 5/6/7/8) and sentiment counts (Phase 7, all zero if none
        scored yet); ml_trend_prediction and ml_top_factors are included
        only if that data exists (a company with no ML prediction yet
        still gets a context dict, just a thinner one — the prompt
        renders "not available" for whatever's missing, and the assistant
        still has price/sentiment data to reason about).
    """
    kpis = get_company_kpis(symbol)
    if kpis is None:
        return None

    sentiment_counts = _fetch_sentiment_counts(symbol)

    context: dict[str, Any] = {
        "symbol": kpis["symbol"],
        "company_name": kpis["company_name"],
        "sector": kpis["sector"],
        "industry": kpis["industry"],
        "current_price": kpis["current_price"],
        "daily_change_pct": kpis["daily_change_pct"],
        "period_low": kpis["period_low"],
        "period_high": kpis["period_high"],
        "pe_ratio": kpis["pe_ratio"],
        "market_cap": kpis["market_cap"],
        "sentiment_score": kpis["sentiment_score"],
        "sentiment_positive_count": sentiment_counts["positive_count"],
        "sentiment_negative_count": sentiment_counts["negative_count"],
        "sentiment_neutral_count": sentiment_counts["neutral_count"],
        "ml_risk_score": kpis["ml_risk_score"],
    }

    prediction = _fetch_latest_prediction(symbol)
    if prediction is not None:
        context["ml_trend_prediction"] = prediction["trend_prediction"]

    try:
        explanation = explain_company_prediction(symbol)
        if explanation is not None:
            context["ml_top_factors"] = explanation["contributions"]
    except ModelNotTrainedError:
        # Models haven't been trained yet — proceed without SHAP factors
        # rather than blocking the whole assistant on Phase 8/9 setup.
        logger.info("No trained models available; building AI context for %s without SHAP factors.", symbol)

    return context


def get_company_ai_insight(symbol: str, force_refresh: bool = False) -> dict[str, Any] | None:
    """
    Get (generating and caching if needed) an AI-written analysis of one
    company, for the Company Detail page's AI Insights section and AI
    Recommendation KPI card.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        force_refresh: If True, bypass the cache and call Gemini again
            even if today's result is already cached.

    Returns:
        None if the symbol isn't a tracked company. Otherwise a dict with
        symbol, outlook ("Bullish"/"Bearish"/"Mixed"/"Cautious"), summary,
        key_considerations (a list of strings), and generated_at (a UTC
        datetime) — cached per (symbol, calendar day), so repeated calls
        on the same day return the same dict without a new API call.

    Raises:
        LLMConfigError: If GEMINI_API_KEY isn't set (src.genai.llm_utils).
        LLMRequestError: If the Gemini API call fails or its response
            can't be parsed (src.genai.llm_utils). Callers on the
            dashboard should catch this and render an empty state, the
            same pattern used for ModelNotTrainedError in Phase 8/9.
    """
    cache_key = (symbol.upper(), date.today())

    if not force_refresh and cache_key in _cache:
        return _cache[cache_key]

    context = _build_context(symbol)
    if context is None:
        return None

    prompt = build_company_analysis_prompt(context)
    analysis = generate_structured_analysis(prompt, SYSTEM_INSTRUCTION, CompanyAnalysis)

    result = {
        "symbol": symbol.upper(),
        "outlook": analysis.outlook,
        "summary": analysis.summary,
        "key_considerations": analysis.key_considerations,
        "generated_at": datetime.now(timezone.utc),
    }

    _cache[cache_key] = result
    return result
