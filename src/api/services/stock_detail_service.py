"""
stock_detail_service.py

Business logic for the Company Detail page (src/api/routes/stock_detail.py).

Composes data from src/analytics/kpi_calculator.py,
src/analytics/technical_indicators.py, and a direct price history query
into the shape the Company Detail template (and its Plotly chart) needs.
"""

from typing import Any

from src.analytics.kpi_calculator import get_company_kpis
from src.analytics.technical_indicators import (
    compute_all_indicators,
    interpret_macd_crossover,
    interpret_rsi,
)
from src.explainability.shap_analysis import explain_company_prediction
from src.genai.llm_utils import LLMConfigError, LLMRequestError
from src.genai.stock_assistant import DISCLAIMER_TEXT, get_company_ai_insight
from src.ml.train_model import ModelNotTrainedError
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of individual articles shown in the Company Detail page's
# News & Sentiment section. The aggregate counts (positive/negative/
# neutral/total) below are computed over ALL scored articles, not just
# this many — only the per-article list is capped, to keep the page from
# growing unbounded for heavily-covered companies.
SENTIMENT_ARTICLE_LIMIT = 10


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


def get_latest_indicator_summary(price_history: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute the full technical indicator suite from a company's price
    history and reduce it to the latest values, for the numeric summary
    shown alongside the chart on the Company Detail page.

    Args:
        price_history: Output of get_price_history() — oldest-first list
            of OHLCV dicts.

    Returns:
        A dict with the latest value of each indicator (None if not
        enough price history exists yet to compute it), plus a
        human-readable signal label for RSI and MACD:
            sma_20, sma_50, ema_12, ema_26, rsi_14, rsi_signal,
            macd_line, macd_signal_line, macd_histogram, macd_signal,
            bollinger_upper, bollinger_middle, bollinger_lower
    """
    closes = [row["close"] for row in price_history]

    if not closes:
        return {
            "sma_20": None, "sma_50": None,
            "ema_12": None, "ema_26": None,
            "rsi_14": None, "rsi_signal": interpret_rsi(None),
            "macd_line": None, "macd_signal_line": None, "macd_histogram": None,
            "macd_signal": interpret_macd_crossover(None, None),
            "bollinger_upper": None, "bollinger_middle": None, "bollinger_lower": None,
        }

    indicators = compute_all_indicators(closes)

    latest_rsi = indicators["rsi_14"][-1]
    latest_macd_line = indicators["macd"]["macd_line"][-1]
    latest_macd_signal_line = indicators["macd"]["signal_line"][-1]

    return {
        "sma_20": indicators["sma_20"][-1],
        "sma_50": indicators["sma_50"][-1],
        "ema_12": indicators["ema_12"][-1],
        "ema_26": indicators["ema_26"][-1],
        "rsi_14": latest_rsi,
        "rsi_signal": interpret_rsi(latest_rsi),
        "macd_line": latest_macd_line,
        "macd_signal_line": latest_macd_signal_line,
        "macd_histogram": indicators["macd"]["histogram"][-1],
        "macd_signal": interpret_macd_crossover(latest_macd_line, latest_macd_signal_line),
        "bollinger_upper": indicators["bollinger"]["upper_band"][-1],
        "bollinger_middle": indicators["bollinger"]["middle_band"][-1],
        "bollinger_lower": indicators["bollinger"]["lower_band"][-1],
    }


def get_company_sentiment(symbol: str) -> dict[str, Any]:
    """
    Fetch aggregated sentiment counts plus a capped list of individual
    scored articles for a company, for the Company Detail page's News &
    Sentiment section (src/sentiment/sentiment_pipeline.py, Phase 7).

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".

    Returns:
        A dict with:
            - positive_count, negative_count, neutral_count,
              total_scored_articles, avg_confidence_score: aggregates
              over every scored article for this company (all 0/None if
              none have been scored yet).
            - articles: a list of up to SENTIMENT_ARTICLE_LIMIT dicts
              (title, source, published_date, url, sentiment,
              confidence_score), most-recently-published first. Empty
              list if none scored yet.
        Companies with news loaded but not yet scored (sentiment_pipeline
        hasn't run) and companies with no news loaded at all both produce
        this same "nothing scored" shape — the template distinguishes
        them, if needed, via the separately-available news presence, but
        both currently render the same empty state.
    """
    aggregate_query = """
        SELECT positive_count, negative_count, neutral_count,
               total_scored_articles, avg_confidence_score
        FROM company_sentiment_summary
        WHERE symbol = %s;
    """
    articles_query = """
        SELECT na.title, na.source, na.published_date, na.url,
               ss.sentiment, ss.confidence_score
        FROM news_articles na
        JOIN sentiment_scores ss ON ss.news_id = na.news_id
        JOIN companies c ON c.company_id = na.company_id
        WHERE c.symbol = %s
        ORDER BY na.published_date DESC NULLS LAST
        LIMIT %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(aggregate_query, (symbol.upper(),))
            aggregate_row = cur.fetchone()

            cur.execute(articles_query, (symbol.upper(), SENTIMENT_ARTICLE_LIMIT))
            article_rows = cur.fetchall()

    if aggregate_row is None:
        aggregates = {
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "total_scored_articles": 0,
            "avg_confidence_score": None,
        }
    else:
        positive_count, negative_count, neutral_count, total_scored_articles, avg_confidence_score = aggregate_row
        aggregates = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "total_scored_articles": total_scored_articles,
            "avg_confidence_score": float(avg_confidence_score) if avg_confidence_score is not None else None,
        }

    articles = [
        {
            "title": title,
            "source": source,
            "published_date": published_date,
            "url": url,
            "sentiment": sentiment,
            "confidence_score": float(confidence_score),
        }
        for title, source, published_date, url, sentiment, confidence_score in article_rows
    ]

    aggregates["articles"] = articles
    return aggregates


def get_company_ml_prediction(symbol: str) -> dict[str, Any] | None:
    """
    Fetch a company's most recent ML trend/risk prediction for the
    Company Detail page's ML Predictions section
    (src/ml/predict.py, Phase 8).

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".

    Returns:
        A dict with prediction_date, trend_prediction ("up"/"down"/
        "flat"), and risk_score (in [0, 1]) — sourced from the
        latest_predictions view (database/views.sql), which already
        keeps only each company's single most recent prediction row.
        None if predict.py hasn't produced a prediction for this company
        yet (distinct from an empty dict, so the template can tell "not
        predicted yet" apart from "predicted, values pending").
    """
    query = """
        SELECT prediction_date, trend_prediction, risk_score
        FROM latest_predictions
        WHERE symbol = %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol.upper(),))
            row = cur.fetchone()

    if row is None:
        return None

    prediction_date, trend_prediction, risk_score = row
    return {
        "prediction_date": prediction_date,
        "trend_prediction": trend_prediction,
        "risk_score": float(risk_score),
    }


def get_company_ml_explanation(symbol: str, top_n: int = 5) -> dict[str, Any] | None:
    """
    Explain a company's latest ML prediction with SHAP feature
    contributions toward the "down" class — the same class risk_score is
    defined over (src/explainability/shap_analysis.py, Phase 9).

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        top_n: How many top-contributing features to return.

    Returns:
        explain_company_prediction()'s output (target_class, symbol,
        prediction_date, contributions), or None if either: this company
        has no usable latest feature row (same condition
        get_company_ml_prediction handles), or the models haven't been
        trained yet at all (src/ml/train_model.py hasn't been run) — in
        which case this is logged, not raised, since a missing
        explanation should render as an empty state on the dashboard, not
        a 500 error.
    """
    try:
        return explain_company_prediction(symbol, top_n=top_n)
    except ModelNotTrainedError:
        logger.info("SHAP explanation unavailable for %s: models not trained yet.", symbol)
        return None


def get_company_ai_analysis(symbol: str) -> dict[str, Any] | None:
    """
    Get an AI-generated research summary for the Company Detail page's AI
    Insights section and AI Recommendation KPI card
    (src/genai/stock_assistant.py, Phase 10).

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".

    Returns:
        None if either: the symbol isn't a tracked company, or generating
        the analysis failed for a configuration or API reason (missing
        GEMINI_API_KEY, a failed/malformed Gemini response) — both are
        logged, not raised, so a missing AI analysis renders as an empty
        state on the dashboard rather than a 500, the same pattern used
        for ModelNotTrainedError in Phase 8/9. Otherwise
        get_company_ai_insight()'s output: symbol, outlook, summary,
        key_considerations, generated_at.
    """
    try:
        return get_company_ai_insight(symbol)
    except LLMConfigError as exc:
        logger.info("AI analysis unavailable for %s: %s", symbol, exc)
        return None
    except LLMRequestError as exc:
        logger.warning("AI analysis failed for %s: %s", symbol, exc)
        return None


def build_company_detail_page_data(symbol: str) -> dict[str, Any] | None:
    """
    Assemble everything the Company Detail template needs in one call.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".

    Returns:
        None if the symbol has no row in the companies table (the route
        should respond with a 404 in that case). Otherwise a dict with
        keys: kpis (from get_company_kpis), price_history (oldest-first
        list, possibly empty if no price data has been loaded yet for an
        otherwise-valid company), indicators (latest technical indicator
        values, from get_latest_indicator_summary() — all None if
        price_history is empty or too short for a given indicator's
        period), and sentiment (aggregate counts + article list, from
        get_company_sentiment() — Phase 7).

        Also includes ml_prediction: the latest trend/risk prediction
        from get_company_ml_prediction() (Phase 8), None if no prediction
        exists yet for this company. And ml_explanation: SHAP feature
        contributions for that same prediction (Phase 9, from
        get_company_ml_explanation()) — None if there's no prediction to
        explain, or if the models haven't been trained yet.

        Also includes ai_analysis: an AI-generated research summary
        (Phase 10, from get_company_ai_analysis()) — None if generating
        it failed (missing API key, API error) or the underlying data was
        too thin to build a context from.
    """
    kpis = get_company_kpis(symbol)
    if kpis is None:
        return None

    price_history = get_price_history(symbol)
    ml_prediction = get_company_ml_prediction(symbol)

    return {
        "kpis": kpis,
        "price_history": price_history,
        "indicators": get_latest_indicator_summary(price_history),
        "sentiment": get_company_sentiment(symbol),
        "ml_prediction": ml_prediction,
        # Only worth computing (rebuilds this company's full feature row)
        # if there's actually a prediction to explain.
        "ml_explanation": get_company_ml_explanation(symbol) if ml_prediction is not None else None,
        "ai_analysis": get_company_ai_analysis(symbol),
        "ai_disclaimer": DISCLAIMER_TEXT,
    }
