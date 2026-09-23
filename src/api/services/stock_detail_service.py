"""
stock_detail_service.py

Business logic for the Company Detail page.
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
from src.rag.rag_pipeline import answer_question
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of articles shown in the News & Sentiment section.
SENTIMENT_ARTICLE_LIMIT = 10


def get_price_history(symbol: str) -> list[dict[str, Any]]:
    """
    Fetch OHLCV price history for a company.

    Returns:
        Price records sorted by date, or an empty list if unavailable.
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
    Compute technical indicators and return their latest values.

    Returns:
        Latest indicator values and signal labels.
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
    Fetch sentiment metrics and recent scored articles for a company.
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
    Fetch the latest ML trend and risk prediction for a company.

    Returns:
        Prediction data, or None if unavailable.
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
    Generate a SHAP explanation for the latest ML prediction.

    Returns:
        SHAP explanation, or None if unavailable.
    """
    try:
        return explain_company_prediction(symbol, top_n=top_n)
    except ModelNotTrainedError:
        logger.info("SHAP explanation unavailable for %s: models not trained yet.", symbol)
        return None


def get_company_ai_analysis(symbol: str) -> dict[str, Any] | None:
    """
    Generate an AI research summary for a company.

    Returns:
        AI analysis, or None if unavailable.
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
    Assemble all data required by the Company Detail page.

    Returns:
        Company page data, or None if the company does not exist.
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


def ask_about_report(symbol: str, question: str, top_k: int = 5) -> dict[str, Any]:
    """
    Answer a question using the company's ingested annual reports.

    Returns:
        Answer, sources, and any error message.
    """
    if not question or not question.strip():
        return {"question": question, "answer": None, "sources": [], "error": "Enter a question first."}

    try:
        result = answer_question(symbol, question.strip(), top_k=top_k)
    except LLMConfigError:
        logger.warning("RAG question for %s skipped: Gemini not configured.", symbol)
        return {
            "question": question, "answer": None, "sources": [],
            "error": "AI assistant is not configured (GEMINI_API_KEY isn't set).",
        }
    except LLMRequestError as exc:
        logger.warning("RAG question for %s failed: %s", symbol, exc)
        return {
            "question": question, "answer": None, "sources": [],
            "error": "The AI request failed. Please try again.",
        }

    if result is None:
        return {
            "question": question, "answer": None, "sources": [],
            "error": f"No ingested report found for {symbol.upper()}. Ingest one first: "
            f"python -m src.rag.rag_pipeline ingest --symbol {symbol.upper()} --pdf <path>",
        }

    return {"question": result.question, "answer": result.answer, "sources": result.sources, "error": None}
