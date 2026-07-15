"""
test_api_routes.py

End-to-end tests for src/api/routes/api.py (Phase 13's JSON API), hitting
the real FastAPI routes through TestClient. Every boundary this module
would otherwise touch — Postgres (via get_connection) and the Gemini API
(via get_company_ai_insight / answer_question) — is mocked, per the
blueprint's rule that tests run against mocked DB/API boundaries when a
live database/API key isn't available in the build environment.

Run with:
    python -m pytest tests/test_api_routes.py -v
"""

import os
from contextlib import contextmanager
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

os.environ.setdefault("FINNHUB_API_KEY", "x")
os.environ.setdefault("NEWSAPI_API_KEY", "x")
os.environ.setdefault("TWELVEDATA_API_KEY", "x")
os.environ.setdefault("GEMINI_API_KEY", "x")

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.services.portfolio_service import PortfolioInputError
from src.genai.llm_utils import LLMConfigError, LLMRequestError
from src.rag.rag_pipeline import RAGAnswer

client = TestClient(app)


class FakeCursor:
    def __init__(self, row=None, rows=None):
        self._row = row
        self._rows = rows or []

    def execute(self, query, params=None):
        pass

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeConnection:
    def __init__(self, row=None, rows=None):
        self._row = row
        self._rows = rows

    def cursor(self):
        return FakeCursor(row=self._row, rows=self._rows)


def _known_symbol_connection():
    """A get_connection() replacement where `SELECT 1 FROM companies ...` finds a match."""
    @contextmanager
    def _cm():
        yield FakeConnection(row=(1,))
    return _cm


def _unknown_symbol_connection():
    """A get_connection() replacement where `SELECT 1 FROM companies ...` finds nothing."""
    @contextmanager
    def _cm():
        yield FakeConnection(row=None)
    return _cm


# ---------------------------------------------------------------------
# /api/stocks
# ---------------------------------------------------------------------

def test_list_stocks_returns_service_output():
    fake_companies = [{
        "symbol": "AAPL", "company_name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
        "market_cap": 3.0e12, "pe_ratio": 30.0, "current_price": 200.0, "daily_change_pct": 1.5,
        "sentiment_label": "positive",
    }]
    with patch("src.api.routes.api.get_filtered_companies", return_value=fake_companies):
        response = client.get("/api/stocks")

    assert response.status_code == 200
    assert response.json() == fake_companies


def test_get_stock_detail_404_for_unknown_symbol():
    with patch("src.api.routes.api.get_company_kpis", return_value=None):
        response = client.get("/api/stocks/NOPE")

    assert response.status_code == 404
    assert "NOPE" in response.json()["detail"]


def test_get_stock_detail_200_includes_indicators():
    fake_kpis = {
        "symbol": "AAPL", "company_name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
        "current_price": 200.0, "previous_close": 198.0, "daily_change_pct": 1.01,
        "market_cap": 3.0e12, "pe_ratio": 30.0, "volume": 50_000_000,
        "period_high": 210.0, "period_low": 150.0,
        "period_start_date": date(2026, 1, 1), "period_end_date": date(2026, 7, 1),
        "sentiment_score": 40.0, "ml_risk_score": 0.3, "ai_recommendation": None,
    }
    fake_indicators = {
        "sma_20": 195.0, "sma_50": 190.0, "ema_12": 196.0, "ema_26": 193.0,
        "rsi_14": 55.0, "rsi_signal": "Neutral",
        "macd_line": 1.2, "macd_signal_line": 1.0, "macd_histogram": 0.2, "macd_signal": "Bullish",
        "bollinger_upper": 205.0, "bollinger_middle": 198.0, "bollinger_lower": 191.0,
    }
    with patch("src.api.routes.api.get_company_kpis", return_value=fake_kpis), \
         patch("src.api.routes.api.get_price_history", return_value=[{"date": date(2026, 7, 1), "close": 200.0}]), \
         patch("src.api.routes.api.get_latest_indicator_summary", return_value=fake_indicators):
        response = client.get("/api/stocks/AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "AAPL"
    assert body["indicators"]["rsi_14"] == 55.0


def test_get_stock_prices_404_for_unknown_symbol():
    with patch("src.api.routes.api.get_connection", _unknown_symbol_connection()):
        response = client.get("/api/stocks/NOPE/prices")

    assert response.status_code == 404


def test_get_stock_prices_200():
    fake_prices = [{"date": "2026-07-01", "open": 198.0, "high": 201.0, "low": 197.5, "close": 200.0, "volume": 1000}]
    with patch("src.api.routes.api.get_connection", _known_symbol_connection()), \
         patch("src.api.routes.api.get_price_history", return_value=fake_prices):
        response = client.get("/api/stocks/AAPL/prices")

    assert response.status_code == 200
    assert response.json() == fake_prices


# ---------------------------------------------------------------------
# /api/news
# ---------------------------------------------------------------------

def test_list_news_no_symbol_filter():
    fake_news = [{
        "symbol": "AAPL", "company_name": "Apple Inc.", "title": "Apple beats estimates",
        "source": "Reuters", "published_date": "2026-07-01T09:00:00", "url": "https://example.com/a",
    }]
    with patch("src.api.routes.api.get_news", return_value=fake_news) as mock_get_news:
        response = client.get("/api/news")

    assert response.status_code == 200
    assert response.json() == fake_news
    mock_get_news.assert_called_once_with(symbol=None, limit=20)


def test_list_news_unknown_symbol_404():
    with patch("src.api.routes.api.get_connection", _unknown_symbol_connection()):
        response = client.get("/api/news?symbol=NOPE")

    assert response.status_code == 404


# ---------------------------------------------------------------------
# /api/company
# ---------------------------------------------------------------------

def test_get_company_profile_subset_of_kpis():
    fake_kpis = {
        "symbol": "AAPL", "company_name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
        "market_cap": 3.0e12, "pe_ratio": 30.0,
        "current_price": 200.0, "previous_close": 198.0, "daily_change_pct": 1.0,
        "volume": 1000, "period_high": 210.0, "period_low": 150.0,
        "period_start_date": None, "period_end_date": None,
        "sentiment_score": None, "ml_risk_score": None, "ai_recommendation": None,
    }
    with patch("src.api.routes.api.get_company_kpis", return_value=fake_kpis):
        response = client.get("/api/company/AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "symbol": "AAPL", "company_name": "Apple Inc.", "sector": "Technology",
        "industry": "Consumer Electronics", "market_cap": 3.0e12, "pe_ratio": 30.0,
    }


# ---------------------------------------------------------------------
# /api/sentiment
# ---------------------------------------------------------------------

def test_get_sentiment_404_for_unknown_symbol():
    with patch("src.api.routes.api.get_connection", _unknown_symbol_connection()):
        response = client.get("/api/sentiment/NOPE")

    assert response.status_code == 404


def test_get_sentiment_200():
    fake_sentiment = {
        "positive_count": 5, "negative_count": 2, "neutral_count": 1,
        "total_scored_articles": 8, "avg_confidence_score": 0.82, "articles": [],
    }
    with patch("src.api.routes.api.get_connection", _known_symbol_connection()), \
         patch("src.api.routes.api.get_company_sentiment", return_value=fake_sentiment):
        response = client.get("/api/sentiment/AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "AAPL"
    assert body["positive_count"] == 5


# ---------------------------------------------------------------------
# /api/predict
# ---------------------------------------------------------------------

def test_get_prediction_null_when_not_yet_predicted():
    with patch("src.api.routes.api.get_connection", _known_symbol_connection()), \
         patch("src.api.routes.api.get_company_ml_prediction", return_value=None):
        response = client.get("/api/predict/AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] is None
    assert body["explanation"] is None


def test_get_prediction_200_with_explanation():
    fake_prediction = {"prediction_date": "2026-07-01", "trend_prediction": "down", "risk_score": 0.72}
    fake_explanation = {"target_class": "down", "contributions": [{"feature": "rsi_14", "contribution": 0.12, "direction": "down"}]}
    with patch("src.api.routes.api.get_connection", _known_symbol_connection()), \
         patch("src.api.routes.api.get_company_ml_prediction", return_value=fake_prediction), \
         patch("src.api.routes.api.get_company_ml_explanation", return_value=fake_explanation):
        response = client.get("/api/predict/AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["trend_prediction"] == "down"
    assert body["explanation"]["contributions"][0]["feature"] == "rsi_14"


# ---------------------------------------------------------------------
# /api/assistant
# ---------------------------------------------------------------------

def test_get_assistant_insight_503_when_unconfigured():
    with patch("src.api.routes.api.get_connection", _known_symbol_connection()), \
         patch("src.api.routes.api.get_company_ai_insight", side_effect=LLMConfigError("GEMINI_API_KEY not set")):
        response = client.get("/api/assistant/AAPL")

    assert response.status_code == 503


def test_get_assistant_insight_502_on_request_failure():
    with patch("src.api.routes.api.get_connection", _known_symbol_connection()), \
         patch("src.api.routes.api.get_company_ai_insight", side_effect=LLMRequestError("Gemini call failed")):
        response = client.get("/api/assistant/AAPL")

    assert response.status_code == 502


def test_get_assistant_insight_200():
    fake_insight = {
        "symbol": "AAPL", "outlook": "Bullish", "summary": "Strong quarter.",
        "key_considerations": ["Revenue beat", "Margin expansion"],
        "generated_at": datetime.now(timezone.utc),
    }
    with patch("src.api.routes.api.get_connection", _known_symbol_connection()), \
         patch("src.api.routes.api.get_company_ai_insight", return_value=fake_insight):
        response = client.get("/api/assistant/AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body["outlook"] == "Bullish"
    assert "disclaimer" in body and body["disclaimer"]


def test_ask_assistant_404_when_no_report_ingested():
    with patch("src.api.routes.api.get_connection", _known_symbol_connection()), \
         patch("src.api.routes.api.answer_question", return_value=None):
        response = client.post("/api/assistant/AAPL/ask", json={"question": "What was FY25 revenue?"})

    assert response.status_code == 404


def test_ask_assistant_200():
    fake_answer = RAGAnswer(
        symbol="AAPL", question="What was FY25 revenue?",
        answer="FY25 revenue was $390B (see excerpt).",
        sources=[{"source_file": "aapl_10k.pdf", "page": 24}],
    )
    with patch("src.api.routes.api.get_connection", _known_symbol_connection()), \
         patch("src.api.routes.api.answer_question", return_value=fake_answer):
        response = client.post("/api/assistant/AAPL/ask", json={"question": "What was FY25 revenue?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("FY25 revenue")
    assert body["sources"][0]["source_file"] == "aapl_10k.pdf"


def test_ask_assistant_rejects_empty_question():
    response = client.post("/api/assistant/AAPL/ask", json={"question": ""})
    assert response.status_code == 422  # Pydantic min_length=1 validation, no DB/LLM call made


# ---------------------------------------------------------------------
# /api/portfolio
# ---------------------------------------------------------------------

def test_get_portfolio_defaults_to_default_user():
    with patch("src.api.routes.api.get_portfolio_holdings", return_value=[]) as mock_holdings, \
         patch("src.api.routes.api.get_portfolio_summary", return_value={
             "position_count": 0, "watch_only_count": 0, "total_market_value": 0.0, "total_cost_value": 0.0,
             "total_unrealized_pl": 0.0, "total_unrealized_pl_pct": None,
             "avg_sentiment_score": None, "avg_ml_risk_score": None,
         }), \
         patch("src.api.routes.api.get_sector_concentration", return_value=[]):
        response = client.get("/api/portfolio")

    assert response.status_code == 200
    assert response.json()["user_name"] == "default"
    mock_holdings.assert_called_once_with("default")


def test_upsert_holding_400_on_invalid_input():
    """Exercises the service-level PortfolioInputError path with a body that passes Pydantic's own schema validation."""
    with patch("src.api.routes.api.add_or_update_holding", side_effect=PortfolioInputError("Symbol is required.")):
        response = client.post("/api/portfolio/holdings", json={
            "user_name": "jane", "symbol": "AAPL", "shares": 10, "avg_cost_basis": 150.0,
        })

    assert response.status_code == 400
    assert "Symbol is required" in response.json()["detail"]


def test_upsert_holding_201_on_success():
    with patch("src.api.routes.api.add_or_update_holding", return_value=None) as mock_upsert:
        response = client.post("/api/portfolio/holdings", json={
            "user_name": "jane", "symbol": "aapl", "shares": 10, "avg_cost_basis": 150.0, "purchased_at": "2025-01-15",
        })

    assert response.status_code == 201
    assert response.json()["status"] == "ok"
    mock_upsert.assert_called_once_with(
        user_name="jane", symbol="aapl", shares="10.0", avg_cost_basis="150.0", purchased_at="2025-01-15",
    )


def test_upsert_holding_rejects_non_positive_shares_at_schema_level():
    response = client.post("/api/portfolio/holdings", json={
        "user_name": "jane", "symbol": "AAPL", "shares": 0, "avg_cost_basis": 150.0,
    })
    assert response.status_code == 422  # Pydantic gt=0, no service call made


def test_add_watch_only_201():
    with patch("src.api.routes.api.add_watch_only", return_value=None) as mock_add:
        response = client.post("/api/portfolio/watchlist", json={"user_name": "jane", "symbol": "tsla"})

    assert response.status_code == 201
    mock_add.assert_called_once_with(user_name="jane", symbol="tsla")


def test_delete_holding_200():
    with patch("src.api.routes.api.remove_holding", return_value=None) as mock_remove:
        response = client.delete("/api/portfolio/holdings/AAPL?user=jane")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    mock_remove.assert_called_once_with(user_name="jane", symbol="AAPL")
