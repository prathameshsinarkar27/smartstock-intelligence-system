"""
test_stock_detail_rag_panel.py

Tests for the "Ask About This Company's Report" panel added to the
Company Detail page: src.api.services.stock_detail_service.ask_about_report()
and the new POST /stocks/{symbol}/ask route. 

Every Gemini/RAG call is mocked — this suite tests the panel's wiring
and graceful-degradation behavior (no report ingested, Gemini not
configured, Gemini call failed), not RAG's retrieval/generation logic
itself.
"""

import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("FINNHUB_API_KEY", "x")
os.environ.setdefault("NEWSAPI_API_KEY", "x")
os.environ.setdefault("TWELVEDATA_API_KEY", "x")
os.environ.setdefault("GEMINI_API_KEY", "x")

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.services.stock_detail_service import ask_about_report
from src.genai.llm_utils import LLMConfigError, LLMRequestError
from src.rag.rag_pipeline import RAGAnswer

client = TestClient(app)


# ---------------------------------------------------------------------
# ask_about_report() — service-level tests
# ---------------------------------------------------------------------

def test_ask_about_report_rejects_blank_question():
    with patch("src.api.services.stock_detail_service.answer_question") as mock_answer:
        result = ask_about_report("AAPL", "   ")

    assert result["answer"] is None
    assert result["error"] == "Enter a question first."
    mock_answer.assert_not_called()


def test_ask_about_report_handles_missing_gemini_config():
    with patch("src.api.services.stock_detail_service.answer_question", side_effect=LLMConfigError("no key")):
        result = ask_about_report("AAPL", "What was revenue?")

    assert result["answer"] is None
    assert "not configured" in result["error"]


def test_ask_about_report_handles_gemini_request_failure():
    with patch("src.api.services.stock_detail_service.answer_question", side_effect=LLMRequestError("timed out")):
        result = ask_about_report("AAPL", "What was revenue?")

    assert result["answer"] is None
    assert "failed" in result["error"]


def test_ask_about_report_handles_no_ingested_report():
    with patch("src.api.services.stock_detail_service.answer_question", return_value=None):
        result = ask_about_report("AAPL", "What was revenue?")

    assert result["answer"] is None
    assert "No ingested report found for AAPL" in result["error"]
    assert "ingest --symbol AAPL" in result["error"]


def test_ask_about_report_success():
    fake_answer = RAGAnswer(
        symbol="AAPL", question="What was FY25 revenue?",
        answer="FY25 revenue was $390B.",
        sources=[{"source_file": "aapl_10k.pdf", "page": 24}],
    )
    with patch("src.api.services.stock_detail_service.answer_question", return_value=fake_answer) as mock_answer:
        result = ask_about_report("AAPL", "What was FY25 revenue?", top_k=3)

    assert result["error"] is None
    assert result["answer"] == "FY25 revenue was $390B."
    assert result["sources"][0]["source_file"] == "aapl_10k.pdf"
    mock_answer.assert_called_once_with("AAPL", "What was FY25 revenue?", top_k=3)


# ---------------------------------------------------------------------
# Route-level tests
# ---------------------------------------------------------------------

_FAKE_PAGE_DATA = {
    "kpis": {
        "symbol": "AAPL", "company_name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
        "current_price": 200.0, "previous_close": 198.0, "daily_change_pct": 1.0, "market_cap": 3.0e12,
        "pe_ratio": 30.0, "volume": 1000, "period_high": 210.0, "period_low": 150.0,
        "period_start_date": None, "period_end_date": None, "sentiment_score": None,
        "ml_risk_score": None, "ai_recommendation": None,
    },
    "price_history": [],
    "indicators": {
        "sma_20": None, "sma_50": None, "ema_12": None, "ema_26": None, "rsi_14": None, "rsi_signal": None,
        "macd_line": None, "macd_signal_line": None, "macd_histogram": None, "macd_signal": None,
        "bollinger_upper": None, "bollinger_middle": None, "bollinger_lower": None,
    },
    "sentiment": {"positive_count": 0, "negative_count": 0, "neutral_count": 0, "total_scored_articles": 0,
                  "avg_confidence_score": None, "articles": []},
    "ml_prediction": None,
    "ml_explanation": None,
    "ai_analysis": None,
    "ai_disclaimer": "Not investment advice.",
}


def test_get_stock_detail_includes_empty_rag_result():
    """A normal page view shouldn't show any RAG answer/error until a question is actually submitted."""
    with patch("src.api.routes.stock_detail.build_company_detail_page_data", return_value=_FAKE_PAGE_DATA):
        response = client.get("/stocks/AAPL")

    assert response.status_code == 200
    assert "Ask About This Company's Report" in response.text


def test_post_ask_404_for_unknown_symbol():
    with patch("src.api.routes.stock_detail.build_company_detail_page_data", return_value=None):
        response = client.post("/stocks/NOPE/ask", data={"question": "anything"})

    assert response.status_code == 404


def test_post_ask_renders_error_message_when_no_report_ingested():
    with patch("src.api.routes.stock_detail.build_company_detail_page_data", return_value=_FAKE_PAGE_DATA), \
         patch("src.api.routes.stock_detail.ask_about_report", return_value={
             "question": "What was revenue?", "answer": None, "sources": [],
             "error": "No ingested report found for AAPL. Ingest one first: python -m src.rag.rag_pipeline ingest --symbol AAPL --pdf <path>",
         }):
        response = client.post("/stocks/AAPL/ask", data={"question": "What was revenue?"})

    assert response.status_code == 200
    assert "No ingested report found for AAPL" in response.text


def test_post_ask_renders_answer_and_sources_on_success():
    with patch("src.api.routes.stock_detail.build_company_detail_page_data", return_value=_FAKE_PAGE_DATA), \
         patch("src.api.routes.stock_detail.ask_about_report", return_value={
             "question": "What was FY25 revenue?",
             "answer": "FY25 revenue was $390B.",
             "sources": [{"source_file": "aapl_10k.pdf", "page": 24}],
             "error": None,
         }):
        response = client.post("/stocks/AAPL/ask", data={"question": "What was FY25 revenue?"})

    assert response.status_code == 200
    assert "FY25 revenue was $390B." in response.text
    assert "aapl_10k.pdf" in response.text
    assert "page 24" in response.text


def test_post_ask_requires_question_field():
    with patch("src.api.routes.stock_detail.build_company_detail_page_data", return_value=_FAKE_PAGE_DATA):
        response = client.post("/stocks/AAPL/ask", data={})

    assert response.status_code == 422  # FastAPI's Form(...) validation, no service call made
