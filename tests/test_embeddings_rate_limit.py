"""
test_embeddings_rate_limit.py

Tests for src/rag/embeddings.py's free-tier rate-limit handling: batch
sizing, RetryInfo-aware backoff, automatic retry-then-succeed behavior,
and giving up after exhausting retries. No live Gemini API calls are
made — src.genai.llm_utils.get_client() is mocked, and time.sleep is
patched out so these tests run instantly despite exercising real
backoff delays.
"""

from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import ClientError, ServerError

from src.genai.llm_utils import LLMRequestError
from src.rag import embeddings


def _make_429(retry_delay: str | None = "3s") -> ClientError:
    """Build a ClientError shaped like a real Gemini free-tier 429 response."""
    detail = {"@type": "type.googleapis.com/google.rpc.RetryInfo"}
    if retry_delay is not None:
        detail["retryDelay"] = retry_delay
    return ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded for embed_content_free_tier_requests.",
                "details": [detail],
            }
        },
    )


def _fake_embedding_result(n: int):
    result = MagicMock()
    result.embeddings = [MagicMock(values=[0.1, 0.2, 0.3]) for _ in range(n)]
    return result


def test_extract_retry_delay_seconds_parses_nested_retry_info():
    exc = _make_429(retry_delay="12.5s")
    assert embeddings._extract_retry_delay_seconds(exc) == 12.5


def test_extract_retry_delay_seconds_returns_none_when_absent():
    exc = _make_429(retry_delay=None)
    assert embeddings._extract_retry_delay_seconds(exc) is None


def test_seconds_until_retry_honors_server_delay():
    exc = _make_429(retry_delay="7s")
    # Server-provided delay + the module's small safety buffer.
    assert embeddings._seconds_until_retry(exc, attempt=1) == 7.5


def test_seconds_until_retry_falls_back_to_exponential_backoff():
    exc = _make_429(retry_delay=None)
    delay_attempt_1 = embeddings._seconds_until_retry(exc, attempt=1)
    delay_attempt_3 = embeddings._seconds_until_retry(exc, attempt=3)
    # Backoff grows with attempt number, and jitter keeps it non-negative.
    assert delay_attempt_1 >= embeddings._BACKOFF_BASE_SECONDS
    assert delay_attempt_3 > delay_attempt_1
    assert delay_attempt_3 <= embeddings._BACKOFF_MAX_SECONDS * 1.25


def test_is_rate_limit_error_distinguishes_429_from_other_errors():
    assert embeddings._is_rate_limit_error(_make_429()) is True

    other = ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "bad input"}})
    assert embeddings._is_rate_limit_error(other) is False


@patch("src.rag.embeddings.time.sleep", return_value=None)
@patch("src.rag.embeddings.get_client")
def test_embed_documents_retries_429_then_succeeds(mock_get_client, mock_sleep):
    """A 429 on the first attempt should be retried automatically and succeed on the second, not raise."""
    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = [
        _make_429(retry_delay="1s"),
        _fake_embedding_result(2),
    ]
    mock_get_client.return_value = mock_client

    vectors = embeddings.embed_documents(["chunk one", "chunk two"])

    assert len(vectors) == 2
    assert mock_client.models.embed_content.call_count == 2
    mock_sleep.assert_called_once()  # slept once for the single retry, then succeeded


@patch("src.rag.embeddings.time.sleep", return_value=None)
@patch("src.rag.embeddings.get_client")
def test_embed_documents_gives_up_after_max_retries(mock_get_client, mock_sleep):
    """Persistent 429s should eventually raise LLMRequestError rather than retrying forever."""
    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = _make_429(retry_delay="0.01s")
    mock_get_client.return_value = mock_client

    with pytest.raises(LLMRequestError, match="still rate-limited"):
        embeddings.embed_documents(["chunk one"])

    # Initial attempt + _MAX_RETRIES retries.
    assert mock_client.models.embed_content.call_count == embeddings._MAX_RETRIES + 1


@patch("src.rag.embeddings.time.sleep", return_value=None)
@patch("src.rag.embeddings.get_client")
def test_embed_documents_does_not_retry_non_rate_limit_errors(mock_get_client, mock_sleep):
    """A non-429 API error (e.g. a real server error) should fail fast, not consume retries."""
    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = ServerError(
        500, {"error": {"code": 500, "status": "INTERNAL", "message": "server exploded"}}
    )
    mock_get_client.return_value = mock_client

    with pytest.raises(LLMRequestError, match="Gemini embedding call failed"):
        embeddings.embed_documents(["chunk one"])

    mock_client.models.embed_content.assert_called_once()
    mock_sleep.assert_not_called()


@patch("src.rag.embeddings.time.sleep", return_value=None)
@patch("src.rag.embeddings.get_client")
def test_embed_documents_splits_into_configured_batch_size(mock_get_client, mock_sleep):
    """More texts than _BATCH_SIZE should result in multiple embed_content calls, not one oversized call."""
    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = [
        _fake_embedding_result(embeddings._BATCH_SIZE),
        _fake_embedding_result(3),
    ]
    mock_get_client.return_value = mock_client

    texts = [f"chunk {i}" for i in range(embeddings._BATCH_SIZE + 3)]
    vectors = embeddings.embed_documents(texts)

    assert len(vectors) == len(texts)
    assert mock_client.models.embed_content.call_count == 2
    first_call_batch = mock_client.models.embed_content.call_args_list[0].kwargs["contents"]
    assert len(first_call_batch) == embeddings._BATCH_SIZE


@patch("src.rag.embeddings.get_client")
def test_embed_documents_empty_input_makes_no_api_call(mock_get_client):
    assert embeddings.embed_documents([]) == []
    mock_get_client.assert_not_called()


def test_module_constants_fall_back_when_settings_missing_new_fields():
    """
    Simulates a stale/partial src/utils/config.py that hasn't been
    updated with the rag_embed_* fields (the AttributeError seen when
    only embeddings.py was copied over without also replacing
    config.py). Reloading embeddings.py against such a settings object
    should fall back to safe defaults instead of raising.
    """
    import importlib

    class StaleSettings:
        """Stands in for a Settings instance built before this fix, with none of the new fields."""

    with patch("src.utils.config.settings", StaleSettings()):
        reloaded = importlib.reload(embeddings)
        try:
            assert reloaded._BATCH_SIZE == 10
            assert reloaded._INTER_BATCH_DELAY_SECONDS == 1.0
            assert reloaded._MAX_RETRIES == 5
        finally:
            # Restore the module to its normal state (real settings) for
            # any tests that run after this one in the same session.
            importlib.reload(embeddings)
