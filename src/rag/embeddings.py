"""
embeddings.py

Wraps Gemini's embedding API for the RAG pipeline, converting report
chunks and user queries into vectors for similarity search.

Uses the shared Gemini client from src.genai.llm_utils and supports
document/query-specific embedding task types with configurable batching
and rate-limit retries.
"""

import random
import re
import time

from google.genai import types
from google.genai.errors import APIError

from src.genai.llm_utils import LLMRequestError, get_client
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "gemini-embedding-001"

# 768 dimensions reduce vector-store size while retaining useful quality.
EMBEDDING_DIMENSIONS = 768

# Keep embedding requests small and predictable.
_BATCH_SIZE = getattr(settings, "rag_embed_batch_size", 10)

# Pause between batches to reduce rate-limit pressure.
_INTER_BATCH_DELAY_SECONDS = getattr(settings, "rag_embed_inter_batch_delay_seconds", 1.0)

# Maximum retries for a rate-limited batch.
_MAX_RETRIES = getattr(settings, "rag_embed_max_retries", 5)

# Exponential backoff base/cap used when the API's 429 response doesn't
# include a RetryInfo delay to honor directly (see _seconds_until_retry).
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_MAX_SECONDS = 60.0

_RETRY_DELAY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*s")


def _seconds_until_retry(exc: APIError, attempt: int) -> float:
    """
    Determine the delay before retrying a rate-limited request.

    Uses the server-provided RetryInfo delay when available, otherwise
    falls back to exponential backoff with jitter.
    """
    retry_delay = _extract_retry_delay_seconds(exc)
    if retry_delay is not None:
        # Add a small buffer for API latency and clock differences.
        return retry_delay + 0.5

    backoff = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
    jitter = random.uniform(0, backoff * 0.25)
    return backoff + jitter


def _extract_retry_delay_seconds(exc: APIError) -> float | None:
    """
    Extract a RetryInfo delay from a Gemini API error.

    Returns:
        Retry delay in seconds, or None if no valid delay is found.
    """
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None

    detail_entries = details.get("details") or details.get("error", {}).get("details") or []

    for entry in detail_entries:
        if not isinstance(entry, dict):
            continue
        entry_type = str(entry.get("@type", ""))
        if "RetryInfo" not in entry_type:
            continue
        retry_delay = entry.get("retryDelay")
        if not retry_delay:
            continue
        match = _RETRY_DELAY_PATTERN.match(str(retry_delay).strip())
        if match:
            return float(match.group(1))

    return None


def _is_rate_limit_error(exc: APIError) -> bool:
    """Return True when an API error represents a 429 response."""
    return getattr(exc, "code", None) == 429


def _embed_batch_with_retry(
    client, batch: list[str], config: "types.EmbedContentConfig", batch_label: str
) -> list[list[float]]:
    """
    Embed one batch with automatic retries for 429 responses.

    Args:
        client: Gemini client instance.
        batch: Texts to embed.
        config: Embedding configuration.
        batch_label: Label used in progress logs.

    Returns:
        Embedding vector for each input text.

    Raises:
        LLMRequestError: If embedding fails or retries are exhausted.
    """
    attempt = 0

    while True:
        try:
            result = client.models.embed_content(model=EMBEDDING_MODEL, contents=batch, config=config)
        except APIError as exc:
            if not _is_rate_limit_error(exc):
                raise LLMRequestError(f"Gemini embedding call failed for {batch_label}: {exc}") from exc

            if attempt >= _MAX_RETRIES:
                raise LLMRequestError(
                    f"Gemini embedding call for {batch_label} was still rate-limited "
                    f"after {_MAX_RETRIES} retries; giving up. Consider lowering "
                    f"RAG_EMBED_BATCH_SIZE or raising RAG_EMBED_INTER_BATCH_DELAY_SECONDS."
                ) from exc

            attempt += 1
            delay = _seconds_until_retry(exc, attempt)
            logger.warning(
                "Rate limited (429) embedding %s (%d text(s)); retry %d/%d in %.1fs.",
                batch_label, len(batch), attempt, _MAX_RETRIES, delay,
            )
            time.sleep(delay)
            continue
        except Exception as exc:
            raise LLMRequestError(f"Gemini embedding call failed for {batch_label}: {exc}") from exc

        if not result.embeddings or len(result.embeddings) != len(batch):
            raise LLMRequestError(
                f"Gemini embedding call for {batch_label} returned "
                f"{len(result.embeddings or [])} vector(s) for {len(batch)} input text(s)."
            )

        if attempt > 0:
            logger.info("Recovered from rate limiting for %s after %d retry(ies).", batch_label, attempt)

        return [embedding.values for embedding in result.embeddings]


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    """
    Embed texts in batches with automatic rate-limit retries.

    Args:
        texts: Texts to embed. Output order matches input order.
        task_type: "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY".

    Returns:
        Embedding vectors in input order.

    Raises:
        LLMConfigError: If the Gemini client is not configured.
        LLMRequestError: If any batch fails.
    """
    if not texts:
        return []

    client = get_client()
    config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS)

    total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE
    all_vectors: list[list[float]] = []

    for batch_index, batch_start in enumerate(range(0, len(texts), _BATCH_SIZE), start=1):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]
        batch_label = f"batch {batch_index}/{total_batches}"

        logger.info("Embedding %s (%d text(s), task_type=%s)...", batch_label, len(batch), task_type)
        all_vectors.extend(_embed_batch_with_retry(client, batch, config, batch_label))

        is_last_batch = batch_index == total_batches
        if _INTER_BATCH_DELAY_SECONDS > 0 and not is_last_batch:
            time.sleep(_INTER_BATCH_DELAY_SECONDS)

    logger.info("Finished embedding %d text(s) across %d batch(es).", len(texts), total_batches)
    return all_vectors


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed report chunks for vector-store storage.

    Args:
        texts: Report chunk texts.

    Returns:
        Embedding vectors in the same order as the input texts.

    Raises:
        LLMConfigError: If Gemini is not configured.
        LLMRequestError: If embedding fails after retries.
    """
    return _embed(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """
    Embed a user question for vector similarity search.

    Args:
        text: User's question.

    Returns:
        A single embedding vector.

    Raises:
        LLMConfigError: If Gemini is not configured.
        LLMRequestError: If embedding fails after retries.
    """
    vectors = _embed([text], task_type="RETRIEVAL_QUERY")
    return vectors[0]

