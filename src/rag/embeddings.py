"""
embeddings.py

Wraps the Gemini API's embedding endpoint (embedContent) for the RAG
pipeline: turning report chunks (src/rag/document_loader.py) into vectors
for storage (src/rag/vector_store.py), and turning a user's question into
a vector for similarity search.

This module does not import `google.genai` itself — it reuses
src.genai.llm_utils.get_client() (see that module's docstring), so there
remains exactly one place in the codebase that constructs a Gemini
client and validates GEMINI_API_KEY.

Two distinct task types matter here, and using the right one for each
side of the pipeline measurably improves retrieval quality: document
chunks are embedded with task_type="RETRIEVAL_DOCUMENT" (optimized for
being *found*), and a question is embedded with
task_type="RETRIEVAL_QUERY" (optimized for *finding* — Gemini's
embedding model produces asymmetric embeddings for retrieval, unlike a
single generic embedding used for both sides).

Rate limiting (added after initial Phase 11 delivery)
------------------------------------------------------
The Gemini free tier enforces low per-minute request/token quotas for
embedContent. A large annual report chunked at the default batch size of
100 could burn through the free-tier quota in a handful of calls and
fail outright with `429 RESOURCE_EXHAUSTED
(embed_content_free_tier_requests)`, aborting ingestion partway through
a document. This module now:
  1. Uses a much smaller default batch size (_BATCH_SIZE = 10) so each
     call is cheaper and a single rate-limit hit doesn't waste a large
     batch's worth of work.
  2. Adds a small fixed pause between successful batches
     (_INTER_BATCH_DELAY_SECONDS) to stay under the free tier's
     requests-per-minute ceiling in the common case, instead of only
     reacting after a 429.
  3. Automatically retries a 429 with exponential backoff, honoring the
     server's suggested `RetryInfo.retryDelay` when the API provides one
     (Gemini's 429 responses generally include this) instead of guessing.
  4. Logs progress per batch (batch N/M, chunk counts, retry attempts)
     so a long ingestion run's status is visible rather than silent.
  5. Only gives up (raising LLMRequestError) after _MAX_RETRIES
     consecutive failures on the *same* batch — a transient rate limit
     no longer aborts the whole document's ingestion the way it did
     before, and callers like src.rag.rag_pipeline.ingest_report() see
     ingestion simply take a bit longer rather than failing.

All of the above is configurable via environment variables so it can be
tuned per API tier without a code change (e.g. raise _BATCH_SIZE back up
and drop the inter-batch delay on a paid tier with higher quotas).
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

# gemini-embedding-001 defaults to 3072 dimensions; Matryoshka
# Representation Learning means truncating to a smaller officially-
# recommended size (768/1536/3072) costs little quality while
# significantly reducing ChromaDB's on-disk footprint and query cost for
# a project-scale document set.
EMBEDDING_DIMENSIONS = 768

# The embedContent endpoint accepts a batch of texts per call, but an
# unbounded batch risks hitting a request size, token, or (on the free
# tier especially) rate limit for a large report. Chunking client-side
# keeps each call small and predictable regardless of how many chunks a
# document produces. Overridable via RAG_EMBED_BATCH_SIZE for API tiers
# with higher quotas.
#
# Free-tier embedContent quotas are also tight on requests-per-minute;
# pausing briefly between successful batches (RAG_EMBED_INTER_BATCH_DELAY_SECONDS)
# keeps a multi-batch ingestion under that ceiling in the common case,
# rather than relying solely on 429 retries to pace things. And
# RAG_EMBED_MAX_RETRIES caps how many times a single batch is retried
# after a 429 before giving up on it.
#
# The three settings below were added to Settings alongside this fix. If
# something in the deployment environment still has a stale/partial
# src/utils/config.py (e.g. only embeddings.py was copied over without
# also replacing config.py, or a cached .pyc from before this fix is
# being picked up), getattr() falls back to the same defaults
# config.py itself uses, rather than crashing the whole
# `ingest`/`ingest-all` command on import with an AttributeError. If you
# hit these fallbacks, re-copy src/utils/config.py from this delivery
# and clear __pycache__/ (see docs/PHASE_11_SETUP_GUIDE.md's
# Troubleshooting table).
_BATCH_SIZE = getattr(settings, "rag_embed_batch_size", 10)
_INTER_BATCH_DELAY_SECONDS = getattr(settings, "rag_embed_inter_batch_delay_seconds", 1.0)
_MAX_RETRIES = getattr(settings, "rag_embed_max_retries", 5)

# Exponential backoff base/cap used when the API's 429 response doesn't
# include a RetryInfo delay to honor directly (see _seconds_until_retry).
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_MAX_SECONDS = 60.0

_RETRY_DELAY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*s")


def _seconds_until_retry(exc: APIError, attempt: int) -> float:
    """
    Determine how long to wait before retrying a rate-limited batch.

    Prefers the server's own suggested delay (Gemini 429 responses
    include a `RetryInfo.retryDelay` such as "38s" in `error.details`)
    over a locally-guessed backoff, since the server knows its own quota
    reset window better than a fixed formula would. Falls back to
    exponential backoff with jitter if no RetryInfo is present (or it
    can't be parsed) — capped at _BACKOFF_MAX_SECONDS so a bug in a
    future API response format can't stall ingestion for an
    unreasonable amount of time.

    Args:
        exc: The APIError raised for the rate-limited call.
        attempt: Which retry attempt this is (1-indexed), used only for
            the exponential-backoff fallback.

    Returns:
        Seconds to sleep before retrying.
    """
    retry_delay = _extract_retry_delay_seconds(exc)
    if retry_delay is not None:
        # Small buffer on top of the server's own suggested delay, since
        # "retry after exactly N seconds" run right at the boundary
        # tends to still get rate-limited by a few hundred milliseconds
        # of clock skew/latency.
        return retry_delay + 0.5

    backoff = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
    jitter = random.uniform(0, backoff * 0.25)
    return backoff + jitter


def _extract_retry_delay_seconds(exc: APIError) -> float | None:
    """
    Pull a RetryInfo `retryDelay` (e.g. "38s") out of a Gemini APIError's
    details, if present.

    Args:
        exc: The APIError to inspect. `exc.details` is the parsed JSON
            error body (see google.genai.errors.APIError.__init__);
            structure is generally
            `{"details": [{"@type": ".../google.rpc.RetryInfo", "retryDelay": "38s"}, ...]}`
            but is treated defensively here since it's an undocumented
            part of the error body, not a stable public schema.

    Returns:
        The delay in seconds, or None if no parseable RetryInfo was
        found.
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
    """True if an APIError represents a 429 rate-limit response (as opposed to any other 4xx/5xx)."""
    return getattr(exc, "code", None) == 429


def _embed_batch_with_retry(
    client, batch: list[str], config: "types.EmbedContentConfig", batch_label: str
) -> list[list[float]]:
    """
    Embed a single batch of texts, automatically retrying on 429
    RESOURCE_EXHAUSTED with backoff up to _MAX_RETRIES times.

    Args:
        client: A Gemini client from src.genai.llm_utils.get_client().
        batch: Texts for this batch only (already sized to _BATCH_SIZE
            or smaller by the caller).
        config: The EmbedContentConfig (task type + output dimensions)
            shared across all batches of this embed_documents()/
            embed_query() call.
        batch_label: A human-readable label like "batch 3/12" used in
            log messages so a multi-batch ingestion's progress is
            visible.

    Returns:
        One embedding vector per input text, in order.

    Raises:
        LLMRequestError: If the batch fails for a non-rate-limit reason,
            or is still rate-limited after _MAX_RETRIES retries.
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
    Embed a batch of texts with a given Gemini task type, in
    _BATCH_SIZE-sized sub-batches, automatically retrying any sub-batch
    that hits a free-tier rate limit rather than aborting the whole call.

    Args:
        texts: Texts to embed. Order is preserved in the output.
        task_type: "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY".

    Returns:
        A list of embedding vectors, one per input text, in the same order.

    Raises:
        LLMConfigError: If GEMINI_API_KEY isn't set.
        LLMRequestError: If any batch fails for a non-rate-limit reason,
            or remains rate-limited after exhausting its retries.
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
    Embed a batch of report chunk texts for storage in the vector store.

    Automatically retries individual sub-batches on a Gemini free-tier
    429 (RESOURCE_EXHAUSTED) with backoff, so a rate limit hit partway
    through a large report slows ingestion down rather than aborting it.

    Args:
        texts: Chunk texts (typically ReportChunk.text values from
            src.rag.document_loader).

    Returns:
        A list of embedding vectors, one per input text, in the same order.

    Raises:
        LLMConfigError: If GEMINI_API_KEY isn't set.
        LLMRequestError: If a batch fails for a non-rate-limit reason, or
            remains rate-limited after exhausting RAG_EMBED_MAX_RETRIES
            retries.
    """
    return _embed(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """
    Embed a single user question for similarity search against the
    vector store.

    Args:
        text: The user's question.

    Returns:
        A single embedding vector.

    Raises:
        LLMConfigError: If GEMINI_API_KEY isn't set.
        LLMRequestError: If the embedding call fails, or remains
            rate-limited after exhausting RAG_EMBED_MAX_RETRIES retries.
    """
    vectors = _embed([text], task_type="RETRIEVAL_QUERY")
    return vectors[0]

