"""
llm_utils.py

Low-level wrapper around the Gemini API (Google Gen AI SDK, the
`google-genai` package). This is the only module in the codebase that
should import `google.genai` directly, mirroring how src/utils/config.py
is the only module that reads environment variables directly — every
other module calls generate_structured_analysis() or generate_text()
instead of touching the SDK itself. (src/rag/embeddings.py, added in
Phase 11, reuses this module's get_client() for embedding calls rather
than importing google.genai a second time, for the same reason.)

Structured output: rather than asking the model for free-form text and
parsing it with regex, generate_structured_analysis() requests JSON
output constrained to a caller-supplied Pydantic schema (Gemini's
response_schema / response_mime_type="application/json" feature). The
response is still parsed defensively from response.text with
json.loads() rather than relying on any SDK auto-parsing convenience,
since the exact response object returned by the API is out of this
codebase's control and a manual, explicit parse is easier to reason
about and to test against a mocked client.

"""

import json
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TEMPERATURE = 0.4

# Deliberately generous: on Gemini 2.5 models, "thinking" tokens are
# deducted from this same budget before the visible answer is generated
# (see module docstring). A tight limit here risks the model exhausting
# the whole budget on thinking alone and returning a truncated (or fully
# empty) response.
DEFAULT_MAX_OUTPUT_TOKENS = 2048

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Cached client, built lazily on first use (see get_client()).
_client: genai.Client | None = None


class LLMConfigError(Exception):
    """Raised when the Gemini API key is missing or otherwise misconfigured."""


class LLMRequestError(Exception):
    """Raised when a Gemini API call fails, or its response can't be parsed."""


def get_client() -> genai.Client:
    """
    Build (on first call) and return a cached Gemini API client.

    Returns:
        A genai.Client configured with settings.gemini_api_key.

    Raises:
        LLMConfigError: If GEMINI_API_KEY isn't set (settings.gemini_api_key
            is empty) — checked explicitly here, rather than letting the
            SDK fail with a less clear error later, so callers get an
            actionable message.
    """
    global _client

    if _client is None:
        if not settings.gemini_api_key:
            raise LLMConfigError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/app/apikey and add it to your .env file."
            )
        _client = genai.Client(api_key=settings.gemini_api_key)
        logger.info("Initialized Gemini API client (model=%s).", settings.gemini_model)

    return _client


def reset_client() -> None:
    """
    Drop the cached client, forcing the next get_client() call to rebuild
    it. Used in tests, and after changing GEMINI_API_KEY at runtime.
    """
    global _client
    _client = None


def _build_thinking_config(model_name: str) -> types.ThinkingConfig | None:
    """
    Build a ThinkingConfig that disables "thinking" for models that
    support fully disabling it, to avoid thinking tokens silently eating
    into max_output_tokens (see module docstring).

    Args:
        model_name: The Gemini model name being called (settings.gemini_model).

    Returns:
        types.ThinkingConfig(thinking_budget=0) for Flash-family models,
        which support disabling thinking entirely. None for other models
        (e.g. Pro-family, which requires a non-zero thinking budget and
        would reject or ignore a request to disable it) — those are left
        on the API's own default dynamic budget instead.
    """
    if "flash" in model_name.lower():
        return types.ThinkingConfig(thinking_budget=0)
    return None


def _check_finish_reason(response: types.GenerateContentResponse) -> None:
    """
    Raise a clear, specific error if a response was cut off by hitting
    max_output_tokens, instead of letting callers fall through to a
    confusing "invalid JSON" or "empty response" error for what's really
    a token-budget problem (see module docstring).

    Args:
        response: The raw response from client.models.generate_content().

    Raises:
        LLMRequestError: If the response's first candidate has
            finish_reason == MAX_TOKENS.
    """
    if not response.candidates:
        return

    finish_reason = response.candidates[0].finish_reason
    if finish_reason == types.FinishReason.MAX_TOKENS:
        usage = response.usage_metadata
        thoughts_tokens = getattr(usage, "thoughts_token_count", None) if usage else None
        raise LLMRequestError(
            "Gemini API response was cut off after hitting max_output_tokens "
            f"(thoughts_token_count={thoughts_tokens!r}). Increase max_output_tokens, "
            "or if you're on a Pro-family model where thinking can't be fully disabled, "
            "increase it further to leave room for both thinking and the answer."
        )


def _strip_code_fence(text: str) -> str:
    """
    Remove a surrounding ```json ... ``` (or bare ``` ... ```) code fence
    from a model response, if present.

    Gemini's JSON mode is expected to return bare JSON, but defending
    against an occasional markdown-wrapped response costs nothing and
    avoids a confusing json.JSONDecodeError for what's otherwise valid
    output.

    Args:
        text: The raw response text.

    Returns:
        `text` with a wrapping code fence removed, and surrounding
        whitespace stripped. Text without a code fence is returned
        unchanged (aside from stripping).
    """
    stripped = text.strip()

    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def generate_structured_analysis(
    prompt: str,
    system_instruction: str,
    schema: type[SchemaT],
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> SchemaT:
    """
    Call Gemini with a prompt and parse its response into a Pydantic model.

    Args:
        prompt: The user-turn content — the fully-built prompt (see
            src/genai/prompts.py), not just a bare question.
        system_instruction: The system-level persona/constraints text
            (see src/genai/prompts.py's SYSTEM_INSTRUCTION).
        schema: A Pydantic BaseModel subclass describing the expected
            JSON shape. Passed to Gemini as response_schema so the model
            is constrained to matching JSON, and used again locally to
            validate the parsed response.
        temperature: Sampling temperature. Lower is more deterministic;
            kept fairly low by default since this is a factual/analytical
            task, not creative writing.
        max_output_tokens: Hard cap on response length, including any
            "thinking" tokens the model spends before its visible answer
            (see module docstring) — keep this generous, not just large
            enough for the expected answer text alone.

    Returns:
        An instance of `schema`, populated from the model's JSON response.

    Raises:
        LLMConfigError: If GEMINI_API_KEY isn't set.
        LLMRequestError: If the API call itself fails, the response was
            cut off by hitting max_output_tokens, or the response can't
            be parsed as JSON or doesn't match `schema`.
    """
    client = get_client()

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=_build_thinking_config(settings.gemini_model),
    )

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        # The SDK can raise several distinct exception types depending on
        # failure mode (auth, quota, network, invalid request). Callers
        # only need to know "the LLM call failed and why," not which SDK
        # exception class it was, so they're normalized to one type here.
        raise LLMRequestError(f"Gemini API call failed: {exc}") from exc

    _check_finish_reason(response)

    if not response.text:
        raise LLMRequestError("Gemini API returned an empty response.")

    cleaned_text = _strip_code_fence(response.text)

    try:
        parsed_json = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise LLMRequestError(
            f"Gemini API response was not valid JSON: {exc}. Raw response: {response.text[:500]!r}"
        ) from exc

    try:
        return schema.model_validate(parsed_json)
    except ValidationError as exc:
        raise LLMRequestError(
            f"Gemini API response didn't match the expected schema: {exc}"
        ) from exc


def generate_text(
    prompt: str,
    system_instruction: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """
    Call Gemini with a prompt and return its raw text response — the
    plain-text counterpart to generate_structured_analysis(), for
    use cases (like the Phase 11 RAG chatbot) where a natural-language
    answer is more appropriate than a rigid JSON schema.

    Args:
        prompt: The user-turn content.
        system_instruction: The system-level persona/constraints text.
        temperature: Sampling temperature.
        max_output_tokens: Hard cap on response length, including any
            "thinking" tokens (see module docstring).

    Returns:
        The model's response text, stripped of leading/trailing
        whitespace.

    Raises:
        LLMConfigError: If GEMINI_API_KEY isn't set.
        LLMRequestError: If the API call fails, the response was cut off
            by hitting max_output_tokens, or it returned an empty response.
    """
    client = get_client()

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=_build_thinking_config(settings.gemini_model),
    )

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        raise LLMRequestError(f"Gemini API call failed: {exc}") from exc

    _check_finish_reason(response)

    if not response.text:
        raise LLMRequestError("Gemini API returned an empty response.")

    return response.text.strip()
