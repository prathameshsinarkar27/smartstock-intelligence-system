"""
llm_utils.py

Low-level wrapper around the Gemini API. This is the only module that
imports google.genai directly. Other modules use generate_structured_analysis()
or generate_text().

Structured responses are requested as JSON using a caller-supplied Pydantic
schema and parsed explicitly with json.loads() before validation.
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

# Generous budget to account for models that use part of the limit for thinking.
DEFAULT_MAX_OUTPUT_TOKENS = 2048

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Lazily initialized client cache.
_client: genai.Client | None = None


class LLMConfigError(Exception):
    """Raised when the Gemini API key is missing or otherwise misconfigured."""


class LLMRequestError(Exception):
    """Raised when a Gemini API call fails, or its response can't be parsed."""


def get_client() -> genai.Client:
    """Build and return the cached Gemini API client."""
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
    """Clear the cached client."""
    global _client
    _client = None


def _build_thinking_config(model_name: str) -> types.ThinkingConfig | None:
    """
    Build thinking configuration based on model capabilities.

    Flash-family models support disabling thinking. Other models use the
    API's default dynamic budget.
    """
    if "flash" in model_name.lower():
        return types.ThinkingConfig(thinking_budget=0)
    return None


def _check_finish_reason(response: types.GenerateContentResponse) -> None:
    """Raise a clear error when the response exceeds the token budget."""
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
    """Remove a surrounding Markdown code fence from a model response."""
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
    Call Gemini and validate its JSON response against a Pydantic schema.
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
        # Normalize SDK-specific failures to a single application exception.
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
    """Call Gemini and return its raw text response."""
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
