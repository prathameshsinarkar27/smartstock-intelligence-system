"""
prompts.py

Prompt templates and structured-output schema for the AI Research Assistant.
This module handles prompt/schema construction only and makes no API calls.
"""

from typing import Literal

from pydantic import BaseModel, Field

OUTLOOK_VALUES = ("Bullish", "Bearish", "Mixed", "Cautious")


class CompanyAnalysis(BaseModel):
    """Structured response schema for company AI insights."""

    outlook: Literal["Bullish", "Bearish", "Mixed", "Cautious"]
    summary: str = Field(..., min_length=1, max_length=800)
    key_considerations: list[str] = Field(..., min_length=1, max_length=5)


SYSTEM_INSTRUCTION = """\
You are an AI research assistant embedded in a stock market analytics dashboard.

Your job is to synthesize the structured data you're given — price action, \
technical indicators, news sentiment, and a machine learning model's trend/risk \
prediction — into a short, clearly-written analysis for someone researching the \
stock. You are not a licensed financial advisor, and the person reading your \
output has not asked you for personalized advice.

Follow these rules strictly:
1. Base your analysis ONLY on the data provided in the prompt. Do not draw on \
   general or historical knowledge about the company, and do not invent facts, \
   figures, or events that are not present in the given data.
2. Never issue a direct instruction to buy, sell, hold, or otherwise trade the \
   stock. Describe an informational "outlook" instead, and frame considerations \
   as things a reader might weigh, not as directives.
3. Be balanced: if signals conflict (e.g. positive sentiment alongside a \
   model-predicted downside risk), say so rather than picking a side to sound \
   more decisive.
4. Respond with ONLY a JSON object matching the required schema — no \
   markdown code fences, no preamble, no text outside the JSON object.
"""


def _fmt(value, prefix: str = "", suffix: str = "", scale: float = 1.0) -> str:
    """Format an optional numeric value for prompt text."""
    if value is None:
        return "not available"
    return f"{prefix}{value * scale:,.2f}{suffix}"


def build_company_analysis_prompt(context: dict) -> str:
    """Build the analysis prompt from a structured company context."""
    lines = [
        f"Company: {context.get('company_name', 'Unknown')} ({context.get('symbol', 'N/A')})",
        f"Sector / Industry: {context.get('sector', 'Unknown')} / {context.get('industry', 'Unknown')}",
        "",
        "PRICE & FUNDAMENTALS",
        f"- Current price: {_fmt(context.get('current_price'), '$')}",
        f"- Daily change: {_fmt(context.get('daily_change_pct'), '', '%')}",
        f"- Recent range: {_fmt(context.get('period_low'), '$')} - {_fmt(context.get('period_high'), '$')}",
        f"- P/E ratio: {_fmt(context.get('pe_ratio'))}",
        f"- Market cap: {_fmt(context.get('market_cap'), '$')}",
        "",
        "NEWS SENTIMENT",
        f"- Sentiment score (-100 bearish .. +100 bullish, from scored news): {_fmt(context.get('sentiment_score'))}",
        f"- Scored articles: {context.get('sentiment_positive_count', 0)} positive, "
        f"{context.get('sentiment_negative_count', 0)} negative, "
        f"{context.get('sentiment_neutral_count', 0)} neutral",
        "",
        "ML MODEL PREDICTION",
        f"- Predicted trend (Random Forest + XGBoost ensemble): {context.get('ml_trend_prediction', 'not available')}",
        f"- Estimated probability of a downside move: {_fmt(context.get('ml_risk_score'), '', '%', scale=100)}",
    ]

    top_factors = context.get("ml_top_factors") or []
    if top_factors:
        lines.append("- Top factors behind that prediction (SHAP, direction relative to downside risk):")
        for factor in top_factors:
            lines.append(
                f"  - {factor['display_name']}: {factor['direction']} "
                f"(relative share {factor['contribution_share']:+.2f})"
            )

    lines += [
        "",
        "Write your analysis of this company now, following the JSON schema and all rules "
        "in your system instructions.",
    ]

    return "\n".join(lines)
