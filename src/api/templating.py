"""
templating.py

Shared Jinja2 template configuration and formatting filters.
"""

from pathlib import Path

from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _format_currency(value: float | None, decimals: int = 2) -> str:
    """Format a number as a US-dollar amount."""
    if value is None:
        return "—"
    return f"${value:,.{decimals}f}"


def _format_large_number(value: float | None) -> str:
    """Format large monetary values using K, M, B, or T notation."""
    if value is None:
        return "—"
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    if abs_value >= 1_000_000_000_000:
        return f"{sign}${abs_value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"{sign}${abs_value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{sign}${abs_value / 1_000_000:.2f}M"
    return f"{sign}${abs_value:,.2f}"


def _format_percent(value: float | None, decimals: int = 2) -> str:
    """Format a number as a signed percentage."""
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def _format_volume(value: int | None) -> str:
    """Format share volume using K, M, or B notation."""
    if value is None:
        return "—"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


templates.env.filters["currency"] = _format_currency
templates.env.filters["large_number"] = _format_large_number
templates.env.filters["percent"] = _format_percent
templates.env.filters["volume"] = _format_volume
