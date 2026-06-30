"""
templating.py

Shared Jinja2Templates instance used by every route module in src/api/routes/.

"""

from pathlib import Path

from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _format_currency(value: float | None, decimals: int = 2) -> str:
    """Jinja2 filter: format a number as a US-dollar amount, e.g. 1234.5 -> \"$1,234.50\"."""
    if value is None:
        return "—"
    return f"${value:,.{decimals}f}"


def _format_large_number(value: float | None) -> str:
    """Jinja2 filter: abbreviate large numbers, e.g. 3_000_000_000_000 -> \"$3.00T\"."""
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
    """Jinja2 filter: format a number as a signed percentage, e.g. -1.2 -> \"-1.20%\"."""
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def _format_volume(value: int | None) -> str:
    """Jinja2 filter: abbreviate share volume, e.g. 1_234_567 -> \"1.23M\"."""
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
