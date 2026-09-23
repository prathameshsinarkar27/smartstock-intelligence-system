"""
portfolio.py

Routes for the Portfolio Analyzer page, including holdings,
P&L, sector concentration, and portfolio form actions.
"""

from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src.api.services.portfolio_service import (
    PortfolioInputError,
    add_or_update_holding,
    add_watch_only,
    build_portfolio_page_data,
    remove_holding,
)
from src.api.templating import templates
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _redirect_to_portfolio(user_name: str, error: str | None = None) -> RedirectResponse:
    """Build a redirect to the portfolio page with an optional error."""
    url = f"/portfolio?user={quote(user_name)}"
    if error:
        url += f"&error={quote(error)}"
    return RedirectResponse(url=url, status_code=303)


@router.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request, user: str | None = None, error: str | None = None):
    """
    Render the Portfolio Analyzer page.

    Args:
        request: FastAPI request used by the Jinja2 template.
        user: Optional portfolio owner. Defaults to "default".
        error: Optional validation error message.

    Returns:
        Rendered portfolio.html template with portfolio data.
    """
    page_data = build_portfolio_page_data(user)
    page_data["error"] = error

    return templates.TemplateResponse(
        request=request,
        name="portfolio.html",
        context=page_data,
    )


@router.post("/portfolio/holdings/add")
async def add_holding(
    user_name: str = Form(...),
    symbol: str = Form(...),
    shares: str = Form(...),
    avg_cost_basis: str = Form(...),
    purchased_at: str = Form(""),
):
    """Add or update a portfolio holding, then redirect back."""
    resolved_user = (user_name or "").strip() or "default"

    try:
        add_or_update_holding(
            user_name=resolved_user,
            symbol=symbol,
            shares=shares,
            avg_cost_basis=avg_cost_basis,
            purchased_at=purchased_at,
        )
    except PortfolioInputError as exc:
        logger.warning("Rejected add/update holding for user '%s': %s", resolved_user, exc)
        return _redirect_to_portfolio(resolved_user, error=str(exc))

    return _redirect_to_portfolio(resolved_user)


@router.post("/portfolio/watchlist/add")
async def add_to_watchlist(user_name: str = Form(...), symbol: str = Form(...)):
    """Add a symbol to the watchlist without a position."""
    resolved_user = (user_name or "").strip() or "default"

    try:
        add_watch_only(user_name=resolved_user, symbol=symbol)
    except PortfolioInputError as exc:
        logger.warning("Rejected watch-only add for user '%s': %s", resolved_user, exc)
        return _redirect_to_portfolio(resolved_user, error=str(exc))

    return _redirect_to_portfolio(resolved_user)


@router.post("/portfolio/holdings/remove")
async def remove_holding_route(user_name: str = Form(...), symbol: str = Form(...)):
    """Remove a holding or watchlist entry, then redirect back."""
    resolved_user = (user_name or "").strip() or "default"
    remove_holding(user_name=resolved_user, symbol=symbol)
    return _redirect_to_portfolio(resolved_user)
