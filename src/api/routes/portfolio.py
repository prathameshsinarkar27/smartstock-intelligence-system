"""
portfolio.py

Routes for the Portfolio Analyzer page: view holdings/P&L/sector
concentration for a user, and add/update/remove positions via simple
HTML forms.
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
    """Build a 303 redirect back to GET /portfolio, preserving the user and optionally an error message."""
    url = f"/portfolio?user={quote(user_name)}"
    if error:
        url += f"&error={quote(error)}"
    return RedirectResponse(url=url, status_code=303)


@router.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request, user: str | None = None, error: str | None = None):
    """
    Render the Portfolio Analyzer page for a given user.

    Args:
        request: Injected by FastAPI; required by Jinja2Templates.
        user: Optional query parameter (?user=jane) identifying whose
            portfolio to display. Defaults to "default" (see
            portfolio_service.DEFAULT_USER) so the page works with zero
            setup — there's no login system in this project.
        error: Optional query parameter carrying a validation error
            message from a failed add/update/remove submission, rendered
            as a dismissible banner.

    Returns:
        The rendered portfolio.html template with holdings, summary
        KPIs, and sector concentration populated (or empty/zeroed states
        if the user has no watchlist entries yet).
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
    """
    Handle the "Add / Update Holding" form: upsert a real position
    (shares + avg_cost_basis) for a user, then redirect back to the
    portfolio page.
    """
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
    """
    Handle the lighter-weight "Watch Symbol" form (no shares/cost basis —
    the original Phase 0-11 watchlist behavior), then redirect back.
    """
    resolved_user = (user_name or "").strip() or "default"

    try:
        add_watch_only(user_name=resolved_user, symbol=symbol)
    except PortfolioInputError as exc:
        logger.warning("Rejected watch-only add for user '%s': %s", resolved_user, exc)
        return _redirect_to_portfolio(resolved_user, error=str(exc))

    return _redirect_to_portfolio(resolved_user)


@router.post("/portfolio/holdings/remove")
async def remove_holding_route(user_name: str = Form(...), symbol: str = Form(...)):
    """Handle the "Remove" button next to a holding/watch entry, then redirect back."""
    resolved_user = (user_name or "").strip() or "default"
    remove_holding(user_name=resolved_user, symbol=symbol)
    return _redirect_to_portfolio(resolved_user)
