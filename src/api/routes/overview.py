"""
overview.py

Routes for the Market Overview page, including KPIs, search,
top movers, sector performance, news, and company data.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src.api.services.overview_service import build_overview_page_data
from src.api.templating import templates
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def market_overview(request: Request, sector: str | None = None, search: str | None = None):
    """
    Render the Market Overview page.

    Args:
        request: FastAPI request used by the Jinja2 template.
        sector: Optional exact sector filter.
        search: Optional symbol or company name filter.

    Returns:
        Rendered overview.html template with market data.
    """
    page_data = build_overview_page_data(sector=sector, search=search)

    return templates.TemplateResponse(
        request=request,
        name="overview.html",
        context=page_data,
    )
