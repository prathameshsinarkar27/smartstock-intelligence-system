"""
overview.py

Route for the Market Overview page (the dashboard's landing page):
market-wide KPI cards, search, top gainers/losers, sector performance,
latest market news, and a filterable company table.

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
        request: Injected by FastAPI; required by Jinja2Templates to build
            url_for() links inside the template.
        sector: Optional query parameter (?sector=Technology) to filter the
            company table by exact sector match.
        search: Optional query parameter (?search=apple) to filter the
            company table by symbol or company name.

    Returns:
        The rendered overview.html template, with all KPI/news/company
        data populated from the warehouse (or showing "no data yet"
        empty states if nothing has been loaded via Phases 1-3.1).
    """
    page_data = build_overview_page_data(sector=sector, search=search)

    return templates.TemplateResponse(
        request=request,
        name="overview.html",
        context=page_data,
    )
