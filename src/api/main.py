"""
main.py

FastAPI application entrypoint for the SmartStock Intelligence Platform.

Run locally with:
    uvicorn src.api.main:app --reload

"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import overview, stock_detail
from src.utils.logger import get_logger

logger = get_logger(__name__)

API_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="SmartStock Intelligence Platform",
    description="AI-powered stock market analytics, research, and decision intelligence dashboard.",
    version="0.1.0",
)

# Serve CSS/JS assets from src/api/static/ at the /static URL path. Bootstrap
# 5 and Plotly.js themselves are loaded via CDN in the base template, not
# bundled here — this directory only holds this project's own small
# stylesheet and page scripts.
app.mount("/static", StaticFiles(directory=str(API_DIR / "static")), name="static")

# Each router owns one functional area of the dashboard, per the
# Blueprint Update's "one router module per functional area" structure
# (docs/01_FOLDER_STRUCTURE.md). Routers are included with no prefix here
# since their own path operations already define the full path (e.g. "/"
# for the overview, "/stocks/{symbol}" for company detail) — this keeps
# URLs human-readable rather than nested under a generic prefix.
app.include_router(overview.router)
app.include_router(stock_detail.router)


@app.on_event("startup")
async def on_startup() -> None:
    """Log a clear startup message so it's obvious the app is ready."""
    logger.info("SmartStock dashboard starting up.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Log a clear shutdown message."""
    logger.info("SmartStock dashboard shutting down.")
