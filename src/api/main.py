"""
main.py

FastAPI application entrypoint for SmartStock.

Run locally with:
    uvicorn src.api.main:app --reload
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import api, overview, portfolio, stock_detail
from src.utils.logger import get_logger

logger = get_logger(__name__)

API_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="SmartStock Intelligence Platform",
    description="AI-powered stock market analytics, research, and decision intelligence dashboard.",
    version="0.1.0",
)

# Serve project CSS and JavaScript assets.
app.mount("/static", StaticFiles(directory=str(API_DIR / "static")), name="static")

# Register dashboard routers.
app.include_router(overview.router)
app.include_router(stock_detail.router)
app.include_router(portfolio.router)
app.include_router(api.router)


@app.on_event("startup")
async def on_startup() -> None:
    """Log when the application starts."""
    logger.info("SmartStock dashboard starting up.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Log when the application shuts down."""
    logger.info("SmartStock dashboard shutting down.")
