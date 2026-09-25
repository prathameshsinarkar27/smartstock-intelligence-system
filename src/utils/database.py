"""
database.py

Centralized PostgreSQL connection management for SmartStock.

Usage:
    from src.utils.database import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            print(cur.fetchone())
"""

from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.extensions import connection as Psycopg2Connection

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when a PostgreSQL connection cannot be established."""


def _build_connection() -> Psycopg2Connection:
    """Open a PostgreSQL connection using the configured settings."""
    try:
        return psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )
    except psycopg2.OperationalError as exc:
        raise DatabaseConnectionError(
            f"Could not connect to PostgreSQL at "
            f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db} "
            f"as user '{settings.postgres_user}'. "
            f"Check that PostgreSQL is running and your .env values are correct. "
            f"Original error: {exc}"
        ) from exc


@contextmanager
def get_connection() -> Generator[Psycopg2Connection, None, None]:
    """
    Yield a PostgreSQL connection with automatic commit, rollback, and close.
    """
    conn = _build_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def test_connection() -> bool:
    """
    Verify database connectivity with a simple query.

    Returns:
        True if the connection and query succeed, otherwise False.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                result = cur.fetchone()
                return result == (1,)
    except DatabaseConnectionError as exc:
        logger.error("Database connection test failed: %s", exc)
        return False


if __name__ == "__main__":
    # Supports `python -m src.utils.database` for a quick connectivity check.
    if test_connection():
        logger.info("Successfully connected to PostgreSQL database '%s'.", settings.postgres_db)
    else:
        logger.error("Failed to connect to PostgreSQL. See error above for details.")
