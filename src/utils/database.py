"""
database.py

Centralized PostgreSQL connection management for the SmartStock Intelligence
Platform.

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
    """Raised when a connection to PostgreSQL cannot be established."""


def _build_connection() -> Psycopg2Connection:
    """
    Open a new raw psycopg2 connection using settings from src.utils.config.

    Returns:
        A new, open psycopg2 connection.

    Raises:
        DatabaseConnectionError: If the connection attempt fails.
    """
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
    Context manager that yields an open PostgreSQL connection and guarantees
    it is closed afterward, committing on success and rolling back on
    exception.

    Yields:
        An open psycopg2 connection.

    Raises:
        DatabaseConnectionError: If the connection cannot be established.

    Example:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM companies;")
                rows = cur.fetchall()
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
    Attempt a trivial query to verify the database is reachable and
    correctly configured. Intended for use in setup verification and the
    Phase 2 testing guide — not called by application code paths.

    Returns:
        True if the connection and a basic query succeed, False otherwise.
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
    # Allows running `python -m src.utils.database` as a quick connectivity
    # check during setup, without needing to write a separate script.
    if test_connection():
        logger.info("Successfully connected to PostgreSQL database '%s'.", settings.postgres_db)
    else:
        logger.error("Failed to connect to PostgreSQL. See error above for details.")
