"""
logger.py

Centralized logging configuration for the SmartStock Intelligence Platform.

"""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Tracks which logger names have already been configured, so repeated calls
# to get_logger() for the same module don't attach duplicate handlers.
_configured_loggers: set[str] = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Args:
        name: Typically the caller's __name__, e.g. "src.ingestion.fetch_stock_data".
        level: Logging level threshold (default: logging.INFO).

    Returns:
        A logging.Logger instance with a console handler attached exactly once.
    """
    logger = logging.getLogger(name)

    if name not in _configured_loggers:
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        # Prevent double-logging if the root logger also has handlers attached.
        logger.propagate = False
        _configured_loggers.add(name)

    return logger
