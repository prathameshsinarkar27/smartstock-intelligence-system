"""
logger.py

Centralized logging configuration for SmartStock.
"""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track configured loggers to prevent duplicate handlers.
_configured_loggers: set[str] = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger with a single console handler.
    """
    logger = logging.getLogger(name)

    if name not in _configured_loggers:
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        # Prevent duplicate output through the root logger.
        logger.propagate = False
        _configured_loggers.add(name)

    return logger
