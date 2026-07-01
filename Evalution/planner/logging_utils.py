"""Structured logging utilities built on top of ``structlog``."""

from __future__ import annotations

import logging
import sys
from typing import Optional

import structlog


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Configure *structlog* for the entire process.

    Parameters:
        level: Standard logging level name (DEBUG, INFO, WARNING, ERROR).
        json_output: When ``True`` emit JSON-formatted logs (ideal for
            production / containerised environments).  When ``False`` use a
            human-readable coloured console renderer.
        log_file: Optional file path.  When provided a :class:`FileHandler`
            is added alongside the console handler.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger identified by *name*.

    This is the preferred way to obtain a logger throughout the package::

        from planner.logging_utils import get_logger

        log = get_logger(__name__)
        log.info("planning_started", criterion_id="CRIT001")
    """
    return structlog.get_logger(name)