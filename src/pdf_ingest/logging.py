"""Structured logging configuration for pdf-ingest.

This module provides:
- JSONFormatter: Structured JSON logging for production
- ColoredFormatter: Human-readable colored console output
- setup_logging(): One-call logging configuration
- LogContext: Context manager for structured log context

Example:
    setup_logging(log_level="DEBUG", json_logs=True, log_file="app.log")
    logger = get_logger(__name__)
    logger.info("Processing started")

    with LogContext(logger, request_id="abc123"):
        logger.info("Processing request")
"""
from __future__ import annotations

import json
import logging
import logging.config
import sys
from pathlib import Path
from typing import Any

from .config import settings


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured fields.

    Outputs log records as JSON with standard fields:
    timestamp, level, logger, message, module, function, line
    Plus any extra_fields attached to the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON string representation
        """
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development.

    Adds ANSI color codes to log level names for readability.
    """

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
        "RESET": "\033[0m",     # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color codes.

        Args:
            record: Log record to format

        Returns:
            Colored formatted string
        """
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def get_logging_config(
    log_level: str = "INFO",
    json_logs: bool = False,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Generate logging configuration dictionary.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Use JSON formatter (True) or colored (False)
        log_file: Optional file path for log output

    Returns:
        Configuration dict for logging.config.dictConfig()
    """
    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "json" if json_logs else "colored",
            "level": log_level,
        }
    }

    formatters = {
        "json": {
            "()": "pdf_ingest.logging.JSONFormatter",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
        "colored": {
            "()": "pdf_ingest.logging.ColoredFormatter",
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%H:%M:%S",
        },
    }

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_file),
            "maxBytes": 10_485_760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
            "level": log_level,
            "encoding": "utf-8",
        }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "root": {
            "level": log_level,
            "handlers": list(handlers.keys()),
        },
        "loggers": {
            "pdf_ingest": {
                "level": log_level,
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "httpx": {
                "level": "WARNING",
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
        },
    }


def setup_logging(
    log_level: str | None = None,
    json_logs: bool | None = None,
    log_file: Path | str | None = None,
) -> None:
    """Configure application logging.

    Args:
        log_level: Logging level (default: settings.log_level)
        json_logs: Use JSON formatter (default: settings.json_logs)
        log_file: Optional log file path (default: settings.log_file)
    """
    level = log_level or getattr(settings, "log_level", "INFO")
    use_json = json_logs if json_logs is not None else getattr(settings, "json_logs", False)
    file_path = Path(log_file) if log_file else getattr(settings, "log_file", None)

    config = get_logging_config(level, use_json, file_path)
    logging.config.dictConfig(config)

    logging.getLogger("pdf_ingest").info(
        "Logging configured",
        extra={"extra_fields": {"level": level, "json_logs": use_json, "log_file": str(file_path) if file_path else None}},
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the pdf_ingest namespace.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"pdf_ingest.{name}")


class LogContext:
    """Context manager for adding structured context to logs.

    Usage:
        with LogContext(logger, request_id="abc123", user="john"):
            logger.info("Processing request")  # Includes request_id and user in JSON output

    Args:
        logger: Logger instance
        **context: Key-value pairs to add to all log records in context
    """

    def __init__(self, logger: logging.Logger, **context: Any):
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self) -> logging.Logger:
        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = self.old_factory(*args, **kwargs)
            record.extra_fields = self.context
            return record

        logging.setLogRecordFactory(record_factory)
        return self.logger

    def __exit__(self, *args: Any) -> None:
        logging.setLogRecordFactory(self.old_factory)