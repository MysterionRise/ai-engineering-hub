"""Structured logging configuration using structlog.

This module provides:
- JSON logging for production (machine-readable)
- Pretty console logging for development (human-readable)
- Automatic context enrichment (timestamps, caller info)
- Integration with standard logging
"""

import logging
import sys
from typing import Any, cast

import structlog

from ai_hub.core.config import get_settings


def setup_logging() -> None:
    """Configure structured logging based on settings.

    Call this once at application startup to configure logging.
    """
    settings = get_settings()

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )

    # Configure structlog processors
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.log_format == "json":
        # Production: JSON output
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Pretty console output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=cast(list[structlog.typing.Processor], processors),
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, settings.log_level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Get a configured logger instance.

    Args:
        name: Optional logger name for identification.

    Returns:
        A bound logger instance.
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary log context.

    Example:
        with LogContext(request_id="abc123", user_id="user1"):
            logger.info("Processing request")
    """

    def __init__(self, **kwargs: Any) -> None:
        self.context = kwargs
        self._token: Any = None

    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_llm_call(
    logger: Any,
    provider: str,
    model: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: float | None = None,
    **kwargs: Any,
) -> None:
    """Log an LLM API call with standard metrics.

    Args:
        logger: The logger instance to use.
        provider: The LLM provider name (e.g., "openai", "anthropic").
        model: The model name used.
        input_tokens: Number of input tokens (if available).
        output_tokens: Number of output tokens (if available).
        latency_ms: Request latency in milliseconds.
        **kwargs: Additional context to log.
    """
    logger.info(
        "llm_call",
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=(input_tokens or 0) + (output_tokens or 0) if input_tokens or output_tokens else None,
        latency_ms=latency_ms,
        **kwargs,
    )
