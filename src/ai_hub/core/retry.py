"""Retry utilities with exponential backoff using tenacity.

This module provides:
- Configurable retry decorators for API calls
- Exponential backoff with jitter to prevent thundering herd
- Automatic handling of rate limits and transient errors
- Logging of retry attempts
"""

import random
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from tenacity import (  # type: ignore[attr-defined]
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ai_hub.core.config import get_settings
from ai_hub.core.errors import RateLimitError, RetryableError, ServiceUnavailableError, TimeoutError
from ai_hub.core.logging import get_logger

P = ParamSpec("P")
T = TypeVar("T")

logger = get_logger(__name__)


# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS = (
    RetryableError,
    RateLimitError,
    TimeoutError,
    ServiceUnavailableError,
    ConnectionError,
    TimeoutError,
)


def create_retry_decorator(
    max_retries: int | None = None,
    min_wait: float | None = None,
    max_wait: float | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = RETRYABLE_EXCEPTIONS,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Create a retry decorator with custom settings.

    Args:
        max_retries: Maximum number of retry attempts. Defaults to config value.
        min_wait: Minimum wait time between retries in seconds.
        max_wait: Maximum wait time between retries in seconds.
        retryable_exceptions: Tuple of exception types to retry on.

    Returns:
        A decorator that adds retry logic to functions.
    """
    settings = get_settings()

    max_retries = max_retries if max_retries is not None else settings.max_retries
    min_wait = min_wait if min_wait is not None else settings.retry_min_wait
    max_wait = max_wait if max_wait is not None else settings.retry_max_wait

    def log_retry(retry_state: Any) -> None:
        """Log retry attempts."""
        if retry_state.attempt_number > 1:
            exception = retry_state.outcome.exception() if retry_state.outcome else None
            logger.warning(
                "retry_attempt",
                attempt=retry_state.attempt_number,
                max_attempts=max_retries + 1,
                exception_type=type(exception).__name__ if exception else None,
                exception_message=str(exception) if exception else None,
            )

    return retry(
        stop=stop_after_attempt(max_retries + 1),  # type: ignore[no-untyped-call]
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait, jitter=min_wait),
        retry=retry_if_exception_type(retryable_exceptions),  # type: ignore[no-untyped-call]
        before_sleep=log_retry,
        reraise=True,
    )


# Default retry decorator using config settings
with_retry = create_retry_decorator()


def retry_with_backoff(
    max_retries: int | None = None,
    min_wait: float | None = None,
    max_wait: float | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that adds retry logic with exponential backoff.

    Example:
        @retry_with_backoff(max_retries=5)
        async def call_api():
            ...

    Args:
        max_retries: Maximum number of retry attempts.
        min_wait: Minimum wait time between retries.
        max_wait: Maximum wait time between retries.

    Returns:
        Decorated function with retry logic.
    """
    return create_retry_decorator(
        max_retries=max_retries,
        min_wait=min_wait,
        max_wait=max_wait,
    )


async def execute_with_retry(
    func: Callable[..., T],
    *args: Any,
    max_retries: int | None = None,
    **kwargs: Any,
) -> T:
    """Execute a function with retry logic.

    This is useful when you can't use a decorator (e.g., for lambdas or
    dynamically created functions).

    Args:
        func: The function to execute.
        *args: Positional arguments for the function.
        max_retries: Maximum number of retry attempts.
        **kwargs: Keyword arguments for the function.

    Returns:
        The function's return value.

    Raises:
        The last exception if all retries fail.
    """
    settings = get_settings()
    max_retries = max_retries if max_retries is not None else settings.max_retries

    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < max_retries:
                # Calculate wait time with jitter
                wait_time = min(
                    settings.retry_max_wait,
                    settings.retry_min_wait * (2**attempt) + random.uniform(0, 1),
                )
                logger.warning(
                    "retry_attempt",
                    attempt=attempt + 1,
                    max_attempts=max_retries + 1,
                    wait_seconds=wait_time,
                    exception_type=type(e).__name__,
                )
                import asyncio

                await asyncio.sleep(wait_time)
            else:
                raise

    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry logic")
