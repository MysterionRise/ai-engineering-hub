"""Custom exception hierarchy for AI Hub.

This module defines a structured exception hierarchy that enables:
- Consistent error handling across providers
- Automatic retry for transient errors
- Clear error messages for debugging
- Type-safe exception handling
"""

from typing import Any


class AIHubError(Exception):
    """Base exception for all AI Hub errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(AIHubError):
    """Raised when configuration is invalid or missing."""

    pass


class ProviderError(AIHubError):
    """Base exception for provider-related errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        super().__init__(message, {"provider": provider, **(details or {})})


class AuthenticationError(ProviderError):
    """Raised when API authentication fails."""

    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded.

    This error is retryable - the system should wait and retry.
    """

    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message,
            provider,
            {"retry_after": retry_after, **(details or {})},
        )


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not available."""

    def __init__(
        self,
        model: str,
        provider: str,
        available_models: list[str] | None = None,
    ) -> None:
        self.model = model
        self.available_models = available_models
        super().__init__(
            f"Model '{model}' not found",
            provider,
            {"model": model, "available_models": available_models},
        )


class ContextLengthError(ProviderError):
    """Raised when input exceeds model's context length."""

    def __init__(
        self,
        message: str,
        provider: str,
        max_tokens: int | None = None,
        requested_tokens: int | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.requested_tokens = requested_tokens
        super().__init__(
            message,
            provider,
            {"max_tokens": max_tokens, "requested_tokens": requested_tokens},
        )


class ContentFilterError(ProviderError):
    """Raised when content is blocked by safety filters."""

    pass


class RetryableError(AIHubError):
    """Base class for errors that can be retried."""

    pass


class TimeoutError(RetryableError):
    """Raised when a request times out."""

    pass


class ServiceUnavailableError(RetryableError):
    """Raised when the service is temporarily unavailable."""

    pass


class ValidationError(AIHubError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
    ) -> None:
        self.field = field
        self.value = value
        super().__init__(message, {"field": field, "value": value})
