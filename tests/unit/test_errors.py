"""Tests for error handling module."""

import pytest

from ai_hub.core.errors import (
    AIHubError,
    AuthenticationError,
    ConfigurationError,
    ContextLengthError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    ValidationError,
)


class TestAIHubError:
    """Tests for base AIHubError."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = AIHubError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details == {}

    def test_error_with_details(self) -> None:
        """Test error with details."""
        error = AIHubError("Error occurred", details={"code": 500, "request_id": "abc123"})

        assert "Error occurred" in str(error)
        assert "code" in str(error)
        assert error.details["code"] == 500


class TestProviderError:
    """Tests for ProviderError."""

    def test_provider_error(self) -> None:
        """Test provider error creation."""
        error = ProviderError("API call failed", provider="openai")

        assert error.provider == "openai"
        assert error.details["provider"] == "openai"

    def test_authentication_error(self) -> None:
        """Test authentication error."""
        error = AuthenticationError("Invalid API key", provider="anthropic")

        assert isinstance(error, ProviderError)
        assert error.provider == "anthropic"


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_rate_limit_error(self) -> None:
        """Test rate limit error with retry_after."""
        error = RateLimitError(
            "Rate limit exceeded",
            provider="openai",
            retry_after=30.0,
        )

        assert error.retry_after == 30.0
        assert error.provider == "openai"
        assert error.details["retry_after"] == 30.0


class TestModelNotFoundError:
    """Tests for ModelNotFoundError."""

    def test_model_not_found(self) -> None:
        """Test model not found error."""
        error = ModelNotFoundError(
            model="gpt-5",
            provider="openai",
            available_models=["gpt-4", "gpt-3.5-turbo"],
        )

        assert error.model == "gpt-5"
        assert "gpt-5" in str(error)
        assert error.available_models == ["gpt-4", "gpt-3.5-turbo"]


class TestContextLengthError:
    """Tests for ContextLengthError."""

    def test_context_length_error(self) -> None:
        """Test context length error."""
        error = ContextLengthError(
            "Context too long",
            provider="openai",
            max_tokens=8192,
            requested_tokens=10000,
        )

        assert error.max_tokens == 8192
        assert error.requested_tokens == 10000


class TestValidationError:
    """Tests for ValidationError."""

    def test_validation_error(self) -> None:
        """Test validation error."""
        error = ValidationError(
            "Invalid temperature",
            field="temperature",
            value=3.5,
        )

        assert error.field == "temperature"
        assert error.value == 3.5


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_provider_error_is_aihub_error(self) -> None:
        """Test that ProviderError inherits from AIHubError."""
        error = ProviderError("Test", provider="test")
        assert isinstance(error, AIHubError)

    def test_auth_error_is_provider_error(self) -> None:
        """Test that AuthenticationError inherits from ProviderError."""
        error = AuthenticationError("Test", provider="test")
        assert isinstance(error, ProviderError)
        assert isinstance(error, AIHubError)

    def test_can_catch_by_base_class(self) -> None:
        """Test that specific errors can be caught by base class."""
        with pytest.raises(AIHubError):
            raise RateLimitError("Test", provider="test")

        with pytest.raises(ProviderError):
            raise ModelNotFoundError("model", provider="test")
