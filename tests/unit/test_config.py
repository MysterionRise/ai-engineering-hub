"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from ai_hub.core.config import Settings, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

            assert settings.default_provider == "openai"
            assert settings.default_model == "gpt-4o"
            assert settings.default_temperature == 0.7
            assert settings.log_level == "INFO"
            assert settings.max_retries == 3

    def test_env_override(self) -> None:
        """Test environment variable overrides."""
        env = {
            "AI_HUB_DEFAULT_PROVIDER": "anthropic",
            "AI_HUB_LOG_LEVEL": "DEBUG",
            "AI_HUB_MAX_RETRIES": "5",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = Settings()

            assert settings.default_provider == "anthropic"
            assert settings.log_level == "DEBUG"
            assert settings.max_retries == 5

    def test_api_key_from_env(self) -> None:
        """Test API key loading from environment."""
        env = {
            "OPENAI_API_KEY": "sk-test-key",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = Settings()

            assert settings.openai_key == "sk-test-key"

    def test_temperature_validation(self) -> None:
        """Test temperature bounds validation."""
        # Valid temperature
        env = {"AI_HUB_DEFAULT_TEMPERATURE": "1.5"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.default_temperature == 1.5

        # Invalid temperature should raise
        env = {"AI_HUB_DEFAULT_TEMPERATURE": "3.0"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(Exception):  # Pydantic ValidationError
                Settings()


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_instance(self) -> None:
        """Test that get_settings returns a Settings instance."""
        # Clear cache for test isolation
        get_settings.cache_clear()

        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_caching(self) -> None:
        """Test that settings are cached."""
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2
