"""Configuration management using Pydantic settings.

This module provides a centralized configuration system that:
- Loads settings from environment variables
- Supports .env files via python-dotenv
- Validates configuration at startup
- Provides type-safe access to settings
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):  # type: ignore[misc]
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables.
    Prefix: AI_HUB_ (e.g., AI_HUB_LOG_LEVEL=DEBUG)
    """

    model_config = SettingsConfigDict(
        env_prefix="AI_HUB_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys (loaded from environment without prefix for compatibility)
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY")

    # Default model settings
    default_provider: Literal["openai", "anthropic", "google"] = "openai"
    default_model: str = "gpt-4o"
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=4096, ge=1, le=128000)

    # Retry settings
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_min_wait: float = Field(default=1.0, ge=0.1)
    retry_max_wait: float = Field(default=60.0, ge=1.0)

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "console"

    # Rate limiting
    requests_per_minute: int = Field(default=60, ge=1)

    @property
    def openai_key(self) -> str | None:
        """Get OpenAI API key as string."""
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None

    @property
    def anthropic_key(self) -> str | None:
        """Get Anthropic API key as string."""
        return self.anthropic_api_key.get_secret_value() if self.anthropic_api_key else None

    @property
    def google_key(self) -> str | None:
        """Get Google API key as string."""
        return self.google_api_key.get_secret_value() if self.google_api_key else None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: The application settings singleton.
    """
    return Settings()
