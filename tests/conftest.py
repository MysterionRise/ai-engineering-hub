"""Pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_env() -> None:
    """Mock environment for all tests."""
    # Clear any cached settings
    from ai_hub.core.config import get_settings

    get_settings.cache_clear()


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client."""
    mock_client = MagicMock()

    # Mock chat completions
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mock response"
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.model = "gpt-4o-mini"

    mock_client.chat.completions.create.return_value = mock_response

    # Mock models list
    mock_models = MagicMock()
    mock_models.data = [
        MagicMock(id="gpt-4o"),
        MagicMock(id="gpt-4o-mini"),
        MagicMock(id="gpt-3.5-turbo"),
    ]
    mock_client.models.list.return_value = mock_models

    return mock_client


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
