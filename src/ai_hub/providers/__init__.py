"""LLM provider implementations.

This module exports provider classes and utilities for working with
multiple LLM providers through a unified interface.
"""

from ai_hub.providers.base import (
    BaseLLMProvider,
    CompletionChunk,
    CompletionResponse,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    Usage,
)
from ai_hub.providers.openai_provider import OpenAIProvider

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "CompletionChunk",
    "CompletionResponse",
    "Message",
    "Role",
    "ToolCall",
    "ToolDefinition",
    "Usage",
    # Providers
    "OpenAIProvider",
    # Factory function
    "get_provider",
]


def get_provider(
    provider_name: str = "openai",
    **kwargs: object,
) -> BaseLLMProvider:
    """Factory function to get a provider instance.

    Args:
        provider_name: Name of the provider ("openai", "anthropic", "google").
        **kwargs: Additional arguments passed to the provider constructor.

    Returns:
        An instance of the requested provider.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    providers = {
        "openai": OpenAIProvider,
    }

    # Lazy load optional providers
    if provider_name == "anthropic":
        try:
            from ai_hub.providers.anthropic_provider import AnthropicProvider

            providers["anthropic"] = AnthropicProvider
        except ImportError as e:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. Install it with: pip install anthropic"
            ) from e

    if provider_name not in providers:
        available = list(providers.keys())
        raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")

    return providers[provider_name](**kwargs)
