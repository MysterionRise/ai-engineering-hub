"""AI Engineering Hub - Production-grade AI patterns and examples.

This package provides a comprehensive toolkit for building AI applications,
including multi-provider LLM support, RAG patterns, and production utilities.

Quick start:
    from ai_hub import OpenAIProvider, Message

    provider = OpenAIProvider()
    response = provider.complete([
        Message.system("You are a helpful assistant."),
        Message.user("Hello!"),
    ])
    print(response.content)
"""

__version__ = "0.1.0"

# Core utilities
from ai_hub.core import (
    Settings,
    get_logger,
    get_settings,
    setup_logging,
)

# Provider interface
from ai_hub.providers import (
    CompletionResponse,
    Message,
    OpenAIProvider,
    Role,
    ToolCall,
    ToolDefinition,
    Usage,
    get_provider,
)

# Token utilities
from ai_hub.utils import (
    count_tokens,
    get_context_window,
    truncate_to_token_limit,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "Settings",
    "get_logger",
    "get_settings",
    "setup_logging",
    # Providers
    "CompletionResponse",
    "Message",
    "OpenAIProvider",
    "Role",
    "ToolCall",
    "ToolDefinition",
    "Usage",
    "get_provider",
    # Utils
    "count_tokens",
    "get_context_window",
    "truncate_to_token_limit",
]
