"""Abstract base class for LLM providers.

This module defines the interface that all LLM providers must implement,
enabling provider-agnostic code and easy switching between providers.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A message in a conversation."""

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> "Message":
        """Create a tool response message."""
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)


@dataclass
class ToolDefinition:
    """Definition of a tool/function the model can call."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tools API format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A tool call made by the model."""

    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class Usage:
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


@dataclass
class CompletionResponse:
    """Response from a completion request."""

    content: str | None
    model: str
    usage: Usage | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    raw_response: Any = None


@dataclass
class CompletionChunk:
    """A chunk from a streaming completion response."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class and
    implement its abstract methods.
    """

    provider_name: str = "base"

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion for the given messages.

        Args:
            messages: The conversation history.
            model: The model to use. Defaults to provider's default.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            tools: Available tools for function calling.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The completion response.
        """
        ...

    @abstractmethod
    async def complete_async(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Async version of complete()."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> Iterator[CompletionChunk]:
        """Stream a completion for the given messages.

        Args:
            messages: The conversation history.
            model: The model to use. Defaults to provider's default.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            tools: Available tools for function calling.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Completion chunks as they arrive.
        """
        ...

    @abstractmethod
    async def stream_async(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        """Async version of stream()."""
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available models for this provider.

        Returns:
            List of model identifiers.
        """
        ...
