"""Anthropic provider implementation.

This module provides the Anthropic Claude implementation of the LLM provider
interface, supporting chat completions, streaming, and tool use.
"""

import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ai_hub.core.config import get_settings
from ai_hub.core.errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    ProviderError,
    RateLimitError,
)
from ai_hub.core.logging import get_logger, log_llm_call
from ai_hub.core.retry import with_retry
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

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):  # type: ignore[misc]
    """Anthropic Claude provider implementation."""

    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. Defaults to environment variable.
            default_model: Default model to use for completions.
        """
        try:
            from anthropic import Anthropic, AsyncAnthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. Install it with: pip install anthropic"
            ) from e

        settings = get_settings()
        self.api_key = api_key or settings.anthropic_key
        self.default_model = default_model
        self.default_temperature = settings.default_temperature
        self.default_max_tokens = settings.default_max_tokens

        self._client = Anthropic(api_key=self.api_key)
        self._async_client = AsyncAnthropic(api_key=self.api_key)

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert Message objects to Anthropic API format.

        Anthropic requires system message to be separate from other messages.

        Returns:
            Tuple of (system_prompt, messages_list)
        """
        system_prompt = None
        converted = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            elif msg.role == Role.TOOL:
                # Anthropic uses tool_result for tool responses
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )
            else:
                converted.append(
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                    }
                )

        return system_prompt, converted

    def _convert_tools(self, tools: list[ToolDefinition] | None) -> list[dict[str, Any]] | None:
        """Convert ToolDefinition objects to Anthropic API format."""
        if not tools:
            return None
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def _handle_error(self, error: Exception) -> None:
        """Convert Anthropic errors to AI Hub errors."""
        from anthropic import APIConnectionError, APIStatusError
        from anthropic import RateLimitError as AnthropicRateLimitError

        if isinstance(error, AnthropicRateLimitError):
            raise RateLimitError(
                str(error),
                provider=self.provider_name,
            )
        elif isinstance(error, APIStatusError):
            if error.status_code == 401:
                raise AuthenticationError(str(error), provider=self.provider_name)
            elif "context" in str(error).lower() or "token" in str(error).lower():
                raise ContextLengthError(str(error), provider=self.provider_name)
            elif "safety" in str(error).lower() or "harmful" in str(error).lower():
                raise ContentFilterError(str(error), provider=self.provider_name)
            else:
                raise ProviderError(str(error), provider=self.provider_name)
        elif isinstance(error, APIConnectionError):
            raise ProviderError(f"Connection error: {error}", provider=self.provider_name)
        else:
            raise ProviderError(str(error), provider=self.provider_name)

    def _extract_tool_calls(self, content: list[Any]) -> tuple[str | None, list[ToolCall]]:
        """Extract text content and tool calls from Anthropic response."""
        import json

        text_parts = []
        tool_calls = []

        for block in content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input),
                    )
                )

        text_content = "\n".join(text_parts) if text_parts else None
        return text_content, tool_calls

    @with_retry  # type: ignore[untyped-decorator]
    def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion using Anthropic Claude."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        system_prompt, converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)

        start_time = time.time()

        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": converted_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }
            if system_prompt:
                request_kwargs["system"] = system_prompt
            if converted_tools:
                request_kwargs["tools"] = converted_tools

            response = self._client.messages.create(**request_kwargs)
        except Exception as e:
            self._handle_error(e)
            raise

        latency_ms = (time.time() - start_time) * 1000

        # Extract content and tool calls
        text_content, tool_calls = self._extract_tool_calls(response.content)

        # Log the call
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        log_llm_call(
            logger,
            provider=self.provider_name,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
        )

        return CompletionResponse(
            content=text_content,
            model=response.model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            raw_response=response,
        )

    async def complete_async(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Async completion using Anthropic Claude."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        system_prompt, converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)

        start_time = time.time()

        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": converted_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }
            if system_prompt:
                request_kwargs["system"] = system_prompt
            if converted_tools:
                request_kwargs["tools"] = converted_tools

            response = await self._async_client.messages.create(**request_kwargs)
        except Exception as e:
            self._handle_error(e)
            raise

        latency_ms = (time.time() - start_time) * 1000

        # Extract content and tool calls
        text_content, tool_calls = self._extract_tool_calls(response.content)

        # Log the call
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        log_llm_call(
            logger,
            provider=self.provider_name,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
        )

        return CompletionResponse(
            content=text_content,
            model=response.model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            raw_response=response,
        )

    def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> Iterator[CompletionChunk]:
        """Stream a completion using Anthropic Claude."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        system_prompt, converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)

        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": converted_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }
            if system_prompt:
                request_kwargs["system"] = system_prompt
            if converted_tools:
                request_kwargs["tools"] = converted_tools

            with self._client.messages.stream(**request_kwargs) as stream:
                for text in stream.text_stream:
                    yield CompletionChunk(content=text)

                # Final chunk with finish reason
                final_message = stream.get_final_message()
                yield CompletionChunk(
                    content=None,
                    finish_reason=final_message.stop_reason,
                )
        except Exception as e:
            self._handle_error(e)
            raise

    async def stream_async(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        """Async stream a completion using Anthropic Claude."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        system_prompt, converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)

        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": converted_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }
            if system_prompt:
                request_kwargs["system"] = system_prompt
            if converted_tools:
                request_kwargs["tools"] = converted_tools

            async with self._async_client.messages.stream(**request_kwargs) as stream:
                async for text in stream.text_stream:
                    yield CompletionChunk(content=text)

                # Final chunk with finish reason
                final_message = await stream.get_final_message()
                yield CompletionChunk(
                    content=None,
                    finish_reason=final_message.stop_reason,
                )
        except Exception as e:
            self._handle_error(e)
            raise

    def list_models(self) -> list[str]:
        """List available Anthropic Claude models."""
        # Anthropic doesn't have a list models API, so return known models
        return [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
