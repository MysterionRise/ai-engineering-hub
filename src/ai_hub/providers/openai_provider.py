"""OpenAI provider implementation.

This module provides the OpenAI implementation of the LLM provider interface,
supporting chat completions, streaming, and function calling with the modern
tools API.
"""

import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI
from openai import RateLimitError as OpenAIRateLimitError

from ai_hub.core.config import get_settings
from ai_hub.core.errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    ModelNotFoundError,
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
    ToolCall,
    ToolDefinition,
    Usage,
)

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):  # type: ignore[misc]
    """OpenAI provider implementation."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gpt-4o",
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. Defaults to environment variable.
            default_model: Default model to use for completions.
        """
        settings = get_settings()
        self.api_key = api_key or settings.openai_key
        self.default_model = default_model
        self.default_temperature = settings.default_temperature
        self.default_max_tokens = settings.default_max_tokens

        self._client = OpenAI(api_key=self.api_key)
        self._async_client = AsyncOpenAI(api_key=self.api_key)

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to OpenAI API format."""
        result = []
        for msg in messages:
            converted: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }
            if msg.name:
                converted["name"] = msg.name
            if msg.tool_call_id:
                converted["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                converted["tool_calls"] = msg.tool_calls
            result.append(converted)
        return result

    def _convert_tools(self, tools: list[ToolDefinition] | None) -> list[dict[str, Any]] | None:
        """Convert ToolDefinition objects to OpenAI API format."""
        if not tools:
            return None
        return [tool.to_openai_format() for tool in tools]

    def _handle_error(self, error: Exception) -> None:
        """Convert OpenAI errors to AI Hub errors."""
        if isinstance(error, OpenAIRateLimitError):
            raise RateLimitError(
                str(error),
                provider=self.provider_name,
                retry_after=getattr(error, "retry_after", None),
            )
        elif isinstance(error, APIStatusError):
            if error.status_code == 401:
                raise AuthenticationError(str(error), provider=self.provider_name)
            elif error.status_code == 404:
                raise ModelNotFoundError(
                    model="unknown",
                    provider=self.provider_name,
                )
            elif error.status_code == 400 and "context_length" in str(error).lower():
                raise ContextLengthError(str(error), provider=self.provider_name)
            elif error.status_code == 400 and "content_filter" in str(error).lower():
                raise ContentFilterError(str(error), provider=self.provider_name)
            else:
                raise ProviderError(str(error), provider=self.provider_name)
        elif isinstance(error, APIConnectionError):
            raise ProviderError(f"Connection error: {error}", provider=self.provider_name)
        else:
            raise ProviderError(str(error), provider=self.provider_name)

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
        """Generate a completion using OpenAI."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        start_time = time.time()

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=cast(Any, self._convert_messages(messages)),
                temperature=temperature,
                max_tokens=max_tokens,
                tools=cast(Any, self._convert_tools(tools)),
                **kwargs,
            )
        except Exception as e:
            self._handle_error(e)
            raise  # Re-raise if not converted

        latency_ms = (time.time() - start_time) * 1000

        # Extract tool calls if present
        tool_calls = []
        choice = response.choices[0]
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                func = getattr(tc, "function", None)
                if func:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=getattr(func, "name", ""),
                            arguments=getattr(func, "arguments", ""),
                        )
                    )

        # Log the call
        usage = None
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
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
            content=choice.message.content,
            model=response.model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
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
        """Async completion using OpenAI."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        start_time = time.time()

        try:
            response = await self._async_client.chat.completions.create(
                model=model,
                messages=cast(Any, self._convert_messages(messages)),
                temperature=temperature,
                max_tokens=max_tokens,
                tools=cast(Any, self._convert_tools(tools)),
                **kwargs,
            )
        except Exception as e:
            self._handle_error(e)
            raise

        latency_ms = (time.time() - start_time) * 1000

        # Extract tool calls if present
        tool_calls = []
        choice = response.choices[0]
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                func = getattr(tc, "function", None)
                if func:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=getattr(func, "name", ""),
                            arguments=getattr(func, "arguments", ""),
                        )
                    )

        # Log the call
        usage = None
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
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
            content=choice.message.content,
            model=response.model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
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
        """Stream a completion using OpenAI."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        try:
            stream = self._client.chat.completions.create(
                model=model,
                messages=cast(Any, self._convert_messages(messages)),
                temperature=temperature,
                max_tokens=max_tokens,
                tools=cast(Any, self._convert_tools(tools)),
                stream=True,
                **kwargs,
            )
        except Exception as e:
            self._handle_error(e)
            raise

        for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta

                # Extract tool calls from delta
                tool_calls = []
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function:
                            tool_calls.append(
                                ToolCall(
                                    id=tc.id or "",
                                    name=tc.function.name or "",
                                    arguments=tc.function.arguments or "",
                                )
                            )

                yield CompletionChunk(
                    content=delta.content,
                    tool_calls=tool_calls,
                    finish_reason=choice.finish_reason,
                )

    async def stream_async(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        """Async stream a completion using OpenAI."""
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        try:
            stream = await self._async_client.chat.completions.create(
                model=model,
                messages=cast(Any, self._convert_messages(messages)),
                temperature=temperature,
                max_tokens=max_tokens,
                tools=cast(Any, self._convert_tools(tools)),
                stream=True,
                **kwargs,
            )
        except Exception as e:
            self._handle_error(e)
            raise

        async for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta

                # Extract tool calls from delta
                tool_calls = []
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function:
                            tool_calls.append(
                                ToolCall(
                                    id=tc.id or "",
                                    name=tc.function.name or "",
                                    arguments=tc.function.arguments or "",
                                )
                            )

                yield CompletionChunk(
                    content=delta.content,
                    tool_calls=tool_calls,
                    finish_reason=choice.finish_reason,
                )

    def list_models(self) -> list[str]:
        """List available OpenAI models."""
        try:
            models = self._client.models.list()
            return sorted([m.id for m in models.data if "gpt" in m.id.lower()])
        except Exception as e:
            logger.warning("Failed to list models", error=str(e))
            # Return commonly available models as fallback
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
            ]
