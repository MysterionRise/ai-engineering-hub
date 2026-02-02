"""Token counting utilities using tiktoken.

This module provides functions for counting and encoding tokens,
useful for managing context windows and estimating costs.
"""

from functools import lru_cache

import tiktoken

# Model to encoding mapping
MODEL_ENCODINGS: dict[str, str] = {
    # GPT-4 models
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    # GPT-3.5 models
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    # Claude models (approximate with cl100k_base)
    "claude-sonnet-4-20250514": "cl100k_base",
    "claude-opus-4-20250514": "cl100k_base",
    "claude-3-5-sonnet-20241022": "cl100k_base",
    "claude-3-5-haiku-20241022": "cl100k_base",
    "claude-3-opus-20240229": "cl100k_base",
}

# Default context window sizes
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "claude-sonnet-4-20250514": 200000,
    "claude-opus-4-20250514": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-opus-20240229": 200000,
}


@lru_cache(maxsize=10)
def get_encoding(encoding_name: str) -> tiktoken.Encoding:
    """Get a cached tiktoken encoding.

    Args:
        encoding_name: Name of the encoding (e.g., "cl100k_base").

    Returns:
        The tiktoken Encoding object.
    """
    return tiktoken.get_encoding(encoding_name)


@lru_cache(maxsize=20)
def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    """Get the tiktoken encoding for a specific model.

    Args:
        model: The model name (e.g., "gpt-4o").

    Returns:
        The tiktoken Encoding object appropriate for the model.
    """
    # Try model-specific encoding first
    if model in MODEL_ENCODINGS:
        return get_encoding(MODEL_ENCODINGS[model])

    # Try tiktoken's built-in model mapping
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Default to cl100k_base for unknown models
        return get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in a text string.

    Args:
        text: The text to count tokens for.
        model: The model to use for encoding (affects token count).

    Returns:
        The number of tokens in the text.
    """
    encoding = get_encoding_for_model(model)
    return len(encoding.encode(text))


def count_tokens_with_encoding(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens using a specific encoding.

    Args:
        text: The text to count tokens for.
        encoding_name: The encoding name to use.

    Returns:
        The number of tokens in the text.
    """
    encoding = get_encoding(encoding_name)
    return len(encoding.encode(text))


def encode(text: str, model: str = "gpt-4o") -> list[int]:
    """Encode text into token IDs.

    Args:
        text: The text to encode.
        model: The model to use for encoding.

    Returns:
        List of token IDs.
    """
    encoding = get_encoding_for_model(model)
    return encoding.encode(text)


def decode(tokens: list[int], model: str = "gpt-4o") -> str:
    """Decode token IDs back to text.

    Args:
        tokens: List of token IDs.
        model: The model to use for decoding.

    Returns:
        The decoded text string.
    """
    encoding = get_encoding_for_model(model)
    return encoding.decode(tokens)


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    model: str = "gpt-4o",
    truncation_marker: str = "...",
) -> str:
    """Truncate text to fit within a token limit.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens allowed.
        model: The model to use for encoding.
        truncation_marker: Marker to append if truncated.

    Returns:
        The truncated text, or original if within limit.
    """
    encoding = get_encoding_for_model(model)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Reserve space for truncation marker
    marker_tokens = encoding.encode(truncation_marker)
    available_tokens = max_tokens - len(marker_tokens)

    if available_tokens <= 0:
        return truncation_marker

    truncated_tokens = tokens[:available_tokens]
    return encoding.decode(truncated_tokens) + truncation_marker


def estimate_message_tokens(
    messages: list[dict[str, str]],
    model: str = "gpt-4o",
) -> int:
    """Estimate tokens for a list of chat messages.

    This includes overhead for message formatting.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        model: The model to use for encoding.

    Returns:
        Estimated total token count.
    """
    encoding = get_encoding_for_model(model)

    # Token overhead per message (varies by model, using GPT-4 estimate)
    tokens_per_message = 3  # <|start|>role<|sep|>
    tokens_per_name = 1  # if name is present

    total = 0
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            total += len(encoding.encode(str(value)))
            if key == "name":
                total += tokens_per_name

    total += 3  # Assistant reply priming
    return total


def get_context_window(model: str) -> int:
    """Get the context window size for a model.

    Args:
        model: The model name.

    Returns:
        The context window size in tokens.
    """
    return MODEL_CONTEXT_WINDOWS.get(model, 8192)


def check_context_fit(
    text: str,
    model: str = "gpt-4o",
    max_output_tokens: int = 4096,
) -> tuple[bool, int, int]:
    """Check if text fits within model's context window.

    Args:
        text: The input text.
        model: The model to check against.
        max_output_tokens: Reserved tokens for output.

    Returns:
        Tuple of (fits, input_tokens, available_tokens).
    """
    input_tokens = count_tokens(text, model)
    context_window = get_context_window(model)
    available = context_window - max_output_tokens
    return input_tokens <= available, input_tokens, available
