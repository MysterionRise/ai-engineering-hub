"""Tests for token utilities."""

from ai_hub.utils.tokens import (
    check_context_fit,
    count_tokens,
    decode,
    encode,
    get_context_window,
    truncate_to_token_limit,
)


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_count_simple_text(self) -> None:
        """Test counting tokens in simple text."""
        text = "Hello, world!"
        tokens = count_tokens(text, "gpt-4o")

        assert isinstance(tokens, int)
        assert tokens > 0
        assert tokens < 10  # Simple text should be few tokens

    def test_count_empty_text(self) -> None:
        """Test counting tokens in empty text."""
        tokens = count_tokens("", "gpt-4o")
        assert tokens == 0

    def test_different_models_may_differ(self) -> None:
        """Test that different models may have different token counts."""
        text = "Hello, world!"
        tokens_gpt4 = count_tokens(text, "gpt-4o")
        tokens_gpt35 = count_tokens(text, "gpt-3.5-turbo")

        # Both should be reasonable counts
        assert 1 <= tokens_gpt4 <= 10
        assert 1 <= tokens_gpt35 <= 10


class TestEncodeDecode:
    """Tests for encode and decode functions."""

    def test_encode_decode_roundtrip(self) -> None:
        """Test that encode/decode is lossless."""
        text = "Hello, world! How are you today?"
        tokens = encode(text, "gpt-4o")
        decoded = decode(tokens, "gpt-4o")

        assert decoded == text

    def test_encode_returns_integers(self) -> None:
        """Test that encode returns list of integers."""
        tokens = encode("Hello", "gpt-4o")

        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)


class TestTruncateToTokenLimit:
    """Tests for truncate_to_token_limit function."""

    def test_no_truncation_needed(self) -> None:
        """Test that short text is not truncated."""
        text = "Hello"
        result = truncate_to_token_limit(text, max_tokens=100, model="gpt-4o")

        assert result == text

    def test_truncation_applied(self) -> None:
        """Test that long text is truncated."""
        text = "Hello world! " * 100
        result = truncate_to_token_limit(text, max_tokens=10, model="gpt-4o")

        assert len(result) < len(text)
        assert result.endswith("...")

    def test_custom_truncation_marker(self) -> None:
        """Test custom truncation marker."""
        text = "Hello world! " * 100
        result = truncate_to_token_limit(
            text,
            max_tokens=10,
            model="gpt-4o",
            truncation_marker=" [truncated]",
        )

        assert result.endswith("[truncated]")


class TestGetContextWindow:
    """Tests for get_context_window function."""

    def test_known_model(self) -> None:
        """Test context window for known model."""
        window = get_context_window("gpt-4o")
        assert window == 128000

    def test_unknown_model_default(self) -> None:
        """Test context window for unknown model uses default."""
        window = get_context_window("unknown-model-xyz")
        assert window == 8192  # Default


class TestCheckContextFit:
    """Tests for check_context_fit function."""

    def test_fits_context(self) -> None:
        """Test text that fits in context."""
        fits, input_tokens, available = check_context_fit(
            "Hello world!",
            model="gpt-4o",
            max_output_tokens=4096,
        )

        assert fits is True
        assert input_tokens < 10
        assert available > input_tokens

    def test_exceeds_context(self) -> None:
        """Test text that exceeds context."""
        # Create very long text
        long_text = "word " * 200000

        fits, input_tokens, available = check_context_fit(
            long_text,
            model="gpt-4",  # 8192 context
            max_output_tokens=4096,
        )

        assert fits is False
        assert input_tokens > available
