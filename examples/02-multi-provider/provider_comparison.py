"""Multi-Provider Comparison Example.

This example demonstrates how to use the provider abstraction layer
to easily switch between different LLM providers (OpenAI, Anthropic, etc.)
while maintaining consistent code.

Features demonstrated:
- Provider factory pattern
- Comparing responses across providers
- Handling provider-specific features
- Graceful degradation when providers are unavailable
"""

from ai_hub import Message, OpenAIProvider, get_logger, setup_logging
from ai_hub.providers import get_provider

setup_logging()
logger = get_logger(__name__)


def compare_providers(prompt: str, providers: list[str]) -> dict[str, str]:
    """Compare responses from multiple providers.

    Args:
        prompt: The prompt to send to each provider.
        providers: List of provider names to compare.

    Returns:
        Dictionary mapping provider name to response.
    """
    results: dict[str, str] = {}

    messages = [
        Message.system("You are a helpful assistant. Keep responses concise."),
        Message.user(prompt),
    ]

    for provider_name in providers:
        try:
            provider = get_provider(provider_name)
            logger.info("trying_provider", provider=provider_name)

            response = provider.complete(messages, temperature=0.7, max_tokens=500)
            results[provider_name] = response.content or "(empty response)"

            if response.usage:
                logger.info(
                    "provider_response",
                    provider=provider_name,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

        except ImportError as e:
            logger.warning("provider_not_installed", provider=provider_name, error=str(e))
            results[provider_name] = f"(not installed: {e})"
        except Exception as e:
            logger.error("provider_failed", provider=provider_name, error=str(e))
            results[provider_name] = f"(error: {e})"

    return results


def provider_specific_features() -> None:
    """Demonstrate provider-specific features.

    Different providers may support different features. This example
    shows how to handle these differences gracefully.
    """
    # OpenAI-specific: using response_format for JSON
    print("--- OpenAI JSON Mode ---")
    openai_provider = OpenAIProvider(default_model="gpt-4o-mini")

    messages = [
        Message.system("You are a data extractor. Respond only with valid JSON."),
        Message.user("Extract key facts from: Python was created by Guido van Rossum in 1991."),
    ]

    response = openai_provider.complete(
        messages,
        response_format={"type": "json_object"},
    )
    print(f"JSON response: {response.content}")

    # Anthropic-specific features would go here when using Claude
    # e.g., extended thinking, tool use with specific formats, etc.


def with_fallback(prompt: str, preferred_providers: list[str]) -> tuple[str, str]:
    """Try providers in order, falling back on failure.

    Args:
        prompt: The prompt to send.
        preferred_providers: Ordered list of providers to try.

    Returns:
        Tuple of (provider_used, response_content).

    Raises:
        RuntimeError: If all providers fail.
    """
    messages = [
        Message.system("You are a helpful assistant."),
        Message.user(prompt),
    ]

    last_error: Exception | None = None

    for provider_name in preferred_providers:
        try:
            provider = get_provider(provider_name)
            response = provider.complete(messages, temperature=0.7)

            logger.info("provider_succeeded", provider=provider_name)
            return provider_name, response.content or ""

        except Exception as e:
            logger.warning("provider_failed_trying_next", provider=provider_name, error=str(e))
            last_error = e
            continue

    raise RuntimeError(f"All providers failed. Last error: {last_error}")


def main() -> None:
    """Run the multi-provider examples."""
    print("=" * 60)
    print("Multi-Provider Examples")
    print("=" * 60)

    # Example 1: Compare providers
    print("\n--- Provider Comparison ---")
    prompt = "Explain quantum entanglement in one sentence."

    # Only use OpenAI by default (Anthropic requires separate API key)
    available_providers = ["openai"]

    results = compare_providers(prompt, available_providers)
    for provider, response in results.items():
        print(f"\n{provider.upper()}:")
        print(f"  {response[:200]}..." if len(response) > 200 else f"  {response}")

    # Example 2: Provider-specific features
    print("\n--- Provider-Specific Features ---")
    provider_specific_features()

    # Example 3: Fallback pattern
    print("\n--- Fallback Pattern ---")
    try:
        provider_used, response = with_fallback(
            "What is 2 + 2?",
            ["openai"],  # Add "anthropic" if API key is configured
        )
        print(f"Used provider: {provider_used}")
        print(f"Response: {response}")
    except RuntimeError as e:
        print(f"All providers failed: {e}")


if __name__ == "__main__":
    main()
