"""Streaming Chat Completion Example.

This example demonstrates how to stream responses from LLM providers,
which is essential for providing real-time feedback in user-facing applications.

Features demonstrated:
- Synchronous streaming with iterators
- Asynchronous streaming with async iterators
- Collecting streamed content
- Progress tracking during streaming
"""

import asyncio
import sys

from ai_hub import Message, OpenAIProvider, get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def stream_response() -> str:
    """Stream a response and print tokens as they arrive.

    Returns:
        The complete response text.
    """
    provider = OpenAIProvider(default_model="gpt-4o-mini")

    messages = [
        Message.system("You are a creative storyteller."),
        Message.user("Write a short story about a robot learning to paint. Keep it under 200 words."),
    ]

    print("Streaming response: ", end="", flush=True)

    collected_content: list[str] = []

    for chunk in provider.stream(messages, temperature=0.9):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            collected_content.append(chunk.content)

        if chunk.finish_reason:
            logger.info("stream_finished", finish_reason=chunk.finish_reason)

    print()  # New line after streaming
    return "".join(collected_content)


async def stream_response_async() -> str:
    """Async version of streaming response.

    Returns:
        The complete response text.
    """
    provider = OpenAIProvider(default_model="gpt-4o-mini")

    messages = [
        Message.system("You are a helpful assistant."),
        Message.user("List 5 interesting facts about the ocean."),
    ]

    print("Async streaming: ", end="", flush=True)

    collected_content: list[str] = []

    async for chunk in provider.stream_async(messages, temperature=0.7):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            collected_content.append(chunk.content)

    print()
    return "".join(collected_content)


def stream_with_progress() -> str:
    """Stream with a simple progress indicator.

    Returns:
        The complete response text.
    """
    provider = OpenAIProvider(default_model="gpt-4o-mini")

    messages = [
        Message.system("You are a technical writer."),
        Message.user("Explain how HTTP works in 3 paragraphs."),
    ]

    collected_content: list[str] = []
    token_count = 0

    for chunk in provider.stream(messages, temperature=0.5):
        if chunk.content:
            collected_content.append(chunk.content)
            token_count += 1

            # Update progress indicator
            if token_count % 10 == 0:
                sys.stderr.write(f"\rTokens received: {token_count}")
                sys.stderr.flush()

    sys.stderr.write(f"\rTokens received: {token_count} (complete)\n")
    return "".join(collected_content)


def main() -> None:
    """Run the streaming examples."""
    print("=" * 60)
    print("Streaming Examples")
    print("=" * 60)

    # Synchronous streaming
    print("\n--- Synchronous Streaming ---")
    result = stream_response()
    print(f"\n[Total length: {len(result)} characters]")

    # Async streaming
    print("\n--- Asynchronous Streaming ---")
    result = asyncio.run(stream_response_async())
    print(f"[Total length: {len(result)} characters]")

    # Streaming with progress
    print("\n--- Streaming with Progress ---")
    result = stream_with_progress()
    print(f"\nResult:\n{result}")


if __name__ == "__main__":
    main()
