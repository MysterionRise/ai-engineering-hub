"""Basic Chat Completion Example.

This example demonstrates the fundamental pattern for chat completions
using the AI Hub provider abstraction layer.

Features demonstrated:
- Creating a provider instance
- Constructing messages with system and user roles
- Handling responses with usage statistics
- Proper error handling and logging
"""

from ai_hub import Message, OpenAIProvider, get_logger, setup_logging

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def chat_completion_example() -> str:
    """Demonstrate basic chat completion.

    Returns:
        The assistant's response content.
    """
    # Initialize the provider
    provider = OpenAIProvider(default_model="gpt-4o")

    # Create the conversation
    messages = [
        Message.system(
            "You are a helpful assistant that provides clear, concise answers. "
            "Keep responses under 100 words unless asked for more detail."
        ),
        Message.user("What are the three laws of thermodynamics? Explain briefly."),
    ]

    logger.info("sending_request", model=provider.default_model)

    # Get the completion
    response = provider.complete(messages, temperature=0.7)

    # Log usage statistics
    if response.usage:
        logger.info(
            "completion_finished",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        )

    return response.content or ""


def multi_turn_conversation() -> list[str]:
    """Demonstrate a multi-turn conversation.

    Returns:
        List of assistant responses from the conversation.
    """
    provider = OpenAIProvider(default_model="gpt-4o-mini")
    responses: list[str] = []

    # Start the conversation
    messages = [
        Message.system("You are a friendly coding tutor."),
        Message.user("What is a Python list?"),
    ]

    # First turn
    response = provider.complete(messages, temperature=0.7)
    responses.append(response.content or "")
    logger.info("turn_1_complete", response_length=len(response.content or ""))

    # Add assistant response and continue conversation
    messages.append(Message.assistant(response.content or ""))
    messages.append(Message.user("Can you show me an example of list comprehension?"))

    # Second turn
    response = provider.complete(messages, temperature=0.7)
    responses.append(response.content or "")
    logger.info("turn_2_complete", response_length=len(response.content or ""))

    return responses


def main() -> None:
    """Run the chat completion examples."""
    print("=" * 60)
    print("Basic Chat Completion Example")
    print("=" * 60)

    # Run single turn example
    print("\n--- Single Turn ---")
    result = chat_completion_example()
    print(f"\nResponse:\n{result}")

    # Run multi-turn example
    print("\n--- Multi-Turn Conversation ---")
    responses = multi_turn_conversation()
    for i, response in enumerate(responses, 1):
        print(f"\nTurn {i}:\n{response[:500]}...")


if __name__ == "__main__":
    main()
