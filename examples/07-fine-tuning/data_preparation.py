"""Fine-Tuning Data Preparation Example.

This example demonstrates how to prepare and validate data for fine-tuning
OpenAI models, including format conversion, validation, and quality checks.

Features demonstrated:
- JSONL format for fine-tuning data
- Data validation and quality checks
- Token counting and cost estimation
- Train/validation split
"""

import json
import random
from pathlib import Path
from typing import Any

from ai_hub import count_tokens, get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class FineTuningDataPreparer:
    """Prepare data for OpenAI fine-tuning."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        """Initialize the data preparer.

        Args:
            model: Target model for token counting.
        """
        self.model = model
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_message(self, message: dict[str, Any]) -> bool:
        """Validate a single message in a conversation.

        Args:
            message: Message dict with 'role' and 'content'.

        Returns:
            True if valid.
        """
        if not isinstance(message, dict):
            self.errors.append(f"Message is not a dict: {type(message)}")
            return False

        if "role" not in message:
            self.errors.append("Message missing 'role' field")
            return False

        if message["role"] not in ["system", "user", "assistant"]:
            self.errors.append(f"Invalid role: {message['role']}")
            return False

        if "content" not in message:
            self.errors.append("Message missing 'content' field")
            return False

        if not message["content"] or not message["content"].strip():
            self.warnings.append("Message has empty content")

        return True

    def validate_example(self, example: dict[str, Any], index: int) -> bool:
        """Validate a training example.

        Args:
            example: Training example with 'messages' key.
            index: Example index for error reporting.

        Returns:
            True if valid.
        """
        if not isinstance(example, dict):
            self.errors.append(f"Example {index}: Not a dict")
            return False

        if "messages" not in example:
            self.errors.append(f"Example {index}: Missing 'messages' field")
            return False

        messages = example["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            self.errors.append(f"Example {index}: 'messages' must be a list with at least 2 messages")
            return False

        # Validate each message
        for i, msg in enumerate(messages):
            if not self.validate_message(msg):
                self.errors.append(f"Example {index}, message {i}: Invalid")
                return False

        # Check for assistant response
        has_assistant = any(m["role"] == "assistant" for m in messages)
        if not has_assistant:
            self.errors.append(f"Example {index}: Must have at least one assistant message")
            return False

        return True

    def estimate_tokens(self, example: dict[str, Any]) -> int:
        """Estimate tokens for a training example.

        Args:
            example: Training example.

        Returns:
            Estimated token count.
        """
        total = 0
        for msg in example.get("messages", []):
            content = msg.get("content", "")
            # Add overhead for message formatting
            total += count_tokens(content, self.model) + 4  # Role tokens + separators
        return total

    def prepare_data(
        self,
        raw_data: list[dict[str, Any]],
        validation_split: float = 0.1,
        min_examples: int = 10,
        max_tokens_per_example: int = 4096,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Prepare and split data for fine-tuning.

        Args:
            raw_data: List of training examples.
            validation_split: Fraction for validation set.
            min_examples: Minimum required examples.
            max_tokens_per_example: Maximum tokens per example.

        Returns:
            Tuple of (training_data, validation_data).
        """
        self.errors = []
        self.warnings = []

        logger.info("validating_data", total_examples=len(raw_data))

        # Validate and filter
        valid_examples = []
        for i, example in enumerate(raw_data):
            if self.validate_example(example, i):
                tokens = self.estimate_tokens(example)
                if tokens > max_tokens_per_example:
                    self.warnings.append(f"Example {i}: {tokens} tokens exceeds limit")
                else:
                    valid_examples.append(example)

        logger.info(
            "validation_complete",
            valid=len(valid_examples),
            errors=len(self.errors),
            warnings=len(self.warnings),
        )

        if len(valid_examples) < min_examples:
            raise ValueError(f"Only {len(valid_examples)} valid examples, minimum is {min_examples}")

        # Shuffle and split
        random.shuffle(valid_examples)
        split_idx = int(len(valid_examples) * (1 - validation_split))

        train_data = valid_examples[:split_idx]
        val_data = valid_examples[split_idx:]

        return train_data, val_data

    def save_jsonl(self, data: list[dict[str, Any]], output_path: Path) -> None:
        """Save data in JSONL format.

        Args:
            data: List of examples.
            output_path: Output file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")

        logger.info("saved_jsonl", path=str(output_path), examples=len(data))

    def estimate_cost(
        self,
        data: list[dict[str, Any]],
        epochs: int = 3,
        cost_per_1k_tokens: float = 0.008,  # GPT-4o-mini fine-tuning cost
    ) -> dict[str, float]:
        """Estimate fine-tuning cost.

        Args:
            data: Training data.
            epochs: Number of training epochs.
            cost_per_1k_tokens: Cost per 1000 training tokens.

        Returns:
            Cost estimation dict.
        """
        total_tokens = sum(self.estimate_tokens(ex) for ex in data)
        training_tokens = total_tokens * epochs

        return {
            "total_examples": len(data),
            "tokens_per_example_avg": total_tokens / len(data) if data else 0,
            "total_tokens": total_tokens,
            "training_tokens": training_tokens,
            "estimated_cost_usd": (training_tokens / 1000) * cost_per_1k_tokens,
        }

    def get_statistics(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Get statistics about the dataset.

        Args:
            data: Training data.

        Returns:
            Statistics dict.
        """
        if not data:
            return {"error": "No data"}

        token_counts = [self.estimate_tokens(ex) for ex in data]
        message_counts = [len(ex.get("messages", [])) for ex in data]

        return {
            "num_examples": len(data),
            "tokens": {
                "min": min(token_counts),
                "max": max(token_counts),
                "mean": sum(token_counts) / len(token_counts),
                "total": sum(token_counts),
            },
            "messages_per_example": {
                "min": min(message_counts),
                "max": max(message_counts),
                "mean": sum(message_counts) / len(message_counts),
            },
        }


def create_sample_data() -> list[dict[str, Any]]:
    """Create sample training data for demonstration."""
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "How do I create a list in Python?"},
                {"role": "assistant", "content": "You can create a list using square brackets: `my_list = [1, 2, 3]`"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "What is a dictionary in Python?"},
                {
                    "role": "assistant",
                    "content": "A dictionary is a collection of key-value pairs: `my_dict = {'key': 'value'}`",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "How do I write a for loop?"},
                {
                    "role": "assistant",
                    "content": "Use `for item in iterable:` followed by indented code block.",
                },
            ]
        },
        # Add more examples...
    ] * 5  # Duplicate to have enough examples


def main() -> None:
    """Run the fine-tuning data preparation example."""
    print("=" * 60)
    print("Fine-Tuning Data Preparation")
    print("=" * 60)

    preparer = FineTuningDataPreparer()

    # Create sample data
    raw_data = create_sample_data()
    print(f"\nRaw data: {len(raw_data)} examples")

    # Prepare and validate
    print("\n--- Validation ---")
    train_data, val_data = preparer.prepare_data(raw_data, validation_split=0.2)

    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    if preparer.errors:
        print(f"\nErrors ({len(preparer.errors)}):")
        for err in preparer.errors[:5]:
            print(f"  - {err}")

    if preparer.warnings:
        print(f"\nWarnings ({len(preparer.warnings)}):")
        for warn in preparer.warnings[:5]:
            print(f"  - {warn}")

    # Statistics
    print("\n--- Dataset Statistics ---")
    stats = preparer.get_statistics(train_data)
    print(f"Training examples: {stats['num_examples']}")
    print(f"Tokens - min: {stats['tokens']['min']}, max: {stats['tokens']['max']}, mean: {stats['tokens']['mean']:.1f}")

    # Cost estimation
    print("\n--- Cost Estimation ---")
    cost = preparer.estimate_cost(train_data, epochs=3)
    print(f"Total tokens: {cost['total_tokens']:,}")
    print(f"Training tokens (3 epochs): {cost['training_tokens']:,}")
    print(f"Estimated cost: ${cost['estimated_cost_usd']:.2f}")

    # Save to JSONL (in scratchpad for demo)
    print("\n--- Saving Data ---")
    output_dir = Path("/private/tmp/claude/fine-tuning-demo")
    preparer.save_jsonl(train_data, output_dir / "train.jsonl")
    preparer.save_jsonl(val_data, output_dir / "validation.jsonl")
    print(f"Data saved to {output_dir}")


if __name__ == "__main__":
    main()
