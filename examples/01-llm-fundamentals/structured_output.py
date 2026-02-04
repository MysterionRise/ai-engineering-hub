"""Structured Output Example.

This example demonstrates how to get structured data from LLMs using:
1. JSON mode with explicit prompting
2. Response parsing with Pydantic models

Features demonstrated:
- Prompting for JSON output
- Parsing and validating LLM responses
- Type-safe structured data extraction
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from ai_hub import Message, OpenAIProvider, get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class MovieReview(BaseModel):
    """Structured movie review data."""

    title: str = Field(description="The movie title")
    year: int = Field(description="Release year")
    rating: float = Field(ge=0, le=10, description="Rating out of 10")
    genre: list[str] = Field(description="List of genres")
    summary: str = Field(description="Brief summary in 1-2 sentences")
    pros: list[str] = Field(description="List of positive aspects")
    cons: list[str] = Field(description="List of negative aspects")


def get_structured_json(prompt: str) -> dict[str, Any]:
    """Get a structured JSON response from the LLM.

    Args:
        prompt: The user prompt.

    Returns:
        Parsed JSON dictionary.
    """
    provider = OpenAIProvider(default_model="gpt-4o-mini")

    messages = [
        Message.system(
            "You are a helpful assistant that always responds in valid JSON format. "
            "Never include markdown code blocks or any text outside the JSON object."
        ),
        Message.user(prompt),
    ]

    response = provider.complete(
        messages,
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    if not response.content:
        raise ValueError("Empty response from LLM")

    return json.loads(response.content)


def get_movie_review(movie_name: str) -> MovieReview:
    """Get a structured movie review.

    Args:
        movie_name: Name of the movie to review.

    Returns:
        Validated MovieReview object.
    """
    provider = OpenAIProvider(default_model="gpt-4o-mini")

    schema_hint = MovieReview.model_json_schema()

    messages = [
        Message.system(
            "You are a movie critic. Respond with a JSON object matching this schema:\n"
            f"{json.dumps(schema_hint, indent=2)}\n\n"
            "Respond ONLY with valid JSON, no markdown or extra text."
        ),
        Message.user(f"Write a review for the movie: {movie_name}"),
    ]

    response = provider.complete(
        messages,
        temperature=0.5,
        response_format={"type": "json_object"},
    )

    if not response.content:
        raise ValueError("Empty response from LLM")

    data = json.loads(response.content)
    return MovieReview.model_validate(data)


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract named entities from text.

    Args:
        text: Text to extract entities from.

    Returns:
        Dictionary of entity types to entity values.
    """
    provider = OpenAIProvider(default_model="gpt-4o-mini")

    messages = [
        Message.system(
            "Extract named entities from the text. Respond with a JSON object with these keys:\n"
            "- people: list of person names\n"
            "- organizations: list of organization names\n"
            "- locations: list of location names\n"
            "- dates: list of dates or time references\n"
            "Respond ONLY with valid JSON."
        ),
        Message.user(text),
    ]

    response = provider.complete(
        messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    if not response.content:
        return {"people": [], "organizations": [], "locations": [], "dates": []}

    return json.loads(response.content)


def main() -> None:
    """Run the structured output examples."""
    print("=" * 60)
    print("Structured Output Examples")
    print("=" * 60)

    # Example 1: Basic JSON extraction
    print("\n--- Basic JSON Extraction ---")
    result = get_structured_json("List 3 programming languages with their year of creation and primary use case.")
    print(json.dumps(result, indent=2))

    # Example 2: Pydantic model validation
    print("\n--- Movie Review (Pydantic) ---")
    review = get_movie_review("The Matrix")
    print(f"Title: {review.title} ({review.year})")
    print(f"Rating: {review.rating}/10")
    print(f"Genres: {', '.join(review.genre)}")
    print(f"Summary: {review.summary}")
    print(f"Pros: {review.pros}")
    print(f"Cons: {review.cons}")

    # Example 3: Entity extraction
    print("\n--- Entity Extraction ---")
    sample_text = """
    Apple CEO Tim Cook announced yesterday that the company will open
    a new research facility in Austin, Texas next March. The $1 billion
    investment was praised by Governor Greg Abbott at a press conference
    in the Texas State Capitol.
    """
    entities = extract_entities(sample_text)
    print(json.dumps(entities, indent=2))


if __name__ == "__main__":
    main()
