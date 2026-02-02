"""Vision/Multimodal Image Analysis Example.

This example demonstrates how to use GPT-4 Vision capabilities to
analyze and understand images using the proper Vision API format.

Features demonstrated:
- Proper image content format for Vision API
- URL-based image analysis
- Base64 image encoding
- Structured image descriptions
"""

import base64
import json
from pathlib import Path
from typing import Any

from ai_hub import Message, OpenAIProvider, get_logger, setup_logging
from ai_hub.providers.base import Role

setup_logging()
logger = get_logger(__name__)


class VisionAnalyzer:
    """Analyzer for image content using GPT-4 Vision."""

    def __init__(self, model: str = "gpt-4o") -> None:
        """Initialize the vision analyzer.

        Args:
            model: Vision-capable model to use.
        """
        self.model = model
        # Direct OpenAI client for vision API
        from openai import OpenAI

        self.client = OpenAI()

    def analyze_image_url(
        self,
        image_url: str,
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 500,
    ) -> str:
        """Analyze an image from a URL.

        Args:
            image_url: URL of the image to analyze.
            prompt: Analysis prompt.
            max_tokens: Maximum response tokens.

        Returns:
            Analysis text.
        """
        logger.info("analyzing_image_url", url=image_url[:50] + "...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": "auto"},
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""

    def analyze_image_base64(
        self,
        image_path: Path,
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 500,
    ) -> str:
        """Analyze a local image file.

        Args:
            image_path: Path to the image file.
            prompt: Analysis prompt.
            max_tokens: Maximum response tokens.

        Returns:
            Analysis text.
        """
        # Determine media type
        suffix = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/jpeg")

        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        logger.info("analyzing_local_image", path=str(image_path), size_kb=len(image_data) // 1024)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""

    def extract_structured_data(
        self,
        image_url: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract structured data from an image.

        Args:
            image_url: URL of the image.
            schema: JSON schema describing the data to extract.

        Returns:
            Extracted data as dictionary.
        """
        prompt = f"""Analyze this image and extract information according to this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with valid JSON matching the schema."""

        logger.info("extracting_structured_data")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return {"error": "Failed to parse response as JSON"}

    def compare_images(
        self,
        image_urls: list[str],
        comparison_prompt: str = "Compare these images and describe their similarities and differences.",
    ) -> str:
        """Compare multiple images.

        Args:
            image_urls: List of image URLs to compare.
            comparison_prompt: Prompt for comparison.

        Returns:
            Comparison text.
        """
        logger.info("comparing_images", count=len(image_urls))

        content: list[dict[str, Any]] = [{"type": "text", "text": comparison_prompt}]

        for url in image_urls:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "auto"},
                }
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1000,
        )

        return response.choices[0].message.content or ""


def main() -> None:
    """Run the vision analysis examples."""
    print("=" * 60)
    print("Vision/Multimodal Image Analysis")
    print("=" * 60)

    analyzer = VisionAnalyzer()

    # Example 1: Analyze image from URL
    print("\n--- URL Image Analysis ---")
    # Using a public domain image
    sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

    try:
        description = analyzer.analyze_image_url(
            sample_url,
            prompt="What do you see in this image? Describe the visual elements.",
        )
        print(f"Description: {description}")
    except Exception as e:
        print(f"Could not analyze URL image: {e}")

    # Example 2: Structured extraction
    print("\n--- Structured Data Extraction ---")
    schema = {
        "type": "object",
        "properties": {
            "main_subject": {"type": "string", "description": "The main subject of the image"},
            "colors": {"type": "array", "items": {"type": "string"}, "description": "Dominant colors"},
            "contains_text": {"type": "boolean"},
            "mood": {"type": "string", "description": "Overall mood or feeling"},
        },
    }

    try:
        structured_data = analyzer.extract_structured_data(sample_url, schema)
        print("Extracted data:")
        print(json.dumps(structured_data, indent=2))
    except Exception as e:
        print(f"Could not extract structured data: {e}")

    # Example 3: Image comparison (would need multiple images)
    print("\n--- Image Comparison ---")
    print("(Requires multiple image URLs - skipping in demo)")

    print("\n--- Usage Notes ---")
    print("1. For local images, use analyze_image_base64() with a file path")
    print("2. Supported formats: JPEG, PNG, GIF, WebP")
    print("3. Images are automatically resized by the API for optimal processing")
    print("4. Use 'detail: low' for faster processing of simple images")


if __name__ == "__main__":
    main()
