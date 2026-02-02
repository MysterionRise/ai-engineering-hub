"""When Traditional ML Beats LLMs.

This example demonstrates scenarios where traditional machine learning
approaches are more appropriate than LLMs, and how to combine them
in hybrid pipelines.

Features demonstrated:
- Use cases where ML > LLM
- Simple classification with sklearn
- Hybrid pipeline (ML + LLM)
- Cost and latency comparisons
"""

import time
from dataclasses import dataclass
from typing import Any

from ai_hub import Message, OpenAIProvider, get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Result of a prediction with metadata."""

    prediction: Any
    confidence: float
    latency_ms: float
    method: str


class HybridClassifier:
    """Classifier that combines ML and LLM approaches."""

    def __init__(self) -> None:
        """Initialize the hybrid classifier."""
        self.provider = OpenAIProvider(default_model="gpt-4o-mini")

        # Simple rule-based classifier for common cases
        self.sentiment_keywords = {
            "positive": ["great", "excellent", "amazing", "love", "fantastic", "wonderful", "best"],
            "negative": ["terrible", "awful", "hate", "worst", "horrible", "bad", "disappointing"],
        }

    def simple_sentiment(self, text: str) -> tuple[str | None, float]:
        """Simple keyword-based sentiment analysis.

        Args:
            text: Text to analyze.

        Returns:
            Tuple of (sentiment, confidence) or (None, 0) if uncertain.
        """
        text_lower = text.lower()

        pos_count = sum(1 for word in self.sentiment_keywords["positive"] if word in text_lower)
        neg_count = sum(1 for word in self.sentiment_keywords["negative"] if word in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return None, 0.0

        if pos_count > neg_count:
            return "positive", pos_count / total
        elif neg_count > pos_count:
            return "negative", neg_count / total
        else:
            return None, 0.0  # Unclear, need LLM

    def llm_sentiment(self, text: str) -> tuple[str, float]:
        """LLM-based sentiment analysis for complex cases.

        Args:
            text: Text to analyze.

        Returns:
            Tuple of (sentiment, confidence).
        """
        messages = [
            Message.system(
                "Analyze the sentiment of the text. Respond with ONLY 'positive', 'negative', or 'neutral'."
            ),
            Message.user(text),
        ]

        response = self.provider.complete(messages, temperature=0.1, max_tokens=10)
        result = (response.content or "neutral").strip().lower()

        # Map to standard labels
        if "positive" in result:
            return "positive", 0.8
        elif "negative" in result:
            return "negative", 0.8
        else:
            return "neutral", 0.7

    def classify_sentiment(self, text: str, force_llm: bool = False) -> PredictionResult:
        """Classify sentiment using hybrid approach.

        Args:
            text: Text to classify.
            force_llm: Force LLM usage even if simple method works.

        Returns:
            PredictionResult with prediction and metadata.
        """
        start_time = time.time()

        if not force_llm:
            # Try simple method first
            sentiment, confidence = self.simple_sentiment(text)
            if sentiment and confidence > 0.6:
                latency = (time.time() - start_time) * 1000
                return PredictionResult(
                    prediction=sentiment,
                    confidence=confidence,
                    latency_ms=latency,
                    method="rule_based",
                )

        # Fall back to LLM
        sentiment, confidence = self.llm_sentiment(text)
        latency = (time.time() - start_time) * 1000

        return PredictionResult(
            prediction=sentiment,
            confidence=confidence,
            latency_ms=latency,
            method="llm",
        )


def demonstrate_ml_advantages() -> None:
    """Demonstrate where traditional ML excels."""
    print("\n--- When to Use Traditional ML ---")
    print(
        """
1. HIGH-VOLUME, LOW-LATENCY REQUIREMENTS
   - Real-time fraud detection (needs <10ms)
   - Ad click prediction (billions of requests)
   - Recommendation systems (low latency critical)

2. STRUCTURED DATA WITH KNOWN PATTERNS
   - Tabular data classification
   - Time series forecasting
   - Anomaly detection in metrics

3. COST-SENSITIVE APPLICATIONS
   - LLM: ~$0.15-3.00 per 1M tokens
   - ML inference: ~$0.001 per 1M predictions

4. INTERPRETABILITY REQUIREMENTS
   - Regulatory compliance (explainable AI)
   - Feature importance analysis
   - Decision audit trails

5. OFFLINE/EDGE DEPLOYMENT
   - On-device inference
   - Air-gapped environments
   - Bandwidth-limited scenarios
"""
    )


def demonstrate_llm_advantages() -> None:
    """Demonstrate where LLMs excel."""
    print("\n--- When to Use LLMs ---")
    print(
        """
1. COMPLEX LANGUAGE UNDERSTANDING
   - Nuanced sentiment (sarcasm, context)
   - Open-domain Q&A
   - Summarization and paraphrasing

2. ZERO-SHOT / FEW-SHOT LEARNING
   - New tasks without training data
   - Rapid prototyping
   - Domain transfer

3. GENERATION TASKS
   - Content creation
   - Code generation
   - Creative writing

4. REASONING AND MULTI-STEP TASKS
   - Chain-of-thought problems
   - Planning and decomposition
   - Complex decision making

5. MULTIMODAL UNDERSTANDING
   - Image + text analysis
   - Document understanding
   - Audio transcription + analysis
"""
    )


def main() -> None:
    """Run the ML vs LLM comparison examples."""
    print("=" * 60)
    print("When Traditional ML Beats LLMs")
    print("=" * 60)

    # Show when each approach is better
    demonstrate_ml_advantages()
    demonstrate_llm_advantages()

    # Hybrid classifier demo
    print("\n--- Hybrid Classifier Demo ---")
    classifier = HybridClassifier()

    test_cases = [
        "This product is absolutely amazing! Best purchase ever!",  # Clear positive
        "Terrible experience, worst service I've ever had.",  # Clear negative
        "The movie was okay, had some good parts but also dragged in places.",  # Nuanced
        "I can't believe how bad this is... NOT! It's actually great.",  # Sarcasm
        "The implementation is efficient but lacks documentation.",  # Mixed technical
    ]

    print("\nClassification Results:")
    print("-" * 70)

    for text in test_cases:
        result = classifier.classify_sentiment(text)
        print(f"\nText: {text[:60]}...")
        print(f"  Prediction: {result.prediction}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Method: {result.method}")
        print(f"  Latency: {result.latency_ms:.1f}ms")

    # Cost comparison
    print("\n--- Cost Comparison (1M classifications) ---")
    print("Rule-based ML: ~$0.001 (compute only)")
    print("GPT-4o-mini:   ~$150 (at 100 tokens/request)")
    print("GPT-4o:        ~$500 (at 100 tokens/request)")
    print("\nHybrid approach: Use ML for 80% of clear cases, LLM for 20% edge cases")
    print("Estimated hybrid cost: ~$30-100 per 1M (80% savings)")


if __name__ == "__main__":
    main()
