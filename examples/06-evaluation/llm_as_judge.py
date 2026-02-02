"""LLM-as-Judge Evaluation Example.

This example demonstrates using an LLM to evaluate the quality of
other LLM outputs, a powerful technique for automated quality assessment.

Features demonstrated:
- Evaluation prompting patterns
- Structured scoring rubrics
- Comparing model outputs
- Aggregating evaluation metrics
"""

import json
from dataclasses import dataclass

from ai_hub import Message, OpenAIProvider, get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result of an LLM evaluation."""

    score: float
    reasoning: str
    criteria_scores: dict[str, float]


class LLMJudge:
    """LLM-as-Judge evaluator."""

    def __init__(self, model: str = "gpt-4o") -> None:
        """Initialize the judge.

        Args:
            model: Model to use for evaluation (should be capable model).
        """
        self.provider = OpenAIProvider(default_model=model)

    def evaluate_response(
        self,
        question: str,
        response: str,
        reference: str | None = None,
        criteria: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate a response to a question.

        Args:
            question: The original question.
            response: The response to evaluate.
            reference: Optional reference answer for comparison.
            criteria: Optional list of evaluation criteria.

        Returns:
            EvaluationResult with scores and reasoning.
        """
        if criteria is None:
            criteria = ["accuracy", "completeness", "clarity", "relevance"]

        criteria_list = "\n".join(f"- {c}" for c in criteria)

        system_prompt = f"""You are an expert evaluator assessing the quality of AI responses.

Evaluate the response based on these criteria:
{criteria_list}

For each criterion, provide a score from 1-5 where:
1 = Very Poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Respond in JSON format:
{{
    "overall_score": <1-5>,
    "reasoning": "<brief explanation>",
    "criteria_scores": {{
        "<criterion>": <score>,
        ...
    }}
}}"""

        user_content = f"Question: {question}\n\nResponse to evaluate:\n{response}"

        if reference:
            user_content += f"\n\nReference answer:\n{reference}"

        messages = [
            Message.system(system_prompt),
            Message.user(user_content),
        ]

        result = self.provider.complete(
            messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(result.content or "{}")
            return EvaluationResult(
                score=float(data.get("overall_score", 0)),
                reasoning=data.get("reasoning", ""),
                criteria_scores=data.get("criteria_scores", {}),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("evaluation_parse_error", error=str(e))
            return EvaluationResult(score=0, reasoning=f"Parse error: {e}", criteria_scores={})

    def compare_responses(
        self,
        question: str,
        response_a: str,
        response_b: str,
    ) -> dict[str, str | float]:
        """Compare two responses and determine which is better.

        Args:
            question: The original question.
            response_a: First response.
            response_b: Second response.

        Returns:
            Dictionary with winner and explanation.
        """
        system_prompt = """You are an expert evaluator comparing two AI responses.

Determine which response (A or B) is better overall. Consider:
- Accuracy of information
- Completeness
- Clarity and readability
- Relevance to the question

Respond in JSON format:
{
    "winner": "A" or "B" or "tie",
    "confidence": <0.0-1.0>,
    "explanation": "<detailed reasoning>"
}"""

        user_content = f"""Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Which response is better?"""

        messages = [
            Message.system(system_prompt),
            Message.user(user_content),
        ]

        result = self.provider.complete(
            messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(result.content or "{}")
            return {
                "winner": data.get("winner", "unknown"),
                "confidence": float(data.get("confidence", 0)),
                "explanation": data.get("explanation", ""),
            }
        except (json.JSONDecodeError, KeyError):
            return {"winner": "error", "confidence": 0, "explanation": "Parse error"}

    def batch_evaluate(
        self,
        test_cases: list[dict[str, str]],
    ) -> dict[str, float]:
        """Evaluate multiple test cases and compute aggregate metrics.

        Args:
            test_cases: List of dicts with 'question' and 'response' keys,
                       optionally 'reference'.

        Returns:
            Aggregate metrics.
        """
        scores: list[float] = []
        criteria_totals: dict[str, list[float]] = {}

        for i, case in enumerate(test_cases):
            logger.info("evaluating_case", case_number=i + 1, total=len(test_cases))

            result = self.evaluate_response(
                question=case["question"],
                response=case["response"],
                reference=case.get("reference"),
            )

            scores.append(result.score)

            for criterion, score in result.criteria_scores.items():
                if criterion not in criteria_totals:
                    criteria_totals[criterion] = []
                criteria_totals[criterion].append(score)

        # Compute aggregates
        metrics = {
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "num_evaluated": len(scores),
        }

        for criterion, criterion_scores in criteria_totals.items():
            metrics[f"mean_{criterion}"] = sum(criterion_scores) / len(criterion_scores)

        return metrics


def main() -> None:
    """Run the LLM-as-Judge examples."""
    print("=" * 60)
    print("LLM-as-Judge Evaluation Example")
    print("=" * 60)

    judge = LLMJudge(model="gpt-4o-mini")  # Use a capable model as judge

    # Example 1: Single response evaluation
    print("\n--- Single Response Evaluation ---")
    question = "What causes rain?"
    response = """Rain is caused by the water cycle. When the sun heats water in oceans,
    lakes, and rivers, it evaporates and rises as water vapor. As this vapor rises and cools,
    it condenses into tiny water droplets that form clouds. When these droplets combine and
    become heavy enough, they fall as rain."""

    result = judge.evaluate_response(question, response)
    print(f"Question: {question}")
    print(f"Overall Score: {result.score}/5")
    print(f"Reasoning: {result.reasoning}")
    print("Criteria Scores:")
    for criterion, score in result.criteria_scores.items():
        print(f"  - {criterion}: {score}/5")

    # Example 2: Compare two responses
    print("\n--- Response Comparison ---")
    question = "What is machine learning?"

    response_a = "Machine learning is a type of AI that learns from data."

    response_b = """Machine learning is a subset of artificial intelligence that enables
    systems to learn and improve from experience without being explicitly programmed.
    It focuses on developing algorithms that can access data, learn from it, and make
    predictions or decisions. Common applications include image recognition, natural
    language processing, and recommendation systems."""

    comparison = judge.compare_responses(question, response_a, response_b)
    print(f"Question: {question}")
    print(f"Winner: Response {comparison['winner']}")
    print(f"Confidence: {comparison['confidence']:.0%}")
    print(f"Explanation: {comparison['explanation']}")

    # Example 3: Batch evaluation
    print("\n--- Batch Evaluation ---")
    test_cases = [
        {
            "question": "What is Python?",
            "response": "Python is a programming language.",
            "reference": "Python is a high-level, interpreted programming language known for its readable syntax.",
        },
        {
            "question": "What is 2+2?",
            "response": "2+2 equals 4.",
        },
        {
            "question": "Explain photosynthesis",
            "response": "Photosynthesis is how plants make food using sunlight, water, and CO2.",
        },
    ]

    metrics = judge.batch_evaluate(test_cases)
    print("\nAggregate Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
