"""ReAct Agent Implementation.

This example demonstrates the ReAct (Reasoning and Acting) pattern,
where the agent iteratively reasons about a problem, takes actions,
and observes results until reaching a conclusion.

Features demonstrated:
- ReAct prompting pattern
- Tool execution loop
- Observation-based reasoning
- Explicit thought chain
"""

import re
from typing import Any

from ai_hub import Message, OpenAIProvider, get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class ReActAgent:
    """ReAct agent that reasons and acts iteratively."""

    def __init__(self, max_iterations: int = 10) -> None:
        """Initialize the ReAct agent.

        Args:
            max_iterations: Maximum reasoning iterations.
        """
        self.provider = OpenAIProvider(default_model="gpt-4o")
        self.max_iterations = max_iterations
        self.tools = self._define_tools()

    def _define_tools(self) -> dict[str, dict[str, Any]]:
        """Define available tools and their implementations."""
        return {
            "search": {
                "description": "Search for information. Args: query (string)",
                "function": self._mock_search,
            },
            "calculate": {
                "description": "Perform calculations. Args: expression (string)",
                "function": self._calculate,
            },
            "lookup": {
                "description": "Look up a specific fact. Args: topic (string)",
                "function": self._mock_lookup,
            },
        }

    def _mock_search(self, query: str) -> str:
        """Mock search function."""
        # In production, this would call a real search API
        mock_results = {
            "python creator": "Python was created by Guido van Rossum in the Netherlands.",
            "eiffel tower height": "The Eiffel Tower is 330 meters (1,083 ft) tall.",
            "speed of light": "The speed of light is approximately 299,792,458 meters per second.",
            "population of japan": "Japan has a population of approximately 125 million people.",
        }

        for key, result in mock_results.items():
            if key in query.lower():
                return result

        return f"Search results for '{query}': No specific results found. Try a more specific query."

    def _calculate(self, expression: str) -> str:
        """Safe calculation function."""
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"

        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _mock_lookup(self, topic: str) -> str:
        """Mock lookup function for specific facts."""
        facts = {
            "python": "Python is a programming language first released in 1991.",
            "einstein": "Albert Einstein developed the theory of relativity.",
            "moon": "The Moon orbits Earth at an average distance of 384,400 km.",
        }

        for key, fact in facts.items():
            if key in topic.lower():
                return fact

        return f"No specific fact found for '{topic}'."

    def _get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions."""
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool['description']}")
        return "\n".join(lines)

    def _parse_action(self, text: str) -> tuple[str | None, str | None]:
        """Parse action and action input from response.

        Args:
            text: The response text to parse.

        Returns:
            Tuple of (action, action_input) or (None, None) if no action found.
        """
        # Look for Action: and Action Input: patterns
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)

        if action_match and input_match:
            return action_match.group(1), input_match.group(1).strip()

        return None, None

    def _execute_action(self, action: str, action_input: str) -> str:
        """Execute a tool action.

        Args:
            action: Tool name to execute.
            action_input: Input for the tool.

        Returns:
            Observation string.
        """
        if action not in self.tools:
            return f"Error: Unknown action '{action}'. Available actions: {list(self.tools.keys())}"

        try:
            result = self.tools[action]["function"](action_input)
            return result
        except Exception as e:
            return f"Error executing {action}: {str(e)}"

    def run(self, question: str) -> str:
        """Run the ReAct agent on a question.

        Args:
            question: The question to answer.

        Returns:
            The final answer.
        """
        system_prompt = f"""You are a helpful assistant that answers questions by reasoning step-by-step.

You have access to the following tools:
{self._get_tool_descriptions()}

Use the following format:

Question: the input question
Thought: reason about what to do
Action: the action to take (one of: {", ".join(self.tools.keys())})
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the question

Begin!"""

        messages = [
            Message.system(system_prompt),
            Message.user(f"Question: {question}"),
        ]

        full_response = ""

        for iteration in range(self.max_iterations):
            logger.info("react_iteration", iteration=iteration + 1)

            # Get model response
            response = self.provider.complete(messages, temperature=0.3)
            response_text = response.content or ""
            full_response += response_text

            # Check if we have a final answer
            if "Final Answer:" in response_text:
                # Extract final answer
                final_match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", response_text, re.DOTALL)
                if final_match:
                    final_answer = final_match.group(1).strip()
                    logger.info("final_answer_found", iterations=iteration + 1)
                    return final_answer

            # Parse and execute action
            action, action_input = self._parse_action(response_text)

            if action and action_input:
                logger.info("executing_action", action=action, input=action_input)
                observation = self._execute_action(action, action_input)

                # Add observation to conversation
                messages.append(Message.assistant(response_text))
                messages.append(Message.user(f"Observation: {observation}"))
                full_response += f"\nObservation: {observation}\n"
            else:
                # No action found, might be confused
                messages.append(Message.assistant(response_text))
                messages.append(
                    Message.user("Please continue with a Thought, then an Action, or provide a Final Answer.")
                )

        logger.warning("max_iterations_reached")
        return "I was unable to find a definitive answer within the allowed iterations."


def main() -> None:
    """Run the ReAct agent examples."""
    print("=" * 60)
    print("ReAct Agent Example")
    print("=" * 60)

    agent = ReActAgent(max_iterations=5)

    # Example questions
    questions = [
        "Who created Python and when was it first released?",
        "What is 15% of 340 plus 25?",
        "How tall is the Eiffel Tower in feet?",
    ]

    for question in questions:
        print(f"\n--- Question: {question} ---")
        answer = agent.run(question)
        print(f"\nFinal Answer: {answer}")


if __name__ == "__main__":
    main()
