"""Modern Function Calling (Tools API) Example.

This example demonstrates the modern OpenAI tools API for function calling,
replacing the deprecated `functions` parameter with the new `tools` parameter.

Features demonstrated:
- Defining tools with JSON Schema
- Handling tool calls from the model
- Executing functions and returning results
- Multi-turn conversations with tool use
"""

import json
from typing import Any

from ai_hub import Message, OpenAIProvider, ToolDefinition, get_logger, setup_logging
from ai_hub.providers.base import Role

setup_logging()
logger = get_logger(__name__)


# Define available tools
def get_weather(location: str, unit: str = "fahrenheit") -> dict[str, Any]:
    """Get the current weather for a location (mock implementation).

    Args:
        location: City and state, e.g., "San Francisco, CA"
        unit: Temperature unit ("celsius" or "fahrenheit")

    Returns:
        Weather information dictionary.
    """
    # Mock weather data
    weather_data = {
        "San Francisco, CA": {"temp": 62, "condition": "foggy"},
        "New York, NY": {"temp": 45, "condition": "cloudy"},
        "Miami, FL": {"temp": 78, "condition": "sunny"},
        "Boston, MA": {"temp": 38, "condition": "snowy"},
    }

    data = weather_data.get(location, {"temp": 70, "condition": "unknown"})

    if unit == "celsius":
        data["temp"] = round((data["temp"] - 32) * 5 / 9)

    return {
        "location": location,
        "temperature": data["temp"],
        "unit": unit,
        "condition": data["condition"],
    }


def calculate(expression: str) -> dict[str, Any]:
    """Safely evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate.

    Returns:
        Result dictionary.
    """
    # Very basic safe evaluation (in production, use a proper math parser)
    allowed_chars = set("0123456789+-*/.(). ")
    if not all(c in allowed_chars for c in expression):
        return {"error": "Invalid characters in expression"}

    try:
        result = eval(expression)  # Safe because we validated input
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


# Tool definitions
WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Get the current weather in a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g., San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use",
            },
        },
        "required": ["location"],
    },
)

CALCULATOR_TOOL = ToolDefinition(
    name="calculate",
    description="Evaluate a mathematical expression",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g., '2 + 2' or '(10 * 5) / 2'",
            },
        },
        "required": ["expression"],
    },
)

# Map tool names to functions
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
}


def execute_tool(tool_name: str, arguments: str) -> str:
    """Execute a tool and return the result as a string.

    Args:
        tool_name: Name of the tool to execute.
        arguments: JSON string of arguments.

    Returns:
        JSON string of the result.
    """
    if tool_name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        args = json.loads(arguments)
        result = TOOL_FUNCTIONS[tool_name](**args)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def chat_with_tools(user_message: str) -> str:
    """Run a conversation that may use tools.

    Args:
        user_message: The user's message.

    Returns:
        The final response from the assistant.
    """
    provider = OpenAIProvider(default_model="gpt-4o-mini")
    tools = [WEATHER_TOOL, CALCULATOR_TOOL]

    messages = [
        Message.system(
            "You are a helpful assistant with access to tools. "
            "Use tools when appropriate to answer questions accurately."
        ),
        Message.user(user_message),
    ]

    # Initial completion
    response = provider.complete(messages, tools=tools, temperature=0.3)

    # Handle tool calls in a loop (model may call multiple tools)
    while response.tool_calls:
        logger.info("tool_calls_received", count=len(response.tool_calls))

        # Add assistant message with tool calls
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=response.content or "",
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in response.tool_calls
                ],
            )
        )

        # Execute each tool and add results
        for tool_call in response.tool_calls:
            logger.info("executing_tool", tool=tool_call.name, arguments=tool_call.arguments)

            result = execute_tool(tool_call.name, tool_call.arguments)

            messages.append(Message.tool(content=result, tool_call_id=tool_call.id))

        # Get next response
        response = provider.complete(messages, tools=tools, temperature=0.3)

    return response.content or ""


def main() -> None:
    """Run the function calling examples."""
    print("=" * 60)
    print("Function Calling (Tools API) Examples")
    print("=" * 60)

    # Example 1: Weather query
    print("\n--- Weather Query ---")
    result = chat_with_tools("What's the weather like in Boston?")
    print(f"Response: {result}")

    # Example 2: Calculator
    print("\n--- Calculator ---")
    result = chat_with_tools("What is 15% of 230?")
    print(f"Response: {result}")

    # Example 3: Multiple tools
    print("\n--- Multiple Tools ---")
    result = chat_with_tools(
        "I'm in San Francisco and need to know two things: "
        "1) What's the weather? 2) What's 72 fahrenheit in celsius?"
    )
    print(f"Response: {result}")

    # Example 4: No tool needed
    print("\n--- No Tool Needed ---")
    result = chat_with_tools("What is the capital of France?")
    print(f"Response: {result}")


if __name__ == "__main__":
    main()
