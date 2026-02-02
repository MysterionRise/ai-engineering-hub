# AI Engineering Hub

Production-grade AI engineering patterns, examples, and best practices for building LLM-powered applications.

[![CI](https://github.com/yourusername/ai-engineering-hub/actions/workflows/ci.yaml/badge.svg)](https://github.com/yourusername/ai-engineering-hub/actions/workflows/ci.yaml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository demonstrates modern AI engineering practices including:

- **Multi-Provider LLM Support** - Unified interface for OpenAI, Anthropic, and other providers
- **Production Patterns** - Rate limiting, retry logic, structured logging, error handling
- **RAG Implementation** - Vector search, document chunking, context augmentation
- **Agent Architectures** - ReAct pattern, tool use, multi-step reasoning
- **Evaluation Framework** - LLM-as-judge, automated quality assessment
- **Hybrid ML/LLM Pipelines** - When to use traditional ML vs. LLMs

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-engineering-hub.git
cd ai-engineering-hub

# Install dependencies
pip install -e ".[dev]"

# Set up your API keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"  # Optional

# Run an example
python examples/01-llm-fundamentals/basic_chat.py
```

## Project Structure

```
ai-engineering-hub/
├── src/ai_hub/              # Core library
│   ├── core/                # Config, errors, logging, retry utilities
│   ├── providers/           # LLM provider abstractions
│   └── utils/               # Token counting, text utilities
├── examples/                # Runnable examples by category
│   ├── 01-llm-fundamentals/ # Chat, streaming, structured output
│   ├── 02-multi-provider/   # Provider comparison and fallbacks
│   ├── 03-function-calling/ # Modern tools API usage
│   ├── 04-rag-patterns/     # Retrieval-augmented generation
│   ├── 05-agents/           # ReAct and agent patterns
│   ├── 06-evaluation/       # LLM-as-judge evaluation
│   ├── 07-fine-tuning/      # Data prep for fine-tuning
│   ├── 08-vision-multimodal/# Image analysis with Vision API
│   ├── 09-traditional-ml/   # When ML beats LLMs
│   └── 10-production-patterns/ # Rate limiting, caching
├── projects/                # Complete mini-projects
│   └── semantic_search/     # Full semantic search implementation
├── tests/                   # Unit and integration tests
└── docs/                    # Documentation
```

## Example Gallery

### 🎯 Fundamentals

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Basic Chat](examples/01-llm-fundamentals/basic_chat.py) | Single and multi-turn conversations | ⭐ |
| [Streaming](examples/01-llm-fundamentals/streaming.py) | Real-time response streaming | ⭐ |
| [Structured Output](examples/01-llm-fundamentals/structured_output.py) | JSON mode and Pydantic validation | ⭐⭐ |

### 🔄 Multi-Provider

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Provider Comparison](examples/02-multi-provider/provider_comparison.py) | Compare responses across providers | ⭐⭐ |

### 🔧 Function Calling

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Basic Tools](examples/03-function-calling/basic_tools.py) | Modern tools API with execution loop | ⭐⭐ |

### 📚 RAG Patterns

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Simple RAG](examples/04-rag-patterns/simple_rag.py) | Vector search with context augmentation | ⭐⭐⭐ |

### 🤖 Agents

| Example | Description | Difficulty |
|---------|-------------|------------|
| [ReAct Agent](examples/05-agents/react_agent.py) | Reasoning and acting pattern | ⭐⭐⭐ |

### 📊 Evaluation

| Example | Description | Difficulty |
|---------|-------------|------------|
| [LLM-as-Judge](examples/06-evaluation/llm_as_judge.py) | Automated quality evaluation | ⭐⭐⭐ |

### 🏭 Production

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Rate Limiting](examples/10-production-patterns/rate_limiting.py) | Token bucket rate limiter | ⭐⭐⭐ |

## Core Library Usage

```python
from ai_hub import OpenAIProvider, Message

# Initialize provider
provider = OpenAIProvider(default_model="gpt-4o")

# Simple completion
response = provider.complete([
    Message.system("You are a helpful assistant."),
    Message.user("What is the capital of France?"),
])
print(response.content)

# With streaming
for chunk in provider.stream([Message.user("Tell me a story")]):
    print(chunk.content, end="", flush=True)

# Multi-provider support
from ai_hub import get_provider

openai = get_provider("openai")
anthropic = get_provider("anthropic")  # Requires anthropic package
```

## Architecture Principles

1. **Provider Abstraction** - Write once, run on any LLM
2. **Production-Ready** - Retry logic, rate limiting, structured logging
3. **Type Safety** - Full type hints with Pydantic validation
4. **Testability** - Mockable interfaces, comprehensive test coverage
5. **Observability** - Structured JSON logging with metrics

## Configuration

Configuration via environment variables (or `.env` file):

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
ANTHROPIC_API_KEY=sk-ant-...
AI_HUB_DEFAULT_MODEL=gpt-4o
AI_HUB_LOG_LEVEL=INFO
AI_HUB_MAX_RETRIES=3
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check .
ruff format .

# Type checking
mypy src/ai_hub

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## For Different Audiences

### 👔 Recruiters
This repository demonstrates:
- Production-grade Python development practices
- Deep understanding of LLM APIs and architectures
- System design for AI applications
- Clean, maintainable, well-documented code

### 💼 Clients
Examples show capability in:
- Building reliable AI-powered features
- Integrating multiple AI providers
- Implementing evaluation and quality control
- Production deployment patterns

### 👩‍💻 Developers
Learn from:
- Modern Python packaging (pyproject.toml, ruff)
- Provider abstraction patterns
- Error handling and retry strategies
- Testing LLM applications

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read the contributing guidelines and submit a PR.

---

Built with modern Python practices and production-grade patterns.
