# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Engineering Hub - A production-grade AI engineering portfolio showcasing LLM patterns, multi-provider support, RAG, agents, and ML best practices.

## Commands

### Setup
```bash
pip install -e ".[dev]"  # Install package with dev dependencies
```

### Running Examples
```bash
python examples/01-llm-fundamentals/basic_chat.py
python examples/03-function-calling/basic_tools.py
python examples/05-agents/react_agent.py
python projects/semantic_search/search_engine.py
```

### Weather Server (legacy, for function calling demo)
```bash
uvicorn weather_server:app --reload
```

### Code Quality
```bash
# Run all checks via pre-commit
pre-commit run --all-files

# Individual tools
ruff check .
ruff format .
mypy src/ai_hub
pytest tests/
```

## Architecture

### Code Organization
```
ai-engineering-hub/
├── src/ai_hub/              # Core library
│   ├── core/                # Config, errors, logging, retry
│   │   ├── config.py        # Pydantic settings management
│   │   ├── errors.py        # Exception hierarchy
│   │   ├── logging.py       # Structured logging
│   │   └── retry.py         # Exponential backoff with tenacity
│   ├── providers/           # LLM provider abstractions
│   │   ├── base.py          # Abstract base class
│   │   ├── openai_provider.py
│   │   └── anthropic_provider.py
│   └── utils/               # Utilities
│       └── tokens.py        # Token counting with tiktoken
├── examples/                # Runnable examples by category
│   ├── 01-llm-fundamentals/ # Chat, streaming, structured output
│   ├── 02-multi-provider/   # Provider comparison
│   ├── 03-function-calling/ # Modern tools API
│   ├── 04-rag-patterns/     # RAG implementations
│   ├── 05-agents/           # ReAct pattern
│   ├── 06-evaluation/       # LLM-as-judge
│   ├── 07-fine-tuning/      # Data preparation
│   ├── 08-vision-multimodal/# Image analysis
│   ├── 09-traditional-ml/   # ML vs LLM patterns
│   └── 10-production-patterns/ # Rate limiting
├── projects/                # Complete mini-projects
├── tests/                   # Unit and integration tests
└── docs/                    # Documentation
```

### Key Patterns

1. **Provider Abstraction**: All LLM providers implement `BaseLLMProvider` for consistent interface
2. **Message Types**: Use `Message.system()`, `Message.user()`, `Message.assistant()` helpers
3. **Configuration**: Environment-based config via `get_settings()` from Pydantic
4. **Logging**: Structured logging via `get_logger()` from structlog
5. **Retry**: Automatic retry with `@with_retry` decorator

### Key Dependencies
- `openai` - OpenAI SDK
- `anthropic` - Anthropic SDK (optional)
- `pydantic` / `pydantic-settings` - Configuration and validation
- `structlog` - Structured logging
- `tenacity` - Retry logic
- `tiktoken` - Token counting

## Code Style

- Line length: 120 characters
- Formatter: Ruff (replaces Black + isort)
- Linter: Ruff (replaces flake8)
- Type checker: mypy (strict mode)
- Commits: Conventional commits via Commitizen

## Testing

```bash
# Run unit tests
pytest tests/unit -v

# Run with coverage
pytest tests/unit --cov=src/ai_hub
```
