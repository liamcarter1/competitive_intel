# CLAUDE.md — Competitive Intelligence Monitor

## Project Overview

A CrewAI-powered competitive intelligence platform with a Gradio web UI. The system orchestrates four AI agents (trend scanner, company analyst, strategy advisor, report writer) in a sequential pipeline to generate executive-ready competitive intelligence briefings. A secondary chat interface allows Q&A against the briefing (via OpenAI) and deep-dive research with live web search (via Anthropic + Serper).

## Architecture

```
competitive_intel/
├── app.py                          # Gradio UI, chat, deep-dive (entry point)
├── pyproject.toml                  # Project metadata and dependencies
├── uv.lock                        # Locked dependency versions
├── output/                         # Generated briefing reports (gitignored)
└── src/competitive_intel/
    ├── __init__.py
    ├── main.py                     # CLI entry point (run/train/replay)
    ├── crew.py                     # CrewAI crew definition and agent/task wiring
    ├── config/
    │   ├── agents.yaml             # Agent roles, goals, backstories, LLM assignments
    │   └── tasks.yaml              # Task descriptions, expected outputs, dependencies
    └── tools/
        └── __init__.py             # Custom tool definitions (currently empty)
```

### Key Components

- **CrewAI Crew** (`crew.py`): Four agents running sequentially — `trend_scanner` (with SerperDevTool) -> `company_analyst` -> `strategy_advisor` -> `report_writer`. Agent configs live in `agents.yaml`, task configs in `tasks.yaml`.
- **Gradio App** (`app.py`): Web UI with briefing generation, report loading, quick chat (OpenAI gpt-4o-mini), and deep-dive research (Serper search + Anthropic Claude synthesis).
- **CLI** (`main.py`): `run()`, `train()`, `replay()` functions callable via `competitive_intel` script entry point.

## Tech Stack

- **Python** >=3.10, <3.14
- **CrewAI** >=0.108.0 (with tools)
- **Gradio** >=5.22.0
- **Anthropic SDK** >=0.40.0
- **OpenAI SDK** (transitive via crewai)
- **Package manager**: uv (with hatchling build backend)

## Environment Variables (Required)

```
OPENAI_API_KEY          # Used by CrewAI agents and quick chat
ANTHROPIC_API_KEY       # Used by deep-dive synthesis
SERPER_API_KEY          # Used by trend_scanner agent and deep-dive web search
```

These MUST be set in the environment before running. Never commit these values.

## Setup and Running

```bash
cd competitive_intel
uv sync                    # Install dependencies
uv run python app.py       # Launch the Gradio web UI
uv run competitive_intel   # Run the CLI crew pipeline
```

## Security Requirements

### API Keys and Secrets
- **Never** commit API keys, tokens, or credentials to the repository.
- **Never** hardcode secrets in source code. All secrets must come from environment variables.
- The `.env` file is gitignored and must stay that way.
- When adding new external service integrations, use `os.environ.get()` or `os.environ[]` — never default to a real key value.

### Input Handling
- All user inputs from the Gradio UI (company, industry, competitors, chat messages) are passed to external LLM APIs. Treat these as untrusted.
- Do not construct shell commands, file paths, or SQL queries from user input.
- User input passed to `_search_web()` goes directly to the Serper API — do not add any filesystem or command execution based on this input.
- Validate that user inputs are non-empty strings before processing (as `run_briefing()` already does).

### Dependency Security
- Keep dependencies pinned via `uv.lock`. Run `uv sync` to install exact locked versions.
- When updating dependencies, review changelogs for security advisories.
- Only add dependencies that are actively maintained and widely trusted.

### Output Handling
- Briefing reports are written to `output/briefing.md`. This directory is gitignored to prevent accidental commit of sensitive competitive intelligence.
- Do not serve the `output/` directory over a network or expose it publicly.
- LLM responses are rendered as markdown in Gradio — Gradio handles sanitization, but do not bypass this by rendering raw HTML.

### Network Security
- All external API calls (OpenAI, Anthropic, Serper) must use HTTPS. Do not downgrade to HTTP.
- Set explicit timeouts on all HTTP requests (as `_search_web()` already does with `timeout=15`).
- Do not add proxy or redirect-following logic that could leak credentials.

## Coding Protocols

### General
- Keep code simple and direct. Avoid unnecessary abstractions.
- Do not add features, refactoring, or "improvements" beyond what is requested.
- Do not add comments, docstrings, or type annotations to code you did not change.
- Use existing patterns in the codebase as the template for new code.

### Python Style
- Follow the existing code style: no strict formatter is enforced, but keep it clean and readable.
- Use `Path` (from `pathlib`) for filesystem operations, not string concatenation.
- Use f-strings for string formatting.
- Imports: standard library first, then third-party, then local. No blank-line separation is enforced but keep it logical.

### CrewAI Patterns
- Agent definitions go in `config/agents.yaml`. Task definitions go in `config/tasks.yaml`. Wire them in `crew.py` using the `@agent` and `@task` decorators.
- Each agent method in `crew.py` returns an `Agent` with `config=self.agents_config['agent_name']`.
- Each task method in `crew.py` returns a `Task` with `config=self.tasks_config['task_name']`.
- The crew runs in `Process.sequential` mode. If adding new agents/tasks, place them in the correct pipeline order.
- Custom tools go in `src/competitive_intel/tools/__init__.py`.

### LLM Model Selection
- CrewAI agents: model is set per-agent in `agents.yaml` via the `llm` field (e.g., `openai/gpt-4o-mini`).
- Quick chat: uses `gpt-4o-mini` with low temperature (0.1).
- Deep-dive synthesis: uses `claude-3-5-sonnet-latest` with low temperature (0.1).
- When changing models, update the relevant config — do not hardcode model names in multiple places.

### Error Handling
- Catch exceptions at UI boundaries (Gradio callbacks) and display user-friendly messages.
- Do not catch broad exceptions silently. Log or surface the error.
- In CrewAI pipeline code (`main.py`), re-raise with context so failures are diagnosable.

### Adding New Agents or Tasks
1. Define the agent config in `agents.yaml` (role, goal, backstory, llm).
2. Define the task config in `tasks.yaml` (description, expected_output, agent, context dependencies).
3. Add corresponding `@agent` and `@task` methods in `crew.py`.
4. The crew auto-discovers agents/tasks via the `@CrewBase` decorator — ordering in `crew.py` determines pipeline order.

### Adding New Tools
1. Create the tool class in `src/competitive_intel/tools/__init__.py`.
2. Import and attach it to the relevant agent in `crew.py` via the `tools=[]` parameter.
3. Tools that call external APIs must use environment variables for credentials and set request timeouts.

## Files to Never Commit

- `.env` — API keys and secrets
- `output/` — generated intelligence reports (may contain sensitive business data)
- `.venv/` — virtual environment
- `__pycache__/` — bytecode cache
- `.idea/` — IDE configuration

## Testing

No test suite exists yet. When adding tests:
- Place tests in a `tests/` directory at the project root.
- Use `pytest` as the test runner.
- Mock all external API calls (OpenAI, Anthropic, Serper) — never make real API calls in tests.
- Test the crew wiring, input validation, and output formatting independently.
