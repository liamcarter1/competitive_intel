# CLAUDE.md — Competitive Intelligence Monitor

## Project Overview

A LangGraph-powered competitive intelligence platform with a Gradio web UI. The system orchestrates four pipeline nodes (trend scanner, company analyst, strategy advisor, report writer) using a fan-out/fan-in graph to generate executive-ready competitive intelligence briefings. Each node uses its own LLM client (OpenAI or Anthropic) with independent message history, enabling multi-provider pipelines. A secondary chat interface allows Q&A against the briefing (via OpenAI) and deep-dive research with live web search (via Anthropic + Serper).

## Architecture

```
competitive_intel/
├── app.py                          # Gradio UI, chat, deep-dive (entry point)
├── pyproject.toml                  # Project metadata and dependencies
├── uv.lock                        # Locked dependency versions
├── output/                         # Generated briefing reports (gitignored)
└── src/competitive_intel/
    ├── __init__.py
    ├── main.py                     # CLI entry point (run)
    ├── graph.py                    # LangGraph StateGraph definition and pipeline nodes
    ├── config/
    │   ├── agents.yaml             # Agent roles, goals, backstories (used as system prompts)
    │   └── tasks.yaml              # Task descriptions and expected outputs (used as user prompts)
    └── tools/
        └── __init__.py             # search_serper() function for web search
```

### Pipeline Graph

```
                    ┌─ scan(competitor_A) ─┐
User Input ──→ fan_out ─→ scan(competitor_B) ─→ fan_in ──→ analyze ──→ recommend ──→ write_briefing
                    └─ scan(competitor_C) ─┘
```

- **Fan-out**: Parallel scan nodes (one per competitor), each calling Serper API then summarizing with GPT-4o
- **Fan-in**: Aggregates all scan results into shared state
- **Sequential**: analyze (Claude Sonnet) → recommend (Claude Sonnet) → write_briefing (GPT-4o-mini)

### Key Components

- **LangGraph Pipeline** (`graph.py`): A `StateGraph` with fan-out/fan-in for parallel competitor scanning, followed by sequential analysis, recommendations, and report writing. Each node makes its own LLM API call with a clean message list — no shared conversation history between nodes.
- **Gradio App** (`app.py`): Web UI with briefing generation, report loading, quick chat (OpenAI gpt-4o-mini), and deep-dive research (Serper search + Anthropic Claude synthesis).
- **CLI** (`main.py`): `run()` function callable via `competitive_intel` script entry point.
- **Config** (`agents.yaml`, `tasks.yaml`): Agent roles/backstories and task descriptions loaded at runtime and interpolated into system/user prompts for each node.

## Tech Stack

- **Python** >=3.10, <3.14
- **LangGraph** >=0.4.0
- **LangChain OpenAI** >=0.3.0
- **LangChain Anthropic** >=0.3.0
- **Gradio** >=5.22.0
- **Anthropic SDK** >=0.40.0
- **OpenAI SDK** >=1.0.0
- **Package manager**: uv (with hatchling build backend)

## Environment Variables (Required)

```
OPENAI_API_KEY          # Used by scan and write_briefing nodes, and quick chat
ANTHROPIC_API_KEY       # Used by analyze and recommend nodes, and deep-dive synthesis
SERPER_API_KEY          # Used by scan nodes and deep-dive web search
```

These MUST be set in the environment before running. Never commit these values.

## Setup and Running

```bash
cd competitive_intel
uv sync                    # Install dependencies
uv run python app.py       # Launch the Gradio web UI
uv run competitive_intel   # Run the CLI pipeline
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
- User input passed to `_search_web()` and `search_serper()` goes directly to the Serper API — do not add any filesystem or command execution based on this input.
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
- Set explicit timeouts on all HTTP requests (as `search_serper()` does with `timeout=15`).
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

### LangGraph Patterns
- Agent prompts (role, goal, backstory) live in `config/agents.yaml`. Task prompts (description, expected_output) live in `config/tasks.yaml`.
- Each graph node in `graph.py` loads its prompts from these YAML configs, interpolates input variables, and makes a direct LLM call.
- The graph uses `Send()` for fan-out (parallel competitor scans) and sequential edges for the analysis pipeline.
- Each node constructs its own message list (system + user) — never pass message history between nodes.
- State is shared via a `GraphState` TypedDict. Use `Annotated[list, operator.add]` for fields that accumulate across parallel nodes (e.g., `scan_results`).

### LLM Model Selection
- Scan nodes: GPT-4o (reliable for search result summarization)
- Analyze node: Claude Sonnet (better analytical reasoning)
- Recommend node: Claude Sonnet (better strategic synthesis)
- Write briefing node: GPT-4o-mini (cost-effective for formatting)
- Quick chat: GPT-4o-mini with low temperature (0.1)
- Deep-dive synthesis: Claude 3.5 Sonnet with low temperature (0.1)
- Model assignments are in `graph.py` node functions. Each node is independent so models can be changed freely.

### Error Handling
- Catch exceptions at UI boundaries (Gradio callbacks) and display user-friendly messages.
- Do not catch broad exceptions silently. Log or surface the error.
- In pipeline code (`main.py`), re-raise with context so failures are diagnosable.

### Adding New Nodes
1. Define the agent config in `agents.yaml` (role, goal, backstory).
2. Define the task config in `tasks.yaml` (description, expected_output).
3. Add a node function in `graph.py` that loads prompts from config, calls an LLM, and returns state updates.
4. Wire the node into the graph in `build_graph()` with appropriate edges.

### Adding New Tools
1. Add the tool function in `src/competitive_intel/tools/__init__.py`.
2. Call it directly from the relevant graph node — no LLM tool-calling needed.
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
- Test the graph node functions, input validation, and output formatting independently.
