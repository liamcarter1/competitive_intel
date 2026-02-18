# CLAUDE.md — Competitive Intelligence Monitor

## Project Overview

A LangGraph-powered competitive intelligence platform with a Gradio web UI. The system orchestrates a two-node News Monitor pipeline (scan_news, compile_digest) using a fan-out/fan-in graph to generate categorized news digests from live web and news search results. The scan node collects raw search results with no LLM summarization; a single compile node deduplicates, categorizes, and formats the digest using GPT-4o-mini. A secondary chat interface allows Q&A against the news digest (via OpenAI) and deep-dive research with live web search (via Anthropic + Serper).

## Architecture

```
competitive_intel/
├── app.py                          # Gradio UI, chat, deep-dive (entry point)
├── pyproject.toml                  # Project metadata and dependencies
├── uv.lock                        # Locked dependency versions
├── output/                         # Generated reports (news_digest.md, annual reports) (gitignored)
└── src/competitive_intel/
    ├── __init__.py
    ├── main.py                     # CLI entry point (run)
    ├── graph.py                    # LangGraph StateGraph definition and pipeline nodes
    ├── config/
    │   ├── agents.yaml             # Agent roles, goals, backstories (news_digest_curator + annual report agents)
    │   └── tasks.yaml              # Task descriptions and expected outputs (compile_news_digest + annual report tasks)
    └── tools/
        └── __init__.py             # search_serper() and search_serper_news() for web/news search
```

### News Monitor Pipeline Graph

```
                         ┌─ scan_news(competitor_A) ─┐
User Input ──→ fan_out ─→ scan_news(competitor_B) ─→ fan_in ──→ compile_digest
                         └─ scan_news(competitor_C) ─┘
```

- **Fan-out**: Parallel scan_news nodes (one per competitor), each starting with an LLM disambiguation call (`_disambiguate_competitor()` via gpt-4o-mini) that generates a search-friendly name, Google exclusion terms, and a context sentence. Then runs 25 Serper News searches + 17 Serper Web searches (42 total per competitor — including trade press, trade shows, electrification/tech trends, distributor/channel, press releases, capex/manufacturing, and site-targeted queries for trade publications and PR wire services) using the disambiguated name + exclusion terms. There is **no LLM summarization** in the scan node — raw Serper results (with real URLs, titles, snippets, and dates) are passed directly to state. Each competitor's result block includes the disambiguation context sentence as a header so compile_digest can filter off-topic results. News results are date-filtered based on the configurable time window before being added to state.
- **Fan-in**: Aggregates all raw news results from parallel scan nodes into shared `news_results` state.
- **Compile**: compile_digest (GPT-4o-mini) receives all raw results and deduplicates, categorizes, and formats them into a structured markdown news digest. Output saved to `output/news_digest.md`.
- **State**: `NewsMonitorState` TypedDict with fields: `company`, `industry`, `competitors`, `current_date`, `time_window`, `news_results` (Annotated list, accumulates across parallel nodes), `digest`.
- **Time window**: Configurable via UI dropdown — `past_week` (Serper `tbs=qdr:w`, code filter 7 days), `past_2_weeks` (`tbs=qdr:w2`, 14 days), `past_month` (`tbs=qdr:m`, 30 days).

### Annual Report Pipeline Graph

```
                              ┌─ scan_annual_report(competitor_A) [+ inline evaluate + retry] ─┐
User Input ──→ fan_out_annual ─→ scan_annual_report(competitor_B) [+ inline evaluate + retry] ─→ combine results
                              └─ scan_annual_report(competitor_C) [+ inline evaluate + retry] ─┘
```

- **Fan-out**: Parallel deep-dive report nodes (one per competitor), each starting with the same LLM disambiguation call, then running 18 Serper searches using the disambiguated name + exclusion terms, then synthesizing with Claude Sonnet. The LLM prompt includes the disambiguation context sentence to help discard results about unrelated companies.
- **Inline evaluation**: Each branch evaluates its own report against a quality rubric and retries up to 2 times with feedback (evaluation happens inside the node, not as a separate graph node, to preserve per-competitor granularity)
- **Output**: Combined markdown report saved to `output/annual_report_analysis.md`

### Key Components

- **LangGraph Pipeline** (`graph.py`): Two separate `StateGraph` definitions — the News Monitor graph (fan-out scan_news per competitor, fan-in, then compile_digest) and the Annual Report graph (unchanged). The scan_news node makes no LLM call (raw Serper results only); compile_digest makes a single GPT-4o-mini call. The annual report nodes each make their own LLM calls with clean message lists.
- **Gradio App** (`app.py`): Web UI with news monitor generation (with real-time progress log and configurable time window), report loading, quick chat (OpenAI gpt-4o-mini), and deep-dive research (Serper search + Anthropic Claude synthesis). The competitor input uses a CheckboxGroup of 10 predefined competitors plus an ad-hoc text field for additional names. Progress updates stream to the UI as each pipeline node completes.
- **CLI** (`main.py`): `run()` function callable via `competitive_intel` script entry point. Uses the same streaming generators as the UI, printing progress to stdout.
- **Config** (`agents.yaml`, `tasks.yaml`): The `news_digest_curator` agent and `compile_news_digest` task are loaded at runtime and interpolated into prompts for the compile_digest node. Annual report agent/task configs remain unchanged.

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
OPENAI_API_KEY          # Used by compile_digest node, disambiguation, and quick chat
ANTHROPIC_API_KEY       # Used by annual report nodes and deep-dive synthesis
SERPER_API_KEY          # Used by scan_news nodes (news + web search) and deep-dive web search
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
- User input passed to `_search_web()`, `search_serper()`, and `search_serper_news()` goes directly to the Serper API — do not add any filesystem or command execution based on this input.
- Validate that user inputs are non-empty strings before processing (as `run_news_monitor_stream()` already does).

### Dependency Security
- Keep dependencies pinned via `uv.lock`. Run `uv sync` to install exact locked versions.
- When updating dependencies, review changelogs for security advisories.
- Only add dependencies that are actively maintained and widely trusted.

### Output Handling
- News digests are written to `output/news_digest.md`. This directory is gitignored to prevent accidental commit of sensitive competitive intelligence.
- Do not serve the `output/` directory over a network or expose it publicly.
- LLM responses are rendered as markdown in Gradio — Gradio handles sanitization, but do not bypass this by rendering raw HTML.

### Network Security
- All external API calls (OpenAI, Anthropic, Serper) must use HTTPS. Do not downgrade to HTTP.
- Serper has two endpoints: `/search` (web results) and `/news` (news articles with date filtering via `tbs` parameter). Both are used by the news monitor scan (42 queries per competitor: 25 news + 17 web). News results are also date-filtered in code based on the configured time window (7, 14, or 30 days), since the `tbs` parameter is not always reliable.
- Competitor names are disambiguated via `_disambiguate_competitor()` (gpt-4o-mini) which generates search-friendly names and Google exclusion operators. All search queries use the disambiguated name + exclusion terms + industry to prevent name ambiguity. LLM prompts include the disambiguation context sentence. The deep dive feature also receives company/industry context for its query generation and synthesis prompts.
- Set explicit timeouts on all HTTP requests (as `search_serper()` and `search_serper_news()` do with `timeout=15`).
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
- The graph uses `Send()` for fan-out (parallel competitor scans) and sequential edges for the compile step.
- Each node constructs its own message list (system + user) — never pass message history between nodes.
- The News Monitor uses a `NewsMonitorState` TypedDict with fields: `company`, `industry`, `competitors`, `current_date`, `time_window`, `news_results` (`Annotated[list, operator.add]`), `digest`. The Annual Report uses its own `AnnualReportState`.
- For UI progress, use `graph.stream(stream_mode="updates")` which yields a dict after each node completes. The `run_news_monitor_stream()` and `run_annual_report_pipeline_stream()` generators wrap this into `("progress", msg)` / `("result", text)` tuples that Gradio consumes via generator yields. The non-streaming `run_news_monitor()` and `run_annual_report_pipeline()` are thin wrappers that print progress to stdout for CLI use.

### LLM Model Selection
- Disambiguation: GPT-4o-mini with low temperature (0.1) — cheap per-competitor call to generate search names and exclusion terms
- scan_news nodes: **No LLM call** — raw Serper results only (42 queries per competitor: 25 news + 17 web)
- compile_digest node: GPT-4o-mini (cost-effective for deduplication, categorization, and formatting; receives raw search results with real URLs)
- Annual report scan nodes: Claude Sonnet (deep analytical synthesis from 18 searches)
- Annual report inline evaluator: Claude Sonnet (same quality gate, runs inside each parallel branch)
- Quick chat: GPT-4o-mini with low temperature (0.1)
- Deep-dive query generation: GPT-4o-mini (with industry context for disambiguation)
- Deep-dive synthesis: Claude Sonnet with low temperature (0.1)
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

## Project Learning Document

For every project, write a detailed `FORliam.md` file that explains the whole project in plain language.

Explain the technical architecture, the structure of the codebase and how the various parts are connected, the technologies used, why we made these technical decisions, and lessons I can learn from it (this should include the bugs we ran into and how we fixed them, potential pitfalls and how to avoid them in the future, new technologies used, how good engineers think and work, best practices, etc).

It should be very engaging to read; don't make it sound like boring technical documentation/textbook. Where appropriate, use analogies and anecdotes to make it more understandable and memorable.

## Testing

No test suite exists yet. When adding tests:
- Place tests in a `tests/` directory at the project root.
- Use `pytest` as the test runner.
- Mock all external API calls (OpenAI, Anthropic, Serper) — never make real API calls in tests.
- Test the graph node functions, input validation, and output formatting independently.
