# Competitive Intelligence Monitor

A multi-agent competitive intelligence platform that generates executive-ready briefing reports on your competitors. Powered by [LangGraph](https://langchain-ai.github.io/langgraph/) for pipeline orchestration with mixed LLM providers (OpenAI + Anthropic), and a [Gradio](https://www.gradio.app/) web interface for interactive use.

Enter a company name, its industry, and a list of competitors — the system dispatches a pipeline of AI nodes to research the web, analyze findings, develop strategic recommendations, and compile everything into a structured briefing document. Once a briefing is generated, you can ask follow-up questions against it or launch deep-dive research with live web search.

## Features

### Briefing Generation

Four pipeline nodes produce a comprehensive competitive intelligence briefing using a fan-out/fan-in architecture:

1. **Trend Scanner** (GPT-4o) — Searches the web for recent competitor activity: product launches, R&D investments, patent filings, hiring patterns, pricing changes, partnerships, funding rounds, and more. Runs multiple targeted searches per competitor via the Serper API. **Runs in parallel** — one scan node per competitor simultaneously.
2. **Company Analyst** (Claude Sonnet) — Analyzes raw findings and segments them for Engineering (capability gaps, tech bets, R&D signals), Sales (positioning shifts, pricing changes, win/loss signals), and Strategy (market trends, threats, opportunities). Categorizes everything by urgency.
3. **Strategy Advisor** (Claude Sonnet) — Synthesizes the analysis into actionable recommendations per business function, each backed by specific competitive evidence with impact ratings, difficulty assessments, and suggested timelines.
4. **Report Writer** (GPT-4o-mini) — Compiles all intelligence into a structured executive briefing with seven sections: Executive Summary, Product & Technology Landscape, Market & Business Intelligence, Competitor Deep Dives, Threat Assessment, Strategic Recommendations, and Watch List.

The final briefing is saved to `output/briefing.md` and displayed in the UI.

### Interactive Q&A

After generating or loading a briefing, two modes of follow-up are available:

- **Quick Chat** — Ask questions about the briefing report. Answers are grounded strictly in the report content using GPT-4o-mini. The system will not speculate or add information beyond what the report contains.
- **Research This (Deep Dive)** — Ask a question that triggers a live web search. The system generates targeted search queries, runs them through the Serper API, then synthesizes a source-cited answer using Claude. Every claim is backed by a URL reference.

### Report Management

- **Load Latest Report** — Reload the most recently generated briefing from disk without re-running the pipeline.

## Prerequisites

- **Python** 3.10 through 3.13
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- API keys for the following services:
  - [OpenAI](https://platform.openai.com/api-keys) — used by scan and report writing nodes, and quick chat
  - [Anthropic](https://console.anthropic.com/) — used by analysis and recommendation nodes, and deep-dive research
  - [Serper](https://serper.dev/) — used for web search by scan nodes and deep-dive research

## Installation

```bash
git clone https://github.com/liamcarter1/competitive_intel.git
cd competitive_intel/competitive_intel
uv sync
```

## Configuration

Create a `.env` file in the `competitive_intel/competitive_intel/` directory (or export the variables in your shell):

```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
SERPER_API_KEY=your-serper-key
```

Do not commit this file — it is included in `.gitignore`.

### LLM Models

Each pipeline node uses its own LLM client, so models can be mixed freely across providers:

| Node | Model | Rationale |
|---|---|---|
| Trend Scanner (scan) | `gpt-4o` | Reliable for search result summarization |
| Company Analyst (analyze) | `claude-sonnet-4-20250514` | Better analytical reasoning |
| Strategy Advisor (recommend) | `claude-sonnet-4-20250514` | Better strategic synthesis |
| Report Writer (write_briefing) | `gpt-4o-mini` | Cost-effective for formatting |

Quick Chat uses `gpt-4o-mini`. Deep Dive synthesis uses `claude-3-5-sonnet-latest`.

Models can be changed by editing the node functions in `graph.py` or the model parameter in `app.py`.

## Usage

### Web UI (Recommended)

```bash
cd competitive_intel/competitive_intel
uv run python app.py
```

This launches a Gradio web interface (default: `http://127.0.0.1:7860`).

1. Enter the **Company Name** you want intelligence for (e.g., "OpenAI").
2. Enter the **Industry** (e.g., "Artificial Intelligence").
3. Enter **Competitors** as a comma-separated list (e.g., "Anthropic, Google DeepMind, Meta AI, Mistral").
4. Click **Generate Briefing**. The pipeline will run — competitors are scanned in parallel, then analysis, recommendations, and report writing run sequentially.
5. Once complete, the briefing appears in the UI and is saved to `output/briefing.md`.
6. Use the **Quick Chat** or **Research This** buttons to ask follow-up questions.

### CLI

```bash
cd competitive_intel/competitive_intel
uv run competitive_intel
```

Runs the pipeline with default inputs (editable in `src/competitive_intel/main.py`) and writes the briefing to `output/briefing.md`.

## Project Structure

```
competitive_intel/
├── CLAUDE.md                           # Development guidelines and coding protocols
├── README.md
└── competitive_intel/
    ├── app.py                          # Gradio web UI and chat/deep-dive logic
    ├── pyproject.toml                  # Project metadata and dependencies
    ├── uv.lock                         # Locked dependency versions
    ├── output/                         # Generated briefing reports
    │   └── briefing.md
    └── src/competitive_intel/
        ├── __init__.py
        ├── main.py                     # CLI entry point (run)
        ├── graph.py                    # LangGraph StateGraph — nodes and pipeline wiring
        ├── config/
        │   ├── agents.yaml             # Agent roles, goals, backstories (system prompts)
        │   └── tasks.yaml              # Task descriptions and expected outputs (user prompts)
        └── tools/
            └── __init__.py             # search_serper() web search function
```

## How It Works

### Pipeline Graph

The system uses a LangGraph StateGraph with fan-out/fan-in for parallel competitor scanning, followed by sequential analysis nodes. Each node makes its own LLM API call with a clean message list — no shared conversation history between nodes.

```
                    ┌─ scan(competitor_A) ─┐
User Input ──→ fan_out ─→ scan(competitor_B) ─→ fan_in ──→ analyze ──→ recommend ──→ write_briefing
                    └─ scan(competitor_C) ─┘
```

- **Fan-out** uses LangGraph `Send()` to dispatch one scan node per competitor in parallel
- **Fan-in** aggregates all scan results into shared `GraphState`
- **Sequential edges** connect analyze → recommend → write_briefing
- Agent prompts (role, goal, backstory) are loaded from `agents.yaml` and task prompts from `tasks.yaml`

### Deep Dive Research Flow

When a user clicks "Research This":

1. GPT-4o-mini generates 3-5 targeted search queries based on the question and briefing context.
2. Each query is sent to the Serper API for Google search results.
3. All search results are passed to Claude (claude-3-5-sonnet-latest) along with the original briefing for context.
4. Claude synthesizes a grounded, source-cited response using only the search results.

### Briefing Report Sections

Generated briefings follow a standardized structure:

1. **Executive Summary** — Top 5-7 critical developments with overall threat level assessment.
2. **Product & Technology Landscape** — Competitor releases, R&D signals, capability comparisons.
3. **Market & Business Intelligence** — Pricing, go-to-market, partnerships, customer movement.
4. **Competitor Deep Dives** — Per-competitor activity summaries with strategic intent assessments.
5. **Threat Assessment** — Ranked threats with urgency levels and recommended responses.
6. **Strategic Recommendations** — Actionable items segmented for Engineering, Sales, and Strategy teams.
7. **Watch List** — Items requiring continued monitoring.

## Extending the System

### Adding a New Node

1. Define the agent config in `src/competitive_intel/config/agents.yaml` with `role`, `goal`, and `backstory`.
2. Define its task in `src/competitive_intel/config/tasks.yaml` with `description` and `expected_output`.
3. Add a node function in `src/competitive_intel/graph.py` that loads prompts from config, calls an LLM, and returns state updates.
4. Wire the node into the graph in `build_graph()` with appropriate edges.

### Adding Custom Tools

Add tool functions in `src/competitive_intel/tools/__init__.py`, then call them directly from the relevant graph node. No LLM tool-calling is needed — nodes call tools as regular Python functions.

## License

This project does not currently specify a license.
