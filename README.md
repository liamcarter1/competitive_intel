# Competitive Intelligence Monitor

A multi-agent competitive intelligence platform that generates executive-ready briefing reports on your competitors. Powered by [CrewAI](https://www.crewai.com/) for agent orchestration, with a [Gradio](https://www.gradio.app/) web interface for interactive use.

Enter a company name, its industry, and a list of competitors — the system dispatches a team of AI agents to research the web, analyze findings, develop strategic recommendations, and compile everything into a structured briefing document. Once a briefing is generated, you can ask follow-up questions against it or launch deep-dive research with live web search.

## Features

### Briefing Generation

Four AI agents work in sequence to produce a comprehensive competitive intelligence briefing:

1. **Trend Scanner** — Searches the web for recent competitor activity: product launches, R&D investments, patent filings, hiring patterns, pricing changes, partnerships, funding rounds, and more. Runs multiple targeted searches per competitor using the Serper API.
2. **Company Analyst** — Analyzes raw findings and segments them for Engineering (capability gaps, tech bets, R&D signals), Sales (positioning shifts, pricing changes, win/loss signals), and Strategy (market trends, threats, opportunities). Categorizes everything by urgency.
3. **Strategy Advisor** — Synthesizes the analysis into actionable recommendations per business function, each backed by specific competitive evidence with impact ratings, difficulty assessments, and suggested timelines.
4. **Report Writer** — Compiles all intelligence into a structured executive briefing with seven sections: Executive Summary, Product & Technology Landscape, Market & Business Intelligence, Competitor Deep Dives, Threat Assessment, Strategic Recommendations, and Watch List.

The final briefing is saved to `output/briefing.md` and displayed in the UI.

### Interactive Q&A

After generating or loading a briefing, two modes of follow-up are available:

- **Quick Chat** — Ask questions about the briefing report. Answers are grounded strictly in the report content using GPT-4o-mini. The system will not speculate or add information beyond what the report contains.
- **Research This (Deep Dive)** — Ask a question that triggers a live web search. The system generates targeted search queries, runs them through the Serper API, then synthesizes a source-cited answer using Claude. Every claim is backed by a URL reference.

### Report Management

- **Load Latest Report** — Reload the most recently generated briefing from disk without re-running the agent pipeline.

## Prerequisites

- **Python** 3.10 through 3.13
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- API keys for the following services:
  - [OpenAI](https://platform.openai.com/api-keys) — used by CrewAI agents and quick chat
  - [Anthropic](https://console.anthropic.com/) — used by deep-dive research synthesis
  - [Serper](https://serper.dev/) — used for web search by the trend scanner agent and deep-dive research

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

The agents use different models depending on their role, configured in `src/competitive_intel/config/agents.yaml`:

| Agent | Model | Rationale |
|---|---|---|
| Trend Scanner | `gpt-4o-mini` | High-volume search synthesis, cost-efficient |
| Company Analyst | `gpt-4o-mini` | Structured analysis from existing findings |
| Strategy Advisor | `gpt-4o` | Higher reasoning for strategic recommendations |
| Report Writer | `gpt-4o-mini` | Document compilation and formatting |

Quick Chat uses `gpt-4o-mini`. Deep Dive synthesis uses `claude-3-5-sonnet-latest`.

Models can be changed by editing the `llm` field in `agents.yaml` or the model parameter in `app.py`.

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
4. Click **Generate Briefing**. The agent pipeline will run — this involves multiple web searches and LLM calls.
5. Once complete, the briefing appears in the UI and is saved to `output/briefing.md`.
6. Use the **Quick Chat** or **Research This** buttons to ask follow-up questions.

### CLI

```bash
cd competitive_intel/competitive_intel
uv run competitive_intel
```

Runs the agent pipeline with default inputs (editable in `src/competitive_intel/main.py`) and writes the briefing to `output/briefing.md`.

Additional CLI commands:

```bash
# Train the crew over multiple iterations
uv run competitive_intel train <n_iterations> <output_filename>

# Replay from a specific task
uv run competitive_intel replay <task_id>
```

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
        ├── main.py                     # CLI entry points (run, train, replay)
        ├── crew.py                     # CrewAI crew definition — agents and tasks
        ├── config/
        │   ├── agents.yaml             # Agent roles, goals, backstories, LLM config
        │   └── tasks.yaml              # Task descriptions, outputs, dependencies
        └── tools/
            └── __init__.py             # Custom CrewAI tool definitions
```

## How It Works

### Agent Pipeline

The system uses CrewAI's sequential process mode. Each agent completes its task before the next begins, with outputs passed forward as context:

```
trend_scanner (web search)
    └──> company_analyst (analysis, segmented by audience)
             └──> strategy_advisor (recommendations with evidence)
                      └──> report_writer (final briefing document)
```

Task dependencies are defined in `config/tasks.yaml` via the `context` field, which determines what prior task outputs each agent receives.

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

### Adding a New Agent

1. Define the agent in `src/competitive_intel/config/agents.yaml` with `role`, `goal`, `backstory`, and `llm`.
2. Define its task in `src/competitive_intel/config/tasks.yaml` with `description`, `expected_output`, `agent`, and `context` (list of upstream task names).
3. Add `@agent` and `@task` decorated methods in `src/competitive_intel/crew.py`. Method ordering determines pipeline position.

### Adding Custom Tools

Create tool classes in `src/competitive_intel/tools/__init__.py` following CrewAI's tool patterns, then attach them to agents in `crew.py` via the `tools=[]` parameter.

## License

This project does not currently specify a license.
