# FORliam.md — Competitive Intelligence Monitor

## What This Project Actually Does

Imagine you're a strategy VP at a tech company. Every week, you need to know what your competitors are up to — who shipped what, who hired whom, who's pivoting their pricing. Normally, this means a junior analyst spending days reading press releases, trawling LinkedIn, and assembling a slide deck that's already stale by Friday.

This project automates that entire workflow. You type in a company name, an industry, and a list of competitors. A pipeline of AI agents fans out across the web, researches each competitor in parallel, then passes everything through an analyst, a strategist, and a report writer — each one a different LLM chosen for the job it's best at. Out the other end comes a structured executive briefing with threat assessments, actionable recommendations, and source citations.

There's also a chat interface bolted on top: you can ask questions about the briefing (grounded strictly in what it says) or hit "Research This" to trigger a live web search with a fully cited deep-dive answer.

---

## The Architecture — And Why It Looks Like This

### The Pipeline: Think Assembly Line, Not Committee

The core of the system is a **graph-based pipeline** built with LangGraph. Picture a car factory:

```
                    ┌─ scan(Anthropic) ────┐
User Input ──→ fan_out ─→ scan(DeepMind) ────→ fan_in ──→ analyze ──→ recommend ──→ write_briefing
                    └─ scan(Mistral) ─────┘
```

1. **Fan-out**: The system splits into parallel tracks — one per competitor. Each track searches the web (via Serper API) and summarizes what it finds. If you have 4 competitors, 4 scans run simultaneously. This is like sending 4 scouts out in different directions instead of one scout visiting 4 locations sequentially.

2. **Fan-in**: All the scout reports land back in one place — a shared `GraphState` dictionary.

3. **Sequential chain**: The combined intelligence flows through three more nodes in order: analysis, recommendations, final report. Each one builds on the previous output.

The key design principle: **each node is an island**. It gets a fresh LLM conversation with its own system prompt and user message. No message history leaks between nodes. This is the whole reason we moved to LangGraph (more on that below).

### The File Structure: Where Everything Lives

```
competitive_intel/
├── app.py              ← Gradio web UI (the thing users see)
├── src/competitive_intel/
│   ├── graph.py        ← The brain: LangGraph pipeline definition
│   ├── main.py         ← CLI entry point (just calls graph.py)
│   ├── config/
│   │   ├── agents.yaml ← WHO each agent is (role, goal, backstory)
│   │   └── tasks.yaml  ← WHAT each agent does (task descriptions)
│   └── tools/
│       └── __init__.py ← search_serper() function
```

The separation between `agents.yaml`/`tasks.yaml` and `graph.py` is deliberate. The YAML files are like job descriptions — you can tweak an agent's personality or task instructions without touching any Python code. The graph.py file is the wiring — it loads those descriptions, plugs them into LLM calls, and connects the nodes together.

`app.py` is the frontend. It doesn't know or care about LangGraph internals — it just calls `run_pipeline(company, industry, competitors)` and gets back a string of markdown.

---

## The Tech Decisions — And the Stories Behind Them

### Why We Ditched CrewAI for LangGraph

This project originally used **CrewAI**, a framework that lets you define AI "agents" with roles and tasks and run them in a pipeline. It worked — until it didn't.

The problem was a **tool-calling message format bug**. When an LLM uses a tool (like a web search), it generates a special `tool_use` message, and the tool's result comes back as a `tool_result` message. These have to be paired correctly. CrewAI passed the full conversation history — including tool call/result pairs from previous agents — into the next agent's context. Different LLM providers (OpenAI, Anthropic) use different formats for these messages. So when Agent A (OpenAI) used a tool and Agent B (Anthropic) received that conversation history, the message format was incompatible. The pipeline would crash.

This is a subtle but important lesson: **frameworks that abstract away LLM communication details can bite you when you need to mix providers**. CrewAI was designed for a world where everyone uses OpenAI. The moment you want Claude for analysis and GPT for search summarization, the abstraction breaks.

LangGraph solved this by giving us explicit control. Each node constructs its own message list from scratch. The only thing shared between nodes is the *output text* — clean strings in the state dictionary, not raw LLM message objects. Think of it as the difference between passing someone a finished report versus forwarding them the entire email thread that produced it.

### Why Different Models for Different Jobs

| Node | Model | Why |
|------|-------|-----|
| Scan | GPT-4o | Solid at summarizing search results, reliable tool-adjacent behavior |
| Analyze | Claude Sonnet | Stronger at structured analytical reasoning, better at segmenting for audiences |
| Recommend | Claude Sonnet | Better at strategic synthesis, produces more actionable outputs |
| Write briefing | GPT-4o-mini | The cheapest option — by this point, the hard thinking is done, and this node just formats |

This is a pattern worth remembering: **match the model to the cognitive demand of the task**. You wouldn't hire a senior architect to paint walls. The scan node does relatively mechanical work (read search results, extract key points), so a cheaper/faster model works fine. The analysis node needs to reason across multiple competitors and segment findings for different audiences — that's where you want the stronger model.

### Why Serper Instead of LLM Tool-Calling

The scan nodes don't use LLM tool-calling to search the web. Instead, they:
1. Generate hardcoded search queries based on the competitor name
2. Call the Serper API directly as a Python function
3. Feed the results into the LLM as user message content

This sidesteps the entire tool-calling format problem. It's also more predictable — you know exactly what searches will run, and the LLM can't decide to skip searching or search for something irrelevant. The tradeoff is less flexibility (the LLM can't dynamically generate creative search queries), but for this use case, the fixed query templates (`"{competitor} latest product news"`, `"{competitor} R&D investments"`, etc.) cover the ground well enough.

**Lesson**: Tool-calling is powerful but adds complexity. If you can achieve the same result with a simple function call + prompt, prefer that. Save tool-calling for cases where the LLM genuinely needs to decide *whether* and *how* to use a tool.

### Why Fan-Out Instead of Sequential Scanning

The original CrewAI version scanned competitors one at a time. With 4 competitors, that's 4 sequential LLM calls + 4 sets of web searches, all waiting on each other. The fan-out approach runs them in parallel using LangGraph's `Send()` primitive.

This is a real-world performance win. Each scan takes maybe 15-30 seconds (network calls + LLM inference). Sequential: 60-120 seconds. Parallel: still 15-30 seconds. The analysis, recommendation, and writing stages have to be sequential (each depends on the previous), but the scanning stage is embarrassingly parallel — each competitor scan is completely independent.

LangGraph handles the fan-in automatically: the `scan_results` field in state uses `Annotated[list[str], operator.add]`, which means results from parallel nodes get concatenated into a single list. This is a nice pattern — you declare the merge strategy in the type annotation, and the framework handles the rest.

---

## Bugs We Hit and How We Fixed Them

### Bug 1: The .env File That Nobody Loaded

**What happened**: After rewriting everything to LangGraph, the pipeline crashed immediately with `OpenAIError: The api_key client option must be set`. The API keys were sitting in a `.env` file, but nothing was loading them into the environment.

**Why it happened**: CrewAI had its own dotenv loading built in (buried somewhere in its internals). When we removed CrewAI, that implicit behavior disappeared. The `.env` file existed, the keys were correct, but `os.environ` had no idea they were there.

**The fix**: Added `python-dotenv` as a dependency and called `load_dotenv()` at the top of both entry points (`main.py` and `app.py`). Two lines of code, but without them, nothing works.

**The lesson**: When you rip out a framework, you lose its implicit behaviors — not just the ones you know about. CrewAI was silently loading env vars, managing message formatting, handling tool execution. Each of those becomes your responsibility. When migrating away from a framework, make a checklist of everything it was doing for you, not just the things you're replacing.

### Bug 2: Windows File Locking During Dependency Swap

**What happened**: Running `uv sync` after changing dependencies from CrewAI to LangGraph failed with `Access is denied (os error 5)` on `.pyd` files inside the virtual environment.

**Why it happened**: On Windows, compiled Python extensions (`.pyd` files, which are basically DLLs) get locked by any process that imports them. If a previous Python process hadn't fully terminated, or if an IDE had the venv's Python loaded, those files can't be deleted or replaced.

**The fix**: Deleted the entire `.venv` directory and re-ran `uv sync` to rebuild from scratch. Clean slate.

**The lesson**: On Windows, when doing major dependency swaps, it's often faster to nuke the venv than to fight file locks. On Linux/macOS this rarely happens because file deletion works differently (you can delete a file while it's open). If you're on Windows and hitting permission errors during dependency changes, close your IDE, kill any Python processes, then try again — or just delete `.venv`.

### Bug 3: The `import warnings` That Became Dead Code

**What happened**: After removing CrewAI, there was a leftover `warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")` line in `app.py`. The `pysbd` module was a CrewAI dependency for sentence boundary detection — it's no longer installed.

**Why it matters**: It's not a crash bug — filtering warnings for a module that doesn't exist is harmless. But it's noise. Dead code creates confusion for the next person reading the file ("what's pysbd? do I need it?"). We removed it during the migration.

**The lesson**: When removing a dependency, grep for its name across the codebase. You'll often find imports, warning filters, config entries, or comments referencing it that should be cleaned up.

---

## Potential Pitfalls and How to Avoid Them

### Pitfall: State Schema Mismatches in LangGraph

LangGraph uses TypedDict for state, and if a node returns a key that doesn't match the schema, or returns the wrong type, you get runtime errors that can be cryptic. The `Annotated[list[str], operator.add]` pattern for `scan_results` is particularly easy to get wrong — if a node returns a plain string instead of a list, the `operator.add` fails silently or concatenates characters.

**Avoidance**: Always return the exact type declared in the state schema. For list fields with `operator.add`, always return a list (even if it's a single-item list like `[result]`).

### Pitfall: YAML Interpolation Failures

The agent/task YAML configs use Python `.format()` interpolation (`{company}`, `{competitor}`, etc.). If a template references a variable that's not in the inputs dict, you get a `KeyError` at runtime — in the middle of a pipeline run, after you've already burned API credits on earlier nodes.

**Avoidance**: When adding new YAML templates, make sure every `{variable}` has a corresponding key in the inputs dict. Test the interpolation locally before running the full pipeline.

### Pitfall: LLM Cost Surprises

The analysis and recommendation nodes send the full scan results (potentially thousands of tokens per competitor) as input context. With 4+ competitors, the input to the `analyze` node can easily be 10,000+ tokens. Claude Sonnet isn't cheap at scale.

**Avoidance**: Monitor token usage. If costs are a concern, consider summarizing scan results before passing them to analysis, or using a cheaper model for analysis with a more detailed prompt.

---

## Technologies Worth Understanding

### LangGraph

LangGraph is a graph-based orchestration library from the LangChain team. Think of it as "state machines for LLM pipelines." You define nodes (functions), edges (transitions), and state (a shared dictionary). It handles parallelism, state management, and checkpointing.

The killer feature for this project is `Send()` — it lets a single node dynamically dispatch multiple copies of another node with different inputs. This is how we do fan-out: one `fan_out` function returns `[Send("scan_competitor", {..., "competitor": c}) for c in competitors]`, and LangGraph runs them all in parallel.

It's a lower-level tool than CrewAI — you write more code, but you understand exactly what's happening. For production systems where you need reliability and debuggability, that tradeoff is usually worth it.

### Serper API

Serper is a Google Search API wrapper. You POST a query, you get back structured search results (title, link, snippet). It's simpler and cheaper than Google's official Custom Search API. The 15-second timeout is important — web search APIs occasionally hang, and you don't want a single slow search to block your entire pipeline.

### Gradio

Gradio is a Python library for building web UIs for ML applications. You define components (textboxes, buttons, markdown displays) and wire them to Python functions. It handles the web server, WebSocket connections, and frontend rendering. For internal tools and demos, it's dramatically faster than building a proper frontend.

---

## How Good Engineers Think About This

### Separation of Concerns

Notice how the system has clean boundaries: `app.py` handles UI, `graph.py` handles orchestration, `tools/__init__.py` handles external API calls, YAML files handle prompts. You can change the UI without touching the pipeline, swap models without touching the UI, or modify prompts without touching any Python code. Each piece has one job.

### Fail Fast, Fail Loud

The pipeline doesn't silently swallow errors. If a Serper search fails, the error message ends up in the scan results ("Search error for '...': ...") and propagates through the pipeline. The analysis node sees it and can work around it. At the UI level, exceptions are caught and displayed to the user. At no point does the system pretend everything is fine when it isn't.

### Pragmatism Over Purity

The scan node uses hardcoded search query templates instead of dynamic LLM-generated queries. The `search_serper()` function is 10 lines of code instead of a proper tool class with retry logic and rate limiting. The YAML configs are loaded once at module level instead of being dependency-injected. These are all "impure" choices that a textbook might frown at — but they make the code simpler, faster to debug, and easier to understand. Good engineers optimize for the team's ability to maintain and modify the code, not for architectural elegance points.

### Know When to Rip and Replace

The move from CrewAI to LangGraph wasn't a refactor — it was a replacement. We didn't try to patch CrewAI's message format handling or wrap it in an adapter layer. When a framework's core abstraction doesn't fit your needs (in this case, the assumption that all agents share a conversation history), it's faster and cleaner to replace it than to fight it. The rewrite took one session. Working around CrewAI's limitations would have been an ongoing tax on every future change.
