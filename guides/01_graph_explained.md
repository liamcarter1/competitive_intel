# graph.py — Line-by-Line Explanation

This is the **brain of the entire application**. It defines a LangGraph pipeline that orchestrates multiple LLM calls in a structured workflow: search the web for competitor intelligence, analyze it, generate recommendations, and write a final report. If you want to understand LangGraph deeply, this is the file to study.

---

## Imports (Lines 1-16)

```python
from __future__ import annotations
```
**Line 1:** This is a Python compatibility trick. It makes all type annotations in this file behave as strings (deferred evaluation). Without this, `Annotated[list[str], operator.add]` on line 37 would fail in Python 3.9 because `list[str]` as a type hint wasn't supported until 3.10. With this import, Python doesn't try to evaluate the annotation at runtime — it just stores it as a string. This is a good habit for any code that uses modern type hints but needs to support slightly older Python versions.

```python
import json
import operator
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict
```
**Lines 3-8:** Standard library imports.
- `json`: Used for serializing search results.
- `operator`: This is the key import for LangGraph's **reducer pattern**. Specifically, `operator.add` is used to tell LangGraph how to merge results from parallel nodes (more on this at line 37).
- `os`: For environment variable access.
- `datetime`: To stamp reports with the current date.
- `Path`: Pythonic file path handling (better than string concatenation like `"config/" + "agents.yaml"`).
- `Annotated, TypedDict`: Typing constructs. `TypedDict` defines the shape of state that flows through the graph. `Annotated` attaches metadata (the reducer function) to a type.

```python
import yaml
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.types import Send
```
**Lines 10-14:** Third-party imports. This is where the LangChain/LangGraph ecosystem comes in.
- `yaml`: Parses the YAML config files that define agent prompts.
- `ChatAnthropic`: LangChain's wrapper around the Anthropic API. It gives you a unified `.invoke()` interface so you can swap between providers without changing your code.
- `ChatOpenAI`: Same thing but for OpenAI's API.
- `END`: A special constant in LangGraph that means "the graph is done, stop executing." Think of it as the exit door.
- `StateGraph`: The core LangGraph class. It's a directed graph where each node is a function and edges define execution order. State flows through the graph and gets updated by each node.
- `Send`: This is LangGraph's mechanism for **fan-out** (parallelism). Instead of a node returning just state updates, it can return `Send` objects that say "run this other node with this specific input." This is how we scan multiple competitors in parallel.

```python
from competitive_intel.tools import search_serper, search_serper_news
```
**Line 16:** Imports our custom web search functions from the tools module. `search_serper` calls the standard `/search` endpoint for web results, while `search_serper_news` calls the `/news` endpoint for recent news articles with date filtering. This keeps the search logic separate from the graph logic — good separation of concerns.

---

## Configuration Loading (Lines 18-27)

```python
CONFIG_DIR = Path(__file__).parent / "config"
```
**Line 18:** Builds the path to the `config/` directory. `Path(__file__)` gives us the path to `graph.py` itself, `.parent` goes up one directory (to `src/competitive_intel/`), then `/ "config"` appends the config folder. This is **relative to the code**, not relative to where you run the script from — which is important because scripts can be run from any working directory.

```python
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
```
**Line 19:** Navigates up three levels from `graph.py` to reach the `competitive_intel/` project root, then into `output/`. The `.resolve()` call converts the path to an absolute path (resolving any symlinks), which prevents weird path issues. The three `.parent` calls go: `competitive_intel/` (from `src/competitive_intel/`) -> `src/` -> `competitive_intel/` (project root).

```python
def _load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
```
**Lines 22-23:** A helper that reads a YAML file and parses it into a Python dictionary. `yaml.safe_load` is important — never use `yaml.load` without a Loader argument because it can execute arbitrary Python code embedded in the YAML (a security vulnerability). `safe_load` only allows basic data types.

```python
AGENTS_CONFIG = _load_yaml("agents.yaml")
TASKS_CONFIG = _load_yaml("tasks.yaml")
```
**Lines 26-27:** Load both config files at **module import time** (when Python first imports `graph.py`). This means the YAML is parsed once and cached in memory as dictionaries. Every node function just reads from these dictionaries rather than re-parsing the files. This is a performance optimization — YAML parsing isn't free, and these configs don't change at runtime.

---

## State Definition (Lines 30-40)

This is one of the most important concepts in LangGraph.

```python
class GraphState(TypedDict):
    company: str
    industry: str
    competitors: str
    current_date: str
    scan_results: Annotated[list[str], operator.add]
    analysis: str
    recommendations: str
    briefing: str
    evaluation_result: str
    evaluation_feedback: str
    retry_count_analysis: int
    retry_count_recommendations: int
```

**Lines 32-44:** `GraphState` defines **everything the graph knows**. Every node receives this state and can read from or write to it. Think of it as a shared whiteboard that all the agents can see.

- `company`, `industry`, `competitors`, `current_date` (lines 33-36): **Input fields.** These are set once at the start and never changed. Every node reads them to know what it's working on.
- `scan_results` (line 37): **This is the most interesting field.** The `Annotated[list[str], operator.add]` syntax is LangGraph's **reducer pattern**. Here's the problem it solves: when multiple scan nodes run in parallel (one per competitor), they each return a `scan_results` list. How should LangGraph combine them? Without a reducer, the last one to finish would overwrite the others. The `operator.add` reducer tells LangGraph: "concatenate all the lists together." So if scan_A returns `["result_A"]` and scan_B returns `["result_B"]`, the final state has `scan_results = ["result_A", "result_B"]`. This is the core mechanism that makes fan-out/fan-in work.
- `analysis`, `recommendations`, `briefing` (lines 38-40): **Output fields.** Each sequential node fills in one of these. They're just regular strings — no reducer needed because only one node writes to each.
- `evaluation_result`, `evaluation_feedback` (lines 41-42): **Evaluator fields.** The evaluate node writes its verdict here (`"pass"`, `"fail_analysis"`, `"fail_recommendations"`, or `"fail_both"`) along with specific feedback. The analyze and recommend nodes read these on retries to know what to fix.
- `retry_count_analysis`, `retry_count_recommendations` (lines 43-44): **Retry counters.** Track how many times each node has been retried. Used by the routing logic to enforce the max retry cap (2 per node).

**Why TypedDict and not a regular class?** LangGraph requires state to be a TypedDict (or a Pydantic BaseModel). TypedDict is essentially a type-annotated dictionary — it gives you IDE autocomplete and type checking while keeping the simplicity of dict access (`state["company"]`). LangGraph inspects these annotations at graph compile time to know which fields exist and which have reducers.

---

## LLM Client Factories (Lines 43-50)

```python
def _openai(model: str = "gpt-4o", temperature: float = 0.2):
    return ChatOpenAI(model=model, temperature=temperature)
```
**Lines 45-46:** Creates an OpenAI LLM client. `ChatOpenAI` is LangChain's wrapper — it reads `OPENAI_API_KEY` from the environment automatically (you never pass the key directly). The `temperature=0.2` means the model is fairly deterministic — lower temperature = less creative/random, higher = more varied. For competitive intelligence, you want consistency, not creativity.

**Why a factory function instead of a global client?** Each node might want a different model or temperature. The factory pattern (`_openai("gpt-4o-mini")`) is cleaner than having many global variables.

```python
def _claude(model: str = "claude-sonnet-4-20250514", temperature: float = 0.2):
    return ChatAnthropic(model=model, temperature=temperature)
```
**Lines 49-50:** Same pattern for Anthropic's Claude. Notice how `ChatOpenAI` and `ChatAnthropic` have the **exact same interface** — both have `.invoke()` that takes a list of messages. This is the power of LangChain's abstraction: you can swap providers by changing one line of code. The graph nodes don't care which provider they're using.

---

## Prompt Builders (Lines 53-67)

```python
def _agent_system_prompt(agent_key: str, inputs: dict) -> str:
    cfg = AGENTS_CONFIG[agent_key]
    role = cfg["role"].format(**inputs).strip()
    goal = cfg["goal"].format(**inputs).strip()
    backstory = cfg["backstory"].format(**inputs).strip()
    return f"Role: {role}\nGoal: {goal}\nBackstory: {backstory}"
```
**Lines 55-60:** Builds a system prompt from the YAML config. Here's what happens step by step:
1. `AGENTS_CONFIG[agent_key]` looks up the agent (e.g., `"trend_scanner"`) in the loaded YAML.
2. `.format(**inputs)` does Python string interpolation. If the YAML contains `{company}` and `inputs` has `{"company": "Danfoss"}`, it becomes `"Danfoss"`. The `**` unpacks the dictionary into keyword arguments.
3. `.strip()` removes leading/trailing whitespace (YAML's `>` block scalar often leaves trailing newlines).
4. The return string combines role, goal, and backstory into a single system prompt.

**Why put prompts in YAML instead of hardcoding them?** Separation of concerns. Product people or prompt engineers can edit `agents.yaml` without touching Python code. It also makes it easy to see all your prompts in one place rather than hunting through code files.

```python
def _task_prompt(task_key: str, inputs: dict) -> tuple[str, str]:
    cfg = TASKS_CONFIG[task_key]
    desc = cfg["description"].format(**inputs).strip()
    expected = cfg["expected_output"].format(**inputs).strip()
    return desc, expected
```
**Lines 63-67:** Same pattern for task prompts. Returns a tuple of (description, expected_output). The description tells the LLM what to do, and the expected_output tells it what format to respond in. Returning a tuple lets the caller unpack it cleanly: `desc, expected = _task_prompt(...)`.

---

## The Scan Node (Lines 70-131)

This is where the real work starts. The scan node is the most complex because it combines web search with LLM summarization.

```python
class ScanState(TypedDict):
    company: str
    industry: str
    competitors: str
    current_date: str
    competitor: str
    scan_results: Annotated[list[str], operator.add]
    analysis: str
    recommendations: str
    briefing: str
```
**Lines 72-81:** `ScanState` extends `GraphState` with one extra field: `competitor` (singular, line 77). When `fan_out` sends work to `scan_competitor`, it adds a `competitor` field specifying which single competitor this particular scan should focus on. LangGraph needs to know the full state shape for type checking, so `ScanState` includes everything from `GraphState` plus the extra field.

```python
def scan_competitor(state: ScanState) -> dict:
    competitor = state["competitor"]
    inputs = {
        "company": state["company"],
        "industry": state["industry"],
        "competitors": state["competitors"],
        "current_date": state["current_date"],
        "competitor": competitor,
    }
```
**Lines 84-92:** The node function. Every LangGraph node takes `state` as input and returns a dict of state updates. It doesn't return the full state — just the fields it wants to change. LangGraph handles merging.

The `inputs` dict is built from state for use in prompt interpolation. This is the data that will fill in `{company}`, `{competitor}`, etc. in the YAML templates.

```python
    system = _agent_system_prompt("trend_scanner", inputs)
    desc, expected = _task_prompt("scan_competitor", inputs)
```
**Lines 94-95:** Build the prompts. After this, `system` contains something like "Role: Hydraulics Competitive Trend Scanner\nGoal: Find the latest competitor news..." and `desc` contains the detailed task description.

```python
    # ── News searches (Serper /news endpoint, filtered to past month) ────────
    news_queries = [
        f"{competitor} {industry} news announcement {year}",
        f"{competitor} product launch release update {year}",
        f"{competitor} acquisition merger partnership deal {year}",
        # ... 9 queries total covering news, products, M&A, pricing,
        #     customer wins, leadership, earnings, regulatory, analyst ratings
    ]

    # ── Web searches (Serper /search endpoint, broader context) ──────────────
    web_queries = [
        f"{competitor} {industry} strategy expansion growth plans {year}",
        f"{competitor} hiring jobs open roles site:linkedin.com OR site:indeed.com {year}",
        # ... 5 queries total covering strategy, product roadmaps,
        #     job postings, patents, regulatory/trade exposure
    ]
```
**Lines 101-125:** The scan uses a **dual-endpoint strategy** — two tiers of search queries designed to catch different types of competitive intelligence:

1. **News queries** (9 queries): Call `search_serper_news()` which hits Serper's `/news` endpoint, filtered to the past month (`tbs="qdr:m"`). These return actual news articles sorted by recency — press releases, earnings reports, product launches, M&A, executive hires, regulatory actions, analyst coverage. Each result includes a publication date.

2. **Web queries** (5 queries): Call `search_serper()` which hits the standard `/search` endpoint. These pick up broader context that news doesn't cover — job postings on LinkedIn/Indeed (leading indicator of strategy), patent filings on USPTO, regulatory exposure, and company strategy pages.

**Why two endpoints?** The regular `/search` endpoint returns a mix of evergreen web content (Wikipedia, company "About" pages) and news, with evergreen often ranking higher. For a competitive intelligence tool, recency is everything — the `/news` endpoint cuts through the noise and surfaces breaking developments. The web queries complement this with signals that don't appear as news articles.

**Why hardcode the queries instead of letting the LLM generate them?** Speed and reliability. Having the LLM generate queries would require an extra API call (adding latency and cost), and the LLM might generate vague or unhelpful queries. These templates cover the key intelligence categories comprehensively.

```python
    # Run news searches (recent news articles, past month)
    for q in news_queries:
        try:
            data = search_serper_news(q, num_results=10, tbs="qdr:m")
            for item in data.get("news", [])[:8]:
                date = item.get("date", "")
                date_str = f" ({date})" if date else ""
                all_results.append(
                    f"- [NEWS{date_str}] [{item.get('title', '')}]({item.get('link', '')}): {item.get('snippet', '')}"
                )
        except Exception as e:
            all_results.append(f"- News search error for '{q}': {e}")

    # Run web searches (broader context, top 8 per query)
    for q in web_queries:
        try:
            data = search_serper(q)
            for item in data.get("organic", [])[:8]:
                all_results.append(
                    f"- [WEB] [{item.get('title', '')}]({item.get('link', '')}): {item.get('snippet', '')}"
                )
        except Exception as e:
            all_results.append(f"- Search error for '{q}': {e}")
```
**Lines 127-151:** Runs both tiers of searches and collects results.
- News results are tagged `[NEWS (date)]` and web results are tagged `[WEB]` — this lets the LLM know which are recent news vs broader web context, and prioritise accordingly.
- Top 8 results per query (up from the original 5) — catches more relevant content that may land lower in results.
- `data.get("news", [])` for news endpoint (different key from the web endpoint's `"organic"`).
- The `try/except` ensures one failed search doesn't crash the entire scan. The error is recorded as a result so the LLM knows something went wrong.
- Total potential results: 9 news queries × 8 + 5 web queries × 8 = **up to 112 results** per competitor (vs the original 15).

```python
    search_context = "\n".join(all_results) if all_results else "No search results found."
```
**Line 116:** Joins all results into a single string. The `if/else` handles the edge case where every search failed.

```python
    llm = _openai("gpt-4o")
    user_msg = (
        f"{desc}\n\n"
        f"Here are the web search results for {competitor}:\n\n"
        f"{search_context}\n\n"
        f"Expected output format:\n{expected}"
    )
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ])
```
**Lines 119-129:** The LLM call. This is the standard LangChain pattern:
1. Create an LLM client with `_openai("gpt-4o")`.
2. Build the user message by combining the task description, the raw search results, and the expected output format.
3. Call `.invoke()` with a list of message dicts. The list has exactly two messages: a system message (who the agent is) and a user message (what to do). This is the **chat completions** format — system sets the persona, user provides the task.

**Why GPT-4o for scanning?** It's good at summarizing search results — a relatively straightforward task. Claude Sonnet is reserved for the harder analytical work later.

```python
    print(f"[scan] Finished scanning {competitor} (gpt-4o)")
    return {"scan_results": [f"## {competitor}\n\n{response.content}"]}
```
**Lines 130-131:** The return value is crucial. It returns a dict with only `scan_results` — the one field this node updates. The value is a **list with one element** (the summary wrapped with a markdown heading). Remember the `operator.add` reducer on `scan_results`? When three scan nodes run in parallel, their lists get concatenated: `["## Parker\n\n..."] + ["## Bosch\n\n..."] + ["## Eaton\n\n..."]`.

---

## Fan-Out (Lines 134-136)

```python
def fan_out(state: GraphState) -> list[Send]:
    competitors = [c.strip() for c in state["competitors"].split(",") if c.strip()]
    return [Send("scan_competitor", {**state, "competitor": c}) for c in competitors]
```

This is a **conditional entry point** — it's the first thing that runs and decides what happens next.

**Line 135:** Splits the comma-separated competitors string into a clean list. `"Parker Hannifin, Bosch Rexroth, Eaton"` becomes `["Parker Hannifin", "Bosch Rexroth", "Eaton"]`. The `if c.strip()` filters out empty strings from trailing commas.

**Line 136:** This is where LangGraph's parallelism magic happens. `Send("scan_competitor", {**state, "competitor": c})` creates a `Send` object that says: "Run the `scan_competitor` node with this specific state." The `{**state, "competitor": c}` creates a new dict that has everything from the current state plus the `competitor` field set to one specific competitor.

If there are 3 competitors, this returns 3 `Send` objects, and LangGraph runs 3 `scan_competitor` instances **in parallel**. When all 3 finish, their `scan_results` lists are combined by the `operator.add` reducer, and execution moves to the next node.

**This is the fan-out/fan-in pattern**: one node fans out into many parallel tasks, and the results fan back in through the reducer.

---

## The Analyze Node (Lines 139-162)

```python
def analyze(state: GraphState) -> dict:
    inputs = {
        "company": state["company"],
        "industry": state["industry"],
        "competitors": state["competitors"],
        "current_date": state["current_date"],
    }

    system = _agent_system_prompt("company_analyst", inputs)
    desc, expected = _task_prompt("analyze_findings", inputs)
```
**Lines 139-148:** Same pattern as `scan_competitor` — extract state into an `inputs` dict, build system and task prompts from YAML config. This time using the `company_analyst` agent and `analyze_findings` task.

```python
    scan_text = "\n\n---\n\n".join(state["scan_results"])
```
**Line 150:** Joins all scan results with markdown horizontal rules as separators. At this point, `state["scan_results"]` contains the combined output from all parallel scan nodes (thanks to the `operator.add` reducer). So this creates one big text block with each competitor's scan separated by `---`.

```python
    feedback = state.get("evaluation_feedback", "")
    eval_result = state.get("evaluation_result", "")
    is_retry = eval_result in ("fail_analysis", "fail_both")
    if is_retry and feedback:
        user_content += (
            f"\n\n---\n"
            f"PREVIOUS ATTEMPT FEEDBACK (address these issues in your revised analysis):\n"
            f"{feedback}\n"
            f"Please revise your analysis to address the feedback above."
        )

    llm = _claude()
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ])
    retry_count = state.get("retry_count_analysis", 0)
    if is_retry:
        retry_count += 1
    return {"analysis": response.content, "retry_count_analysis": retry_count}
```
**Lines 158-178:** This node does double duty — it runs on the first pass and on retries. The retry logic works by checking the `evaluation_result` field in state. If the evaluator flagged the analysis as failing (`"fail_analysis"` or `"fail_both"`), the node appends the evaluator's specific feedback to the prompt. This way, the LLM sees both the original task and concrete guidance on what to fix (e.g., "missing Key Patterns section"). The retry counter is incremented so the routing logic knows when to stop retrying.

**Why Claude for analysis?** Claude Sonnet is generally better at analytical reasoning and structured thinking. The scan nodes use GPT-4o for simpler summarization; the heavier analytical work goes to Claude.

---

## The Recommend Node (Lines 165-186)

```python
def recommend(state: GraphState) -> dict:
    # ... same pattern: build inputs, get prompts ...

    llm = _claude()
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"{desc}\n\n"
            f"COMPETITIVE ANALYSIS:\n\n{state['analysis']}\n\n"
            f"Expected output format:\n{expected}"
        )},
    ])
    print(f"[recommend] Finished recommendations (claude-sonnet-4-20250514)")
    return {"recommendations": response.content}
```
**Lines 165-186:** Nearly identical structure to `analyze`. Key difference: instead of passing raw scan results, it passes `state['analysis']` — the output of the previous node. This is the **sequential pipeline** in action: each node builds on the previous node's output. The `strategy_advisor` agent generates actionable recommendations based on the analyst's work.

---

## The Write Briefing Node (Lines 189-215)

```python
def write_briefing(state: GraphState) -> dict:
    # ... build inputs, get prompts ...

    llm = _openai("gpt-4o-mini")
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"{desc}\n\n"
            f"COMPETITIVE ANALYSIS:\n\n{state['analysis']}\n\n"
            f"STRATEGIC RECOMMENDATIONS:\n\n{state['recommendations']}\n\n"
            f"Expected output format:\n{expected}"
        )},
    ])

    briefing = response.content
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "briefing.md").write_text(briefing, encoding="utf-8")
    print(f"[write_briefing] Finished briefing (gpt-4o-mini) -> output/briefing.md")
    return {"briefing": briefing}
```
**Lines 189-215:** The final node. It takes both the analysis and recommendations and compiles them into a polished briefing document. Uses `gpt-4o-mini` because this is essentially a formatting task — the hard thinking was done by Claude in the previous nodes. Using a cheaper model here saves cost.

**Lines 212-213:** A side effect — the briefing is saved to disk as `output/briefing.md`. `exist_ok=True` on `mkdir` means "don't error if the directory already exists." The `encoding="utf-8"` is important on Windows where the default encoding might not handle all characters.

---

## The Evaluate Node (Lines 254-288)

```python
def evaluate(state: GraphState) -> dict:
    inputs = {
        "company": state["company"],
        "industry": state["industry"],
        "competitors": state["competitors"],
        "current_date": state["current_date"],
        "analysis": state["analysis"],
        "recommendations": state["recommendations"],
    }

    system = _agent_system_prompt("quality_evaluator", inputs)
    desc, expected = _task_prompt("evaluate_quality", inputs)

    llm = _claude()
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": f"{desc}\n\n{expected}"},
    ])

    raw = response.content.strip()
    try:
        parsed = json.loads(raw)
        result = parsed.get("evaluation_result", "pass")
        feedback = parsed.get("evaluation_feedback", "")
    except (json.JSONDecodeError, AttributeError):
        result = "pass"
        feedback = ""

    return {"evaluation_result": result, "evaluation_feedback": feedback}
```

This is the **quality gate** — an LLM-as-judge that checks whether the analysis and recommendations meet the rubrics defined in `tasks.yaml`. The evaluator returns a JSON object with a verdict (`"pass"`, `"fail_analysis"`, `"fail_recommendations"`, or `"fail_both"`) and specific feedback explaining what's wrong.

**Key design: fail-open on parse errors.** If the evaluator LLM returns malformed JSON (which LLMs sometimes do — wrapping in markdown fences, adding preamble text), the code defaults to `"pass"`. The evaluator is a safety net, not a brick wall. A formatting hiccup in the quality gate shouldn't block the entire pipeline.

---

## Retry Routing (Lines 291-319)

```python
MAX_RETRIES = 2

def route_after_evaluation(state: GraphState) -> str:
    result = state.get("evaluation_result", "pass")
    retries_analysis = state.get("retry_count_analysis", 0)
    retries_recommendations = state.get("retry_count_recommendations", 0)

    if result == "pass":
        return "write_briefing"

    if result == "fail_analysis" and retries_analysis < MAX_RETRIES:
        return "retry_analyze"

    if result == "fail_recommendations" and retries_recommendations < MAX_RETRIES:
        return "retry_recommend"

    if result == "fail_both":
        if retries_analysis < MAX_RETRIES:
            return "retry_analyze"
        if retries_recommendations < MAX_RETRIES:
            return "retry_recommend"

    return "write_briefing"  # Max retries exhausted, proceed anyway
```

This is a **conditional routing function** — LangGraph calls it after the evaluate node to decide where to go next. The logic:

1. **Pass** → proceed to `write_briefing`.
2. **Fail analysis** → route back to `retry_analyze` (which is just the `analyze` function registered under a different node name). The analyze node sees the feedback in state and incorporates it into the retry prompt.
3. **Fail recommendations** → route back to `retry_recommend`.
4. **Fail both** → fix analysis first (since recommendations depend on it), then recommendations.
5. **Max retries exhausted** → proceed anyway with a warning. The pipeline always completes.

**Why fix analysis before recommendations when both fail?** Recommendations depend on analysis. If the analysis is vague, the recommendations will be too — retrying recommendations alone won't help. Fix the foundation first.

---

## Graph Construction (Lines 322-348)

This is where everything comes together.

```python
def build_graph():
    graph = StateGraph(GraphState)
```
Creates a new `StateGraph` parameterized with `GraphState`. This tells LangGraph the shape of the state that will flow through the graph.

```python
    graph.add_node("scan_competitor", scan_competitor)
    graph.add_node("analyze", analyze)
    graph.add_node("recommend", recommend)
    graph.add_node("evaluate", evaluate)
    graph.add_node("retry_analyze", analyze)
    graph.add_node("retry_recommend", recommend)
    graph.add_node("write_briefing", write_briefing)
```
Registers each function as a named node. Notice that `retry_analyze` and `retry_recommend` point to the **same functions** as `analyze` and `recommend` — they're registered under different names so the graph can route to them separately, but the function logic is identical. The functions themselves check the state to know if they're on a retry (by reading `evaluation_result` and `evaluation_feedback`).

```python
    graph.set_conditional_entry_point(fan_out, ["scan_competitor"])
```
This is the entry point of the graph. `set_conditional_entry_point` calls `fan_out` to dynamically decide where to go. `fan_out` returns `Send` objects that route to `scan_competitor`. The `["scan_competitor"]` argument tells LangGraph which nodes could potentially be targeted (for validation).

```python
    graph.add_edge("scan_competitor", "analyze")
    graph.add_edge("analyze", "recommend")
    graph.add_edge("recommend", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluation, {
        "write_briefing": "write_briefing",
        "retry_analyze": "retry_analyze",
        "retry_recommend": "retry_recommend",
    })
    graph.add_edge("retry_analyze", "recommend")
    graph.add_edge("retry_recommend", "evaluate")
    graph.add_edge("write_briefing", END)
```
The pipeline edges, now including the evaluator loop:
- `scan_competitor` → `analyze` → `recommend` → `evaluate` (the main path)
- `evaluate` uses **conditional edges** — `route_after_evaluation` decides the next node based on the verdict
- `retry_analyze` → `recommend` (re-run recommendations on the improved analysis, then back to evaluate)
- `retry_recommend` → `evaluate` (re-evaluate the new recommendations)
- `write_briefing` → `END`

This creates a **loop** in the graph: evaluate can route back to retry nodes, which eventually route back to evaluate. The loop terminates when the evaluator passes or retries are exhausted.

```python
    return graph.compile()
```
`.compile()` validates the graph (checks for missing nodes, unreachable edges, etc.) and returns a **runnable** object. Before compilation, it's just a blueprint. After compilation, you can call `.invoke()` on it.

---

## Pipeline Runner (Lines 237-249)

```python
def run_pipeline(company: str, industry: str, competitors: str) -> str:
    graph = build_graph()
    result = graph.invoke({
        "company": company,
        "industry": industry,
        "competitors": competitors,
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "scan_results": [],
        "analysis": "",
        "recommendations": "",
        "briefing": "",
    })
    return result["briefing"]
```
**Lines 237-249:** The public API of this module. Builds the graph, invokes it with initial state, and returns the final briefing.

The initial state dict must include all fields from `GraphState`. The empty values (`[]`, `""`) are placeholders that the nodes will fill in. `scan_results` starts as `[]` because the `operator.add` reducer will append to it.

`graph.invoke()` runs the entire pipeline synchronously — it blocks until all nodes have finished. The return value is the final state dict after all nodes have run. We extract just the `briefing` field since that's all the caller needs.

---

## The Execution Flow

When you call `run_pipeline("Danfoss", "Hydraulics", "Parker, Bosch, Eaton")`:

1. `fan_out` splits "Parker, Bosch, Eaton" into 3 `Send` objects
2. Three `scan_competitor` nodes run **in parallel**, each searching the web for one competitor and summarizing with GPT-4o
3. Their `scan_results` lists are **merged** via `operator.add`
4. `analyze` reads all scan results and produces analysis with Claude Sonnet
5. `recommend` reads the analysis and generates recommendations with Claude Sonnet
6. `evaluate` checks both deliverables against rubrics with Claude Sonnet
7. If evaluation passes → `write_briefing` combines analysis + recommendations into a final report with GPT-4o-mini
8. If evaluation fails → the failing node is retried with feedback (up to 2 times), then back to step 6
9. The briefing is saved to disk and returned

---

## The Annual Report Pipeline (Lines 370-477)

This is a separate pipeline for generating deep-dive competitor intelligence reports. It uses the same fan-out pattern as the main pipeline but with a different structure: there's no sequential analysis/recommendation chain — each competitor gets a comprehensive 15-section report produced in parallel.

### Annual Report State

```python
class AnnualReportState(TypedDict):
    company: str
    industry: str
    competitors: str
    current_date: str
    competitor: str
    report_results: Annotated[list[str], operator.add]
```

Simpler than `GraphState` — no analysis, recommendations, or briefing fields. Just the input fields plus `report_results` with the same `operator.add` reducer for collecting parallel results.

### The scan_annual_report Node

```python
def scan_annual_report(state: AnnualReportState) -> dict:
```

This is the workhorse node. It does three things:

1. **Web search**: Runs 15 targeted Serper queries (vs 3 in the main pipeline) covering official website, SEC filings, annual reports, LinkedIn, customer reviews, market share, M&A, patents, and more. The queries use both the current year and previous year to catch the most recent data available.

2. **LLM synthesis**: Feeds all search results to Claude Sonnet with a detailed task description requiring 15 specific sections (Company Overview, Product Portfolio, Pricing, Customers, Go-to-Market, R&D, Financials, Team, Customer Sentiment, Market Position, M&A, Geographic Presence, Patents, Regulatory Risks, Strategic Assessment).

3. **Inline evaluation + retry**: After generating the report, the node evaluates it against a quality rubric and retries up to 2 times with feedback if it fails.

### Why Inline Evaluation (Not a Separate Graph Node)

This is the key architectural difference from the main pipeline. The annual report pipeline uses `Send()` fan-out — all parallel `scan_annual_report` branches converge after that single node. If you added a graph-level evaluate node, it would run **once after all scans complete**, not per-competitor. You'd lose the ability to retry individual competitors.

So instead, the evaluation loop lives inside the node function:

```python
    # Inline evaluation + retry loop
    for attempt in range(MAX_RETRIES + 1):
        # Evaluate the report with Claude
        eval_response = _claude().invoke([...])

        # Parse JSON verdict (fail-open on parse errors)
        if result == "pass":
            break

        if attempt < MAX_RETRIES:
            # Retry: re-invoke the LLM with same search context + feedback
            retry_msg = user_msg + f"\n\nFEEDBACK: {feedback}\nRevise..."
            response = llm.invoke([...])
            report_text = response.content
```

Key details:
- **Web searches are NOT re-run on retry** — only the LLM synthesis is retried with the same search context plus the evaluator's feedback. Web results don't change between retries, and re-fetching them wastes time and API credits.
- **Fail-open**: JSON parse errors default to `"pass"`.
- **Same retry cap (2)** as the main pipeline — consistency across pipelines.
- **Each parallel branch retries independently** — if Competitor A's report fails but Competitor B's passes, only A retries.

### Annual Report Graph Construction

```python
def build_annual_report_graph():
    graph = StateGraph(AnnualReportState)
    graph.add_node("scan_annual_report", scan_annual_report)
    graph.set_conditional_entry_point(fan_out_annual_reports, ["scan_annual_report"])
    graph.add_edge("scan_annual_report", END)
    return graph.compile()
```

Much simpler than the main pipeline — just fan-out and done. All the complexity (evaluation, retries) is encapsulated inside the node function rather than in the graph topology.
