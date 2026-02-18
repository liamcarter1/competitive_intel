# graph.py — Line-by-Line Explanation

This is the **brain of the entire application**. It defines two LangGraph pipelines — a Latest News Monitor and an Annual Report Deep Dive — that orchestrate web searches and LLM calls in structured workflows. The news monitor fans out across competitors to gather raw search results, then compiles them into a formatted digest. If you want to understand LangGraph deeply, this is the file to study.

---

## Imports (Lines 1-18)

```python
from __future__ import annotations
```
**Line 1:** This is a Python compatibility trick. It makes all type annotations in this file behave as strings (deferred evaluation). Without this, `Annotated[list[str], operator.add]` on line 143 would fail in Python 3.9 because `list[str]` as a type hint wasn't supported until 3.10. With this import, Python doesn't try to evaluate the annotation at runtime — it just stores it as a string. This is a good habit for any code that uses modern type hints but needs to support slightly older Python versions.

```python
import json
import operator
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, TypedDict
```
**Lines 3-10:** Standard library imports.
- `json`: Used for serializing search results and parsing disambiguation LLM output.
- `operator`: This is the key import for LangGraph's **reducer pattern**. Specifically, `operator.add` is used to tell LangGraph how to merge results from parallel nodes (more on this at the state definition).
- `os`: For environment variable access.
- `re`: Regular expressions, used by the news date parser to match patterns like "3 days ago".
- `ThreadPoolExecutor, as_completed`: From `concurrent.futures` — used to run Serper search requests in parallel within a single scan node. This is *intra-node* parallelism (multiple HTTP requests at once), distinct from LangGraph's *inter-node* parallelism (multiple scan nodes at once). Without this, 42 search queries per competitor would run sequentially, taking far too long.
- `datetime, timedelta`: To stamp reports with the current date and to calculate age of news results for freshness filtering.
- `Path`: Pythonic file path handling (better than string concatenation like `"config/" + "agents.yaml"`).
- `Annotated, TypedDict`: Typing constructs. `TypedDict` defines the shape of state that flows through the graph. `Annotated` attaches metadata (the reducer function) to a type.

```python
import yaml
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.types import Send
```
**Lines 12-16:** Third-party imports. This is where the LangChain/LangGraph ecosystem comes in.
- `yaml`: Parses the YAML config files that define agent prompts.
- `ChatAnthropic`: LangChain's wrapper around the Anthropic API. It gives you a unified `.invoke()` interface so you can swap between providers without changing your code.
- `ChatOpenAI`: Same thing but for OpenAI's API.
- `END`: A special constant in LangGraph that means "the graph is done, stop executing." Think of it as the exit door.
- `StateGraph`: The core LangGraph class. It's a directed graph where each node is a function and edges define execution order. State flows through the graph and gets updated by each node.
- `Send`: This is LangGraph's mechanism for **fan-out** (parallelism). Instead of a node returning just state updates, it can return `Send` objects that say "run this other node with this specific input." This is how we scan multiple competitors in parallel.

```python
from competitive_intel.tools import search_serper, search_serper_news
```
**Line 18:** Imports our custom web search functions from the tools module. `search_serper` calls the standard `/search` endpoint for web results, while `search_serper_news` calls the `/news` endpoint for recent news articles with date filtering. This keeps the search logic separate from the graph logic — good separation of concerns.

---

## Configuration Loading (Lines 20-29)

```python
CONFIG_DIR = Path(__file__).parent / "config"
```
**Line 20:** Builds the path to the `config/` directory. `Path(__file__)` gives us the path to `graph.py` itself, `.parent` goes up one directory (to `src/competitive_intel/`), then `/ "config"` appends the config folder. This is **relative to the code**, not relative to where you run the script from — which is important because scripts can be run from any working directory.

```python
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
```
**Line 21:** Navigates up three levels from `graph.py` to reach the `competitive_intel/` project root, then into `output/`. The `.resolve()` call converts the path to an absolute path (resolving any symlinks), which prevents weird path issues. The three `.parent` calls go: `competitive_intel/` (from `src/competitive_intel/`) -> `src/` -> `competitive_intel/` (project root).

```python
def _load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
```
**Lines 24-25:** A helper that reads a YAML file and parses it into a Python dictionary. `yaml.safe_load` is important — never use `yaml.load` without a Loader argument because it can execute arbitrary Python code embedded in the YAML (a security vulnerability). `safe_load` only allows basic data types.

```python
AGENTS_CONFIG = _load_yaml("agents.yaml")
TASKS_CONFIG = _load_yaml("tasks.yaml")
```
**Lines 28-29:** Load both config files at **module import time** (when Python first imports `graph.py`). This means the YAML is parsed once and cached in memory as dictionaries. Every node function just reads from these dictionaries rather than re-parsing the files. This is a performance optimization — YAML parsing isn't free, and these configs don't change at runtime.

---

## Constants and Date Parsing Helpers (Lines 31-71)

```python
NEWS_MAX_AGE_DAYS = 45
_SEARCH_WORKERS = 12
_DISAMBIG_CACHE: dict[tuple, dict] = {}
```
**Lines 32-39:** Module-level constants and caches.
- `NEWS_MAX_AGE_DAYS`: The global default for how old a news result can be before it gets discarded (45 days). Individual pipelines can override this — the news monitor uses its own `max_age_days` derived from the selected time window.
- `_SEARCH_WORKERS`: Maximum concurrent threads for Serper HTTP requests within a single scan node. Set to 12 so each competitor's 42 queries don't serialize into 42 sequential HTTP calls (which would take minutes).
- `_DISAMBIG_CACHE`: An in-memory cache that prevents re-disambiguating the same competitor if both pipelines run in the same process. The key is a `(competitor, industry, company)` tuple.

```python
_RELATIVE_RE = re.compile(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", re.I)

def _parse_serper_date(date_str: str) -> datetime | None:
    ...

def _is_recent_news(date_str: str, max_age_days: int = NEWS_MAX_AGE_DAYS) -> bool:
    ...
```
**Lines 41-70:** Date parsing utilities for news freshness filtering.
- `_RELATIVE_RE`: A compiled regex that matches Serper's relative date strings like "3 days ago", "1 hour ago", "2 months ago". The `re.I` flag makes it case-insensitive. Compiling the regex once (at module load) is faster than recompiling on every call.
- `_parse_serper_date()`: Parses both relative dates ("3 days ago" → `datetime.now() - 3 days`) and absolute dates ("Jan 15, 2026" → a `datetime` object). Tries five different date formats. Returns `None` if nothing works.
- `_is_recent_news()`: The gatekeeper function. Returns `True` if the date is within `max_age_days` or if the date can't be parsed (benefit of the doubt). Accepts `max_age_days` as a parameter so callers can override the global default — this is how the news monitor passes in the time-window-specific age limit.

---

## Disambiguation Helper (Lines 73-133)

```python
def _disambiguate_competitor(competitor: str, industry: str, company: str) -> dict:
```

This function calls gpt-4o-mini to generate a search-friendly name, Google exclusion operators, and a one-sentence identity description for a competitor. It returns `{"search_name": str, "exclude_terms": str, "context": str}`. If the LLM call fails for any reason, it falls back to the bare competitor name with no exclusions.

**Why LLM disambiguation?** Company names are frequently ambiguous. Searching for "ATOS" returns results about Atos SE (a French IT company), not ATOS SpA (an Italian hydraulic valve manufacturer). The disambiguation call turns "ATOS" into "ATOS SpA hydraulic valves" with exclusion operators like `-"Atos SE" -"Eviden"`, dramatically improving result quality. At ~$0.001 per call, it's a cheap investment.

Key implementation details:
- **Caching** (lines 80-83): Results are stored in `_DISAMBIG_CACHE` so the same competitor isn't re-disambiguated if both the news monitor and annual report pipelines run in the same process.
- **Normalize list output** (lines 120-122): The LLM sometimes returns `exclude_terms` as a list instead of a string. The code handles both by joining lists with spaces.
- **Fail-open** (lines 130-132): Any exception (LLM timeout, JSON parse error, API key missing) falls back to `{"search_name": competitor, "exclude_terms": "", "context": ""}`. The pipeline continues with less precise searches rather than crashing.

---

## State Definition (Lines 135-168)

This is one of the most important concepts in LangGraph.

```python
class NewsMonitorState(TypedDict):
    company: str
    industry: str
    competitors: str
    current_date: str
    time_window: str                                    # "past_week" | "past_2_weeks" | "past_month"
    news_results: Annotated[list[str], operator.add]    # raw tagged results per competitor
    digest: str                                         # final formatted output
```

**Lines 137-144:** `NewsMonitorState` defines **everything the news monitor graph knows**. Every node receives this state and can read from or write to it. Think of it as a shared whiteboard that all the agents can see.

- `company`, `industry`, `competitors`, `current_date` (lines 138-141): **Input fields.** These are set once at the start and never changed. Every node reads them to know what it's working on.
- `time_window` (line 142): **The recency selector.** A string key — `"past_week"`, `"past_2_weeks"`, or `"past_month"` — that controls how far back the news search looks. This flows through state so the `scan_news` node can configure its Serper `tbs` parameter and date filter accordingly. The UI presents this as a dropdown; the default is `"past_2_weeks"`.
- `news_results` (line 143): **This is the key field.** The `Annotated[list[str], operator.add]` syntax is LangGraph's **reducer pattern**. Here's the problem it solves: when multiple scan nodes run in parallel (one per competitor), they each return a `news_results` list. How should LangGraph combine them? Without a reducer, the last one to finish would overwrite the others. The `operator.add` reducer tells LangGraph: "concatenate all the lists together." So if scan_A returns `["## Parker\n\n..."]` and scan_B returns `["## Bosch\n\n..."]`, the final state has `news_results = ["## Parker\n\n...", "## Bosch\n\n..."]`. This is the core mechanism that makes fan-out/fan-in work.
- `digest` (line 144): **The final output.** The `compile_digest` node writes the formatted news digest here. It's a regular string — no reducer needed because only one node writes to it.

**Why TypedDict and not a regular class?** LangGraph requires state to be a TypedDict (or a Pydantic BaseModel). TypedDict is essentially a type-annotated dictionary — it gives you IDE autocomplete and type checking while keeping the simplicity of dict access (`state["company"]`). LangGraph inspects these annotations at graph compile time to know which fields exist and which have reducers.

**Contrast with the old briefing pipeline:** The old `GraphState` had fields for `scan_results`, `raw_search_results`, `analysis`, `recommendations`, `briefing`, `evaluation_result`, `evaluation_feedback`, and retry counters. The news monitor is dramatically simpler — there's no analysis chain, no evaluator loop, no retry counters. Raw search results flow straight from `scan_news` to `compile_digest`, which formats them in a single LLM call. Fewer moving parts means fewer failure modes.

```python
class NewsScanState(TypedDict):
    company: str
    industry: str
    competitors: str
    current_date: str
    time_window: str
    competitor: str
    news_results: Annotated[list[str], operator.add]
    digest: str
```

**Lines 147-155:** `NewsScanState` extends `NewsMonitorState` with one extra field: `competitor` (singular, line 153). When `fan_out_news` sends work to `scan_news`, it adds a `competitor` field specifying which single competitor this particular scan should focus on. LangGraph needs to know the full state shape for type checking, so `NewsScanState` includes everything from `NewsMonitorState` plus the extra field.

```python
_TIME_WINDOW_PARAMS = {
    "past_week":    {"tbs": "qdr:w",  "max_age_days": 7,  "label": "Past week"},
    "past_2_weeks": {"tbs": "qdr:w2", "max_age_days": 14, "label": "Past 2 weeks"},
    "past_month":   {"tbs": "qdr:m",  "max_age_days": 30, "label": "Past month"},
}

def _time_window_params(time_window: str) -> dict:
    """Return tbs, max_age_days, label for a time_window key."""
    return _TIME_WINDOW_PARAMS.get(time_window, _TIME_WINDOW_PARAMS["past_2_weeks"])
```

**Lines 158-167:** The **time window configuration system**. This is how the news monitor supports configurable recency.

- `_TIME_WINDOW_PARAMS`: A lookup dict mapping user-friendly keys to search parameters. Each entry has three values:
  - `tbs`: The Google time-based search parameter that Serper forwards. `"qdr:w"` means "past week", `"qdr:w2"` means "past 2 weeks", `"qdr:m"` means "past month". This is sent to Serper's `/news` endpoint.
  - `max_age_days`: The code-side date filter cutoff. This is passed to `_is_recent_news()` as a backup, because Serper's `tbs` parameter isn't always reliable — some old results slip through.
  - `label`: A human-readable string (e.g., "Past 2 weeks") used in the digest prompt so the LLM knows what time period it's curating for.
- `_time_window_params()`: A safe accessor that defaults to `"past_2_weeks"` if an unknown key is passed. Defensive coding — the UI should only send valid keys, but a default prevents crashes.

**Why both `tbs` and `max_age_days`?** Belt and suspenders. The Serper `tbs` parameter tells Google to filter server-side, but it's not 100% reliable — some old articles sneak through, especially from smaller publications. The `max_age_days` check in `_is_recent_news()` catches anything the server-side filter misses. Neither alone is sufficient; together they're robust.

---

## LLM Client Factories (Lines 170-177)

```python
def _openai(model: str = "gpt-4o", temperature: float = 0.2):
    return ChatOpenAI(model=model, temperature=temperature)
```
**Lines 172-173:** Creates an OpenAI LLM client. `ChatOpenAI` is LangChain's wrapper — it reads `OPENAI_API_KEY` from the environment automatically (you never pass the key directly). The `temperature=0.2` means the model is fairly deterministic — lower temperature = less creative/random, higher = more varied. For competitive intelligence, you want consistency, not creativity.

**Why a factory function instead of a global client?** Each node might want a different model or temperature. The factory pattern (`_openai("gpt-4o-mini")`) is cleaner than having many global variables.

```python
def _claude(model: str = "claude-sonnet-4-20250514", temperature: float = 0.2):
    return ChatAnthropic(model=model, temperature=temperature)
```
**Lines 176-177:** Same pattern for Anthropic's Claude. Notice how `ChatOpenAI` and `ChatAnthropic` have the **exact same interface** — both have `.invoke()` that takes a list of messages. This is the power of LangChain's abstraction: you can swap providers by changing one line of code. The graph nodes don't care which provider they're using.

---

## Prompt Builders (Lines 180-194)

```python
def _agent_system_prompt(agent_key: str, inputs: dict) -> str:
    cfg = AGENTS_CONFIG[agent_key]
    role = cfg["role"].format(**inputs).strip()
    goal = cfg["goal"].format(**inputs).strip()
    backstory = cfg["backstory"].format(**inputs).strip()
    return f"Role: {role}\nGoal: {goal}\nBackstory: {backstory}"
```
**Lines 182-187:** Builds a system prompt from the YAML config. Here's what happens step by step:
1. `AGENTS_CONFIG[agent_key]` looks up the agent (e.g., `"news_digest_curator"`) in the loaded YAML.
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
**Lines 190-194:** Same pattern for task prompts. Returns a tuple of (description, expected_output). The description tells the LLM what to do, and the expected_output tells it what format to respond in. Returning a tuple lets the caller unpack it cleanly: `desc, expected = _task_prompt(...)`.

---

## The scan_news Node (Lines 199-315)

This is the workhorse of the news monitor pipeline. It runs 42 search queries for a single competitor and collects all results — but critically, it does **not** call an LLM to summarize them. It just gathers the raw data.

```python
def scan_news(state: NewsScanState) -> dict:
    """Run 42 search queries for a single competitor — no LLM summarization."""
    competitor = state["competitor"]
    year = state["current_date"][:4]
    industry = state["industry"]
    company = state["company"]
```
**Lines 199-204:** The node function extracts the key values from state. The `year` extraction (`[:4]` slices "2026-02-18" to "2026") is used in search queries to anchor results to the current year.

```python
    tw = _time_window_params(state.get("time_window", "past_2_weeks"))
    tbs = tw["tbs"]
    max_age = tw["max_age_days"]
```
**Lines 206-208:** Looks up the time window configuration. The `state.get()` with a default handles the case where `time_window` might not be set (defensive coding). After this, `tbs` holds the Serper parameter (e.g., `"qdr:w2"` for past 2 weeks) and `max_age` holds the code-side cutoff (e.g., 14 days). Both are used later during search execution.

```python
    # LLM-powered disambiguation
    disambig = _disambiguate_competitor(competitor, industry, company)
    sn = disambig["search_name"]
    ex = disambig["exclude_terms"]
```
**Lines 210-213:** Same disambiguation step as in the old briefing scan — calls gpt-4o-mini to get a search-friendly name, exclusion terms, and a one-sentence context description. The `sn` and `ex` variables are used in every search query below. The `context` field is used later when building the raw results block (see the return value section).

```python
    # ── News searches (25 queries) ───────────────────────────────────────────
    news_queries = [
        f"{sn} {industry} news announcement {year} {ex}",
        f"{sn} {industry} product launch release update {year} {ex}",
        f"{sn} {industry} acquisition merger partnership deal {year} {ex}",
        f"{sn} {industry} pricing changes new model tier {year} {ex}",
        f"{sn} {industry} customer win contract award {year} {ex}",
        f"{sn} {industry} executive appointment leadership hire CEO {year} {ex}",
        f"{sn} {industry} earnings revenue financial results {year} {ex}",
        f"{sn} {industry} regulatory lawsuit patent filing {year} {ex}",
        f"{sn} {industry} stock analyst upgrade downgrade guidance {year} {ex}",
        f'{sn} "fluid power" OR "hydraulic" product launch news {year} {ex}',
        f'{sn} IFPE OR bauma OR ConExpo OR "Hannover Messe" {year} {ex}',
        f'{sn} electrification OR electrohydraulic OR "electric actuator" OR "digital hydraulic" {year} {ex}',
        f"{sn} {industry} distributor dealer channel OEM supply {year} {ex}",
        f"{sn} {industry} press release announcement new {year} {ex}",
        f"{sn} {industry} factory plant expansion investment manufacturing {year} {ex}",
        f"{sn} {industry} supply chain disruption shortage logistics {year} {ex}",
        f"{sn} {industry} sustainability ESG carbon emissions environmental {year} {ex}",
        f"{sn} {industry} safety recall defect incident OSHA {year} {ex}",
        f"{sn} {industry} award recognition innovation winner {year} {ex}",
        f"{sn} {industry} R&D technology innovation research development {year} {ex}",
        f"{sn} {industry} layoff restructuring cost cutting workforce reduction {year} {ex}",
        f"{sn} {industry} government contract military defense infrastructure {year} {ex}",
        f'{sn} "industrial automation" OR "Industry 4.0" OR "smart manufacturing" {year} {ex}',
        f'{sn} "construction equipment" OR "mobile machinery" OR "off-highway" {year} {ex}',
        f"{sn} {industry} tariff trade regulation import duty {year} {ex}",
    ]
```
**Lines 216-243:** The news queries — **25 queries** hitting Serper's `/news` endpoint. This is a significant expansion from the old briefing scan's 15 news queries. The first 9 are general-purpose (news/announcements, product launches, M&A/partnerships, pricing, customer wins, leadership changes, earnings, regulatory/legal, analyst coverage). The remaining 16 cover increasingly specific intelligence categories:

- **Trade press & fluid power** (lines 227-229): Queries with hydraulic terminology, trade show names (IFPE, bauma, ConExpo, Hannover Messe), and electrification/electrohydraulic technology keywords.
- **Channel & supply chain** (lines 232-233): Distributor/dealer/OEM moves and press releases.
- **Operations** (lines 234-235): Factory expansion, capex, and supply chain disruptions.
- **ESG & safety** (lines 236-237): Sustainability initiatives and safety recalls.
- **Innovation & workforce** (lines 238-240): Awards, R&D, and layoff/restructuring news.
- **Government & end-market** (lines 241-243): Government contracts, industrial automation, construction/off-highway equipment, and tariff/trade regulation.

```python
    # ── Web searches (17 queries) ────────────────────────────────────────────
    web_queries = [
        f"{sn} {industry} strategy expansion growth plans {year} {ex}",
        f"{sn} {industry} new product features roadmap {year} {ex}",
        f"{sn} {industry} hiring jobs open roles site:linkedin.com OR site:indeed.com {year} {ex}",
        f"{sn} {industry} patent USPTO OR Espacenet {year} {ex}",
        f"{sn} site:hydraulicspneumatics.com OR site:fluidpowerworld.com OR site:fluidpowerjournal.com",
        f"{sn} site:mobilehydraulictips.com OR site:powermotiontech.com OR site:oemoffhighway.com",
        f"{sn} site:ifpe.com OR site:fluidpowernet.com OR site:dieselprogress.com",
        f"{sn} site:automationworld.com OR site:controleng.com OR site:plantengineering.com",
        f"{sn} site:equipmentworld.com OR site:forconstructionpros.com OR site:constructionequipment.com",
        f"{sn} {industry} site:prnewswire.com OR site:businesswire.com OR site:globenewswire.com {year}",
        f"{sn} site:sec.gov 10-K OR 10-Q OR 8-K {year}",
        f"{sn} {industry} site:reuters.com OR site:bloomberg.com OR site:ft.com {year}",
        f"{sn} {industry} site:glassdoor.com OR site:ziprecruiter.com {year}",
        f"{sn} site:patents.google.com {year}",
        f"{sn} {industry} site:linkedin.com/posts OR site:linkedin.com/pulse {year}",
        f"{sn} site:nfpa.com OR site:fpda.org OR site:nahad.org",
        f"{sn} site:machinedesign.com OR site:designworldonline.com OR site:theengineer.co.uk",
    ]
```
**Lines 245-264:** The web queries — **17 queries** hitting Serper's standard `/search` endpoint. Also a big expansion from the old briefing scan's 8 web queries. The first 4 are general-purpose (strategy, product roadmaps, job postings, patents). The rest are **site-targeted**, ensuring coverage of specific source categories:

- **Hydraulics trade publications** (lines 251-253): Three queries covering 9 specialist publications by domain — Hydraulics & Pneumatics, Fluid Power World, Fluid Power Journal, Mobile Hydraulic Tips, Power & Motion, OEM Off-Highway, IFPE, Fluid Power Net, Diesel Progress.
- **Industrial automation & controls** (line 254): Automation World, Control Engineering, Plant Engineering.
- **Construction & equipment** (line 255): Equipment World, For Construction Pros, Construction Equipment.
- **PR wire services** (line 256): PRNewswire, BusinessWire, GlobeNewsWire.
- **Financial filings & tier-1 business press** (lines 257-258): SEC.gov, Reuters, Bloomberg, Financial Times.
- **Employment signals** (lines 259-261): Glassdoor, ZipRecruiter, Google Patents, LinkedIn posts.
- **Industry associations** (line 262): NFPA, FPDA, NAHAD.
- **General engineering publications** (line 263): Machine Design, Design World, The Engineer.

**Why 42 queries?** The news monitor is designed to be comprehensive. Unlike the old briefing scan, which had an LLM summarize the results, the news monitor passes raw results directly to a formatting step. More queries means more coverage of niche intelligence categories. The cost is purely in Serper API calls (no per-query LLM cost for summarization), so the marginal cost of additional queries is low.

**Why two endpoints?** The regular `/search` endpoint returns a mix of evergreen web content (Wikipedia, company "About" pages) and news, with evergreen often ranking higher. For a competitive intelligence tool, recency is everything — the `/news` endpoint cuts through the noise and surfaces breaking developments. The web queries complement this with signals that don't appear as news articles (job postings, SEC filings, patent databases, trade association pages).

**Why hardcode the queries instead of letting the LLM generate them?** Speed and reliability. Having the LLM generate queries would require an extra API call (adding latency and cost), and the LLM might generate vague or unhelpful queries. These templates cover the key intelligence categories comprehensively. The one exception is the disambiguation call, which *does* use an LLM — but it runs once per competitor (not per query) and addresses a problem that static templates can't solve.

```python
    all_results = []
    skipped_old = 0

    def _fetch_news(q):
        results, skipped = [], 0
        try:
            data = search_serper_news(q, num_results=10, tbs=tbs)
            for item in data.get("news", [])[:8]:
                date = item.get("date", "")
                if not _is_recent_news(date, max_age_days=max_age):
                    skipped += 1
                    continue
                date_str = f" ({date})" if date else ""
                results.append(
                    f"- [NEWS{date_str}] [{item.get('title', '')}]({item.get('link', '')}): {item.get('snippet', '')}"
                )
        except Exception as e:
            results.append(f"- News search error for '{q}': {e}")
        return results, skipped

    def _fetch_web(q):
        results = []
        try:
            data = search_serper(q)
            for item in data.get("organic", [])[:8]:
                results.append(
                    f"- [WEB] [{item.get('title', '')}]({item.get('link', '')}): {item.get('snippet', '')}"
                )
        except Exception as e:
            results.append(f"- Search error for '{q}': {e}")
        return results
```
**Lines 266-296:** Two inner functions that handle individual search queries. They're defined inside `scan_news` so they can be dispatched to the thread pool.

- `_fetch_news()`: Calls `search_serper_news()` with the time-window-specific `tbs` parameter, takes up to 8 results per query, and applies `_is_recent_news()` with the time-window-specific `max_age_days`. Results are tagged `[NEWS (date)]` with the publication date. Returns a tuple of `(results_list, skipped_count)`.
- `_fetch_web()`: Calls `search_serper()` for standard web results, takes up to 8 per query. Results are tagged `[WEB]`. Returns just the results list (no date filtering needed for web results).
- Both functions catch exceptions per-query — one failed search doesn't crash the entire scan.

```python
    with ThreadPoolExecutor(max_workers=_SEARCH_WORKERS) as executor:
        news_futures = {executor.submit(_fetch_news, q): q for q in news_queries}
        web_futures = {executor.submit(_fetch_web, q): q for q in web_queries}

        for future in as_completed(news_futures):
            results, skipped = future.result()
            all_results.extend(results)
            skipped_old += skipped

        for future in as_completed(web_futures):
            all_results.extend(future.result())
```
**Lines 298-308:** The **concurrent execution** block. All 42 queries are submitted to a thread pool with up to 12 workers. `as_completed()` processes results as they arrive, rather than waiting for all to finish in order. This means the first few results appear quickly and the thread pool stays busy.

Without threading, 42 sequential HTTP requests at ~0.5-1s each would take 21-42 seconds per competitor. With 12 threads, the wall-clock time drops to roughly 4-7 seconds. This is intra-node parallelism, separate from LangGraph's inter-node parallelism that runs scan nodes for different competitors simultaneously.

```python
    if skipped_old:
        print(f"[scan_news] {competitor}: filtered out {skipped_old} results older than {max_age} days")

    print(f"[scan_news] Finished {competitor} — {len(all_results)} results from {len(news_queries)} news + {len(web_queries)} web queries")
    ctx = disambig["context"]
    header = f"## {competitor}\n\n> Disambiguation: {ctx}\n" if ctx else f"## {competitor}\n"
    raw_block = header + "\n" + "\n".join(all_results) if all_results else f"## {competitor}\n\nNo search results found."
    return {"news_results": [raw_block]}
```
**Lines 310-317:** Diagnostics and return value.
- The skip count is logged for monitoring (helps detect when the time window filter is too aggressive).
- The result count is logged so you can see how many results survived filtering.
- The disambiguation context sentence is included as a blockquote header (e.g., `> Disambiguation: Sun Hydraulics (now Helios Technologies) is a Florida-based manufacturer of hydraulic cartridge valves and manifolds`). This is critical for the `compile_digest` node — it gives GPT-4o-mini per-competitor identity information to confidently discard off-topic results that slipped through the search queries. Without it, generic company names like "SUN" can produce noise (e.g., bike rack articles) that the LLM has no basis to filter.
- All results are joined into a single markdown block under the heading + context header.
- The return value is `{"news_results": [raw_block]}` — a list with one string. The `operator.add` reducer on `news_results` will concatenate these lists across parallel scan nodes.

**The critical difference from the old `scan_competitor`:** This node does **not** call an LLM. The old node fed all results into GPT-4o for summarization, which cost time and money and also lost the original URLs (the LLM would paraphrase and sometimes hallucinate links). `scan_news` just collects raw results with their real URLs, titles, and snippets. The single LLM call happens later in `compile_digest`, operating on all competitors' results at once rather than per-competitor.

Total potential results per competitor: 25 news queries x 8 + 17 web queries x 8 = **up to 336 results**, minus any filtered out by the date check.

---

## The compile_digest Node (Lines 318-343)

This is where the raw search results become a readable news digest. It's a single LLM call that deduplicates, categorizes, and formats all the raw results from all competitors.

```python
def compile_digest(state: NewsMonitorState) -> dict:
    """Single GPT-4o-mini call to deduplicate, categorize, and format the digest."""
    tw = _time_window_params(state.get("time_window", "past_2_weeks"))
    inputs = {
        "company": state["company"],
        "industry": state["industry"],
        "competitors": state["competitors"],
        "current_date": state["current_date"],
        "time_window_label": tw["label"],
        "raw_results": "\n\n---\n\n".join(state.get("news_results", [])),
    }
```
**Lines 318-328:** Builds the inputs dict for prompt interpolation. Two things worth noting:
- `time_window_label` (line 326): The human-readable time window string (e.g., "Past 2 weeks") is passed into the prompt so the LLM knows what recency period it's curating for.
- `raw_results` (line 327): All competitor result blocks are joined with `---` separators. At this point, `state["news_results"]` contains the merged output from all parallel `scan_news` nodes (thanks to the `operator.add` reducer). This is the full dataset the LLM will work with.

```python
    system = _agent_system_prompt("news_digest_curator", inputs)
    desc, expected = _task_prompt("compile_news_digest", inputs)

    llm = _openai("gpt-4o-mini")
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": f"{desc}\n\nExpected output format:\n{expected}"},
    ])
```
**Lines 330-337:** The LLM call. Uses the `news_digest_curator` agent config and `compile_news_digest` task config from the YAML files. The system prompt sets the persona (a news digest curator), and the user message provides the task description with all the raw results embedded.

**Why GPT-4o-mini?** This is fundamentally a formatting and deduplication task. The raw results already contain the intelligence — the LLM just needs to organize them into categories, remove duplicates, and format them into a readable digest. GPT-4o-mini is fast and cheap for this kind of structured formatting work. There's no deep analytical reasoning required (unlike the annual report's Claude Sonnet synthesis).

```python
    digest = response.content
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "news_digest.md").write_text(digest, encoding="utf-8")
    print(f"[compile_digest] Finished digest (gpt-4o-mini) -> output/news_digest.md")
    return {"digest": digest}
```
**Lines 339-343:** Saves the digest to `output/news_digest.md` and returns it in state. The `exist_ok=True` on `mkdir` means "don't error if the directory already exists." The `encoding="utf-8"` is important on Windows where the default encoding might not handle all characters.

**Contrast with the old pipeline:** The old pipeline had four sequential LLM calls after scanning: `analyze` (Claude Sonnet), `recommend` (Claude Sonnet), `evaluate` (Claude Sonnet), and `write_briefing` (GPT-4o-mini). The news monitor collapses all of that into a single `compile_digest` call. This is possible because the news monitor's goal is different — it's producing a curated news digest, not a deep strategic analysis. The raw search results are the product; the LLM is just the formatter.

---

## Fan-Out (Lines 346-348)

```python
def fan_out_news(state: NewsMonitorState) -> list[Send]:
    competitors = [c.strip() for c in state["competitors"].split(",") if c.strip()]
    return [Send("scan_news", {**state, "competitor": c}) for c in competitors]
```

This is a **conditional entry point** — it's the first thing that runs and decides what happens next.

**Line 347:** Splits the comma-separated competitors string into a clean list. `"Parker Hannifin, Bosch Rexroth, ATOS"` becomes `["Parker Hannifin", "Bosch Rexroth", "ATOS"]`. The `if c.strip()` filters out empty strings from trailing commas.

**Line 348:** This is where LangGraph's parallelism magic happens. `Send("scan_news", {**state, "competitor": c})` creates a `Send` object that says: "Run the `scan_news` node with this specific state." The `{**state, "competitor": c}` creates a new dict that has everything from the current state plus the `competitor` field set to one specific competitor.

If there are 3 competitors, this returns 3 `Send` objects, and LangGraph runs 3 `scan_news` instances **in parallel**. When all 3 finish, their `news_results` lists are combined by the `operator.add` reducer, and execution moves to the next node (`compile_digest`).

**This is the fan-out/fan-in pattern**: one node fans out into many parallel tasks, and the results fan back in through the reducer.

---

## Graph Construction (Lines 351-363)

This is where everything comes together — and it's remarkably simple.

```python
def build_news_graph():
    graph = StateGraph(NewsMonitorState)
```
Creates a new `StateGraph` parameterized with `NewsMonitorState`. This tells LangGraph the shape of the state that will flow through the graph.

```python
    graph.add_node("scan_news", scan_news)
    graph.add_node("compile_digest", compile_digest)
```
Registers each function as a named node. Just two nodes — that's it. Compare this to the old briefing pipeline which had seven nodes (`scan_competitor`, `analyze`, `recommend`, `evaluate`, `retry_analyze`, `retry_recommend`, `write_briefing`).

```python
    graph.set_conditional_entry_point(fan_out_news, ["scan_news"])
```
This is the entry point of the graph. `set_conditional_entry_point` calls `fan_out_news` to dynamically decide where to go. `fan_out_news` returns `Send` objects that route to `scan_news`. The `["scan_news"]` argument tells LangGraph which nodes could potentially be targeted (for validation).

```python
    graph.add_edge("scan_news", "compile_digest")
    graph.add_edge("compile_digest", END)
```
The pipeline edges — dead simple:
- `scan_news` -> `compile_digest` -> `END`

When all parallel `scan_news` instances complete and their results merge, `compile_digest` runs once, formats the digest, and the graph ends. No evaluation loop, no retry routing, no conditional edges. The entire graph is a straight line (with fan-out at the start).

```python
    return graph.compile()
```
`.compile()` validates the graph (checks for missing nodes, unreachable edges, etc.) and returns a **runnable** object. Before compilation, it's just a blueprint. After compilation, you can call `.invoke()` on it.

**Why so much simpler than the old graph?** The old briefing pipeline had a multi-stage analysis chain (scan -> analyze -> recommend -> evaluate -> write) with a conditional retry loop. The news monitor eliminates all the intermediate analysis — raw results go straight to formatting. There's no quality gate because the "quality" of a news digest is primarily about completeness of search coverage (handled by the 42 queries) and formatting (handled by the single `compile_digest` LLM call). If the formatting is off, it's cheaper to just re-run than to build an evaluator.

---

## Pipeline Runners — Streaming and Non-Streaming

The module exposes two layers: a **streaming generator** (used by the Gradio UI for real-time progress) and a **blocking wrapper** (used by the CLI).

### The Streaming Generator: `run_news_monitor_stream()`

```python
_NEWS_MONITOR_NODE_LABELS = {
    "scan_news": "Scanned",
    "compile_digest": "News digest compiled",
}

def run_news_monitor_stream(company: str, industry: str, competitors: str,
                            time_window: str = "past_2_weeks"):
    """Generator that yields (type, message) tuples as each graph node completes."""
    graph = build_news_graph()
    inputs = {
        "company": company,
        "industry": industry,
        "competitors": competitors,
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "time_window": time_window,
        "news_results": [],
        "digest": "",
    }
```
**Lines 366-384:** Sets up the graph and initial state. The `time_window` parameter defaults to `"past_2_weeks"` and flows into state so `scan_news` can look it up. All accumulator fields (`news_results`, `digest`) are initialized to empty values.

```python
    final_state = {}
    for chunk in graph.stream(inputs, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            final_state.update(node_output)
            label = _NEWS_MONITOR_NODE_LABELS.get(node_name, node_name)

            if node_name == "scan_news":
                news_results = node_output.get("news_results", [])
                if news_results:
                    first_line = news_results[0].split("\n", 1)[0]
                    comp_name = first_line.lstrip("# ").strip()
                else:
                    comp_name = "unknown"
                yield ("progress", f"  ✓ {label} {comp_name}")

            elif node_name == "compile_digest":
                yield ("progress", f"  ✓ {label} → output/news_digest.md")

            else:
                yield ("progress", f"  ✓ {label}")

    yield ("result", final_state.get("digest", ""))
```
**Lines 386-407:** The streaming loop. The key is `graph.stream(inputs, stream_mode="updates")`. Instead of `graph.invoke()` which blocks until completion, `.stream()` returns a generator that yields a dict chunk **after each graph node completes**. Each chunk is `{node_name: node_output}` — the same data the node returned to LangGraph's state.

The `_NEWS_MONITOR_NODE_LABELS` dict maps internal node names to user-friendly messages. Special handling:
- `scan_news`: Extracts the competitor name from the result (the text starts with `## CompetitorName`) and includes it in the message, e.g., `"  ✓ Scanned Parker Hannifin"`.
- `compile_digest`: Reports the output file path.

Each iteration yields a `("progress", message)` tuple. At the end, it yields `("result", digest_text)`. The Gradio UI iterates over this generator and pushes each progress message to the browser in real time.

### The Blocking Wrapper: `run_news_monitor()`

```python
def run_news_monitor(company: str, industry: str, competitors: str,
                     time_window: str = "past_2_weeks") -> str:
    result = None
    for msg_type, msg in run_news_monitor_stream(company, industry, competitors, time_window):
        if msg_type == "progress":
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode("ascii", errors="replace").decode("ascii"))
        elif msg_type == "result":
            result = msg
    return result
```

**Lines 410-421:** This is what the CLI uses. It consumes the same stream but prints progress to stdout instead of yielding to a UI. Same pipeline, same progress messages, different output target. The `UnicodeEncodeError` handling is a Windows-specific safety net — some terminal encodings can't handle certain Unicode characters (like the checkmark in the progress messages), so it falls back to ASCII with replacement characters.

---

## The Execution Flow

When you call `run_news_monitor_stream("Danfoss", "Hydraulics", "Parker, Bosch, ATOS")`:

1. `fan_out_news` splits "Parker, Bosch, ATOS" into 3 `Send` objects
2. Three `scan_news` nodes run **in parallel**, each running 42 search queries (25 news + 17 web) with up to 12 concurrent threads per node — no LLM summarization
3. As each scan completes, the stream yields a progress message like `"✓ Scanned Parker Hannifin"`
4. Their `news_results` lists are **merged** via `operator.add`
5. `compile_digest` receives all raw results and produces a formatted, deduplicated digest with a single GPT-4o-mini call -> yields `"✓ News digest compiled → output/news_digest.md"`
6. The digest is saved to disk and yielded as the final `("result", ...)` message

That's it — two stages, one LLM call (plus one disambiguation call per competitor). Compared to the old briefing pipeline's 6+ LLM calls (disambiguation, scan summary, analysis, recommendations, evaluation, write briefing — with potential retries), the news monitor is dramatically faster and cheaper.

---

## The Annual Report Pipeline (Lines 424-640)

This is a separate pipeline for generating deep-dive competitor intelligence reports. It uses the same fan-out pattern as the news monitor but with a different structure: each competitor gets a comprehensive 15-section report produced in parallel, with inline evaluation and retry.

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

Simpler than `NewsMonitorState` — no time window or digest fields. Just the input fields plus `report_results` with the same `operator.add` reducer for collecting parallel results.

### The scan_annual_report Node

```python
def scan_annual_report(state: AnnualReportState) -> dict:
```

This is the workhorse node. It does four things:

1. **LLM disambiguation**: Calls `_disambiguate_competitor()` (same as the news monitor scan) to get a search-friendly name, Google exclusion operators, and a context sentence. This is essential for the annual report too — without it, 18 searches for an ambiguous name like "ATOS" would return mostly wrong-company results.

2. **Web search**: Runs 18 targeted Serper queries (using the disambiguated name + exclusion terms) covering official website, SEC filings, annual reports, LinkedIn, customer reviews, market share, M&A, patents, and more. The queries use both the current year and previous year to catch the most recent data available.

3. **LLM synthesis**: Feeds all search results to Claude Sonnet with a detailed task description requiring 15 specific sections (Company Overview, Product Portfolio, Pricing, Customers, Go-to-Market, R&D, Financials, Team, Customer Sentiment, Market Position, M&A, Geographic Presence, Patents, Regulatory Risks, Strategic Assessment). The disambiguation context sentence is included in the prompt.

4. **Inline evaluation + retry**: After generating the report, the node evaluates it against a quality rubric and retries up to 2 times with feedback if it fails.

### Why Inline Evaluation (Not a Separate Graph Node)

This is the key architectural difference from the news monitor. The annual report pipeline uses `Send()` fan-out — all parallel `scan_annual_report` branches converge after that single node. If you added a graph-level evaluate node, it would run **once after all scans complete**, not per-competitor. You'd lose the ability to retry individual competitors.

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
- **Same retry cap (2)** as `MAX_RETRIES` — consistency across pipelines.
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

Much simpler than even the news monitor — just fan-out and done. All the complexity (evaluation, retries) is encapsulated inside the node function rather than in the graph topology.

### Annual Report Streaming: `run_annual_report_pipeline_stream()`

Same pattern as the news monitor — uses `graph.stream(stream_mode="updates")` and yields `("progress", msg)` tuples as each `scan_annual_report` node completes. Each message includes the competitor name extracted from the report text (e.g., `"✓ Finished deep dive — Parker Hannifin"`).

One important detail: the streaming function must manually accumulate `report_results` across parallel nodes. With `graph.invoke()`, LangGraph handles the `operator.add` reducer automatically. But with `graph.stream()`, each chunk contains only the individual node's output — there's no automatic merging. So the function maintains its own `all_report_results` list and `extend()`s it with each chunk.

After all nodes complete, the function assembles the combined report (header + all competitor reports), saves it to disk, and yields the final `("result", combined_text)`.

The blocking `run_annual_report_pipeline()` wrapper works identically to the news monitor one — prints progress, returns the final result.

**Note on inline evaluation visibility**: Because the annual report evaluation + retry loop happens *inside* `scan_annual_report()` (not as separate graph nodes), those steps don't produce separate stream events. The stream only yields once per competitor — when the entire node finishes (including any retries). This is acceptable for v1; surfacing intra-node progress would require LangGraph's `stream_mode="custom"` with `get_stream_writer()`.
