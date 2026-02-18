# FORliam.md — Competitive Intelligence Monitor

## What This Project Actually Does

Imagine you're a strategy VP at a tech company. Every week, you need to know what your competitors are up to — who shipped what, who hired whom, who's pivoting their pricing. Normally, this means a junior analyst spending days reading press releases, trawling LinkedIn, and assembling a slide deck that's already stale by Friday.

This project automates the *news gathering* part of that workflow. You pick competitors from a predefined list (or type in ad-hoc ones), choose a time window (past week, 2 weeks, or month), and hit go. A pipeline fans out across the web, blasting 42 search queries per competitor in parallel — 25 news searches and 17 web searches — collecting every relevant result with real titles, URLs, dates, and snippets. No LLM touches the raw search results. Then a single GPT-4o-mini call deduplicates, categorizes, and formats everything into a clean news digest with source links. The whole thing runs in about 60-90 seconds and costs roughly $0.10.

There's also a chat interface bolted on top: you can ask questions about the digest (grounded strictly in what it says) or hit "Research This" to trigger a live web search with a fully cited deep-dive answer.

---

## The Architecture — And Why It Looks Like This

### The Pipeline: Think News Wire, Not Committee

The core of the system is a **graph-based pipeline** built with LangGraph. The news monitor pipeline is intentionally simple — two stages, no loops:

```
                         ┌─ scan_news(Competitor A) ─┐
User Input ──→ fan_out_news ─→ scan_news(Competitor B) ─→ fan_in ──→ compile_digest ──→ done
                         └─ scan_news(Competitor C) ─┘
```

1. **Fan-out**: The system splits into parallel tracks — one per competitor. Each track fires off 42 search queries (25 news + 17 web) via the Serper API and collects every result as a raw tagged line item — title, URL, date, snippet. **No LLM is involved.** If you have 4 competitors, 4 scans run simultaneously. This is like sending 4 scouts out in different directions, each one just taking photographs instead of writing a report.

2. **Fan-in**: All the raw results land back in one place — a shared `NewsMonitorState` dictionary.

3. **Compile digest**: A single GPT-4o-mini call reads *all* the raw results, deduplicates them, categorizes them by topic, and formats them into a clean markdown digest with real source links. This is the only LLM call in the entire pipeline (aside from the cheap disambiguation calls).

That's it. No analysis node, no recommendations node, no evaluator, no retry loop. The old briefing pipeline had five sequential stages with a quality gate that could loop — this one has two stages and always runs forward. The result: ~60-90 seconds instead of 3-5 minutes, ~$0.10 instead of ~$1.00+.

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
│       └── __init__.py ← search_serper() + search_serper_news() functions
```

The separation between `agents.yaml`/`tasks.yaml` and `graph.py` is deliberate. The YAML files are like job descriptions — you can tweak an agent's personality or task instructions without touching any Python code. The graph.py file is the wiring — it loads those descriptions, plugs them into LLM calls, and connects the nodes together.

`app.py` is the frontend. It doesn't know or care about LangGraph internals — it calls `run_news_monitor_stream(company, industry, competitors, time_window)` and iterates over a generator that yields progress messages as each scan finishes, then the final digest text. The UI has a CheckboxGroup of 10 predefined competitors (plus an ad-hoc text field for custom ones) and a time window dropdown. The progress log shows each competitor scan completing in real time.

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
| scan_news | None (no LLM) | Pure search — fires 42 Serper queries and collects raw results. No summarization, no token cost. |
| compile_digest | GPT-4o-mini | Cheap and fast — the only LLM call in the news monitor. Deduplicates, categorizes, and formats raw results into a digest. |
| Disambiguation | GPT-4o-mini (temp 0.1) | ~$0.001 per competitor to generate search-friendly names and exclusion terms. |

This is a pattern worth remembering: **match the model to the cognitive demand of the task**. The previous briefing pipeline used Claude Sonnet for analysis and recommendations — models chosen for their reasoning ability. The news monitor doesn't *need* reasoning; it needs to collect data and format it cleanly. A single GPT-4o-mini call handles the formatting, and the search nodes don't use an LLM at all. The result is roughly 90% cheaper per run.

### Why Serper Instead of LLM Tool-Calling

The scan nodes don't use LLM tool-calling to search the web. Instead, they:
1. Generate hardcoded search queries based on the competitor name
2. Call the Serper API directly as a Python function
3. Feed the results into the LLM as user message content

This sidesteps the entire tool-calling format problem. It's also more predictable — you know exactly what searches will run, and the LLM can't decide to skip searching or search for something irrelevant. The tradeoff is less flexibility (the LLM can't dynamically generate creative search queries), but for this use case, the fixed query templates cover the ground well enough.

**Lesson**: Tool-calling is powerful but adds complexity. If you can achieve the same result with a simple function call + prompt, prefer that. Save tool-calling for cases where the LLM genuinely needs to decide *whether* and *how* to use a tool.

### The Dual-Endpoint Search Strategy

The news monitor scan uses **two Serper endpoints** to get the best of both worlds — 42 queries per competitor total:

1. **`/news` endpoint** (25 queries per competitor, filtered by time window): This returns actual news articles sorted by recency — press releases, earnings reports, product announcements, M&A news, executive appointments, regulatory actions, analyst coverage, trade show news (IFPE, bauma, ConExpo, Hannover Messe), electrification/electrohydraulic technology trends, distributor/channel moves, capex/manufacturing investments, supply chain disruption, sustainability/ESG, safety recalls, R&D innovation, layoffs/restructuring, government contracts, industrial automation, construction/off-highway, tariffs/trade regulation, and fluid power-specific press. Each result comes with a publication date. The `tbs` parameter is set according to the user's chosen time window (`qdr:w` for past week, `qdr:w2` for past 2 weeks, `qdr:m` for past month) — but this isn't always reliable (see Bug 5 below), so we also filter in code.

2. **`/search` endpoint** (17 queries per competitor, no date filter): This picks up broader context that news doesn't cover — strategy/growth plans, product roadmaps, job postings on LinkedIn/Indeed/Glassdoor/ZipRecruiter, patent filings on USPTO and Google Patents, SEC filings (10-K, 10-Q, 8-K), coverage from Reuters/Bloomberg/FT, LinkedIn posts/articles, and **site-targeted searches** for key hydraulics trade publications (Hydraulics & Pneumatics, Fluid Power World, Fluid Power Journal, Mobile Hydraulic Tips, Power & Motion, OEM Off-Highway, IFPE, Diesel Progress, Fluid Power Net), automation/engineering publications (Automation World, Control Engineering, Plant Engineering, Machine Design, Design World, The Engineer), construction publications (Equipment World, For Construction Pros, Construction Equipment), industry associations (NFPA, FPDA, NAHAD), and PR wire services (PRNewswire, BusinessWire, GlobeNewsWire).

**The time window is user-configurable.** The UI offers three options: past week, past 2 weeks (default), and past month. This controls both the Serper `tbs` parameter and the code-level `_is_recent_news()` filter threshold. Shorter windows produce tighter, more actionable digests; longer windows catch more but include more noise.

**Every query includes the industry term** (except the site-targeted queries, which are already scoped to specific publications by domain). This is critical for disambiguation — a search for "ATOS product launch" returns the French IT company Atos SE; a search for "ATOS Hydraulics & Mobile Machinery product launch" returns the right one. Without industry context, common or ambiguous company names pollute results with the wrong entity (see Bug 6 below).

Results are tagged `[NEWS (date)]` or `[WEB]` so the compile_digest LLM can prioritise recent news over older web content. Since the news monitor has no intermediate LLM summarization steps, every result that `compile_digest` sees carries its original Serper URL — no risk of URL hallucination from lossy LLM hops.

**Two layers of freshness defence** keep stale news out of the digest:
1. **API layer**: `tbs` parameter set per the chosen time window (unreliable but helps)
2. **Code layer**: `_is_recent_news()` in `graph.py` parses each result's date string (relative like "3 days ago" or absolute like "Jan 15, 2024") and discards anything older than the time window's max_age_days before it reaches `compile_digest`

**Disambiguation works at two layers in the news monitor:**
1. **LLM disambiguation layer**: Before any searches run, `_disambiguate_competitor()` calls gpt-4o-mini to generate a search-friendly name (e.g., "ATOS SpA hydraulic valves" instead of "ATOS"), Google exclusion operators (e.g., `-"Atos SE" -"Eviden"`), and a one-sentence identity description. This is cheap (~$0.001 per call) and dramatically improves query precision for ambiguous names.
2. **Query layer**: Every search query uses the disambiguated search name + exclusion terms + `{industry}` to anchor results to the right company

(The annual report pipeline adds a third layer — the LLM summarization prompt includes the disambiguation context sentence so it can discard wrong-company results during synthesis.)

This matters because using only the regular `/search` endpoint was the app's biggest blind spot. Google's web search returns a mix of evergreen content (company "About" pages, Wikipedia) and actual news — and the evergreen stuff often ranks higher. The `/news` endpoint cuts through that noise and surfaces the breaking developments a strategy manager actually cares about.

**Lesson**: When your use case is "tell me what happened recently," use a news-specific search endpoint if one exists. Generic web search is optimised for relevance, not recency — and for competitive intelligence, recency *is* relevance. But don't trust any single layer — the `tbs` parameter and the code-level date filter reinforce each other. Defence in depth beats relying on one mechanism.

### Real-Time Progress: Streaming Node Completions to the UI

Here's a UX problem that's easy to overlook: even though the news monitor is fast (60-90 seconds), scanning 10 competitors still takes long enough that users wonder if anything is happening. This is the "spinning beach ball" problem — the system is working fine, but it *feels* broken if you stare at a static spinner for a minute.

The fix uses a LangGraph feature called **streaming mode**. Instead of `graph.invoke()` (which blocks until everything is done and returns the final state), you call `graph.stream(inputs, stream_mode="updates")`. This returns a generator that yields a chunk after *each graph node completes*. Each chunk is a dict like `{"scan_news": {"news_results": [...]}}` — the node name and its output.

The pipeline runners (`run_news_monitor_stream()` and `run_annual_report_pipeline_stream()`) wrap this into a cleaner interface. They iterate over the stream chunks, map each node name to a human-readable message (e.g., `"scan_news"` for Parker Hannifin yields `"✓ Scanned Parker Hannifin"`), and yield `("progress", message)` tuples. At the end, they yield `("result", digest_text)`.

On the Gradio side, the `on_generate()` callback is already a generator (it uses `yield` to update the UI incrementally). It just iterates over the stream and yields a new UI state after each progress message. Gradio's generator pattern handles the rest — each `yield` pushes an update to the browser. Users see each competitor's scan finishing in real time, then the "digest compiled" message at the end.

The non-streaming `run_news_monitor()` and `run_annual_report_pipeline()` functions still exist as thin wrappers that print progress to stdout. The CLI uses these. Same underlying stream, different output target.

**Lesson**: When your backend already processes work in discrete steps (graph nodes, pipeline stages, batch items), surfacing those steps to the user is almost free — you just need the framework to yield between steps instead of blocking until the end. LangGraph's `stream_mode="updates"` does exactly this. The key insight is that "progress feedback" doesn't require streaming tokens from the LLM (which is complex and async) — it just requires knowing when each *stage* finishes, which is much simpler.

### Why Fan-Out Instead of Sequential Scanning

The original CrewAI version scanned competitors one at a time. With 4 competitors, that's 4 sequential LLM calls + 4 sets of web searches, all waiting on each other. The fan-out approach runs them in parallel using LangGraph's `Send()` primitive.

This is a real-world performance win. Each scan takes maybe 15-30 seconds (network calls to Serper). Sequential: 60-120 seconds. Parallel: still 15-30 seconds. The scanning stage is embarrassingly parallel — each competitor scan is completely independent, and the only sequential step (`compile_digest`) runs once after all scans converge.

LangGraph handles the fan-in automatically: the `news_results` field in state uses `Annotated[list[str], operator.add]`, which means results from parallel nodes get concatenated into a single list. This is a nice pattern — you declare the merge strategy in the type annotation, and the framework handles the rest.

### The Annual Report Evaluator: An LLM Quality Gate

Here's a problem with any LLM pipeline: the output quality is inconsistent. Sometimes the synthesis produces a beautifully structured document with all 15 sections, specific facts, and cited sources. Other times, it generates a wall of vague observations with no structure. Without a quality gate, whatever comes out goes straight to the user.

The annual report pipeline fixes this with an **LLM-as-judge** pattern. Think of it like a code review before merging: after the LLM generates a competitor report, an evaluator checks it against a rubric (all 15 sections present? specific facts cited? diverse sources? no padding?), and returns a verdict: pass or fail.

If it fails, the pipeline **retries** — but this time, the evaluator's specific feedback gets injected into the prompt. Instead of the LLM seeing just the original task description, it also sees something like: *"PREVIOUS ATTEMPT FEEDBACK: Missing Patents section. Only 2 sources used — need more diversity. Customer Sentiment section is padded with generic observations."* The LLM gets a second chance to get it right, with concrete guidance on what was wrong.

A few design decisions worth understanding:

**Why fail-open on parse errors?** The evaluator is asked to return JSON (`{"evaluation_result": "pass", "evaluation_feedback": "..."}`). But LLMs aren't reliable JSON producers — sometimes they wrap it in markdown fences, sometimes they add preamble text. If JSON parsing fails, the evaluator defaults to `"pass"` and logs a warning. The alternative — blocking the entire pipeline because the evaluator messed up its formatting — is worse than letting a potentially imperfect report through. The evaluator is a safety net, not a brick wall.

**Why max 2 retries?** Each retry costs an LLM call. Beyond 2 retries, the output is unlikely to improve significantly — if the LLM can't get it right in 3 attempts, a 4th probably won't help. Better to let the imperfect report through and let the human reader notice the gaps.

**Why not evaluate the news monitor's digest?** The news monitor's `compile_digest` is a formatting/dedup task on raw search results — the "quality" is primarily about search coverage (handled by the 42 queries) and formatting (handled by GPT-4o-mini). If the formatting is off, it's cheaper to just re-run than to build an evaluator. The evaluator pattern makes sense for the annual report because that pipeline asks an LLM to *synthesize* a deep analytical document where quality genuinely varies between runs.

**Why inline evaluation instead of a separate graph node?** The annual report pipeline uses `Send()` fan-out: each competitor gets its own parallel `scan_annual_report` branch, and all branches converge after that single node. If you added a graph-level evaluate node, it would run once after *all* scans complete and would have to evaluate all reports in a single pass — losing the ability to retry individual competitors. Imagine a quality inspector at the end of a factory line who can only reject the entire batch, not individual items.

So instead, the annual report evaluation happens **inline** — inside `scan_annual_report()` itself. After the LLM generates a competitor report, the same function immediately evaluates it against a rubric (all 15 sections present? specific facts cited? diverse sources?). If it fails, the function retries the LLM call with the feedback appended — all within the same function invocation. Each parallel branch independently evaluates and retries its own report, without affecting the others.

The key details:
- **No new graph nodes or edges** — the evaluation loop is a Python `for` loop inside the existing node function
- **Per-competitor retries** — if Competitor A's report fails but Competitor B's passes, only A retries
- **Web searches not re-run** — retries re-invoke only the LLM with the same search context plus feedback. The Serper results don't change between attempts, and re-fetching them would waste time and API credits
- **Same retry cap (2)** and **same fail-open behavior** — consistency makes the system easier to reason about

This is a useful pattern to remember: **when your pipeline uses fan-out parallelism, quality gates must live inside the parallel branches, not after convergence**. A graph-level evaluator after fan-in would lose per-item granularity. Inline evaluation preserves it.

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

### Bug 4: The Retired Model That Broke Deep Dive

**What happened**: Everything in the pipeline worked — scans, analysis, recommendations, briefing generation, quick chat. But the deep dive feature crashed with a `404 Not Found` error: `model: claude-3-5-sonnet-latest`.

**Why it happened**: The deep dive synthesis in `app.py` was hardcoded to use `claude-3-5-sonnet-latest`. Anthropic had retired that model alias — it no longer resolved to anything. The pipeline nodes (which we wrote fresh during the LangGraph rewrite) used `claude-sonnet-4-20250514`, but `app.py`'s deep dive code was pre-existing and still referenced the old model name. We tested the pipeline thoroughly but almost missed this because deep dive is a separate code path that doesn't go through the graph at all.

**The fix**: Changed `claude-3-5-sonnet-latest` to `claude-sonnet-4-20250514` in `app.py`.

**The lesson**: Model aliases like `*-latest` feel convenient but are a trap. They're mutable pointers — the provider can retire or redirect them at any time, and your code breaks with no warning. Pin to a specific model version (like `claude-sonnet-4-20250514`) so you control when you upgrade. And when you're testing a system with multiple LLM integration points, test *every* code path that calls an LLM, not just the main pipeline. The deep dive was a completely separate call to Anthropic that happened to use a different model name — easy to overlook because it wasn't part of the LangGraph rewrite.

### Bug 5: The "Past Month" Filter That Returned 2023 Articles

**What happened**: The briefing's "Latest News & Developments (Past 30 Days)" section contained articles from 2024 and 2023. Users clicked the links expecting recent news and found year-old press releases.

**Why it happened**: The Serper news API accepts a `tbs="qdr:m"` parameter that's supposed to filter to the past month. But Serper passes this through to Google, and Google doesn't always respect it — especially for niche industry searches with limited recent results. When there isn't enough recent news for "ATOS hydraulics," Google fills in with older articles that match the keywords.

**The fix**: Three layers of defence. First, a code-level date filter (`_is_recent_news()`) that parses each result's date string — Serper returns dates as "3 days ago", "Jan 15, 2024", etc. — and discards anything older than 45 days before it reaches the LLM. Second, the scan node's LLM prompt now includes today's date and explicit instructions to check freshness. Third, the `write_briefing` task has a DATE FRESHNESS RULE that tells the report writer to only include genuinely recent news and to state clearly when no recent news exists rather than padding with old articles.

**The lesson**: Never trust a single layer of filtering, especially when it's a third-party API parameter you can't inspect. The `tbs` parameter is a *hint* to Google, not a guarantee. When freshness matters, validate dates in your own code. This is defence in depth — the same principle as input validation on both client and server.

### Bug 6: The Wrong ATOS — Competitor Name Disambiguation

**What happened**: Searching for "ATOS" as a hydraulics competitor returned results about Atos SE, a large French IT services company. The briefing contained analysis of cloud computing strategies and digital transformation initiatives — completely irrelevant to hydraulic machinery.

**Why it happened**: Seven of the nine news search queries didn't include the industry term. Queries like `"ATOS product launch release update 2026"` matched the much more prominent Atos SE (a Fortune 500 IT company) rather than ATOS the Italian hydraulics manufacturer. Google's ranking algorithm favoured the more well-known entity.

**The fix (v1)**: Added `{industry}` to every search query — so `"ATOS Hydraulics & Mobile Machinery product launch"` instead of bare `"ATOS product launch"`. This helped, but wasn't enough — Atos SE was so prominent in Google's index that even industry-qualified queries still returned wrong-company results.

**The fix (v2 — LLM-powered disambiguation)**: Added a cheap pre-scan step: `_disambiguate_competitor()` calls gpt-4o-mini with the competitor name, industry, and company context, and gets back three things: (1) a more specific search name like `"ATOS SpA hydraulic valves"`, (2) Google exclusion operators like `-"Atos SE" -"Eviden" -"IT services"`, and (3) a one-sentence identity description used in the LLM prompt. Every search query now uses the disambiguated name + exclusion terms instead of the bare competitor name. This costs ~$0.001 per competitor and dramatically cleans up results for ambiguous names while having zero effect on already-unambiguous names like "Parker Hannifin".

The deep dive feature also got the same treatment — `deep_dive()` now receives company/industry/competitors context and includes it in both the query generation prompt and the synthesis prompt, so clicking "Research This" on an ATOS topic no longer generates generic queries that return 100% wrong-company results.

**The lesson**: Ambiguous entity names are a classic search problem — and it's worse with LLMs because they'll confidently summarise whatever results they get, even if half are about the wrong company. When static disambiguation (adding industry terms) isn't enough, use a cheap LLM call to generate entity-specific search operators. The $0.001 cost per competitor is negligible compared to the wasted API spend and bad output quality from polluted search results.

### Bug 7: The Hallucinated URLs in the Briefing

**What happened**: The "Latest News & Developments" section of the briefing contained plausible-looking URLs that didn't actually exist. Clicking them returned 404 errors. The URLs looked realistic — correct domain names, reasonable paths — but they were fabricated by the LLM.

**Why it happened**: The `write_briefing` node (GPT-4o-mini) was asked to produce `[Read more →](URL)` links for every news item, but it never had access to the original search results. Here's the data flow:

1. `scan_competitor` gets real URLs from Serper, feeds them to GPT-4o → some URLs survive in the summary, some don't
2. `analyze` reads the scan summaries (already 1 LLM hop from real data), produces analysis → URLs further degraded
3. `write_briefing` reads analysis + recommendations (2-3 LLM hops from real URLs) → no real URLs left, but the prompt *demands* clickable links

When you tell an LLM "every news item MUST have a source link" but give it no real URLs to work with, it does what LLMs do: it confidently generates plausible-looking URLs that don't exist. The domain names and path patterns looked right because the LLM had seen real URLs from those domains during training — it was pattern-matching, not citing.

**The fix**: Added a `raw_search_results` field to the graph state that carries the original Serper results (with real URLs) directly from `scan_competitor()` to `write_briefing()`, bypassing all intermediate LLM summarization hops. The write_briefing prompt explicitly says "Extract URLs ONLY from the RAW SEARCH RESULTS section — NEVER invent or guess a URL." The task YAML was updated with the same instruction in both the description and expected output.

The key insight: `raw_search_results` uses the same `Annotated[list[str], operator.add]` reducer as `scan_results`, so results from parallel scan nodes get merged automatically. But unlike `scan_results` (which holds LLM-summarized text), `raw_search_results` holds the original search data untouched — it's a direct pipe from Serper to the report writer.

**The lesson**: When your pipeline has multiple LLM hops, specific data (URLs, numbers, dates) degrades with each hop. LLMs are lossy compressors — they preserve meaning but not exact details. If downstream nodes need exact data from upstream sources, thread that data through the state directly rather than expecting it to survive LLM summarization. This is the same principle as passing structured data alongside natural language in any pipeline: don't rely on prose to preserve machine-readable information.

---

## Potential Pitfalls and How to Avoid Them

### Pitfall: State Schema Mismatches in LangGraph

LangGraph uses TypedDict for state, and if a node returns a key that doesn't match the schema, or returns the wrong type, you get runtime errors that can be cryptic. The `Annotated[list[str], operator.add]` pattern for `scan_results` is particularly easy to get wrong — if a node returns a plain string instead of a list, the `operator.add` fails silently or concatenates characters.

**Avoidance**: Always return the exact type declared in the state schema. For list fields with `operator.add`, always return a list (even if it's a single-item list like `[result]`).

### Pitfall: YAML Interpolation Failures

The agent/task YAML configs use Python `.format()` interpolation (`{company}`, `{competitor}`, etc.). If a template references a variable that's not in the inputs dict, you get a `KeyError` at runtime — in the middle of a pipeline run, after you've already burned API credits on earlier nodes.

**Avoidance**: When adding new YAML templates, make sure every `{variable}` has a corresponding key in the inputs dict. Test the interpolation locally before running the full pipeline.

### Pitfall: LLM-as-Judge JSON Fragility

The evaluator asks Claude to return a raw JSON object. LLMs frequently violate this: they wrap the JSON in ```json fences, add "Here's my evaluation:" preamble, or produce slightly malformed JSON (trailing commas, single quotes). The evaluator handles this by catching `json.JSONDecodeError` and defaulting to "pass" — but a stricter evaluator that tried to fail-hard on bad JSON would block the pipeline on formatting issues, not quality issues.

**Avoidance**: When asking an LLM to produce structured output, always have a fallback. Parse optimistically, fail open, and log the raw output when parsing fails so you can tune the prompt later. Don't let a formatting hiccup in a quality gate shut down the whole pipeline.

### Pitfall: Retry Loops and Cost Multiplication

The evaluator adds at least 1 LLM call per run (the evaluation itself). In the worst case — both analysis and recommendations fail twice — it adds 5 extra calls: 2 evaluation calls, 2 retry-analyze calls, and 1 retry-recommend call (plus the recommend nodes that follow each retry-analyze). This can double the cost and latency of a run.

**Avoidance**: Monitor retry rates in production. If the evaluator fails outputs more than ~20% of the time, the problem is the upstream prompts, not the quality gate. Tighten the task descriptions in `tasks.yaml` so the LLM produces passing output on the first attempt more often. The evaluator should be a safety net, not the primary quality mechanism.

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

### Design for Graceful Degradation

The evaluator embodies a principle that runs through this whole project: **never let a quality mechanism become a reliability risk**. The evaluator defaults to "pass" when it can't parse JSON. Retries are capped at 2 so the pipeline always completes. If all retries are exhausted, the pipeline proceeds with a warning instead of crashing. At every decision point, the question is "what happens if this goes wrong?" and the answer is always "we continue with what we have and flag the issue" rather than "we halt and catch fire."

This is a pattern from production systems engineering: monitoring and quality tools should never be the thing that takes down the system. A broken smoke detector shouldn't set the building on fire.

### Know When to Rip and Replace

The move from CrewAI to LangGraph wasn't a refactor — it was a replacement. We didn't try to patch CrewAI's message format handling or wrap it in an adapter layer. When a framework's core abstraction doesn't fit your needs (in this case, the assumption that all agents share a conversation history), it's faster and cleaner to replace it than to fight it. The rewrite took one session. Working around CrewAI's limitations would have been an ongoing tax on every future change.
