from __future__ import annotations

import json
import operator
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, TypedDict

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from competitive_intel.tools import search_serper, search_serper_news

CONFIG_DIR = Path(__file__).parent / "config"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


AGENTS_CONFIG = _load_yaml("agents.yaml")
TASKS_CONFIG = _load_yaml("tasks.yaml")

# Maximum age (in days) for news results — anything older is discarded.
NEWS_MAX_AGE_DAYS = 45

# Max concurrent Serper search threads per scan node.
_SEARCH_WORKERS = 12

# Cache disambiguation results so the same competitor isn't re-disambiguated
# if both the briefing and annual-report pipelines run in the same process.
_DISAMBIG_CACHE: dict[tuple, dict] = {}

_RELATIVE_RE = re.compile(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", re.I)


def _parse_serper_date(date_str: str) -> datetime | None:
    """Parse Serper date strings like '3 days ago' or 'Jan 15, 2026'."""
    if not date_str:
        return None
    # Relative: "3 days ago", "2 hours ago", etc.
    m = _RELATIVE_RE.search(date_str)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        delta = {"minute": timedelta(minutes=n), "hour": timedelta(hours=n),
                 "day": timedelta(days=n), "week": timedelta(weeks=n),
                 "month": timedelta(days=n * 30), "year": timedelta(days=n * 365)}
        return datetime.now() - delta.get(unit, timedelta())
    # Absolute: "Jan 15, 2026" / "February 3, 2025" / "15 Jan 2026"
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _is_recent_news(date_str: str, max_age_days: int = NEWS_MAX_AGE_DAYS) -> bool:
    """Return True if the date is within max_age_days, or if unparseable (benefit of doubt)."""
    parsed = _parse_serper_date(date_str)
    if parsed is None:
        return True  # can't parse → let the LLM decide
    return (datetime.now() - parsed).days <= max_age_days


def _disambiguate_competitor(competitor: str, industry: str, company: str) -> dict:
    """Use gpt-4o-mini to generate a search-friendly name, Google exclusion terms,
    and a one-sentence identity description for a competitor.

    Returns {"search_name": str, "exclude_terms": str, "context": str}.
    Falls back to bare competitor name on any failure.
    """
    cache_key = (competitor, industry, company)
    if cache_key in _DISAMBIG_CACHE:
        print(f"[disambiguate] Cache hit for {competitor}")
        return _DISAMBIG_CACHE[cache_key]

    fallback = {"search_name": competitor, "exclude_terms": "", "context": ""}
    try:
        llm = _openai("gpt-4o-mini", temperature=0.1)
        response = llm.invoke([
            {"role": "system", "content": (
                "You help disambiguate company names for Google searches. "
                "Given a competitor name, the industry it operates in, and the company "
                "it competes with, return a JSON object with exactly three keys:\n"
                '- "search_name": a more specific, search-friendly version of the '
                "competitor name that adds identifying details (e.g. parent company, "
                'product category, legal entity suffix) to avoid confusion with '
                'unrelated companies that share a similar name. Keep it concise '
                '(2-5 words).\n'
                '- "exclude_terms": Google search exclusion operators to filter out '
                "the most prominent WRONG-COMPANY results that share a similar name "
                "but operate in a DIFFERENT industry. Each exclusion MUST be a quoted "
                'phrase like -"Atos SE" -"Eviden". NEVER exclude the competitor itself, '
                "its parent company, its own brand names, or the competing company. "
                "NEVER use bare unquoted exclusions like -Parker. "
                "Only exclude specific named entities from other industries that "
                "would pollute search results. If the competitor name is already "
                "unambiguous (e.g. 'Parker Hannifin', 'Bosch Rexroth'), return an "
                "empty string — do NOT invent exclusions.\n"
                '- "context": a one-sentence description of who this competitor is '
                "(industry, products, headquarters) for use in LLM prompts.\n\n"
                "Return ONLY valid JSON, no markdown fences, no explanation."
            )},
            {"role": "user", "content": (
                f"Competitor: {competitor}\n"
                f"Industry: {industry}\n"
                f"Competing with: {company}"
            )},
        ])
        parsed = json.loads(response.content.strip())
        # Normalize exclude_terms: LLM may return a list or a string
        raw_exclude = parsed.get("exclude_terms", "")
        if isinstance(raw_exclude, list):
            raw_exclude = " ".join(raw_exclude)
        result = {
            "search_name": parsed.get("search_name", competitor),
            "exclude_terms": raw_exclude,
            "context": parsed.get("context", ""),
        }
        _DISAMBIG_CACHE[cache_key] = result
        return result
    except Exception as e:
        print(f"[disambiguate] WARNING: Failed for {competitor}: {e}. Using bare name.")
        return fallback


# ── State ────────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    company: str
    industry: str
    competitors: str
    current_date: str
    scan_results: Annotated[list[str], operator.add]
    raw_search_results: Annotated[list[str], operator.add]
    analysis: str
    recommendations: str
    briefing: str
    evaluation_result: str
    evaluation_feedback: str
    retry_count_analysis: int
    retry_count_recommendations: int


# ── LLM clients ─────────────────────────────────────────────────────────────

def _openai(model: str = "gpt-4o", temperature: float = 0.2):
    return ChatOpenAI(model=model, temperature=temperature)


def _claude(model: str = "claude-sonnet-4-20250514", temperature: float = 0.2):
    return ChatAnthropic(model=model, temperature=temperature)


# ── Helper: build system prompt from YAML config ────────────────────────────

def _agent_system_prompt(agent_key: str, inputs: dict) -> str:
    cfg = AGENTS_CONFIG[agent_key]
    role = cfg["role"].format(**inputs).strip()
    goal = cfg["goal"].format(**inputs).strip()
    backstory = cfg["backstory"].format(**inputs).strip()
    return f"Role: {role}\nGoal: {goal}\nBackstory: {backstory}"


def _task_prompt(task_key: str, inputs: dict) -> tuple[str, str]:
    cfg = TASKS_CONFIG[task_key]
    desc = cfg["description"].format(**inputs).strip()
    expected = cfg["expected_output"].format(**inputs).strip()
    return desc, expected


# ── Nodes ────────────────────────────────────────────────────────────────────

class ScanState(TypedDict):
    company: str
    industry: str
    competitors: str
    current_date: str
    competitor: str
    scan_results: Annotated[list[str], operator.add]
    raw_search_results: Annotated[list[str], operator.add]
    analysis: str
    recommendations: str
    briefing: str


def scan_competitor(state: ScanState) -> dict:
    competitor = state["competitor"]
    inputs = {
        "company": state["company"],
        "industry": state["industry"],
        "competitors": state["competitors"],
        "current_date": state["current_date"],
        "competitor": competitor,
    }

    system = _agent_system_prompt("trend_scanner", inputs)
    desc, expected = _task_prompt("scan_competitor", inputs)

    year = state["current_date"][:4]
    industry = state["industry"]
    company = state["company"]

    # LLM-powered disambiguation: get a search-friendly name and exclusion terms
    disambig = _disambiguate_competitor(competitor, industry, company)
    sn = disambig["search_name"]  # e.g. "ATOS SpA hydraulic valves"
    ex = disambig["exclude_terms"]  # e.g. -"Atos SE" -"Eviden"
    ctx = disambig["context"]  # one-sentence identity for LLM prompt
    print(f"[scan] {competitor} disambiguation: search_name={sn!r}, exclude={ex!r}")

    # ── News searches (Serper /news endpoint, filtered to past month) ────────
    news_queries = [
        f"{sn} {industry} news announcement {year} {ex}",
        f"{sn} {industry} product launch release update {year} {ex}",
        f"{sn} {industry} acquisition merger partnership deal {year} {ex}",
        f"{sn} {industry} pricing changes new model tier {year} {ex}",
        f"{sn} {industry} customer win contract award {year} {ex}",
        f"{sn} {industry} executive appointment leadership hire {year} {ex}",
        f"{sn} {industry} earnings revenue financial results {year} {ex}",
        f"{sn} {industry} regulatory lawsuit patent filing {year} {ex}",
        f"{sn} {industry} stock analyst upgrade downgrade guidance {year} {ex}",
        # Trade press & fluid power specific
        f"{sn} \"fluid power\" OR \"hydraulic\" product launch news {year} {ex}",
        f"{sn} IFPE OR bauma OR ConExpo OR \"Hannover Messe\" {year} {ex}",
        f"{sn} electrification OR electrohydraulic OR \"electric actuator\" OR \"digital hydraulic\" {year} {ex}",
        f"{sn} {industry} distributor dealer channel OEM supply {year} {ex}",
        f"{sn} {industry} press release announcement new {year} {ex}",
        f"{sn} {industry} factory plant expansion investment manufacturing {year} {ex}",
    ]

    # ── Web searches (Serper /search endpoint, broader context) ──────────────
    web_queries = [
        f"{sn} {industry} strategy expansion growth plans {year} {ex}",
        f"{sn} {industry} new product features roadmap {year} {ex}",
        f"{sn} {industry} hiring jobs open roles site:linkedin.com OR site:indeed.com {year} {ex}",
        f"{sn} {industry} patent USPTO OR Espacenet {year} {ex}",
        f"{sn} {industry} tariff trade regulatory compliance {year} {ex}",
        # Trade publications (site-targeted)
        f"{sn} site:hydraulicspneumatics.com OR site:fluidpowerworld.com OR site:fluidpowerjournal.com",
        f"{sn} site:mobilehydraulictips.com OR site:powermotiontech.com OR site:oemoffhighway.com",
        # Press wire services
        f"{sn} {industry} site:prnewswire.com OR site:businesswire.com OR site:globenewswire.com {year}",
    ]

    all_results = []
    skipped_old = 0

    def _fetch_news(q):
        """Fetch a single news query, return (results_list, skipped_count)."""
        results, skipped = [], 0
        try:
            data = search_serper_news(q, num_results=10, tbs="qdr:m")
            for item in data.get("news", [])[:8]:
                date = item.get("date", "")
                if not _is_recent_news(date):
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
        """Fetch a single web query, return results_list."""
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

    # Run all news + web searches concurrently
    with ThreadPoolExecutor(max_workers=_SEARCH_WORKERS) as executor:
        news_futures = {executor.submit(_fetch_news, q): q for q in news_queries}
        web_futures = {executor.submit(_fetch_web, q): q for q in web_queries}

        for future in as_completed(news_futures):
            results, skipped = future.result()
            all_results.extend(results)
            skipped_old += skipped

        for future in as_completed(web_futures):
            all_results.extend(future.result())

    if skipped_old:
        print(f"[scan] {competitor}: filtered out {skipped_old} news results older than {NEWS_MAX_AGE_DAYS} days")

    search_context = "\n".join(all_results) if all_results else "No search results found."

    # Summarize with LLM
    llm = _openai("gpt-4o")
    current_date = state["current_date"]
    user_msg = (
        f"{desc}\n\n"
        f"DISAMBIGUATION: {ctx if ctx else f'{competitor} is a {industry} company competing with {company}'}. "
        f"DISCARD any search results about unrelated companies that "
        f"happen to share a similar name but operate in a different industry. "
        f"Only include findings you can confidently attribute to {competitor} "
        f"in the {industry} sector.\n\n"
        f"DATE FRESHNESS: Today is {current_date}. For items tagged [NEWS], "
        f"only report them as recent news if the date shown is within the last "
        f"45 days. If a news item's date is from a previous year or clearly "
        f"outdated, do NOT present it as a recent development. Preserve the "
        f"original date in your output so readers can judge recency.\n\n"
        f"Here are the web search results for {competitor}:\n\n"
        f"{search_context}\n\n"
        f"Expected output format:\n{expected}"
    )
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ])
    print(f"[scan] Finished scanning {competitor} (gpt-4o) — {len(all_results)} results from {len(news_queries)} news + {len(web_queries)} web queries")
    raw_block = f"## {competitor} — Raw Search Results\n\n" + "\n".join(all_results)
    return {
        "scan_results": [f"## {competitor}\n\n{response.content}"],
        "raw_search_results": [raw_block],
    }


def fan_out(state: GraphState) -> list[Send]:
    competitors = [c.strip() for c in state["competitors"].split(",") if c.strip()]
    return [Send("scan_competitor", {**state, "competitor": c}) for c in competitors]


def analyze(state: GraphState) -> dict:
    inputs = {
        "company": state["company"],
        "industry": state["industry"],
        "competitors": state["competitors"],
        "current_date": state["current_date"],
    }

    system = _agent_system_prompt("company_analyst", inputs)
    desc, expected = _task_prompt("analyze_findings", inputs)

    scan_text = "\n\n---\n\n".join(state["scan_results"])

    user_content = (
        f"{desc}\n\n"
        f"RAW COMPETITIVE INTELLIGENCE:\n\n{scan_text}\n\n"
        f"Expected output format:\n{expected}"
    )

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
    print(f"[analyze] Finished analysis (claude-sonnet-4-20250514) retry_count={retry_count}")
    return {"analysis": response.content, "retry_count_analysis": retry_count}


def recommend(state: GraphState) -> dict:
    inputs = {
        "company": state["company"],
        "industry": state["industry"],
        "competitors": state["competitors"],
        "current_date": state["current_date"],
    }

    system = _agent_system_prompt("strategy_advisor", inputs)
    desc, expected = _task_prompt("strategic_recommendations", inputs)

    user_content = (
        f"{desc}\n\n"
        f"COMPETITIVE ANALYSIS:\n\n{state['analysis']}\n\n"
        f"Expected output format:\n{expected}"
    )

    feedback = state.get("evaluation_feedback", "")
    eval_result = state.get("evaluation_result", "")
    is_retry = eval_result in ("fail_recommendations", "fail_both")
    if is_retry and feedback:
        user_content += (
            f"\n\n---\n"
            f"PREVIOUS ATTEMPT FEEDBACK (address these issues in your revised recommendations):\n"
            f"{feedback}\n"
            f"Please revise your recommendations to address the feedback above."
        )

    llm = _claude()
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ])
    retry_count = state.get("retry_count_recommendations", 0)
    if is_retry:
        retry_count += 1
    print(f"[recommend] Finished recommendations (claude-sonnet-4-20250514) retry_count={retry_count}")
    return {"recommendations": response.content, "retry_count_recommendations": retry_count}


def write_briefing(state: GraphState) -> dict:
    inputs = {
        "company": state["company"],
        "industry": state["industry"],
        "competitors": state["competitors"],
        "current_date": state["current_date"],
    }

    system = _agent_system_prompt("report_writer", inputs)
    desc, expected = _task_prompt("write_briefing", inputs)

    raw_text = "\n\n---\n\n".join(state.get("raw_search_results", []))

    llm = _openai("gpt-4o-mini")
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"{desc}\n\n"
            f"COMPETITIVE ANALYSIS:\n\n{state['analysis']}\n\n"
            f"STRATEGIC RECOMMENDATIONS:\n\n{state['recommendations']}\n\n"
            f"RAW SEARCH RESULTS WITH SOURCE URLS:\n"
            f"Use these for the Latest News section — extract real URLs, do NOT invent URLs.\n\n"
            f"{raw_text}\n\n"
            f"Expected output format:\n{expected}"
        )},
    ])

    briefing = response.content
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "briefing.md").write_text(briefing, encoding="utf-8")
    print(f"[write_briefing] Finished briefing (gpt-4o-mini) -> output/briefing.md")
    return {"briefing": briefing}


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
        print(f"[evaluate] WARNING: Could not parse evaluator JSON, defaulting to pass. Raw: {raw[:200]}")
        result = "pass"
        feedback = ""

    if result not in ("pass", "fail_analysis", "fail_recommendations", "fail_both"):
        print(f"[evaluate] WARNING: Unknown evaluation_result '{result}', defaulting to pass")
        result = "pass"

    print(f"[evaluate] Result: {result}")
    return {"evaluation_result": result, "evaluation_feedback": feedback}


MAX_RETRIES = 2


def route_after_evaluation(state: GraphState) -> str:
    result = state.get("evaluation_result", "pass")
    retries_analysis = state.get("retry_count_analysis", 0)
    retries_recommendations = state.get("retry_count_recommendations", 0)

    if result == "pass":
        return "write_briefing"

    if result == "fail_analysis" and retries_analysis < MAX_RETRIES:
        print(f"[evaluate] Routing to retry_analyze (attempt {retries_analysis + 1}/{MAX_RETRIES})")
        return "retry_analyze"

    if result == "fail_recommendations" and retries_recommendations < MAX_RETRIES:
        print(f"[evaluate] Routing to retry_recommend (attempt {retries_recommendations + 1}/{MAX_RETRIES})")
        return "retry_recommend"

    if result == "fail_both":
        if retries_analysis < MAX_RETRIES:
            print(f"[evaluate] Both failed — routing to retry_analyze first (attempt {retries_analysis + 1}/{MAX_RETRIES})")
            return "retry_analyze"
        if retries_recommendations < MAX_RETRIES:
            print(f"[evaluate] Analysis retries exhausted — routing to retry_recommend (attempt {retries_recommendations + 1}/{MAX_RETRIES})")
            return "retry_recommend"

    print(f"[evaluate] WARNING: Max retries exhausted (analysis={retries_analysis}, recommendations={retries_recommendations}). Proceeding to write_briefing.")
    return "write_briefing"


# ── Graph construction ───────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("scan_competitor", scan_competitor)
    graph.add_node("analyze", analyze)
    graph.add_node("recommend", recommend)
    graph.add_node("evaluate", evaluate)
    graph.add_node("retry_analyze", analyze)
    graph.add_node("retry_recommend", recommend)
    graph.add_node("write_briefing", write_briefing)

    graph.set_conditional_entry_point(fan_out, ["scan_competitor"])
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

    return graph.compile()


_BRIEFING_NODE_LABELS = {
    "scan_competitor": "Scanned",
    "analyze": "Competitive analysis complete",
    "recommend": "Strategic recommendations complete",
    "evaluate": "Quality evaluation complete",
    "retry_analyze": "Re-running analysis (evaluator feedback)",
    "retry_recommend": "Re-running recommendations (evaluator feedback)",
    "write_briefing": "Final briefing written",
}


def run_pipeline_stream(company: str, industry: str, competitors: str):
    """Generator that yields (type, message) tuples as each graph node completes.

    type is "progress" for status updates or "result" for the final briefing text.
    """
    graph = build_graph()
    inputs = {
        "company": company,
        "industry": industry,
        "competitors": competitors,
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "scan_results": [],
        "raw_search_results": [],
        "analysis": "",
        "recommendations": "",
        "briefing": "",
        "evaluation_result": "",
        "evaluation_feedback": "",
        "retry_count_analysis": 0,
        "retry_count_recommendations": 0,
    }

    final_state = {}
    for chunk in graph.stream(inputs, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            final_state.update(node_output)
            label = _BRIEFING_NODE_LABELS.get(node_name, node_name)

            if node_name == "scan_competitor":
                # Extract competitor name from the scan result (starts with "## CompetitorName")
                scan_results = node_output.get("scan_results", [])
                if scan_results:
                    first_line = scan_results[0].split("\n", 1)[0]
                    comp_name = first_line.lstrip("# ").strip()
                else:
                    comp_name = "unknown"
                num_results = scan_results[0].count("- [") if scan_results else 0
                yield ("progress", f"  ✓ {label} {comp_name}")

            elif node_name == "evaluate":
                eval_result = node_output.get("evaluation_result", "pass")
                if eval_result == "pass":
                    yield ("progress", f"  ✓ Quality check passed")
                else:
                    yield ("progress", f"  ⚠ Quality check: {eval_result}")

            elif node_name in ("retry_analyze", "retry_recommend"):
                yield ("progress", f"  ⟳ {label}")

            elif node_name == "analyze":
                yield ("progress", f"  ✓ {label}")

            elif node_name == "recommend":
                yield ("progress", f"  ✓ {label}")

            elif node_name == "write_briefing":
                yield ("progress", f"  ✓ {label} → output/briefing.md")

            else:
                yield ("progress", f"  ✓ {label}")

    yield ("result", final_state.get("briefing", ""))


def run_pipeline(company: str, industry: str, competitors: str) -> str:
    result = None
    for msg_type, msg in run_pipeline_stream(company, industry, competitors):
        if msg_type == "progress":
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode("ascii", errors="replace").decode("ascii"))
        elif msg_type == "result":
            result = msg
    return result


# ── Annual Report Deep Dive Pipeline ─────────────────────────────────────────

class AnnualReportState(TypedDict):
    company: str
    industry: str
    competitors: str
    current_date: str
    competitor: str
    report_results: Annotated[list[str], operator.add]


def scan_annual_report(state: AnnualReportState) -> dict:
    competitor = state["competitor"]
    year = state["current_date"][:4]
    prev_year = str(int(year) - 1)
    inputs = {
        "company": state["company"],
        "industry": state["industry"],
        "competitors": state["competitors"],
        "current_date": state["current_date"],
        "competitor": competitor,
    }

    system = _agent_system_prompt("annual_report_analyst", inputs)
    desc, expected = _task_prompt("scan_annual_report", inputs)

    company = state["company"]
    industry = state["industry"]

    # LLM-powered disambiguation
    disambig = _disambiguate_competitor(competitor, industry, company)
    sn = disambig["search_name"]
    ex = disambig["exclude_terms"]
    ctx = disambig["context"]
    print(f"[scan_annual_report] {competitor} disambiguation: search_name={sn!r}, exclude={ex!r}")

    queries = [
        f"{sn} {industry} official website products solutions {ex}",
        f"{sn} {industry} about company revenue employees {ex}",
        f"{sn} {industry} latest annual report {year} OR {prev_year} {ex}",
        f"{sn} {industry} 10-K SEC filing investor relations {year} OR {prev_year} {ex}",
        f"{sn} {industry} earnings revenue financial results {year} OR {prev_year} {ex}",
        f"{sn} {industry} product catalog pricing customers case studies {ex}",
        f"{sn} {industry} LinkedIn employees hiring jobs {year} {ex}",
        f"{sn} {industry} customer reviews complaints {year} {ex}",
        f"{sn} {industry} market share ranking {year} OR {prev_year} {ex}",
        f"{sn} {industry} acquisition merger partnership {year} OR {prev_year} {ex}",
        f"{sn} {industry} revenue by region geographic expansion {year} OR {prev_year} {ex}",
        f"{sn} {industry} OEM contracts customer wins {year} {ex}",
        f"{sn} {industry} patent USPTO OR Espacenet {year} OR {prev_year} {ex}",
        f"{sn} {industry} tariff regulatory compliance risk {year} {ex}",
        f"{sn} {industry} news press release announcement {year} {ex}",
        f"{sn} {industry} product catalog model series specifications datasheet {ex}",
        f"{competitor} vs {company} {industry} comparison review {ex}",
        f"{sn} {industry} OEM customer wins named accounts case study {ex}",
    ]

    def _fetch_annual(q):
        """Fetch a single web query for annual report, return results_list."""
        results = []
        try:
            data = search_serper(q)
            for item in data.get("organic", [])[:5]:
                results.append(
                    f"- [{item.get('title', '')}]({item.get('link', '')}): {item.get('snippet', '')}"
                )
        except Exception as e:
            results.append(f"- Search error for '{q}': {e}")
        return results

    all_results = []
    with ThreadPoolExecutor(max_workers=_SEARCH_WORKERS) as executor:
        futures = {executor.submit(_fetch_annual, q): q for q in queries}
        for future in as_completed(futures):
            all_results.extend(future.result())

    search_context = "\n".join(all_results) if all_results else "No search results found."

    llm = _claude()
    user_msg = (
        f"{desc}\n\n"
        f"CRITICAL DISAMBIGUATION REMINDER: "
        f"{ctx if ctx else f'{competitor} is a {industry} company competing with {company}'}. "
        f"Many search results below may be about a "
        f"DIFFERENT company with a similar name in another industry. You MUST "
        f"discard any result that is not about {competitor} in the {industry} "
        f"sector. When in doubt, leave it out.\n\n"
        f"Here are the web search results for {competitor}:\n\n"
        f"{search_context}\n\n"
        f"Expected output format:\n{expected}"
    )
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ])
    report_text = response.content

    # Inline evaluation + retry loop
    for attempt in range(MAX_RETRIES + 1):
        eval_inputs = {**inputs, "report_text": report_text}
        eval_system = _agent_system_prompt("quality_evaluator", eval_inputs)
        eval_desc, eval_expected = _task_prompt("evaluate_annual_report", eval_inputs)

        print(f"[scan_annual_report] Evaluating {competitor} (attempt {attempt + 1}/{MAX_RETRIES + 1})...")
        eval_response = _claude().invoke([
            {"role": "system", "content": eval_system},
            {"role": "user", "content": f"{eval_desc}\n\n{eval_expected}"},
        ])

        raw = eval_response.content.strip()
        try:
            parsed = json.loads(raw)
            result = parsed.get("evaluation_result", "pass")
            feedback = parsed.get("evaluation_feedback", "")
        except (json.JSONDecodeError, AttributeError):
            print(f"[scan_annual_report] WARNING: Could not parse evaluator JSON for {competitor}, defaulting to pass. Raw: {raw[:200]}")
            result = "pass"
            feedback = ""

        if result not in ("pass", "fail"):
            print(f"[scan_annual_report] WARNING: Unknown evaluation_result '{result}' for {competitor}, defaulting to pass")
            result = "pass"

        print(f"[scan_annual_report] {competitor} evaluation result: {result}")

        if result == "pass":
            break

        if attempt < MAX_RETRIES:
            print(f"[scan_annual_report] Retrying {competitor} with feedback (retry {attempt + 1}/{MAX_RETRIES})")
            retry_msg = (
                f"{user_msg}\n\n"
                f"---\n"
                f"FEEDBACK FROM QUALITY REVIEW (address these issues in your revised report):\n"
                f"{feedback}\n"
                f"Please revise your report to address the feedback above."
            )
            response = llm.invoke([
                {"role": "system", "content": system},
                {"role": "user", "content": retry_msg},
            ])
            report_text = response.content
        else:
            print(f"[scan_annual_report] WARNING: Max retries exhausted for {competitor}. Proceeding with current report.")

    print(f"[scan_annual_report] Finished {competitor} (claude-sonnet, retries={attempt})")
    return {"report_results": [f"# {competitor}\n\n{report_text}"]}


def fan_out_annual_reports(state: AnnualReportState) -> list[Send]:
    competitors = [c.strip() for c in state["competitors"].split(",") if c.strip()]
    return [Send("scan_annual_report", {**state, "competitor": c}) for c in competitors]


def build_annual_report_graph():
    graph = StateGraph(AnnualReportState)
    graph.add_node("scan_annual_report", scan_annual_report)
    graph.set_conditional_entry_point(fan_out_annual_reports, ["scan_annual_report"])
    graph.add_edge("scan_annual_report", END)
    return graph.compile()


def run_annual_report_pipeline_stream(company: str, industry: str, competitors: str):
    """Generator that yields (type, message) tuples as each annual report node completes."""
    graph = build_annual_report_graph()
    inputs = {
        "company": company,
        "industry": industry,
        "competitors": competitors,
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "report_results": [],
    }

    all_report_results = []
    for chunk in graph.stream(inputs, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            if node_name == "scan_annual_report":
                report_results = node_output.get("report_results", [])
                all_report_results.extend(report_results)
                if report_results:
                    first_line = report_results[0].split("\n", 1)[0]
                    comp_name = first_line.lstrip("# ").strip()
                else:
                    comp_name = "unknown"
                yield ("progress", f"  ✓ Finished deep dive — {comp_name}")
            else:
                yield ("progress", f"  ✓ {node_name}")

    current_date = datetime.now().strftime("%Y-%m-%d")
    header = (
        f"# Annual Report Deep Dive — Competitive Intelligence\n\n"
        f"**Company:** {company} | **Industry:** {industry} | **Date:** {current_date}\n\n"
        f"**Competitors analyzed:** {competitors}\n\n---\n\n"
    )
    report_results = all_report_results
    combined = header + "\n\n---\n\n".join(report_results)

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "annual_report_analysis.md").write_text(combined, encoding="utf-8")
    yield ("progress", f"  ✓ Report saved → output/annual_report_analysis.md")
    yield ("result", combined)


def run_annual_report_pipeline(company: str, industry: str, competitors: str) -> str:
    result = None
    for msg_type, msg in run_annual_report_pipeline_stream(company, industry, competitors):
        if msg_type == "progress":
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode("ascii", errors="replace").decode("ascii"))
        elif msg_type == "result":
            result = msg
    return result
