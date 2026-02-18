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

class NewsMonitorState(TypedDict):
    company: str
    industry: str
    competitors: str
    current_date: str
    time_window: str                                    # "past_week" | "past_2_weeks" | "past_month"
    news_results: Annotated[list[str], operator.add]    # raw tagged results per competitor
    digest: str                                         # final formatted output


class NewsScanState(TypedDict):
    company: str
    industry: str
    competitors: str
    current_date: str
    time_window: str
    competitor: str
    news_results: Annotated[list[str], operator.add]
    digest: str


_TIME_WINDOW_PARAMS = {
    "past_week":    {"tbs": "qdr:w",  "max_age_days": 7,  "label": "Past week"},
    "past_2_weeks": {"tbs": "qdr:w2", "max_age_days": 14, "label": "Past 2 weeks"},
    "past_month":   {"tbs": "qdr:m",  "max_age_days": 30, "label": "Past month"},
}


def _time_window_params(time_window: str) -> dict:
    """Return tbs, max_age_days, label for a time_window key."""
    return _TIME_WINDOW_PARAMS.get(time_window, _TIME_WINDOW_PARAMS["past_2_weeks"])


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

def scan_news(state: NewsScanState) -> dict:
    """Run 42 search queries for a single competitor — no LLM summarization."""
    competitor = state["competitor"]
    year = state["current_date"][:4]
    industry = state["industry"]
    company = state["company"]

    tw = _time_window_params(state.get("time_window", "past_2_weeks"))
    tbs = tw["tbs"]
    max_age = tw["max_age_days"]

    # LLM-powered disambiguation
    disambig = _disambiguate_competitor(competitor, industry, company)
    sn = disambig["search_name"]
    ex = disambig["exclude_terms"]
    print(f"[scan_news] {competitor} disambiguation: search_name={sn!r}, exclude={ex!r}")

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
        print(f"[scan_news] {competitor}: filtered out {skipped_old} results older than {max_age} days")

    print(f"[scan_news] Finished {competitor} — {len(all_results)} results from {len(news_queries)} news + {len(web_queries)} web queries")
    ctx = disambig["context"]
    header = f"## {competitor}\n\n> Disambiguation: {ctx}\n" if ctx else f"## {competitor}\n"
    raw_block = header + "\n" + "\n".join(all_results) if all_results else f"## {competitor}\n\nNo search results found."
    return {"news_results": [raw_block]}


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

    system = _agent_system_prompt("news_digest_curator", inputs)
    desc, expected = _task_prompt("compile_news_digest", inputs)

    llm = _openai("gpt-4o-mini")
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": f"{desc}\n\nExpected output format:\n{expected}"},
    ])

    digest = response.content
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "news_digest.md").write_text(digest, encoding="utf-8")
    print(f"[compile_digest] Finished digest (gpt-4o-mini) -> output/news_digest.md")
    return {"digest": digest}


def fan_out_news(state: NewsMonitorState) -> list[Send]:
    competitors = [c.strip() for c in state["competitors"].split(",") if c.strip()]
    return [Send("scan_news", {**state, "competitor": c}) for c in competitors]


# ── Graph construction ───────────────────────────────────────────────────────

def build_news_graph():
    graph = StateGraph(NewsMonitorState)

    graph.add_node("scan_news", scan_news)
    graph.add_node("compile_digest", compile_digest)

    graph.set_conditional_entry_point(fan_out_news, ["scan_news"])
    graph.add_edge("scan_news", "compile_digest")
    graph.add_edge("compile_digest", END)

    return graph.compile()


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


MAX_RETRIES = 2


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
