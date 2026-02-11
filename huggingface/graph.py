from __future__ import annotations

import json
import operator
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.types import Send

# HF Spaces deployment: files are at root level
from tools import search_serper

CONFIG_DIR = Path(__file__).parent / "config"
# HF Spaces: output directory at root level
OUTPUT_DIR = Path(__file__).parent / "output"


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


AGENTS_CONFIG = _load_yaml("agents.yaml")
TASKS_CONFIG = _load_yaml("tasks.yaml")


# ── State ────────────────────────────────────────────────────────────────────

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

    # Generate search queries
    queries = [
        f"{competitor} latest product news {state['industry']} {state['current_date'][:4]}",
        f"{competitor} R&D investment partnerships acquisitions {state['current_date'][:4]}",
        f"{competitor} pricing changes strategy {state['industry']}",
    ]

    # Run searches
    all_results = []
    for q in queries:
        try:
            data = search_serper(q)
            for item in data.get("organic", [])[:5]:
                all_results.append(
                    f"- [{item.get('title', '')}]({item.get('link', '')}): {item.get('snippet', '')}"
                )
        except Exception as e:
            all_results.append(f"- Search error for '{q}': {e}")

    search_context = "\n".join(all_results) if all_results else "No search results found."

    # Summarize with LLM
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
    print(f"[scan] Finished scanning {competitor} (gpt-4o)")
    return {"scan_results": [f"## {competitor}\n\n{response.content}"]}


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
        "evaluation_result": "",
        "evaluation_feedback": "",
        "retry_count_analysis": 0,
        "retry_count_recommendations": 0,
    })
    return result["briefing"]


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
    queries = [
        f"{competitor} {industry} official website products solutions",
        f"{competitor} {industry} about company revenue employees",
        f"{competitor} {industry} latest annual report {year} OR {prev_year}",
        f"{competitor} {industry} 10-K SEC filing investor relations {year} OR {prev_year}",
        f"{competitor} {industry} earnings revenue financial results {year} OR {prev_year}",
        f"{competitor} {industry} product catalog pricing customers case studies",
        f"{competitor} {industry} LinkedIn employees hiring jobs {year}",
        f"{competitor} {industry} customer reviews complaints {year}",
        f"{competitor} {industry} market share ranking {year} OR {prev_year}",
        f"{competitor} {industry} acquisition merger partnership {year} OR {prev_year}",
        f"{competitor} {industry} revenue by region geographic expansion {year} OR {prev_year}",
        f"{competitor} {industry} OEM contracts customer wins {year}",
        f"{competitor} hydraulic patent USPTO OR Espacenet {year} OR {prev_year}",
        f"{competitor} {industry} tariff regulatory compliance risk {year}",
        f"{competitor} {industry} news press release announcement {year}",
        f"{competitor} {industry} product catalog model series specifications datasheet",
        f"{competitor} vs {company} {industry} comparison review",
        f"{competitor} {industry} OEM customer wins named accounts case study",
    ]

    all_results = []
    for q in queries:
        try:
            data = search_serper(q)
            for item in data.get("organic", [])[:5]:
                all_results.append(
                    f"- [{item.get('title', '')}]({item.get('link', '')}): {item.get('snippet', '')}"
                )
        except Exception as e:
            all_results.append(f"- Search error for '{q}': {e}")

    search_context = "\n".join(all_results) if all_results else "No search results found."

    llm = _claude()
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


def run_annual_report_pipeline(company: str, industry: str, competitors: str) -> str:
    graph = build_annual_report_graph()
    result = graph.invoke({
        "company": company,
        "industry": industry,
        "competitors": competitors,
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "report_results": [],
    })

    current_date = datetime.now().strftime("%Y-%m-%d")
    header = (
        f"# Annual Report Deep Dive — Competitive Intelligence\n\n"
        f"**Company:** {company} | **Industry:** {industry} | **Date:** {current_date}\n\n"
        f"**Competitors analyzed:** {competitors}\n\n---\n\n"
    )
    combined = header + "\n\n---\n\n".join(result["report_results"])

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "annual_report_analysis.md").write_text(combined, encoding="utf-8")
    print(f"[annual_report] Complete -> output/annual_report_analysis.md")
    return combined
