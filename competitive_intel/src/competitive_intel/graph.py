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

from competitive_intel.tools import search_serper

CONFIG_DIR = Path(__file__).parent / "config"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


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

    llm = _claude()
    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"{desc}\n\n"
            f"RAW COMPETITIVE INTELLIGENCE:\n\n{scan_text}\n\n"
            f"Expected output format:\n{expected}"
        )},
    ])
    print(f"[analyze] Finished analysis (claude-sonnet-4-20250514)")
    return {"analysis": response.content}


def recommend(state: GraphState) -> dict:
    inputs = {
        "company": state["company"],
        "industry": state["industry"],
        "competitors": state["competitors"],
        "current_date": state["current_date"],
    }

    system = _agent_system_prompt("strategy_advisor", inputs)
    desc, expected = _task_prompt("strategic_recommendations", inputs)

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


# ── Graph construction ───────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("scan_competitor", scan_competitor)
    graph.add_node("analyze", analyze)
    graph.add_node("recommend", recommend)
    graph.add_node("write_briefing", write_briefing)

    graph.set_conditional_entry_point(fan_out, ["scan_competitor"])
    graph.add_edge("scan_competitor", "analyze")
    graph.add_edge("analyze", "recommend")
    graph.add_edge("recommend", "write_briefing")
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
    })
    return result["briefing"]
