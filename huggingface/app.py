import json
import os
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

os.environ.pop("LANGSMITH_TRACING", None)
os.environ.pop("LANGCHAIN_TRACING_V2", None)

import gradio as gr
from anthropic import Anthropic
from openai import OpenAI

from graph import run_pipeline

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

openai_client = OpenAI()
anthropic_client = Anthropic()

QUICK_CHAT_SYSTEM = """You are a competitive intelligence analyst assistant. You answer questions
about a competitive intelligence briefing report that was generated for the user.

RULES — follow these strictly:
1. ONLY use information that appears in the briefing report below. Do not add outside knowledge.
2. If the report does not contain enough information to answer the question, say exactly:
   "The briefing does not cover this in detail. Use the 'Research This' button for a deep dive with live web search."
3. Quote or reference specific sections of the report when answering.
4. Be concise and direct. Use bullet points for clarity.
5. Never speculate or infer beyond what the report states.

BRIEFING REPORT:
{briefing}"""

DEEP_DIVE_SYSTEM = """You are a competitive intelligence research analyst conducting a deep dive
investigation. You have been given a user's question, the original briefing report for context,
and fresh web search results.

RULES — follow these strictly:
1. ONLY use facts from the web search results provided below. Do not use your training knowledge.
2. For every claim or fact, cite the source URL in parentheses immediately after the statement.
3. If the search results do not contain relevant information, say so explicitly rather than guessing.
4. Structure your response with clear headers and bullet points.
5. Start with a brief summary, then provide detailed findings.
6. At the end, include a "Sources" section listing all URLs referenced.
7. Never speculate. If information is ambiguous or conflicting across sources, note that explicitly.

ORIGINAL BRIEFING (for context only — prioritize fresh search results):
{briefing}

WEB SEARCH RESULTS:
{search_results}"""


def run_briefing(company: str, industry: str, competitors: str) -> str:
    """Kick off the competitive intelligence crew and return the briefing."""
    if not company or not industry or not competitors:
        return "Please fill in all fields."

    inputs = {
        "company": company.strip(),
        "industry": industry.strip(),
        "competitors": competitors.strip(),
        "current_date": datetime.now().strftime("%Y-%m-%d"),
    }

    result = run_pipeline(
        company=inputs["company"],
        industry=inputs["industry"],
        competitors=inputs["competitors"],
    )
    return result


def list_reports() -> str:
    """Return contents of the most recent briefing."""
    briefing = OUTPUT_DIR / "briefing.md"
    if briefing.exists():
        return briefing.read_text(encoding="utf-8")
    return "No reports generated yet."


def quick_chat(message: str, history: list, briefing_text: str):
    """Answer questions grounded strictly in the briefing report."""
    if not briefing_text:
        return "No briefing loaded yet. Generate or load a briefing first."

    messages = [
        {"role": "system", "content": QUICK_CHAT_SYSTEM.format(briefing=briefing_text)},
    ]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content


def _search_web(queries: list[str]) -> str:
    """Run multiple search queries via Serper and return combined results."""
    import requests
    import os

    api_key = os.environ.get("SERPER_API_KEY", "")
    all_results = []

    for query in queries:
        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": 10},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                timeout=15,
            )
            data = resp.json()
            for item in data.get("organic", []):
                all_results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "query": query,
                })
        except Exception as e:
            all_results.append({"error": str(e), "query": query})

    return json.dumps(all_results, indent=2)


def deep_dive(question: str, briefing_text: str) -> str:
    """Research a question thoroughly using live web search, grounded in sources."""
    if not briefing_text:
        return "No briefing loaded yet. Generate or load a briefing first."

    if not question.strip():
        return "Please enter a question to research."

    # Step 1: Ask the LLM to generate targeted search queries
    query_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You generate web search queries for competitive intelligence research. "
                "Given a user question and briefing context, produce 3-5 specific, varied "
                "search queries that would find detailed, factual information. "
                "Return ONLY a JSON array of query strings, nothing else."
            )},
            {"role": "user", "content": (
                f"Briefing context (for reference):\n{briefing_text[:3000]}\n\n"
                f"User question: {question}\n\n"
                "Generate 3-5 targeted search queries:"
            )},
        ],
        temperature=0.2,
    )

    try:
        queries = json.loads(query_response.choices[0].message.content)
    except json.JSONDecodeError:
        queries = [question]

    # Step 2: Run the searches
    search_results = _search_web(queries)

    # Step 3: Synthesize a grounded answer using Claude
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=0.1,
        system=DEEP_DIVE_SYSTEM.format(
            briefing=briefing_text[:4000],
            search_results=search_results,
        ),
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


# --- Gradio UI ---

danfoss_theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#fef2f2", c100="#fee2e2", c200="#fecaca", c300="#fca5a5",
        c400="#f87171", c500="#E2000F", c600="#dc2626", c700="#b91c1c",
        c800="#991b1b", c900="#7f1d1d", c950="#450a0a",
    ),
)

with gr.Blocks(title="Danfoss Power Solutions — Competitive Intelligence Monitor") as app:
    briefing_state = gr.State("")

    gr.Image(
        value=str(Path(__file__).parent / "csm_logo_danfoss_per_sito_web__5e8f30e4f0.png"),
        show_label=False,
        height=80,
        width=200,
        container=False,
    )
    gr.Markdown("# Danfoss Power Solutions — Competitive Intelligence Monitor")
    gr.Markdown("Enter a company, its industry, and key competitors to generate a strategic intelligence briefing.")

    with gr.Row():
        company = gr.Textbox(label="Company Name", placeholder="e.g. Danfoss Power Solutions")
        industry = gr.Textbox(label="Industry", placeholder="e.g. Hydraulics & Mobile Machinery")

    competitors = gr.Textbox(
        label="Competitors (comma-separated)",
        placeholder="e.g. Parker Hannifin, Bosch Rexroth, Eaton Hydraulics",
    )

    generate_btn = gr.Button("Generate Briefing", variant="primary")
    status = gr.Markdown("*Ready to generate.*")
    output = gr.Markdown(label="Briefing Output")

    def on_generate(company, industry, competitors):
        yield {status: "*Generating briefing — this may take several minutes...*", output: "", briefing_state: ""}
        try:
            result = run_briefing(company, industry, competitors)
            yield {status: "*Briefing complete!*", output: result, briefing_state: result}
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log_path = OUTPUT_DIR / "error.log"
            log_path.write_text(tb, encoding="utf-8")
            yield {status: f"*Error: {e}*", output: f"Full traceback written to {log_path}", briefing_state: ""}

    generate_btn.click(
        fn=on_generate,
        inputs=[company, industry, competitors],
        outputs=[status, output, briefing_state],
    )

    gr.Markdown("---")
    gr.Markdown("## Previous Reports")
    report_btn = gr.Button("Load Latest Report")
    report_output = gr.Markdown()

    def on_load_report():
        text = list_reports()
        return text, text

    report_btn.click(fn=on_load_report, outputs=[report_output, briefing_state])

    # --- Chat & Deep Dive Section ---
    gr.Markdown("---")
    gr.Markdown("## Ask About the Briefing")
    gr.Markdown(
        "**Quick Chat** answers from the report only. "
        "**Research This** runs a live web search for a thorough, source-cited deep dive."
    )

    chatbot = gr.Chatbot(label="Briefing Q&A", height=400)
    chat_input = gr.Textbox(
        label="Your question",
        placeholder="e.g. What are the key product changes from Parker this quarter?",
        lines=2,
    )

    with gr.Row():
        chat_btn = gr.Button("Quick Chat", variant="secondary")
        dive_btn = gr.Button("Research This", variant="primary")

    def on_quick_chat(message, history, briefing_text):
        if not message.strip():
            return history, ""
        history = history + [{"role": "user", "content": message}]
        answer = quick_chat(message, history[:-1], briefing_text)
        history = history + [{"role": "assistant", "content": answer}]
        return history, ""

    def on_deep_dive(message, history, briefing_text):
        if not message.strip():
            return history, ""
        history = history + [{"role": "user", "content": f"[Deep Dive] {message}"}]
        answer = deep_dive(message, briefing_text)
        history = history + [{"role": "assistant", "content": answer}]
        return history, ""

    chat_btn.click(
        fn=on_quick_chat,
        inputs=[chat_input, chatbot, briefing_state],
        outputs=[chatbot, chat_input],
    )
    dive_btn.click(
        fn=on_deep_dive,
        inputs=[chat_input, chatbot, briefing_state],
        outputs=[chatbot, chat_input],
    )


if __name__ == "__main__":
    app.launch(theme=danfoss_theme)
