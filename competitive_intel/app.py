import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

os.environ.pop("LANGSMITH_TRACING", None)
os.environ.pop("LANGCHAIN_TRACING_V2", None)

import gradio as gr
import markdown as md_lib
from anthropic import Anthropic
from fpdf import FPDF
from openai import OpenAI

from competitive_intel.graph import run_pipeline, run_annual_report_pipeline

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

openai_client = OpenAI()
anthropic_client = Anthropic()


LOGO_PATH = Path(__file__).parent / "Vickers_by_Danfoss-Logo.png"
FONT_DIR = Path(__file__).parent / "fonts"
DANFOSS_RED = (226, 0, 15)
DANFOSS_DARK = (50, 50, 50)
DANFOSS_GREY = (120, 120, 120)


def _markdown_to_pdf(md_text: str, output_path: Path) -> Path:
    from fpdf.fonts import TextStyle

    html = md_lib.markdown(md_text, extensions=["tables", "fenced_code"])
    html = re.sub(r"!\[.*?\]\(.*?\)", "", html)
    # Convert bare URLs (not already inside href="") into clickable links
    html = re.sub(
        r'(?<!href=")(https?://[^\s<)\]]+)',
        r'<a href="\1">\1</a>',
        html,
    )
    # Decode HTML entities in headings before title-casing so &amp; becomes &
    import html as html_lib
    def _title_case_heading(match):
        tag = match.group(1)
        content = match.group(2)
        # Decode HTML entities first (e.g. &amp; -> &)
        decoded = html_lib.unescape(content)
        # Strip any inner HTML tags for the uppercase check
        plain = re.sub(r"<[^>]+>", "", decoded)
        if sum(1 for c in plain if c.isupper()) > len(plain) * 0.5:
            decoded = decoded.title()
        return f"<{tag}>{decoded}</{tag}>"
    html = re.sub(r"<(h[1-6])>(.*?)</\1>", _title_case_heading, html)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margin(20)

    # Register fonts
    if FONT_DIR.exists():
        pdf.add_font("report", "", str(FONT_DIR / "DejaVuSans.ttf"), uni=True)
        pdf.add_font("report", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"), uni=True)
        pdf.add_font("report", "I", str(FONT_DIR / "DejaVuSans-Oblique.ttf"), uni=True)
        pdf.add_font("report", "BI", str(FONT_DIR / "DejaVuSans-BoldOblique.ttf"), uni=True)
        font = "report"
    else:
        font = "Helvetica"

    tag_styles = {
        "h1": TextStyle(font_family=font, font_style="B", font_size_pt=18,
                        color=DANFOSS_RED, t_margin=12, b_margin=4),
        "h2": TextStyle(font_family=font, font_style="B", font_size_pt=14,
                        color=DANFOSS_DARK, t_margin=10, b_margin=3),
        "h3": TextStyle(font_family=font, font_style="B", font_size_pt=12,
                        color=DANFOSS_DARK, t_margin=8, b_margin=2),
        "h4": TextStyle(font_family=font, font_style="B", font_size_pt=10,
                        color=DANFOSS_DARK, t_margin=6, b_margin=2),
        "p": TextStyle(font_family=font, font_size_pt=9,
                       color=DANFOSS_DARK, b_margin=3),
        "li": TextStyle(font_family=font, font_size_pt=9,
                        color=DANFOSS_DARK),
        "a": TextStyle(font_family=font, font_size_pt=9,
                       color=(0, 80, 160)),
        "strong": TextStyle(font_family=font, font_style="B", font_size_pt=9,
                            color=DANFOSS_DARK),
        "em": TextStyle(font_family=font, font_style="I", font_size_pt=9,
                        color=DANFOSS_GREY),
    }

    # First page with logo and header
    pdf.add_page()
    logo_h = 20
    if LOGO_PATH.exists():
        pdf.image(str(LOGO_PATH), x=20, y=10, h=logo_h)
    line_y = 10 + logo_h + 3
    pdf.set_draw_color(*DANFOSS_RED)
    pdf.set_line_width(0.8)
    pdf.line(20, line_y, pdf.w - 20, line_y)
    pdf.set_y(line_y + 4)

    pdf.set_font(font, size=9)
    pdf.write_html(html, tag_styles=tag_styles, li_prefix_color=DANFOSS_RED)
    pdf.output(str(output_path))
    return output_path

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

with gr.Blocks(title="Danfoss Power Solutions — Competitive Intelligence Monitor", theme=danfoss_theme) as app:
    briefing_state = gr.State("")

    gr.Image(
        value=str(Path(__file__).parent / "Vickers_by_Danfoss-Logo.png"),
        show_label=False,
        height=100,
        width=300,
        container=False,
    )
    gr.Markdown("# Danfoss Power Solutions — Competitive Intelligence Monitor")
    gr.Markdown("Enter a company, its industry, and key competitors to generate a strategic intelligence briefing.")

    with gr.Row():
        company = gr.Textbox(label="Company Name", placeholder="e.g. Danfoss Power Solutions")
        industry = gr.Textbox(label="Industry", placeholder="e.g. Hydraulics & Mobile Machinery")

    competitors = gr.Textbox(
        label="Competitors (comma-separated)",
        placeholder="e.g. Parker Hannifin, Bosch Rexroth, ATOS",
    )

    gr.Markdown("---")
    gr.Markdown(
        "### Two independent pipelines — run either or both at the same time\n\n"
        "Both pipelines use the company, industry, and competitors entered above. "
        "They search different sources and produce separate reports, so you can launch them concurrently."
    )

    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown(
                "#### Competitive Briefing\n"
                "Scans recent news, product launches, pricing changes, and market moves. "
                "Produces an executive-ready weekly intelligence briefing with strategic recommendations."
            )
            generate_btn = gr.Button("Generate Briefing", variant="primary", size="lg")
            status = gr.Markdown("*Ready to generate.*")

        with gr.Column():
            gr.Markdown(
                "#### Annual Report Deep Dive\n"
                "Analyses the latest annual reports, SEC filings, financial health, market share, "
                "M&A activity, patents, hiring, customer reviews, and exploitable weaknesses. "
                "Covers 15 strategic focus areas per competitor."
            )
            annual_btn = gr.Button("Run Annual Report Analysis", variant="primary", size="lg")
            annual_status = gr.Markdown("*Ready to run.*")

    # --- Briefing Output ---
    gr.Markdown("---")
    gr.Markdown("### Competitive Briefing Output")
    output = gr.Markdown(label="Briefing Output")
    briefing_pdf_btn = gr.Button("Download Briefing as PDF", variant="secondary", visible=False)
    briefing_pdf_file = gr.File(label="Briefing PDF", visible=False)

    def on_generate(company, industry, competitors):
        yield {
            status: "*Generating briefing — this may take several minutes...*",
            output: "", briefing_state: "",
            briefing_pdf_btn: gr.update(visible=False),
            briefing_pdf_file: gr.update(visible=False, value=None),
        }
        try:
            result = run_briefing(company, industry, competitors)
            yield {
                status: "*Briefing complete!*",
                output: result, briefing_state: result,
                briefing_pdf_btn: gr.update(visible=True),
                briefing_pdf_file: gr.update(visible=False, value=None),
            }
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log_path = OUTPUT_DIR / "error.log"
            log_path.write_text(tb, encoding="utf-8")
            yield {
                status: f"*Error: {e}*",
                output: f"Full traceback written to {log_path}", briefing_state: "",
                briefing_pdf_btn: gr.update(visible=False),
                briefing_pdf_file: gr.update(visible=False, value=None),
            }

    generate_btn.click(
        fn=on_generate,
        inputs=[company, industry, competitors],
        outputs=[status, output, briefing_state, briefing_pdf_btn, briefing_pdf_file],
    )

    def on_briefing_pdf(briefing_text):
        if not briefing_text:
            return gr.update(visible=False, value=None)
        path = _markdown_to_pdf(briefing_text, OUTPUT_DIR / "briefing.pdf")
        return gr.update(visible=True, value=str(path))

    briefing_pdf_btn.click(
        fn=on_briefing_pdf,
        inputs=[briefing_state],
        outputs=[briefing_pdf_file],
    )

    # --- Annual Report Output ---
    gr.Markdown("---")
    gr.Markdown("### Annual Report Deep Dive Output")
    annual_output = gr.Markdown(label="Annual Report Analysis")
    annual_pdf_btn = gr.Button("Download Annual Report as PDF", variant="secondary", visible=False)
    annual_pdf_file = gr.File(label="Annual Report PDF", visible=False)

    annual_report_state = gr.State("")

    def run_annual_report_analysis(company, industry, competitors):
        if not company or not industry or not competitors:
            yield {
                annual_status: "*Please fill in all fields above.*",
                annual_output: "", annual_report_state: "",
                annual_pdf_btn: gr.update(visible=False),
                annual_pdf_file: gr.update(visible=False, value=None),
            }
            return
        yield {
            annual_status: "*Running annual report deep dive — this may take several minutes...*",
            annual_output: "", annual_report_state: "",
            annual_pdf_btn: gr.update(visible=False),
            annual_pdf_file: gr.update(visible=False, value=None),
        }
        try:
            result = run_annual_report_pipeline(
                company=company.strip(),
                industry=industry.strip(),
                competitors=competitors.strip(),
            )
            yield {
                annual_status: "*Annual report analysis complete!*",
                annual_output: result, annual_report_state: result,
                annual_pdf_btn: gr.update(visible=True),
                annual_pdf_file: gr.update(visible=False, value=None),
            }
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log_path = OUTPUT_DIR / "annual_report_error.log"
            log_path.write_text(tb, encoding="utf-8")
            yield {
                annual_status: f"*Error: {e}*",
                annual_output: f"Full traceback written to {log_path}",
                annual_report_state: "",
                annual_pdf_btn: gr.update(visible=False),
                annual_pdf_file: gr.update(visible=False, value=None),
            }

    annual_btn.click(
        fn=run_annual_report_analysis,
        inputs=[company, industry, competitors],
        outputs=[annual_status, annual_output, annual_report_state, annual_pdf_btn, annual_pdf_file],
    )

    def on_annual_pdf(report_text):
        if not report_text:
            return gr.update(visible=False, value=None)
        path = _markdown_to_pdf(report_text, OUTPUT_DIR / "annual_report_analysis.pdf")
        return gr.update(visible=True, value=str(path))

    annual_pdf_btn.click(
        fn=on_annual_pdf,
        inputs=[annual_report_state],
        outputs=[annual_pdf_file],
    )

    # --- Previous Reports ---
    gr.Markdown("---")
    gr.Markdown("## Previous Reports")
    report_btn = gr.Button("Load Latest Briefing")
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
    app.launch()
