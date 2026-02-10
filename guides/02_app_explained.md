# app.py — Line-by-Line Explanation

This file is the **user-facing layer** of the application. It builds a web interface using Gradio and connects it to three different AI capabilities: briefing generation (via the LangGraph pipeline), quick Q&A (via OpenAI), and deep-dive research (via web search + Claude). While `graph.py` is about orchestrating agents, `app.py` is about connecting those agents to a real user interface.

---

## Imports and Environment Setup (Lines 1-17)

```python
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
```
**Lines 1-5:** Standard library imports.
- `json`: For serializing/deserializing search results in the deep-dive feature.
- `os`: For managing environment variables.
- `warnings`: Imported but not explicitly used here (may suppress warnings from dependencies).
- `datetime`: To timestamp briefing inputs.
- `Path`: For building file paths to the logo image and output directory.

```python
from dotenv import load_dotenv
load_dotenv()
```
**Lines 7-8:** Loads environment variables from a `.env` file into the process environment. `load_dotenv()` searches for a `.env` file starting from the current working directory and going up. This is the standard way to manage secrets in development — you put your API keys in `.env` (which is gitignored) and `load_dotenv()` makes them available via `os.environ`. In production (like Hugging Face Spaces), you set these as environment variables directly, and `load_dotenv()` is a harmless no-op if no `.env` file exists.

**Why call it immediately at the top of the file?** Because the OpenAI and Anthropic clients (lines 22-23) read their API keys from environment variables when they're instantiated. If `load_dotenv()` ran later, those clients would fail with missing key errors.

```python
os.environ.pop("LANGSMITH_TRACING", None)
os.environ.pop("LANGCHAIN_TRACING_V2", None)
```
**Lines 10-11:** Removes LangSmith tracing variables if they exist. LangSmith is LangChain's observability platform — if these variables are set (even to `"false"`), LangChain tries to connect to LangSmith servers, which slows things down and can cause errors if you don't have an account. `.pop(key, None)` removes the key if it exists and does nothing if it doesn't (the `None` is the default return value when the key is missing — without it, `.pop()` would raise a `KeyError`).

```python
import gradio as gr
from anthropic import Anthropic
from openai import OpenAI
```
**Lines 13-15:** Third-party imports. These are placed **after** `load_dotenv()` intentionally — some libraries read environment variables at import time.
- `gradio`: The web UI framework. `gr` is the conventional alias.
- `Anthropic`: The official Anthropic Python SDK client. Note this is the **raw SDK**, not LangChain's wrapper. The chat and deep-dive features use the raw SDKs directly because they don't need LangGraph's orchestration — they're simple single-call interactions.
- `OpenAI`: The official OpenAI Python SDK client.

```python
from competitive_intel.graph import run_pipeline
```
**Line 17:** Imports the pipeline runner from `graph.py`. This is the bridge between the UI and the LangGraph pipeline. When a user clicks "Generate Briefing," this function gets called.

---

## Module-Level Setup (Lines 19-23)

```python
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
```
**Lines 19-20:** Creates the output directory path and ensures it exists. `Path(__file__).parent` gives the directory containing `app.py`. `exist_ok=True` means "don't raise an error if the directory already exists." This runs at import time, so the output directory is always ready.

```python
openai_client = OpenAI()
anthropic_client = Anthropic()
```
**Lines 22-23:** Creates SDK clients. Both read their API keys from environment variables automatically (`OPENAI_API_KEY` and `ANTHROPIC_API_KEY`). These are **module-level singletons** — one client instance is shared across all requests. This is efficient because the clients maintain connection pools internally.

**Why raw SDK clients here instead of LangChain wrappers?** The chat and deep-dive features are simple request-response interactions. LangChain's value is in orchestration (chains, graphs, agents). For a single API call, the raw SDK is simpler and has less overhead.

---

## System Prompts (Lines 25-56)

```python
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
```
**Lines 25-37:** The system prompt for the quick chat feature. The `{briefing}` placeholder gets filled with the actual briefing text at runtime via `.format()`. The rules are designed to keep the LLM **grounded** — it can only answer from the report, not from its training data. This is a form of Retrieval-Augmented Generation (RAG) where the "retrieved" document is the briefing report. Rule 2 is particularly clever — it directs users to the deep-dive feature when the chat can't help, creating a natural workflow escalation.

```python
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
```
**Lines 39-56:** The system prompt for deep-dive research. This has **two** placeholders — `{briefing}` for context and `{search_results}` for fresh web data. The rules enforce citation discipline (rule 2) and source grounding (rule 1). The briefing is included as context so Claude understands what the user is already working with, but the prompt explicitly says to prioritize fresh search results.

---

## run_briefing Function (Lines 59-76)

```python
def run_briefing(company: str, industry: str, competitors: str) -> str:
    """Kick off the competitive intelligence crew and return the briefing."""
    if not company or not industry or not competitors:
        return "Please fill in all fields."
```
**Lines 59-62:** Input validation. `if not company` catches both empty strings (`""`) and `None`. This is the **UI boundary** — the place where untrusted user input enters the system. The validation is simple here (non-empty check), but it prevents the pipeline from running with missing data, which would produce garbage output and waste API credits.

```python
    inputs = {
        "company": company.strip(),
        "industry": industry.strip(),
        "competitors": competitors.strip(),
        "current_date": datetime.now().strftime("%Y-%m-%d"),
    }
```
**Lines 64-69:** Cleans input and adds the current date. `.strip()` removes leading/trailing whitespace — users often accidentally add spaces. `strftime("%Y-%m-%d")` formats the date as `"2026-02-01"`.

```python
    result = run_pipeline(
        company=inputs["company"],
        industry=inputs["industry"],
        competitors=inputs["competitors"],
    )
    return result
```
**Lines 71-76:** Calls the LangGraph pipeline from `graph.py` and returns the briefing text. This is where the Gradio UI connects to the agent pipeline. The function blocks until the entire pipeline completes (all scans, analysis, recommendations, and writing).

---

## list_reports Function (Lines 79-84)

```python
def list_reports() -> str:
    """Return contents of the most recent briefing."""
    briefing = OUTPUT_DIR / "briefing.md"
    if briefing.exists():
        return briefing.read_text(encoding="utf-8")
    return "No reports generated yet."
```
**Lines 79-84:** Reads the most recent briefing from disk. The pipeline saves briefings to `output/briefing.md`, and this function reads them back. The `.exists()` check prevents a `FileNotFoundError` if no briefing has been generated yet.

---

## quick_chat Function (Lines 87-104)

```python
def quick_chat(message: str, history: list, briefing_text: str):
    """Answer questions grounded strictly in the briefing report."""
    if not briefing_text:
        return "No briefing loaded yet. Generate or load a briefing first."
```
**Lines 87-90:** Guard clause. If there's no briefing loaded, there's nothing to chat about.

```python
    messages = [
        {"role": "system", "content": QUICK_CHAT_SYSTEM.format(briefing=briefing_text)},
    ]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})
```
**Lines 92-97:** Builds the message list for the OpenAI API. This is the **chat completions** format:
1. First message is `system` — sets the persona and includes the full briefing text.
2. Then all previous messages from the conversation history are replayed. This gives the model **memory** of the conversation — without this, each message would be answered in isolation.
3. Finally, the new user message is appended.

**Why replay the full history?** LLMs are stateless. Every API call is independent. To create the illusion of a conversation, you must send the entire conversation history every time. This is how all chatbots work under the hood.

```python
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content
```
**Lines 99-104:** Calls GPT-4o-mini with very low temperature (0.1) for consistent, factual answers. `response.choices[0].message.content` navigates the OpenAI response structure: `choices` is a list (usually with one element), each choice has a `message`, and the `message` has `content` (the actual text).

**Why GPT-4o-mini for chat?** It's fast and cheap. Quick chat is supposed to be snappy — users don't want to wait 10 seconds for a simple lookup. The heavy model (Claude) is reserved for deep-dive research.

---

## _search_web Function (Lines 107-134)

```python
def _search_web(queries: list[str]) -> str:
    """Run multiple search queries via Serper and return combined results."""
    import requests
    import os
```
**Lines 107-110:** The leading underscore in `_search_web` is a Python convention meaning "private" — this function is only used internally by `deep_dive()`, not called from outside. The imports inside the function (rather than at the top of the file) are a minor optimization — `requests` is only loaded when this function is actually called, not when the module is imported. This slightly speeds up app startup.

```python
    api_key = os.environ.get("SERPER_API_KEY", "")
    all_results = []
```
**Lines 112-113:** Gets the API key with a fallback to empty string. Using `.get()` instead of `os.environ["SERPER_API_KEY"]` avoids a `KeyError` crash if the key isn't set — the API call will just fail with an auth error instead, which is caught by the try/except below.

```python
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
```
**Lines 115-132:** Loops through each query, calls the Serper API, and collects results as dicts. Key details:
- `requests.post(... json=...)`: The `json` parameter automatically serializes the dict to JSON and sets the Content-Type header (though we set it explicitly too).
- `timeout=15`: Gives up after 15 seconds. Without this, a hanging request could freeze the entire app indefinitely.
- `data.get("organic", [])`: Gets organic search results. No `[:5]` limit here (unlike in `graph.py`'s scan) — deep-dive wants comprehensive results.
- Each result is stored as a dict with `title`, `link`, `snippet`, and the `query` that produced it.
- The `try/except` catches network errors, API errors, JSON parse errors — anything. The error is recorded as a result dict so the LLM sees it and can mention that some searches failed.

```python
    return json.dumps(all_results, indent=2)
```
**Line 134:** Converts the results list to a formatted JSON string. `indent=2` makes it human-readable (and LLM-readable). This string gets injected into the deep-dive system prompt.

---

## deep_dive Function (Lines 137-183)

This is the most sophisticated function in `app.py`. It implements a three-step pipeline: LLM generates search queries -> Serper runs the searches -> Claude synthesizes the answer.

```python
def deep_dive(question: str, briefing_text: str) -> str:
    if not briefing_text:
        return "No briefing loaded yet. Generate or load a briefing first."
    if not question.strip():
        return "Please enter a question to research."
```
**Lines 137-143:** Guard clauses for empty inputs.

```python
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
```
**Lines 146-162:** **Step 1: Query Generation.** Instead of hardcoding search queries (like `graph.py` does for scans), this uses an LLM to generate queries tailored to the user's specific question. This makes sense here because the user's question is unpredictable — you can't pre-template it.

- `briefing_text[:3000]`: Only sends the first 3000 characters of the briefing as context. This keeps the prompt small (saving tokens/cost) while giving the LLM enough context to generate relevant queries.
- `"Return ONLY a JSON array"`: This is a structured output instruction. We want machine-parseable output, not prose.
- `temperature=0.2`: Low randomness for consistent query quality.

```python
    try:
        queries = json.loads(query_response.choices[0].message.content)
    except json.JSONDecodeError:
        queries = [question]
```
**Lines 164-167:** Parses the LLM's response as JSON. If the LLM didn't return valid JSON (it usually does, but LLMs can be unpredictable), the fallback is to just use the original question as the search query. This is a **graceful degradation** pattern — the feature still works even if one step partially fails.

```python
    search_results = _search_web(queries)
```
**Line 170:** **Step 2: Search.** Runs all the generated queries through the Serper API.

```python
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
```
**Lines 173-183:** **Step 3: Synthesis.** Claude takes the search results and produces a cited, structured answer.

Notice the Anthropic SDK has a slightly different API than OpenAI:
- `system` is a separate parameter (not a message in the list). This is an Anthropic API design choice — they treat the system prompt as a first-class parameter.
- `max_tokens=4096`: Maximum response length. Claude requires this parameter (OpenAI defaults it).
- `response.content[0].text`: Anthropic's response structure uses `.content` (a list of content blocks) and `.text` (the text of a block), vs OpenAI's `.choices[0].message.content`.

---

## Gradio UI (Lines 186-299)

This section builds the entire web interface.

```python
danfoss_theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#fef2f2", c100="#fee2e2", c200="#fecaca", c300="#fca5a5",
        c400="#f87171", c500="#E2000F", c600="#dc2626", c700="#b91c1c",
        c800="#991b1b", c900="#7f1d1d", c950="#450a0a",
    ),
)
```
**Lines 188-194:** Defines a custom Gradio theme. `gr.themes.Soft` is a built-in base theme. The `primary_hue` override changes the primary color to a red spectrum (Danfoss brand red is `#E2000F` at the c500 midpoint). The c50-c950 values are light-to-dark shades used for different UI elements (hover states, borders, backgrounds, etc.).

```python
with gr.Blocks(title="Danfoss Power Solutions — Competitive Intelligence Monitor", theme=danfoss_theme) as app:
```
**Line 196:** `gr.Blocks` is Gradio's flexible layout system (vs `gr.Interface` which is simpler but less customizable). The `with ... as app` context manager means all components created inside the block are children of this app. `title` sets the browser tab title. `theme` applies our custom color scheme.

```python
    briefing_state = gr.State("")
```
**Line 197:** `gr.State` is Gradio's mechanism for **session state** — data that persists across interactions but is scoped to one user session. It's invisible in the UI. Here it stores the current briefing text so the chat and deep-dive features can access it. The initial value is `""` (no briefing loaded).

**Why not use a global variable?** If two users use the app simultaneously, a global variable would be shared between them. `gr.State` is per-session, so each user has their own briefing.

```python
    gr.Image(
        value=str(Path(__file__).parent / "Vickers_by_Danfoss-Logo.png"),
        show_label=False,
        height=100,
        width=300,
        container=False,
    )
```
**Lines 199-205:** Displays the company logo. `value` is the file path to the image. `show_label=False` hides the "Image" label that Gradio adds by default. `container=False` removes the bounding box/border around the image.

```python
    gr.Markdown("# Danfoss Power Solutions — Competitive Intelligence Monitor")
    gr.Markdown("Enter a company, its industry, and key competitors to generate a strategic intelligence briefing.")
```
**Lines 206-207:** Markdown components render as formatted text. The `#` creates an H1 heading.

```python
    with gr.Row():
        company = gr.Textbox(label="Company Name", placeholder="e.g. Danfoss Power Solutions")
        industry = gr.Textbox(label="Industry", placeholder="e.g. Hydraulics & Mobile Machinery")
```
**Lines 209-211:** `gr.Row()` places its children side by side (instead of stacked vertically). So the Company and Industry text boxes appear on the same row. `label` is the text above the input, `placeholder` is the greyed-out hint text inside it.

```python
    competitors = gr.Textbox(
        label="Competitors (comma-separated)",
        placeholder="e.g. Parker Hannifin, Bosch Rexroth, Eaton Hydraulics",
    )
```
**Lines 213-216:** This is outside the `gr.Row()` block, so it gets its own full-width row.

```python
    generate_btn = gr.Button("Generate Briefing", variant="primary")
    status = gr.Markdown("*Ready to generate.*")
    output = gr.Markdown(label="Briefing Output")
```
**Lines 218-220:** A button, a status indicator, and an output area. `variant="primary"` makes the button visually prominent (uses the theme's primary color — Danfoss red). The `status` Markdown starts with italicized "Ready to generate."

```python
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
```
**Lines 222-232:** The callback for the Generate button. This is a **generator function** (uses `yield` instead of `return`). In Gradio, generator callbacks enable **streaming updates** — each `yield` updates the UI immediately without waiting for the function to finish.

- First `yield` (line 223): Updates status to "Generating..." and clears output. The user sees this immediately.
- Second `yield` (line 226): After the pipeline completes, shows the result and stores it in `briefing_state`.
- The dict syntax `{status: "...", output: "..."}` maps Gradio components to their new values. This is how you update multiple components in one yield.
- The `except` block (lines 227-232): Catches any pipeline error, writes the full traceback to a log file for debugging, and shows a user-friendly error message.

```python
    generate_btn.click(
        fn=on_generate,
        inputs=[company, industry, competitors],
        outputs=[status, output, briefing_state],
    )
```
**Lines 234-238:** Wires the button to the callback. When clicked: read values from `inputs` components, pass them to `fn`, and update `outputs` components with the results. This is Gradio's **event system** — it's declarative, not imperative.

```python
    report_btn.click(fn=on_load_report, outputs=[report_output, briefing_state])
```
**Line 249:** The "Load Latest Report" button. `on_load_report` reads the saved briefing file and returns its contents to both the display and the state (so the chat features can use it).

```python
    chatbot = gr.Chatbot(label="Briefing Q&A", height=400, type="messages")
```
**Line 259:** A chat UI component. `type="messages"` means it expects the OpenAI-style `[{"role": "user", "content": "..."}]` format (vs the older tuple format). `height=400` sets the scrollable area height in pixels.

```python
    def on_quick_chat(message, history, briefing_text):
        if not message.strip():
            return history, ""
        history = history + [{"role": "user", "content": message}]
        answer = quick_chat(message, history[:-1], briefing_text)
        history = history + [{"role": "assistant", "content": answer}]
        return history, ""
```
**Lines 270-276:** The quick chat callback.
- Line 272: Ignore empty messages.
- Line 273: Append the user message to history. Uses `history + [...]` (creates new list) instead of `history.append(...)` (mutates in place). This is important in Gradio — mutating state objects directly can cause bugs.
- Line 274: Call `quick_chat` with `history[:-1]` — excluding the message we just added, because `quick_chat` will add it itself.
- Line 275: Append the assistant's response.
- Line 276: Return updated history and `""` to clear the input box.

```python
    def on_deep_dive(message, history, briefing_text):
        if not message.strip():
            return history, ""
        history = history + [{"role": "user", "content": f"[Deep Dive] {message}"}]
        answer = deep_dive(message, briefing_text)
        history = history + [{"role": "assistant", "content": answer}]
        return history, ""
```
**Lines 278-284:** Same pattern as quick chat, but the user message is prefixed with `[Deep Dive]` so the conversation history shows which messages were regular chat vs deep-dive research.

```python
if __name__ == "__main__":
    app.launch()
```
**Lines 298-299:** Starts the Gradio server when the script is run directly. `app.launch()` starts a local web server (default port 7860) and opens the browser. The `if __name__ == "__main__"` guard ensures this only happens when running `python app.py` directly, not when importing the module.
