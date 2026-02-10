# main.py — Line-by-Line Explanation

This is the **CLI entry point** — a simple script that runs the pipeline with hardcoded example inputs. It's useful for testing the pipeline without launching the full Gradio UI.

---

```python
#!/usr/bin/env python
```
**Line 1:** A **shebang line**. On Unix/macOS, if you make this file executable (`chmod +x main.py`) and run it directly (`./main.py`), the OS uses this line to know which interpreter to use. `/usr/bin/env python` means "find `python` on the system PATH" — this is more portable than hardcoding `/usr/bin/python3` because Python might be installed in different locations on different systems. On Windows, this line is ignored (Windows uses file extensions to determine how to run files).

```python
import os
from dotenv import load_dotenv
load_dotenv()
```
**Lines 2-4:** Same pattern as `app.py` — loads environment variables from `.env` before any API clients are created. This is repeated here (and not just in `app.py`) because `main.py` is an **independent entry point**. When you run `uv run competitive_intel` (which calls `main.py`), `app.py` is never loaded, so its `load_dotenv()` never runs. Each entry point must load its own environment.

```python
os.environ.pop("LANGSMITH_TRACING", None)
os.environ.pop("LANGCHAIN_TRACING_V2", None)
```
**Lines 6-7:** Disables LangSmith tracing, same as in `app.py`. Again, repeated because this is a separate entry point.

```python
from competitive_intel.graph import run_pipeline
```
**Line 9:** Imports the pipeline. This import is **after** `load_dotenv()` because importing `graph.py` triggers imports of `langchain_openai` and `langchain_anthropic`, which may read API keys from the environment. The import also triggers the YAML config loading (lines 26-27 of `graph.py`).

```python
def run():
    """Run the competitive intelligence pipeline."""
    try:
        result = run_pipeline(
            company="OpenAI",
            industry="Artificial Intelligence",
            competitors="Anthropic, Google DeepMind, Meta AI, Mistral",
        )
        print(result)
    except Exception as e:
        raise Exception(f"An error occurred while running the pipeline: {e}")
```
**Lines 12-22:** The main function.

- **Lines 15-18:** Calls `run_pipeline` with hardcoded example values. In a real production CLI, you'd accept these as command-line arguments (using `argparse` or `click`). The hardcoded values make this a quick demo/test runner.
- **Line 20:** Prints the final briefing to stdout. Since `run_pipeline` returns a string (the markdown briefing), this outputs the full report to your terminal.
- **Lines 21-22:** Catches any exception and re-raises it with added context. The `f"An error occurred..."` wrapping makes it clear the error came from the pipeline, not from some other part of the system. Using `raise Exception(...)` instead of just `raise` creates a new exception — this means you lose the original traceback, which is a tradeoff (cleaner message vs less debugging info). Using `raise ... from e` would be better practice as it preserves the chain.

```python
if __name__ == "__main__":
    run()
```
**Lines 25-26:** Standard Python idiom. `__name__` is `"__main__"` only when the file is run directly (e.g., `python main.py`). When imported by another module, `__name__` is `"competitive_intel.main"`, so `run()` doesn't execute.

This function is also registered as a **console script** entry point in `pyproject.toml`. When you install the package with `uv sync`, the `competitive_intel` command becomes available on your PATH and calls `run()` directly (bypassing the `if __name__` check).

---

## Why This File Is Important for Learning

Even though it's short, `main.py` demonstrates several patterns:

1. **Entry point hygiene**: Each entry point (`app.py`, `main.py`) independently loads its environment and disables tracing. Don't assume another file has already done setup.

2. **Separation of concerns**: `main.py` doesn't know anything about the graph internals. It calls `run_pipeline()` with simple string arguments. The entire LangGraph complexity is hidden behind a clean function interface.

3. **Error boundaries**: The try/except at the entry point catches all pipeline errors in one place. Individual nodes in the graph don't need their own error handling for the CLI case — it bubbles up here.
