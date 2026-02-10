# tools/__init__.py — Line-by-Line Explanation

This is the **web search tool** — a small but critical utility that gives the LangGraph agents access to live web data. Without this, the agents would only have their training data (which is months old). With it, they can find what competitors did *yesterday*.

---

```python
import os
import requests
```
**Lines 1-2:** Two imports.
- `os`: To read the `SERPER_API_KEY` environment variable.
- `requests`: The most popular Python HTTP library. It's a third-party package (not in the standard library) that makes HTTP calls simple and readable. The standard library alternative (`urllib.request`) is more verbose and harder to use.

```python
def search_serper(query: str, num_results: int = 10) -> dict:
    """Search the web via Serper API."""
```
**Lines 5-6:** Function signature with type hints. `query` is the search string, `num_results` defaults to 10 (how many results to return). Returns a `dict` (the parsed JSON response from Serper).

**What is Serper?** It's a paid API that wraps Google Search. You send it a query, it returns Google search results as structured JSON. It's much cheaper and more reliable than scraping Google directly (which violates Google's ToS and gets you blocked). At ~$1 per 1000 searches, it's a cost-effective way to give LLM agents web access.

```python
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "Content-Type": "application/json",
        },
        json={"q": query, "num": num_results},
        timeout=15,
    )
```
**Lines 7-14:** The HTTP request. Let's break down each part:

- `requests.post(...)`: Makes an HTTP POST request. Serper uses POST (not GET) for search queries — this is their API design choice (POST bodies can be larger than URL query strings).
- `"https://google.serper.dev/search"`: The Serper API endpoint. Always HTTPS — never send API keys over unencrypted HTTP.
- `"X-API-KEY": os.environ["SERPER_API_KEY"]`: Authentication via a custom HTTP header. Note this uses `os.environ["SERPER_API_KEY"]` (with square brackets), not `.get()`. This means it will raise a **`KeyError`** if the key isn't set. This is intentional — it's better to fail loudly and immediately than to send an unauthenticated request and get a confusing 401 error back. Compare this to `app.py`'s `_search_web` which uses `.get("SERPER_API_KEY", "")` — a more lenient approach that lets the error happen at the API level instead.
- `"Content-Type": "application/json"`: Tells the server we're sending JSON in the request body.
- `json={"q": query, "num": num_results}`: The request body. The `json=` parameter in requests automatically serializes the dict to JSON and sets the Content-Type header (so our explicit Content-Type header is technically redundant, but it's good to be explicit).
- `timeout=15`: Give up after 15 seconds. This is **critical** in production code. Without a timeout, if Serper's server hangs, your entire application hangs forever. The 15-second choice is a balance — long enough for a slow search, short enough that users don't wait too long.

```python
    resp.raise_for_status()
```
**Line 16:** This is a `requests` pattern that checks the HTTP status code. If the response is 4xx (client error) or 5xx (server error), it raises an `HTTPError` exception. Without this line, a failed request would silently return bad data. For example, if your API key is wrong, Serper returns a 401 response — `raise_for_status()` turns that into an exception that gets caught and reported.

Common status codes you might see:
- 200: Success
- 401: Invalid API key
- 429: Rate limited (too many requests)
- 500: Serper's servers are down

```python
    return resp.json()
```
**Line 17:** Parses the response body as JSON and returns it as a Python dict. The Serper response looks something like:
```json
{
  "organic": [
    {
      "title": "Parker Hannifin Launches New Hydraulic Pump",
      "link": "https://example.com/...",
      "snippet": "Parker announced today..."
    },
    ...
  ]
}
```

The calling code in `graph.py` (line 109) accesses `data.get("organic", [])` to get the list of search results.

---

## Design Decisions Worth Noting

**1. No retry logic.** If a search fails, it fails. The calling code in `graph.py` wraps the call in a try/except and records the error. Adding retries here would be an over-engineering decision unless search failures are common. Keep it simple.

**2. No caching.** Every call hits the API. For a competitive intelligence tool, this is correct — you want fresh results every time. For other use cases, you might add caching to avoid paying for repeated identical queries.

**3. Stateless function.** No class, no instance variables, no global state. The function takes inputs and returns outputs — pure and simple. This makes it easy to test, easy to understand, and easy to call from anywhere.

**4. This is a "tool" in the conceptual sense, not in the LangChain tool-calling sense.** LangChain has a formal `@tool` decorator system where the LLM decides when to call tools. This project skips that — the graph nodes call `search_serper()` directly in their Python code, with hardcoded search queries. This is simpler and more predictable. The LLM never decides whether to search or what to search for (in the pipeline — the deep-dive in `app.py` does let the LLM generate queries, but that's outside LangGraph).
