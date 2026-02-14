# tools/__init__.py — Line-by-Line Explanation

This is the **web search module** — two small but critical functions that give the LangGraph agents access to live web data. Without these, the agents would only have their training data (which is months old). With them, they can find what competitors did *yesterday*.

The module provides two Serper endpoints:
- `search_serper()` — standard Google web search (`/search` endpoint)
- `search_serper_news()` — dedicated news search (`/news` endpoint) with time-based filtering

---

```python
import os
import requests
```
**Lines 1-2:** Two imports.
- `os`: To read the `SERPER_API_KEY` environment variable.
- `requests`: The most popular Python HTTP library. It's a third-party package (not in the standard library) that makes HTTP calls simple and readable. The standard library alternative (`urllib.request`) is more verbose and harder to use.

## search_serper() — Web Search

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
- `"https://google.serper.dev/search"`: The Serper web search endpoint. Always HTTPS — never send API keys over unencrypted HTTP.
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
**Line 17:** Parses the response body as JSON and returns it as a Python dict. The Serper `/search` response contains an `"organic"` key with the results:
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

The calling code in `graph.py` accesses `data.get("organic", [])` to get the list of search results.

---

## search_serper_news() — News Search

```python
def search_serper_news(query: str, num_results: int = 10, tbs: str = "qdr:m") -> dict:
```
**Line 20:** The news search function. It has one extra parameter compared to `search_serper`:
- `tbs`: Time-based search filter. This is Google's internal syntax for date filtering:
  - `"qdr:w"` = past week
  - `"qdr:m"` = past month (the default)
  - `"qdr:y"` = past year

This is critical for competitive intelligence — you don't want last year's news mixed in with this week's announcements.

```python
    resp = requests.post(
        "https://google.serper.dev/news",
        ...
        json={"q": query, "num": num_results, "tbs": tbs},
        timeout=15,
    )
```
**Key differences from `search_serper`:**
- Endpoint is `/news` instead of `/search` — this returns news articles rather than general web results
- The `tbs` parameter is included in the request body for date filtering

The `/news` response contains a `"news"` key (not `"organic"`), and each result includes a `"date"` field:
```json
{
  "news": [
    {
      "title": "Parker Hannifin Acquires XYZ Corp",
      "link": "https://reuters.com/...",
      "snippet": "Parker announced today...",
      "date": "2 hours ago"
    },
    ...
  ]
}
```

The calling code in `graph.py` accesses `data.get("news", [])` and includes the date in the tagged output: `[NEWS (2 hours ago)] [Title](URL): Snippet`.

---

## Why Two Endpoints?

The briefing scan uses **both** functions — 9 news queries + 5 web queries per competitor. The reason: Google's regular web search returns a mix of evergreen content (company "About" pages, Wikipedia articles, old blog posts) and actual news. For competitive intelligence, you need *recency* — and the `/news` endpoint is optimised for exactly that. It surfaces breaking news, press releases, and recent articles that would be buried on page 2 of web search results.

The web queries complement the news with signals that don't appear as news articles: job postings (leading indicator of strategy), patent filings, regulatory documents, and company strategy pages.

---

## Design Decisions Worth Noting

**1. No retry logic.** If a search fails, it fails. The calling code in `graph.py` wraps the call in a try/except and records the error. Adding retries here would be an over-engineering decision unless search failures are common. Keep it simple.

**2. No caching.** Every call hits the API. For a competitive intelligence tool, this is correct — you want fresh results every time. For other use cases, you might add caching to avoid paying for repeated identical queries.

**3. Stateless functions.** No class, no instance variables, no global state. Both functions take inputs and return outputs — pure and simple. This makes them easy to test, easy to understand, and easy to call from anywhere.

**4. These are "tools" in the conceptual sense, not in the LangChain tool-calling sense.** LangChain has a formal `@tool` decorator system where the LLM decides when to call tools. This project skips that — the graph nodes call `search_serper()` and `search_serper_news()` directly in their Python code, with hardcoded search queries. This is simpler and more predictable. The LLM never decides whether to search or what to search for (in the pipeline — the deep-dive in `app.py` does let the LLM generate queries, but that's outside LangGraph).

**5. Same timeout and auth pattern for both endpoints.** Consistency across functions reduces cognitive load and makes bugs easier to spot. Both use `os.environ["SERPER_API_KEY"]` (fail-fast on missing key), both have `timeout=15`, both call `raise_for_status()`.
