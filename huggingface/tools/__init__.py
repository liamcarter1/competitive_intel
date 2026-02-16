import os
import requests


def search_serper(query: str, num_results: int = 10) -> dict:
    """Search the web via Serper API."""
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "Content-Type": "application/json",
        },
        json={"q": query, "num": num_results},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def search_serper_news(query: str, num_results: int = 10, tbs: str = "qdr:m") -> dict:
    """Search news articles via Serper News API.

    Args:
        query: Search query string.
        num_results: Number of results to return.
        tbs: Time-based filter. 'qdr:w' = past week, 'qdr:m' = past month,
             'qdr:y' = past year. Defaults to past month.
    """
    resp = requests.post(
        "https://google.serper.dev/news",
        headers={
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "Content-Type": "application/json",
        },
        json={"q": query, "num": num_results, "tbs": tbs},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()
