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
