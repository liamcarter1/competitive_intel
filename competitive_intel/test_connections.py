"""
Test script to diagnose connection issues with API services.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

def test_api_key(name: str, env_var: str) -> bool:
    """Test if an API key is set."""
    key = os.environ.get(env_var)
    if not key:
        print(f"[X] {name}: API key not found (env var: {env_var})")
        return False
    print(f"[OK] {name}: API key found ({len(key)} chars)")
    return True

def test_openai():
    """Test OpenAI API connection."""
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
        print("[OK] OpenAI: Connection successful")
        return True
    except Exception as e:
        print(f"[X] OpenAI: Connection failed - {e}")
        return False

def test_anthropic():
    """Test Anthropic API connection."""
    try:
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}],
        )
        print("[OK] Anthropic: Connection successful")
        return True
    except Exception as e:
        print(f"[X] Anthropic: Connection failed - {e}")
        return False

def test_serper():
    """Test Serper API connection."""
    try:
        import requests
        api_key = os.environ.get("SERPER_API_KEY", "")

        # Test web search endpoint
        resp = requests.post(
            "https://google.serper.dev/search",
            json={"q": "test", "num": 1},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        print("[OK] Serper /search: Connection successful")

        # Test news endpoint
        resp = requests.post(
            "https://google.serper.dev/news",
            json={"q": "test", "num": 1, "tbs": "qdr:m"},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        print("[OK] Serper /news: Connection successful")
        return True
    except Exception as e:
        print(f"[X] Serper: Connection failed - {e}")
        return False

def test_network():
    """Test basic network connectivity."""
    try:
        import requests
        response = requests.get("https://www.google.com", timeout=5)
        print("[OK] Network: Internet connection available")
        return True
    except Exception as e:
        print(f"[X] Network: No internet connection - {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("COMPETITIVE INTEL - CONNECTION DIAGNOSTIC")
    print("=" * 60)
    print()

    print("1. Checking API Keys...")
    print("-" * 60)
    has_openai = test_api_key("OpenAI", "OPENAI_API_KEY")
    has_anthropic = test_api_key("Anthropic", "ANTHROPIC_API_KEY")
    has_serper = test_api_key("Serper", "SERPER_API_KEY")
    print()

    print("2. Testing Network Connectivity...")
    print("-" * 60)
    has_network = test_network()
    print()

    if not has_network:
        print("[!] No internet connection. Check your network settings.")
        sys.exit(1)

    print("3. Testing API Connections...")
    print("-" * 60)

    results = {
        "OpenAI": test_openai() if has_openai else False,
        "Anthropic": test_anthropic() if has_anthropic else False,
        "Serper": test_serper() if has_serper else False,
    }
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_ok = all(results.values())
    if all_ok:
        print("[OK] All API connections working!")
        print("\nYou're ready to run the competitive intelligence pipeline.")
    else:
        print("[!] Some connections failed:")
        for service, ok in results.items():
            if not ok:
                print(f"  - {service}")
        print("\nCommon fixes:")
        print("  1. Check your .env file has all three API keys")
        print("  2. Verify API keys are valid (not expired/revoked)")
        print("  3. Check for firewall/proxy blocking API requests")
        print("  4. Verify you have API credits remaining")

    sys.exit(0 if all_ok else 1)
