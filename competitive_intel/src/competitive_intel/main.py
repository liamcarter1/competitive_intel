#!/usr/bin/env python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ.pop("LANGSMITH_TRACING", None)
os.environ.pop("LANGCHAIN_TRACING_V2", None)

from competitive_intel.graph import run_pipeline


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


if __name__ == "__main__":
    run()
