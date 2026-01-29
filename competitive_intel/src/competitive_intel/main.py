#!/usr/bin/env python
import sys
import warnings
from datetime import datetime

from competitive_intel.crew import CompetitiveIntel

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """Run the crew."""
    inputs = {
        'company': 'OpenAI',
        'industry': 'Artificial Intelligence',
        'competitors': 'Anthropic, Google DeepMind, Meta AI, Mistral',
        'current_date': datetime.now().strftime('%Y-%m-%d'),
    }
    try:
        CompetitiveIntel().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """Train the crew for a given number of iterations."""
    inputs = {
        'company': 'OpenAI',
        'industry': 'Artificial Intelligence',
        'competitors': 'Anthropic, Google DeepMind, Meta AI, Mistral',
        'current_date': datetime.now().strftime('%Y-%m-%d'),
    }
    try:
        CompetitiveIntel().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """Replay the crew execution from a specific task."""
    try:
        CompetitiveIntel().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


if __name__ == "__main__":
    run()
