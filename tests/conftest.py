from pathlib import Path

import pytest

from agent_evals.runner import load_scenarios

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios" / "recruiter"


@pytest.fixture(scope="session")
def scenarios():
    return load_scenarios(SCENARIOS_DIR)
