from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from .agents import Agent
from .checks import run_check, validate
from .models import CheckResult, Report, Scenario, ScenarioResult, Transcript, TrialResult
from .recruiting import RecruitingWorld
from .session import AgentError, ToolSession, World

WORLDS: dict[str, Callable[..., World]] = {"recruiting": RecruitingWorld}


def load_scenarios(path: str | Path) -> list[Scenario]:
    files = sorted(Path(path).glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no scenario files (*.yaml) in {path}")
    scenarios = []
    for file in files:
        scenario = Scenario.model_validate(yaml.safe_load(file.read_text(encoding="utf-8")))
        if scenario.world_type not in WORLDS:
            raise ValueError(f"{file.name}: unknown world_type {scenario.world_type!r}")
        for spec in scenario.checks:
            try:
                validate(spec)
            except ValueError as exc:
                raise ValueError(f"{file.name}: {exc}") from exc
        scenarios.append(scenario)
    return scenarios


def new_session(scenario: Scenario) -> tuple[World, ToolSession]:
    """A fresh world for the scenario, and the session that records what happens to it."""
    world = WORLDS[scenario.world_type](**scenario.world)
    return world, ToolSession(world, max_steps=scenario.max_steps)


def run_trial(agent: Agent, scenario: Scenario) -> TrialResult:
    # A fresh world per trial: nothing an earlier run did can leak into this one.
    world, session = new_session(scenario)
    transcript = Transcript()
    try:
        transcript.final_answer = agent.run(scenario.prompt, session)
    except AgentError as exc:
        # Only the agent's own failures are graded. Anything else (network, auth,
        # a bug in the harness) propagates: a broken run must not look like a score.
        transcript.error = str(exc)
    transcript.calls = session.calls

    state: dict[str, Any] = world.snapshot()
    results = [
        CheckResult(
            check="completed", passed=transcript.error is None, detail=transcript.error or ""
        )
    ]
    results += [run_check(spec, transcript, state) for spec in scenario.checks]
    return TrialResult(
        passed=all(r.passed for r in results), checks=results, transcript=transcript, state=state
    )


def run_suite(agent: Agent, scenarios: list[Scenario], trials: int = 1) -> Report:
    """Run every scenario `trials` times. Repeat trials to measure a non-deterministic agent."""
    return Report(
        agent=agent.name,
        trials=trials,
        scenarios=[
            ScenarioResult(
                scenario_id=s.id,
                description=s.description,
                trials=[run_trial(agent, s) for _ in range(trials)],
            )
            for s in scenarios
        ],
    )
