from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agents import build_agent
from .models import Report
from .runner import load_scenarios, run_suite

app = FastAPI(title="agent-eval-harness", version="0.1.0")


def _scenarios_dir() -> str:
    return os.environ.get("AGENT_EVALS_SCENARIOS", "scenarios/recruiter")


class RunRequest(BaseModel):
    agent: str = "baseline"
    trials: int = Field(default=1, ge=1, le=20)
    scenario_ids: list[str] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/scenarios")
def scenarios() -> list[dict[str, str]]:
    return [{"id": s.id, "description": s.description} for s in load_scenarios(_scenarios_dir())]


# Sync on purpose: a suite run blocks, so FastAPI runs it in its threadpool.
@app.post("/runs")
def create_run(request: RunRequest) -> Report:
    suite = load_scenarios(_scenarios_dir())
    if request.scenario_ids is not None:
        unknown = set(request.scenario_ids) - {s.id for s in suite}
        if unknown:
            raise HTTPException(404, f"unknown scenario ids: {', '.join(sorted(unknown))}")
        suite = [s for s in suite if s.id in request.scenario_ids]
    try:
        agent = build_agent(request.agent)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return run_suite(agent, suite, trials=request.trials)
