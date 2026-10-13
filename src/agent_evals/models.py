from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ToolCall(BaseModel):
    """One tool call as the harness saw it. Exactly one of result/error is set."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None


class Transcript(BaseModel):
    calls: list[ToolCall] = Field(default_factory=list)
    final_answer: str = ""
    # Set when the agent did not finish on its own (step limit, refusal, truncated output).
    error: str | None = None


class CheckSpec(BaseModel):
    """A check as written in a scenario file: `type` plus that check's own parameters."""

    model_config = ConfigDict(extra="allow")

    type: str

    @property
    def params(self) -> dict[str, Any]:
        return self.model_extra or {}

    @property
    def label(self) -> str:
        """`state_equals(interviews)`: the type plus its scalar parameters."""
        scalars = [str(v) for v in self.params.values() if not isinstance(v, list | dict)]
        return f"{self.type}({', '.join(scalars)})"


class Scenario(BaseModel):
    id: str
    description: str
    prompt: str
    world_type: str = "recruiting"
    world: dict[str, Any]
    max_steps: int = 20
    checks: list[CheckSpec]


class CheckResult(BaseModel):
    check: str
    passed: bool
    detail: str = ""


class TrialResult(BaseModel):
    passed: bool
    checks: list[CheckResult]
    transcript: Transcript
    state: dict[str, Any]


class ScenarioResult(BaseModel):
    scenario_id: str
    description: str
    trials: list[TrialResult]

    @computed_field
    @property
    def pass_rate(self) -> float:
        return sum(t.passed for t in self.trials) / len(self.trials)


class Report(BaseModel):
    agent: str
    trials: int
    scenarios: list[ScenarioResult]

    @computed_field
    @property
    def pass_rate(self) -> float:
        """Share of trials that passed, across every scenario."""
        results = [t.passed for s in self.scenarios for t in s.trials]
        return sum(results) / len(results) if results else 0.0
