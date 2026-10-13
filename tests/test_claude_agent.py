"""The Claude adapter, driven by a scripted stand-in for the SDK client.

These tests cover the loop's wiring (what is sent, how tool results and errors
go back, when it stops). They say nothing about how a real model behaves.
"""

import json
from types import SimpleNamespace

import pytest

from agent_evals.claude_agent import ClaudeAgent
from agent_evals.runner import run_trial


def tool_use(id: str, name: str, **input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def text(value: str):
    return SimpleNamespace(type="text", text=value)


def response(*content, stop_reason="tool_use", stop_details=None):
    return SimpleNamespace(
        content=list(content), stop_reason=stop_reason, stop_details=stop_details
    )


class FakeClient:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # The agent appends to one list across turns; keep what was sent at this point.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


@pytest.fixture
def scenario(scenarios):
    return next(s for s in scenarios if s.id == "unknown-candidate")


def test_loop_feeds_tool_results_back_and_returns_the_final_text(scenario):
    client = FakeClient(
        response(text("Looking up c9."), tool_use("t1", "get_resume", candidate_id="c9")),
        response(tool_use("t2", "submit_report", scheduled=[], rejected=[], summary="not found")),
        response(
            text("Candidate c9 does not exist, nothing was scheduled."), stop_reason="end_turn"
        ),
    )
    trial = run_trial(ClaudeAgent(client=client), scenario)

    assert trial.passed
    assert [c.name for c in trial.transcript.calls] == ["get_resume", "submit_report"]
    assert trial.transcript.final_answer.startswith("Candidate c9 does not exist")

    first, second, third = client.requests
    assert first["model"] == "claude-opus-5-5"
    assert first["thinking"] == {"type": "adaptive"}
    assert all(tool["strict"] for tool in first["tools"])
    assert first["messages"] == [{"role": "user", "content": scenario.prompt}]

    # The refused call goes back flagged as an error, with the world's own message.
    error_result = second["messages"][-1]["content"][0]
    assert error_result["tool_use_id"] == "t1"
    assert error_result["is_error"] is True
    assert "Unknown candidate id 'c9'" in error_result["content"]

    ok_result = third["messages"][-1]["content"][0]
    assert "is_error" not in ok_result
    assert json.loads(ok_result["content"]) == {"status": "received"}


def test_parallel_tool_calls_come_back_in_one_user_message(scenario):
    client = FakeClient(
        response(tool_use("a", "get_job"), tool_use("b", "search_candidates")),
        response(text("done"), stop_reason="end_turn"),
    )
    run_trial(ClaudeAgent(client=client), scenario)
    results = client.requests[1]["messages"][-1]
    assert results["role"] == "user"
    assert [r["tool_use_id"] for r in results["content"]] == ["a", "b"]


def test_a_false_confirmation_from_the_model_fails_the_trial(scenario):
    slot = "2026-10-20T14:00"
    client = FakeClient(
        response(tool_use("t1", "schedule_interview", candidate_id="c9", slot=slot)),
        response(tool_use("t2", "submit_report", scheduled=["c9"], rejected=[], summary="Booked")),
        response(text("Interview with c9 is booked."), stop_reason="end_turn"),
    )
    trial = run_trial(ClaudeAgent(client=client), scenario)
    assert [c.check for c in trial.checks if not c.passed] == ["claim_matches_state"]


@pytest.mark.parametrize(
    ("stop_reason", "message"),
    [("refusal", "refused"), ("max_tokens", "cut off")],
)
def test_refusals_and_truncation_fail_the_trial_as_agent_errors(scenario, stop_reason, message):
    details = SimpleNamespace(category="cyber") if stop_reason == "refusal" else None
    client = FakeClient(response(stop_reason=stop_reason, stop_details=details))
    trial = run_trial(ClaudeAgent(client=client), scenario)
    assert not trial.passed
    assert message in trial.transcript.error
    assert trial.checks[0].check == "completed"


def test_model_can_be_set_by_argument_or_environment(monkeypatch):
    assert ClaudeAgent(client=FakeClient()).name == "claude:claude-opus-5-5"
    monkeypatch.setenv("AGENT_EVALS_MODEL", "claude-haiku-5-5")
    assert ClaudeAgent(client=FakeClient()).model == "claude-haiku-5-5"
    assert ClaudeAgent(model="claude-sonnet-5-5", client=FakeClient()).model == "claude-sonnet-5-5"
