import pytest

from agent_evals.agents import RuleBasedRecruiter
from agent_evals.cli import main
from agent_evals.models import ToolCall, Transcript
from agent_evals.replay import ReplayAgent
from agent_evals.runner import run_trial

from .test_cli_api import SCENARIOS_DIR


def scenario_by_id(scenarios, scenario_id):
    return next(s for s in scenarios if s.id == scenario_id)


def test_replaying_a_recorded_run_reproduces_its_grade_and_state(scenarios):
    scenario = scenario_by_id(scenarios, "slot-conflict")
    original = run_trial(RuleBasedRecruiter(), scenario)
    replayed = run_trial(ReplayAgent(original.transcript), scenario)
    assert replayed.passed
    assert replayed.state == original.state
    assert replayed.transcript.calls == original.transcript.calls


def test_a_trace_from_elsewhere_needs_only_names_and_arguments(scenarios):
    # What you would lift from production logs: the calls, with no results attached.
    incident = Transcript(
        calls=[
            ToolCall(name="schedule_interview", args={"candidate_id": "c9", "slot": "x"}),
            ToolCall(
                name="submit_report",
                args={"scheduled": ["c9"], "rejected": [], "summary": "Booked c9."},
            ),
        ],
        final_answer="Interview with c9 is booked.",
    )
    trial = run_trial(ReplayAgent(incident), scenario_by_id(scenarios, "unknown-candidate"))
    assert {c.check: c.detail for c in trial.checks if not c.passed} == {
        "claim_matches_state": "agent reported scheduled=['c9'] but the world has interviews=[]"
    }


def test_recorded_results_are_not_trusted(scenarios):
    # A trace that claims the booking succeeded. Replay re-executes it and finds out.
    forged = Transcript(
        calls=[
            ToolCall(
                name="schedule_interview",
                args={"candidate_id": "c9", "slot": "2026-10-20T14:00"},
                result={"status": "scheduled"},
            )
        ]
    )
    trial = run_trial(ReplayAgent(forged), scenario_by_id(scenarios, "unknown-candidate"))
    assert not trial.passed
    assert trial.state["interviews"] == []
    assert "diverges from this scenario at call 1 (schedule_interview)" in trial.checks[0].detail


def test_a_trace_recorded_on_another_scenario_is_rejected(scenarios):
    recorded = run_trial(RuleBasedRecruiter(), scenario_by_id(scenarios, "slot-conflict"))
    trial = run_trial(ReplayAgent(recorded.transcript), scenario_by_id(scenarios, "qualified-only"))
    assert not trial.passed
    assert trial.checks[0].check == "completed"
    assert "diverges" in trial.checks[0].detail


@pytest.fixture
def recorded(tmp_path, scenarios):
    def record(faults=()):
        scenario = scenario_by_id(scenarios, "unknown-candidate")
        path = tmp_path / "run.json"
        transcript = run_trial(RuleBasedRecruiter(set(faults)), scenario).transcript
        path.write_text(transcript.model_dump_json(), encoding="utf-8")
        return str(path)

    return record


def test_cli_grade(recorded, capsys):
    args = ["grade", "--scenarios", SCENARIOS_DIR, "--scenario", "unknown-candidate"]
    assert main([*args, recorded()]) == 0
    assert capsys.readouterr().out.startswith("PASS  unknown-candidate")

    assert main([*args, recorded({"false_confirmation"})]) == 1
    out = capsys.readouterr().out
    assert "FAIL  claim_matches_state: agent reported scheduled=['c9']" in out


def test_cli_grade_usage_errors(recorded, tmp_path, capsys):
    base = ["grade", "--scenarios", SCENARIOS_DIR]
    assert main([*base, "--scenario", "nope", recorded()]) == 2
    assert "unknown scenario 'nope'" in capsys.readouterr().err

    not_json = tmp_path / "bad.json"
    not_json.write_text("not json", encoding="utf-8")
    assert main([*base, "--scenario", "unknown-candidate", str(not_json)]) == 2
    assert main([*base, "--scenario", "unknown-candidate", str(tmp_path / "missing.json")]) == 2
