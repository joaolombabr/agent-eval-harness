import pytest

from agent_evals.agents import FAULTS, RuleBasedRecruiter, build_agent
from agent_evals.runner import load_scenarios, run_suite, run_trial


def failing(report) -> set[str]:
    return {s.scenario_id for s in report.scenarios if s.pass_rate < 1}


def test_the_correct_agent_passes_every_scenario(scenarios):
    report = run_suite(RuleBasedRecruiter(), scenarios)
    assert failing(report) == set()
    assert report.pass_rate == 1


# The suite's own test: each failure mode, injected alone, must be caught, and
# only by the scenarios written for it. A fault that slips through is a hole in
# the suite; an unrelated scenario failing means a check is too strict.
CAUGHT_BY = {
    "trusts_search": {
        "qualified-only",
        "nobody-qualifies",
        "stale-headline",
        "slot-conflict",
        "resume-injection",
    },
    "best_available": {"nobody-qualifies"},
    "obeys_resume": {"resume-injection"},
    "false_confirmation": {"slot-conflict", "unknown-candidate"},
    "invents_id": {"qualified-only", "stale-headline", "slot-conflict", "resume-injection"},
}


def test_every_fault_has_an_expectation():
    assert CAUGHT_BY.keys() == FAULTS.keys()


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_each_fault_is_caught_by_exactly_its_scenarios(scenarios, fault):
    report = run_suite(RuleBasedRecruiter({fault}), scenarios)
    assert failing(report) == CAUGHT_BY[fault]


def test_recovering_from_a_tool_error_is_not_a_failure(scenarios):
    scenario = next(s for s in scenarios if s.id == "slot-conflict")
    trial = run_trial(RuleBasedRecruiter(), scenario)
    assert any(c.error for c in trial.transcript.calls)  # it did hit the taken slot
    assert trial.passed


def test_false_confirmation_is_reported_with_the_evidence(scenarios):
    scenario = next(s for s in scenarios if s.id == "unknown-candidate")
    trial = run_trial(RuleBasedRecruiter({"false_confirmation"}), scenario)
    failed = {c.check: c.detail for c in trial.checks if not c.passed}
    assert failed == {
        "claim_matches_state": "agent reported scheduled=['c9'] but the world has interviews=[]"
    }


class Looper:
    name = "looper"

    def run(self, prompt, session):
        while True:
            session.call("get_job", {})


def test_an_agent_that_never_finishes_fails_on_the_step_limit(scenarios):
    trial = run_trial(Looper(), scenarios[0])
    assert not trial.passed
    assert trial.checks[0].check == "completed"
    assert "20-call limit" in trial.checks[0].detail
    assert len(trial.transcript.calls) == 20


class Crasher:
    name = "crasher"

    def run(self, prompt, session):
        raise ConnectionError("network down")


def test_infrastructure_errors_abort_the_run_instead_of_scoring_it(scenarios):
    with pytest.raises(ConnectionError):
        run_suite(Crasher(), scenarios)


class BooksOnce:
    """Books c1 in the first slot and stops. Only works on a world nobody touched."""

    name = "books-once"

    def run(self, prompt, session):
        slot = session.call("get_job", {}).result["interview_slots"][0]
        booking = session.call("schedule_interview", {"candidate_id": "c1", "slot": slot})
        return "ok" if booking.error is None else "slot taken"


def test_every_trial_starts_from_a_fresh_world(scenarios):
    scenario = next(s for s in scenarios if s.id == "qualified-only")
    trials = run_suite(BooksOnce(), [scenario], trials=3).scenarios[0].trials
    assert [t.transcript.final_answer for t in trials] == ["ok"] * 3
    assert [t.state["interviews"] for t in trials] == [["c1"]] * 3


def test_build_agent_names():
    assert build_agent("baseline").name == "baseline"
    assert build_agent("regressed").faults == frozenset(FAULTS)
    assert build_agent("faulty:obeys_resume").faults == {"obeys_resume"}
    with pytest.raises(ValueError):
        build_agent("faulty:nope")
    with pytest.raises(ValueError):
        build_agent("gpt")


def test_scenario_files_with_a_bad_check_are_rejected_on_load(tmp_path):
    (tmp_path / "bad.yaml").write_text(
        "id: bad\ndescription: d\nprompt: p\nworld: {job: {}, candidates: []}\n"
        "checks:\n  - type: state_equal\n    key: interviews\n    expected: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bad.yaml: unknown check type 'state_equal'"):
        load_scenarios(tmp_path)


def test_an_empty_scenario_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scenarios(tmp_path)
