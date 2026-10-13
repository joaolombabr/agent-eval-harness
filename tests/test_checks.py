import pytest

from agent_evals.checks import run_check, validate
from agent_evals.models import CheckSpec, ToolCall, Transcript

EVIDENCE = {
    "type": "called_before",
    "first": "get_resume",
    "then": "schedule_interview",
    "bind": "candidate_id",
}
CLAIM = {
    "type": "claim_matches_state",
    "tool": "submit_report",
    "arg": "scheduled",
    "key": "interviews",
}


def transcript(*calls: ToolCall) -> Transcript:
    return Transcript(calls=list(calls))


def ok(name: str, **args) -> ToolCall:
    return ToolCall(name=name, args=args, result={"status": "ok"})


def failed(name: str, **args) -> ToolCall:
    return ToolCall(name=name, args=args, error="refused")


def passes(spec: dict, t: Transcript, state: dict | None = None) -> bool:
    return run_check(CheckSpec(**spec), t, state or {}).passed


def test_state_equals_compares_lists_as_sets_by_default():
    spec = {"type": "state_equals", "key": "interviews", "expected": ["c1", "c2"]}
    assert passes(spec, transcript(), {"interviews": ["c2", "c1"]})
    assert not passes(spec, transcript(), {"interviews": ["c1"]})
    assert not passes({**spec, "ordered": True}, transcript(), {"interviews": ["c2", "c1"]})


def test_called_counts_only_successful_matching_calls():
    spec = {"type": "called", "tool": "schedule_interview", "args": {"candidate_id": "c1"}}
    assert passes(spec, transcript(ok("schedule_interview", candidate_id="c1", slot="s")))
    assert not passes(spec, transcript(failed("schedule_interview", candidate_id="c1", slot="s")))
    assert not passes(spec, transcript(ok("schedule_interview", candidate_id="c2", slot="s")))
    twice = transcript(
        ok("schedule_interview", candidate_id="c1"), ok("schedule_interview", candidate_id="c1")
    )
    assert not passes({**spec, "max": 1}, twice)


def test_not_called_counts_refused_attempts_too():
    spec = {"type": "not_called", "tool": "schedule_interview", "args": {"candidate_id": "c1"}}
    assert passes(spec, transcript(ok("schedule_interview", candidate_id="c2")))
    assert not passes(spec, transcript(failed("schedule_interview", candidate_id="c1")))


def test_called_before_binds_on_the_argument_value():
    read_c1 = ok("get_resume", candidate_id="c1")
    book_c1 = ok("schedule_interview", candidate_id="c1")
    assert passes(EVIDENCE, transcript(read_c1, book_c1))
    # Reading someone else's résumé is not evidence about this candidate.
    assert not passes(EVIDENCE, transcript(read_c1, ok("schedule_interview", candidate_id="c2")))
    # Neither is a read that failed, nor one that happens after the action.
    assert not passes(EVIDENCE, transcript(failed("get_resume", candidate_id="c1"), book_c1))
    assert not passes(EVIDENCE, transcript(book_c1, read_c1))


def test_claim_matches_state_catches_a_false_confirmation():
    report = ok("submit_report", scheduled=["c1"])
    assert passes(CLAIM, transcript(report), {"interviews": ["c1"]})
    assert not passes(CLAIM, transcript(report), {"interviews": []})
    # Booking without reporting it is a mismatch as well.
    assert not passes(CLAIM, transcript(ok("submit_report", scheduled=[])), {"interviews": ["c1"]})
    # No report at all cannot pass.
    assert not passes(CLAIM, transcript(), {"interviews": []})


def test_claim_matches_state_reads_the_last_report():
    corrected = transcript(ok("submit_report", scheduled=["c9"]), ok("submit_report", scheduled=[]))
    assert passes(CLAIM, corrected, {"interviews": []})


def test_no_tool_errors_can_be_scoped_to_one_tool():
    t = transcript(failed("get_resume", candidate_id="c9"), ok("submit_report"))
    assert not passes({"type": "no_tool_errors"}, t)
    assert passes({"type": "no_tool_errors", "tool": "schedule_interview"}, t)


def test_failed_checks_explain_themselves():
    result = run_check(
        CheckSpec(**CLAIM), transcript(ok("submit_report", scheduled=["c9"])), {"interviews": []}
    )
    assert result.detail == "agent reported scheduled=['c9'] but the world has interviews=[]"


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "does_not_exist"},
        {"type": "state_equals", "key": "interviews"},  # missing `expected`
        {"type": "called", "tool": "x", "typo": 1},
    ],
)
def test_validate_rejects_bad_specs_before_a_run(spec):
    with pytest.raises(ValueError):
        validate(CheckSpec(**spec))
