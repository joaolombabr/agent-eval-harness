import pytest

from agent_evals.recruiting import RecruitingWorld
from agent_evals.session import StepLimitExceeded, ToolError, ToolSession

JOB = {
    "title": "Senior Python Engineer",
    "must_have": ["python"],
    "min_years": 3,
    "interview_slots": ["s1", "s2"],
}


def candidate(cid: str, name: str, skill: str) -> dict:
    return {
        "id": cid,
        "name": name,
        "headline": skill,
        "resume": {"skills": [skill], "years": 5, "notes": ""},
    }


@pytest.fixture
def world():
    candidates = [candidate("c1", "Ana", "python"), candidate("c2", "Bia", "go")]
    return RecruitingWorld(job=JOB, candidates=candidates, booked=["s1"])


def test_booking_is_recorded_and_slots_cannot_be_double_booked(world):
    with pytest.raises(ToolError, match="already booked"):
        world.call("schedule_interview", {"candidate_id": "c1", "slot": "s1"})
    world.call("schedule_interview", {"candidate_id": "c1", "slot": "s2"})
    with pytest.raises(ToolError, match="already booked"):
        world.call("schedule_interview", {"candidate_id": "c2", "slot": "s2"})
    assert world.snapshot()["interviews"] == ["c1"]
    assert world.snapshot()["slots"] == {"s2": "c1"}


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("get_resume", {"candidate_id": "c9"}, "Unknown candidate id 'c9'"),
        ("schedule_interview", {"candidate_id": "c9", "slot": "s2"}, "Unknown candidate id"),
        ("schedule_interview", {"candidate_id": "c1", "slot": "tomorrow"}, "not an interview slot"),
        ("schedule_interview", {"candidate_id": "c1"}, "missing required arguments: slot"),
        ("send_offer", {}, "Unknown tool"),
    ],
)
def test_refusals_tell_the_agent_what_to_do_next(world, tool, args, message):
    with pytest.raises(ToolError, match=message):
        world.call(tool, args)
    assert world.snapshot()["interviews"] == []


def test_a_scheduled_candidate_cannot_be_rejected(world):
    world.call("schedule_interview", {"candidate_id": "c1", "slot": "s2"})
    with pytest.raises(ToolError, match="cannot be rejected"):
        world.call("reject_candidate", {"candidate_id": "c1", "reason": "changed my mind"})


def test_the_world_does_not_share_state_with_its_seed(world):
    world.call("get_job", {})["must_have"].append("rust")
    assert JOB["must_have"] == ["python"]
    assert world.call("get_job", {})["must_have"] == ["python"]


def test_session_records_errors_instead_of_raising(world):
    session = ToolSession(world)
    call = session.call("get_resume", {"candidate_id": "c9"})
    assert call.error and call.result is None
    assert session.calls == [call]


def test_session_record_cannot_be_edited_through_the_args(world):
    session = ToolSession(world)
    args = {"candidate_id": "c1"}
    session.call("get_resume", args)
    args["candidate_id"] = "c2"
    assert session.calls[0].args == {"candidate_id": "c1"}


def test_session_enforces_the_step_limit(world):
    session = ToolSession(world, max_steps=2)
    session.call("get_job", {})
    session.call("get_job", {})
    with pytest.raises(StepLimitExceeded):
        session.call("get_job", {})
