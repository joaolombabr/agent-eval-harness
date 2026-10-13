from agent_evals import audit as suite_audit
from agent_evals.agents import FAULTS, RuleBasedRecruiter
from agent_evals.cli import main
from agent_evals.models import CheckSpec

from .test_cli_api import SCENARIOS_DIR


def run_audit(scenarios):
    return suite_audit.audit(scenarios, FAULTS, RuleBasedRecruiter)


def test_the_shipped_suite_catches_every_fault_and_accepts_the_reference_agent(scenarios):
    report = run_audit(scenarios)
    assert report.reference_failures == []
    assert report.survivors == []
    assert report.mutation_score == 1
    assert report.ok()


# The known gaps, written down. These checks have never been seen to fail, so
# nothing proves they can. Closing one (or opening another) must change this
# list and the README section that quotes it.
UNPROVEN = {
    ("qualified-only", "claim_matches_state(submit_report, scheduled, interviews)"),
    ("nobody-qualifies", "claim_matches_state(submit_report, scheduled, interviews)"),
    ("stale-headline", "claim_matches_state(submit_report, scheduled, interviews)"),
    ("resume-injection", "claim_matches_state(submit_report, scheduled, interviews)"),
    ("unknown-candidate", "state_equals(interviews)"),
}


def test_checks_that_never_failed_are_reported_not_hidden(scenarios):
    report = run_audit(scenarios)
    assert {(c.scenario_id, c.check) for c in report.unproven} == UNPROVEN
    assert report.check_coverage == 17 / 22
    assert report.ok() and not report.ok(strict=True)


def test_a_missing_fault_model_leaves_its_checks_unproven(scenarios):
    # Without a fault that makes up ids, nothing can show that no_tool_errors works.
    without = {name: text for name, text in FAULTS.items() if name != "invents_id"}
    report = suite_audit.audit(scenarios, without, RuleBasedRecruiter)
    unproven = {(c.scenario_id, c.check) for c in report.unproven}
    assert {
        (s, "no_tool_errors()") for s in ("qualified-only", "stale-headline", "resume-injection")
    } <= unproven


def test_a_hole_in_the_suite_shows_up_as_a_surviving_fault(scenarios):
    without_injection = [s for s in scenarios if s.id != "resume-injection"]
    report = run_audit(without_injection)
    assert report.survivors == ["obeys_resume"]
    assert report.mutation_score == 4 / 5
    assert not report.ok()
    assert "NOTHING" in suite_audit.to_text(report)


def test_a_check_that_rejects_correct_behaviour_fails_the_audit(scenarios):
    # The reference agent legitimately hits the taken slot and recovers.
    too_strict = [
        s.model_copy(update={"checks": [*s.checks, CheckSpec(type="no_tool_errors")]})
        if s.id == "slot-conflict"
        else s
        for s in scenarios
    ]
    report = run_audit(too_strict)
    assert report.reference_failures == ["slot-conflict"]
    assert not report.ok()


def test_detectors_name_the_exact_checks(scenarios):
    report = run_audit(scenarios)
    best_available = next(f for f in report.faults if f.fault == "best_available")
    assert [(d.scenario_id, d.check) for d in best_available.detectors] == [
        ("nobody-qualifies", "state_equals(interviews)"),
        ("nobody-qualifies", "not_called(schedule_interview)"),
    ]


def test_cli_audit_exit_codes_and_kill_matrix(tmp_path, capsys):
    out = tmp_path / "audit.md"
    assert main(["audit", "--scenarios", SCENARIOS_DIR, "--markdown", str(out)]) == 0
    assert "mutation score: 5/5 faults caught" in capsys.readouterr().out
    matrix = out.read_text(encoding="utf-8")
    assert "| `obeys_resume` | · | · | · | · | 2 | · |" in matrix
    assert main(["audit", "--scenarios", SCENARIOS_DIR, "--strict"]) == 1
