"""Audit the suite, not the agent.

A passing eval says the agent is fine only if the eval could have failed. This
module measures that: it injects known failure modes into a correct agent, one
at a time, and records which checks notice.

Two numbers come out. The mutation score is the share of faults some check
caught. Check coverage is the share of checks that caught at least one fault; a
check outside that share has never been seen to fail, so nothing here shows it
can.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from pydantic import BaseModel, computed_field

from .agents import Agent
from .models import Scenario
from .runner import run_trial


class Detector(BaseModel):
    scenario_id: str
    check: str


class FaultResult(BaseModel):
    fault: str
    description: str
    detectors: list[Detector]

    @computed_field
    @property
    def killed(self) -> bool:
        return bool(self.detectors)

    @property
    def scenarios(self) -> list[str]:
        return list(dict.fromkeys(d.scenario_id for d in self.detectors))


class CheckUsage(BaseModel):
    scenario_id: str
    check: str
    caught: list[str]  # the faults this check failed on


class AuditReport(BaseModel):
    scenario_ids: list[str]
    # Scenarios the fault-free agent fails. Any entry means a check is wrong or too strict.
    reference_failures: list[str]
    faults: list[FaultResult]
    checks: list[CheckUsage]

    @property
    def survivors(self) -> list[str]:
        return [f.fault for f in self.faults if not f.killed]

    @property
    def unproven(self) -> list[CheckUsage]:
        return [c for c in self.checks if not c.caught]

    @computed_field
    @property
    def mutation_score(self) -> float:
        return sum(f.killed for f in self.faults) / len(self.faults) if self.faults else 0.0

    @computed_field
    @property
    def check_coverage(self) -> float:
        return 1 - len(self.unproven) / len(self.checks) if self.checks else 0.0

    def ok(self, strict: bool = False) -> bool:
        sound = not self.reference_failures and not self.survivors
        return sound and not (strict and self.unproven)


def audit(
    scenarios: list[Scenario],
    faults: Mapping[str, str],
    make_agent: Callable[[frozenset[str]], Agent],
) -> AuditReport:
    """`make_agent(faults)` builds the reference agent with those faults switched on."""
    reference = make_agent(frozenset())
    reference_failures = [s.id for s in scenarios if not run_trial(reference, s).passed]

    usage = {
        (s.id, i): CheckUsage(scenario_id=s.id, check=spec.label, caught=[])
        for s in scenarios
        for i, spec in enumerate(s.checks)
    }
    results = []
    for fault, description in faults.items():
        agent = make_agent(frozenset({fault}))
        detectors = []
        for scenario in scenarios:
            trial = run_trial(agent, scenario)
            # trial.checks[0] is the implicit `completed` check; the rest line up with the specs.
            if not trial.checks[0].passed:
                detectors.append(Detector(scenario_id=scenario.id, check="completed()"))
            for i, result in enumerate(trial.checks[1:]):
                if not result.passed:
                    usage[scenario.id, i].caught.append(fault)
                    detectors.append(
                        Detector(scenario_id=scenario.id, check=scenario.checks[i].label)
                    )
        results.append(FaultResult(fault=fault, description=description, detectors=detectors))

    return AuditReport(
        scenario_ids=[s.id for s in scenarios],
        reference_failures=reference_failures,
        faults=results,
        checks=list(usage.values()),
    )


def to_text(report: AuditReport) -> str:
    n_faults, n_checks = len(report.faults), len(report.checks)
    lines = [
        f"suite audit: {len(report.scenario_ids)} scenarios, {n_checks} checks, "
        f"{n_faults} injected faults",
        "",
    ]
    if report.reference_failures:
        lines.append(
            "reference agent FAILS: "
            + ", ".join(report.reference_failures)
            + "  (check too strict)"
        )
    else:
        lines.append("reference agent: passes every scenario")
    lines += ["", "fault                 caught by"]
    for fault in report.faults:
        if fault.killed:
            where = f"{len(fault.detectors)} check(s) in {', '.join(fault.scenarios)}"
        else:
            where = "NOTHING  <- the suite cannot see this failure"
        lines.append(f"  {fault.fault:<20}{where}")
    killed = n_faults - len(report.survivors)
    lines += ["", f"mutation score: {killed}/{n_faults} faults caught"]

    proven = n_checks - len(report.unproven)
    lines.append(f"check coverage: {proven}/{n_checks} checks have caught a fault")
    if report.unproven:
        lines += ["", "never seen failing (no injected fault reaches them):"]
        lines += [f"  {c.scenario_id:<20}{c.check}" for c in report.unproven]
    return "\n".join(lines)


def to_markdown(report: AuditReport) -> str:
    """The kill matrix: one row per fault, one column per scenario, cells count failed checks."""
    header = "| Fault | " + " | ".join(f"`{s}`" for s in report.scenario_ids) + " |"
    lines = [
        "## Suite audit",
        "",
        f"**Mutation score: {report.mutation_score:.0%}** · "
        f"**check coverage: {report.check_coverage:.0%}** "
        f"({len(report.checks) - len(report.unproven)}/{len(report.checks)} checks have caught "
        "a fault)",
        "",
        header,
        "| --- |" + " :-: |" * len(report.scenario_ids),
    ]
    for fault in report.faults:
        counts = {s: 0 for s in report.scenario_ids}
        for detector in fault.detectors:
            counts[detector.scenario_id] += 1
        cells = " | ".join(str(counts[s]) if counts[s] else "·" for s in report.scenario_ids)
        mark = "" if fault.killed else " **(survived)**"
        lines.append(f"| `{fault.fault}`{mark} | {cells} |")
    if report.unproven:
        lines += ["", "Checks never seen failing:", ""]
        lines += [f"- `{c.scenario_id}`: `{c.check}`" for c in report.unproven]
    return "\n".join(lines) + "\n"
