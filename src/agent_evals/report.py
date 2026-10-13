from __future__ import annotations

from .models import Report, ScenarioResult


def _failures(scenario: ScenarioResult) -> list[str]:
    """Distinct failed checks across trials, in first-seen order."""
    seen: dict[str, None] = {}
    for trial in scenario.trials:
        for result in trial.checks:
            if not result.passed:
                seen.setdefault(f"{result.check}: {result.detail}")
    return list(seen)


def _score(scenario: ScenarioResult) -> str:
    passed = sum(t.passed for t in scenario.trials)
    return f"{passed}/{len(scenario.trials)}"


def to_text(report: Report) -> str:
    lines = [f"agent: {report.agent}   trials per scenario: {report.trials}", ""]
    for scenario in report.scenarios:
        mark = "PASS" if scenario.pass_rate == 1 else "FAIL"
        lines.append(f"{mark}  {_score(scenario):>5}  {scenario.scenario_id}")
        lines += [f"             - {failure}" for failure in _failures(scenario)]
    lines += ["", f"pass rate: {report.pass_rate:.0%}"]
    return "\n".join(lines)


def to_markdown(report: Report) -> str:
    lines = [
        f"## Agent evals: `{report.agent}`",
        "",
        f"**Pass rate: {report.pass_rate:.0%}** over {len(report.scenarios)} scenarios, "
        f"{report.trials} trial(s) each.",
        "",
        "| Scenario | Passed | What failed |",
        "| --- | --- | --- |",
    ]
    for scenario in report.scenarios:
        failures = "<br>".join(f.replace("|", "\\|") for f in _failures(scenario)) or "-"
        lines.append(f"| `{scenario.scenario_id}` | {_score(scenario)} | {failures} |")
    return "\n".join(lines) + "\n"
