from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from . import audit as suite_audit
from .agents import FAULTS, RuleBasedRecruiter, build_agent
from .models import Scenario, Transcript
from .replay import ReplayAgent
from .report import to_markdown, to_text, trial_to_text
from .runner import load_scenarios, run_suite, run_trial

DEFAULT_SCENARIOS = "scenarios/recruiter"


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-evals", description="Deterministic evals for tool-calling agents."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        cmd = sub.add_parser(name, help=help)
        cmd.add_argument(
            "--scenarios", default=DEFAULT_SCENARIOS, help="directory of *.yaml scenarios"
        )
        return cmd

    run = command("run", "run a scenario suite against an agent")
    run.add_argument(
        "--agent",
        default="baseline",
        help="baseline | regressed | faulty:<fault>[,<fault>] | claude[:<model>]",
    )
    run.add_argument("--trials", type=int, default=1, help="runs per scenario")
    run.add_argument(
        "--min-pass-rate",
        type=float,
        default=1.0,
        help="exit non-zero when the pass rate is below this (0-1)",
    )
    run.add_argument("--json", type=Path, help="write the full report, transcripts included")
    run.add_argument("--markdown", type=Path, help="write a summary table")

    command("list", "list the scenarios in a suite")

    audit = command("audit", "test the suite itself by injecting known failure modes")
    audit.add_argument(
        "--strict", action="store_true", help="also fail when a check has never caught a fault"
    )
    audit.add_argument("--json", type=Path, help="write the full audit")
    audit.add_argument("--markdown", type=Path, help="write the kill matrix")

    grade = command("grade", "grade a recorded transcript by replaying it on a fresh sandbox")
    grade.add_argument("transcript", type=Path, help="JSON file with the recorded tool calls")
    grade.add_argument("--scenario", required=True, help="id of the scenario it was recorded on")

    serve = command("serve-mcp", "expose one scenario's sandbox as an MCP server on stdio")
    serve.add_argument("--scenario", required=True, help="scenario id")
    serve.add_argument("--out", type=Path, required=True, help="where to write the transcript")
    return parser


def _pick(scenarios: list[Scenario], scenario_id: str) -> Scenario:
    for scenario in scenarios:
        if scenario.id == scenario_id:
            return scenario
    known = ", ".join(s.id for s in scenarios)
    raise ValueError(f"unknown scenario {scenario_id!r}; available: {known}")


def _run(args: argparse.Namespace, scenarios: list[Scenario]) -> int:
    try:
        agent = build_agent(args.agent)
    except ValueError as exc:
        return _fail(str(exc))
    except ImportError:
        return _fail("the claude agent needs `pip install agent-eval-harness[claude]`")

    try:
        report = run_suite(agent, scenarios, trials=args.trials)
    except Exception as exc:  # noqa: BLE001 - top-level boundary: no score from a broken run
        return _fail(f"run aborted, no report written: {type(exc).__name__}: {exc}")
    print(to_text(report))
    if args.json:
        args.json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(to_markdown(report), encoding="utf-8")
    return 0 if report.pass_rate >= args.min_pass_rate else 1


def _audit(args: argparse.Namespace, scenarios: list[Scenario]) -> int:
    report = suite_audit.audit(scenarios, FAULTS, RuleBasedRecruiter)
    print(suite_audit.to_text(report))
    if args.json:
        args.json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(suite_audit.to_markdown(report), encoding="utf-8")
    return 0 if report.ok(strict=args.strict) else 1


def _grade(args: argparse.Namespace, scenarios: list[Scenario]) -> int:
    try:
        scenario = _pick(scenarios, args.scenario)
        transcript = Transcript.model_validate_json(args.transcript.read_text(encoding="utf-8"))
    except (ValueError, ValidationError, OSError) as exc:
        return _fail(str(exc))
    trial = run_trial(ReplayAgent(transcript), scenario)
    print(trial_to_text(scenario.id, trial))
    return 0 if trial.passed else 1


def _serve(args: argparse.Namespace, scenarios: list[Scenario]) -> int:
    try:
        scenario = _pick(scenarios, args.scenario)
    except ValueError as exc:
        return _fail(str(exc))
    try:
        from .mcp_server import serve
    except ImportError:
        return _fail("the MCP server needs `pip install agent-eval-harness[mcp]`")
    serve(scenario, args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        scenarios = load_scenarios(args.scenarios)
    except (FileNotFoundError, ValueError) as exc:
        return _fail(str(exc))

    if args.command == "list":
        for scenario in scenarios:
            print(f"{scenario.id}: {scenario.description}")
        return 0
    handlers = {"run": _run, "audit": _audit, "grade": _grade, "serve-mcp": _serve}
    return handlers[args.command](args, scenarios)


if __name__ == "__main__":
    sys.exit(main())
