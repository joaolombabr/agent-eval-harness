from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agents import build_agent
from .report import to_markdown, to_text
from .runner import load_scenarios, run_suite

DEFAULT_SCENARIOS = "scenarios/recruiter"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-evals", description="Deterministic evals for tool-calling agents."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a scenario suite against an agent")
    run.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="directory of *.yaml scenarios")
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

    lst = sub.add_parser("list", help="list the scenarios in a suite")
    lst.add_argument("--scenarios", default=DEFAULT_SCENARIOS)

    args = parser.parse_args(argv)
    try:
        scenarios = load_scenarios(args.scenarios)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "list":
        for scenario in scenarios:
            print(f"{scenario.id}: {scenario.description}")
        return 0

    try:
        agent = build_agent(args.agent)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ImportError:
        print(
            "error: the claude agent needs `pip install agent-eval-harness[claude]`",
            file=sys.stderr,
        )
        return 2

    try:
        report = run_suite(agent, scenarios, trials=args.trials)
    except Exception as exc:  # noqa: BLE001 - top-level boundary: no score from a broken run
        print(
            f"error: run aborted, no report written: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 2
    print(to_text(report))
    if args.json:
        args.json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(to_markdown(report), encoding="utf-8")
    return 0 if report.pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
