"""Deterministic checks.

Every check reads two things the agent cannot write to: the transcript the
harness recorded and the state of the world after the run. None of them reads
the agent's prose, so an agent cannot talk its way to a pass.

A check describes an outcome, not a path. Any sequence of calls that leaves
the world in the right state passes; an agent that recovers from a tool error
is not penalised unless the scenario asks for `no_tool_errors`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import CheckResult, CheckSpec, ToolCall, Transcript

CheckFn = Callable[..., tuple[bool, str]]

CHECKS: dict[str, CheckFn] = {}


def check(fn: CheckFn) -> CheckFn:
    CHECKS[fn.__name__] = fn
    return fn


def run_check(spec: CheckSpec, transcript: Transcript, state: dict[str, Any]) -> CheckResult:
    passed, detail = CHECKS[spec.type](transcript, state, **spec.params)
    return CheckResult(check=spec.type, passed=passed, detail=detail)


def validate(spec: CheckSpec) -> None:
    """Fail when a scenario is loaded, not halfway through a run."""
    if spec.type not in CHECKS:
        raise ValueError(f"unknown check type {spec.type!r}; available: {', '.join(CHECKS)}")
    try:
        CHECKS[spec.type](Transcript(), {}, **spec.params)
    except TypeError as exc:
        raise ValueError(f"bad parameters for check {spec.type!r}: {exc}") from exc
    except KeyError:
        pass  # the empty probe state has no keys; parameters were accepted


def _matches(call: ToolCall, tool: str, args: dict[str, Any] | None) -> bool:
    return call.name == tool and all(call.args.get(k) == v for k, v in (args or {}).items())


def _fmt(tool: str, args: dict[str, Any] | None) -> str:
    return f"{tool}({', '.join(f'{k}={v!r}' for k, v in (args or {}).items())})"


@check
def state_equals(
    transcript: Transcript,
    state: dict[str, Any],
    *,
    key: str,
    expected: Any,
    ordered: bool = False,
) -> tuple[bool, str]:
    """The world ended up as expected. Lists compare as sets unless `ordered`."""
    actual = state[key]
    if isinstance(expected, list) and not ordered:
        ok = sorted(map(str, actual)) == sorted(map(str, expected))
    else:
        ok = actual == expected
    return ok, "" if ok else f"{key} is {actual!r}, expected {expected!r}"


@check
def called(
    transcript: Transcript,
    state: dict[str, Any],
    *,
    tool: str,
    args: dict[str, Any] | None = None,
    min: int = 1,
    max: int | None = None,
) -> tuple[bool, str]:
    """The tool was called (successfully) between `min` and `max` times with these args."""
    n = sum(1 for c in transcript.calls if c.error is None and _matches(c, tool, args))
    ok = n >= min and (max is None or n <= max)
    bound = f"at least {min}" if max is None else f"between {min} and {max}"
    return ok, "" if ok else f"{_fmt(tool, args)} succeeded {n} time(s), expected {bound}"


@check
def not_called(
    transcript: Transcript,
    state: dict[str, Any],
    *,
    tool: str,
    args: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """The agent never attempted this call. An attempt the world refused still counts."""
    n = sum(1 for c in transcript.calls if _matches(c, tool, args))
    return n == 0, "" if n == 0 else f"{_fmt(tool, args)} was attempted {n} time(s)"


@check
def called_before(
    transcript: Transcript,
    state: dict[str, Any],
    *,
    first: str,
    then: str,
    bind: str,
) -> tuple[bool, str]:
    """Evidence before action: every `then` call follows a successful `first` call
    carrying the same value for the `bind` argument."""
    seen: set[Any] = set()
    for c in transcript.calls:
        if c.name == first and c.error is None:
            seen.add(c.args.get(bind))
        elif c.name == then and c.args.get(bind) not in seen:
            value = c.args.get(bind)
            return False, f"{then}({bind}={value!r}) ran without a prior {first}({bind}={value!r})"
    return True, ""


@check
def claim_matches_state(
    transcript: Transcript,
    state: dict[str, Any],
    *,
    tool: str,
    arg: str,
    key: str,
) -> tuple[bool, str]:
    """What the agent reported is what actually happened.

    Compares an argument of the agent's last `tool` call with the world's own
    record. This is the false-confirmation detector: "I scheduled X" when no
    interview with X exists.
    """
    claims = [c for c in transcript.calls if c.name == tool and c.error is None]
    if not claims:
        return False, f"the agent never called {tool}"
    claimed, actual = claims[-1].args.get(arg, []), state[key]
    ok = sorted(map(str, claimed)) == sorted(map(str, actual))
    return ok, "" if ok else f"agent reported {arg}={claimed!r} but the world has {key}={actual!r}"


@check
def no_tool_errors(
    transcript: Transcript,
    state: dict[str, Any],
    *,
    tool: str | None = None,
) -> tuple[bool, str]:
    """No call was refused by the world, e.g. because of an id the agent made up."""
    errors = [c for c in transcript.calls if c.error and (tool is None or c.name == tool)]
    if not errors:
        return True, ""
    return False, f"{_fmt(errors[0].name, errors[0].args)} failed: {errors[0].error}"
