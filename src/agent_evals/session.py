from __future__ import annotations

import copy
from typing import Any, Protocol

from .models import ToolCall


class ToolError(Exception):
    """A tool refused the call. The message goes back to the agent: write it as an instruction."""


class AgentError(Exception):
    """The agent stopped without finishing the task (refusal, truncated output, ...)."""


class StepLimitExceeded(AgentError):
    pass


class World(Protocol):
    """The sandbox an agent acts on. Rebuilt from the scenario seed for every trial."""

    def tool_specs(self) -> list[dict[str, Any]]: ...

    def call(self, name: str, args: dict[str, Any]) -> Any: ...

    def snapshot(self) -> dict[str, Any]: ...


class ToolSession:
    """The only door between an agent and the world.

    The harness records every call here, so the transcript the checks read is
    never something the agent wrote about itself.
    """

    def __init__(self, world: World, max_steps: int = 20) -> None:
        self._world = world
        self._max_steps = max_steps
        self.calls: list[ToolCall] = []

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._world.tool_specs()

    def call(self, name: str, args: dict[str, Any]) -> ToolCall:
        if len(self.calls) >= self._max_steps:
            raise StepLimitExceeded(f"exceeded the {self._max_steps}-call limit")
        # Copy the args: the agent keeps its reference and must not be able to edit the record.
        call = ToolCall(name=name, args=copy.deepcopy(args))
        try:
            call.result = self._world.call(name, args)
        except ToolError as exc:
            call.error = str(exc)
        self.calls.append(call)
        return call
