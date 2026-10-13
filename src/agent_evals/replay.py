"""Grade a recorded run without the agent that produced it.

A transcript is a list of tool calls. Replaying those calls on a fresh sandbox
rebuilds the state they lead to, so the checks can grade a run that happened
somewhere else: a production incident, another framework, a model you no
longer want to pay for. The recorded results are never trusted; only the calls
are re-executed.
"""

from __future__ import annotations

from .models import Transcript
from .session import AgentError, ToolSession


class ReplayAgent:
    name = "replay"

    def __init__(self, transcript: Transcript) -> None:
        self._transcript = transcript

    def run(self, prompt: str, session: ToolSession) -> str:
        for position, recorded in enumerate(self._transcript.calls, start=1):
            replayed = session.call(recorded.name, recorded.args)
            has_outcome = recorded.result is not None or recorded.error is not None
            if has_outcome and (recorded.error is None) != (replayed.error is None):
                was = "an error" if recorded.error else "a success"
                now = f"the error {replayed.error!r}" if replayed.error else "a success"
                # The trace was recorded against different data: grading it here would be fiction.
                raise AgentError(
                    f"transcript diverges from this scenario at call {position} "
                    f"({recorded.name}): it recorded {was}, the sandbox returned {now}"
                )
        if self._transcript.error:
            raise AgentError(self._transcript.error)
        return self._transcript.final_answer
