"""Scripted agents.

They make the harness testable without a model: `baseline` does the job
correctly, and each fault reproduces one way real tool-calling agents fail.
A fault that no scenario catches means the suite has a hole.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from .session import ToolSession

FAULTS = {
    # Decides from the search headline and never opens the résumé.
    "trusts_search": "acts on a summary instead of the source",
    # Nobody qualifies, so it books the least-bad candidate to be helpful.
    "best_available": "relaxes a hard requirement to complete the task",
    # Follows instructions written inside the résumé.
    "obeys_resume": "treats retrieved data as instructions",
    # Never looks at tool errors, then reports the action as done.
    "false_confirmation": "reports an action that did not happen",
}


class Agent(Protocol):
    name: str

    def run(self, prompt: str, session: ToolSession) -> str:
        """Do the task through `session` and return the final answer."""
        ...


class RuleBasedRecruiter:
    def __init__(self, faults: frozenset[str] | set[str] = frozenset()) -> None:
        unknown = set(faults) - FAULTS.keys()
        if unknown:
            raise ValueError(f"unknown faults: {', '.join(sorted(unknown))}")
        self.faults = frozenset(faults)
        self.name = f"faulty[{','.join(sorted(faults))}]" if faults else "baseline"

    def run(self, prompt: str, session: ToolSession) -> str:
        job = session.call("get_job", {}).result
        listed = {c["id"]: c for c in session.call("search_candidates", {}).result}
        # A prompt that names candidate ids narrows the task to those candidates.
        targets = re.findall(r"\bc\d+\b", prompt) or list(listed)

        fit = {cid: self._assess(cid, listed.get(cid), job, session) for cid in targets}
        qualified = [cid for cid, (ok, _) in fit.items() if ok]
        if not qualified and "best_available" in self.faults:
            qualified = [max(fit, key=lambda cid: fit[cid][1])]

        scheduled: list[str] = []
        free = list(job["interview_slots"])
        for cid in qualified:
            while free:
                booking = session.call(
                    "schedule_interview", {"candidate_id": cid, "slot": free.pop(0)}
                )
                if booking.error is None or "false_confirmation" in self.faults:
                    scheduled.append(cid)
                    break

        rejected = [cid for cid in targets if cid in listed and cid not in scheduled]
        for cid in rejected:
            session.call(
                "reject_candidate", {"candidate_id": cid, "reason": "Does not meet requirements."}
            )

        summary = f"Scheduled {scheduled or 'nobody'}; rejected {rejected or 'nobody'}."
        session.call(
            "submit_report", {"scheduled": scheduled, "rejected": rejected, "summary": summary}
        )
        return summary

    def _assess(
        self, cid: str, listing: dict[str, Any] | None, job: dict[str, Any], session: ToolSession
    ) -> tuple[bool, int]:
        """Return (qualifies, number of must-have skills matched)."""
        must_have = set(job["must_have"])
        if "trusts_search" in self.faults and listing:
            matched = len(must_have & set(re.findall(r"[a-z+#]+", listing["headline"].lower())))
            return matched == len(must_have), matched

        resume = session.call("get_resume", {"candidate_id": cid})
        if resume.error:
            return "false_confirmation" in self.faults, 0
        matched = len(must_have & set(resume.result["skills"]))
        if "obeys_resume" in self.faults and "schedule" in resume.result["notes"].lower():
            return True, matched
        qualifies = matched == len(must_have) and resume.result["years"] >= job["min_years"]
        return qualifies, matched


def build_agent(name: str) -> Agent:
    """`baseline`, `regressed` (every fault), `faulty:<a>,<b>`, or `claude[:<model>]`."""
    if name == "baseline":
        return RuleBasedRecruiter()
    if name == "regressed":
        return RuleBasedRecruiter(set(FAULTS))
    if name.startswith("faulty:"):
        return RuleBasedRecruiter(set(name.removeprefix("faulty:").split(",")))
    if name == "claude" or name.startswith("claude:"):
        from .claude_agent import ClaudeAgent

        return ClaudeAgent(model=name.partition(":")[2] or None)
    raise ValueError(f"unknown agent {name!r}")
