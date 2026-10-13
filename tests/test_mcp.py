"""The sandbox as an MCP server, driven by the MCP SDK's own client."""

import json
import sys

import anyio
from mcp import Client, StdioServerParameters

from agent_evals.cli import main
from agent_evals.mcp_server import build_server
from agent_evals.models import Transcript
from agent_evals.runner import new_session

from .test_cli_api import SCENARIOS_DIR


def scenario_by_id(scenarios, scenario_id):
    return next(s for s in scenarios if s.id == scenario_id)


def test_tools_task_and_errors_are_exposed_over_mcp(scenarios, tmp_path):
    scenario = scenario_by_id(scenarios, "unknown-candidate")
    out = tmp_path / "run.json"
    _, session = new_session(scenario)

    async def drive():
        async with Client(build_server(scenario, session, out)) as client:
            tools = await client.list_tools()
            job = await client.call_tool("get_job", {})
            missing = await client.call_tool("get_resume", {"candidate_id": "c9"})
            return client.instructions, tools, job, missing

    instructions, tools, job, missing = anyio.run(drive)

    assert instructions == scenario.prompt
    assert [t.name for t in tools.tools] == [s["name"] for s in session.tools]
    assert tools.tools[2].input_schema["required"] == ["candidate_id"]
    assert json.loads(job.content[0].text)["must_have"] == ["python", "fastapi"]
    assert missing.is_error
    assert "Unknown candidate id 'c9'" in missing.content[0].text

    # Every call went through the recording session and is already on disk.
    saved = Transcript.model_validate_json(out.read_text(encoding="utf-8"))
    assert [c.name for c in saved.calls] == ["get_job", "get_resume"]
    assert saved.calls[1].error


def test_the_step_limit_reaches_the_client_and_the_transcript(scenarios, tmp_path):
    scenario = scenario_by_id(scenarios, "unknown-candidate").model_copy(update={"max_steps": 1})
    out = tmp_path / "run.json"
    _, session = new_session(scenario)

    async def drive():
        async with Client(build_server(scenario, session, out)) as client:
            await client.call_tool("get_job", {})
            return await client.call_tool("get_job", {})

    over_limit = anyio.run(drive)
    assert over_limit.is_error and "1-call limit" in over_limit.content[0].text
    assert "1-call limit" in Transcript.model_validate_json(out.read_text(encoding="utf-8")).error


def test_end_to_end_over_stdio_then_grade(tmp_path, capsys):
    """A real subprocess on a real pipe: what an external MCP agent would connect to."""
    out = tmp_path / "run.json"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_evals.cli", "serve-mcp", "--scenarios", SCENARIOS_DIR]
        + ["--scenario", "unknown-candidate", "--out", str(out)],
    )

    async def act_as_a_careful_agent():
        async with Client(server) as client:
            lookup = await client.call_tool("get_resume", {"candidate_id": "c9"})
            assert lookup.is_error
            report = {"scheduled": [], "rejected": [], "summary": "c9 does not exist."}
            await client.call_tool("submit_report", report)

    anyio.run(act_as_a_careful_agent)

    grade = ["grade", "--scenarios", SCENARIOS_DIR, "--scenario", "unknown-candidate", str(out)]
    assert main(grade) == 0
    assert capsys.readouterr().out.startswith("PASS  unknown-candidate")
