"""Serve one scenario's sandbox over the Model Context Protocol.

Any agent that speaks MCP can then be evaluated with no adapter code: point it
at this server, let it do the task, and grade the transcript the server wrote.
The task itself is delivered as the server's instructions.
"""

from __future__ import annotations

import json
from pathlib import Path

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from . import __version__
from .models import Scenario, Transcript
from .runner import new_session
from .session import StepLimitExceeded, ToolSession


def build_server(scenario: Scenario, session: ToolSession, out: Path | None = None) -> Server:
    error: str | None = None

    def save() -> None:
        # Written after every call, so the record survives a client that just disconnects.
        if out:
            transcript = Transcript(calls=session.calls, error=error)
            out.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=spec["name"],
                    description=spec["description"],
                    input_schema=spec["input_schema"],
                )
                for spec in session.tools
            ]
        )

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        nonlocal error
        try:
            call = session.call(params.name, params.arguments or {})
        except StepLimitExceeded as exc:
            error = str(exc)
            save()
            return _text(f"{exc}. Stop calling tools.", is_error=True)
        save()
        if call.error:
            return _text(call.error, is_error=True)
        return _text(json.dumps(call.result))

    save()
    return Server(
        "agent-evals-sandbox",
        version=__version__,
        instructions=scenario.prompt,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def _text(text: str, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)], is_error=is_error
    )


def serve(scenario: Scenario, out: Path) -> None:
    """Run the server on stdio until the client disconnects."""
    _, session = new_session(scenario)
    server = build_server(scenario, session, out)

    async def main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)
