import json

import pytest
from unittest.mock import Mock

from coding_assistant.agents.callbacks import NullCallbacks
from coding_assistant.agents.execution import run_agent_loop
from coding_assistant.agents.tests.helpers import (
    FakeCompleter,
    FakeFunction,
    FakeMessage,
    FakeToolCall,
    make_test_agent,
    make_ui_mock,
)
from coding_assistant.tools.mcp import MCPServer
from coding_assistant.tools.tools import FinishTaskTool, ShortenConversation


@pytest.mark.asyncio
async def test_start_message_includes_mcp_instructions():
    # Prepare an MCP server with instructions and an async-capable session
    class FakeSession:
        async def list_tools(self):
            class R:
                tools = []

            return R()

    server = MCPServer(name="fs", session=FakeSession(), instructions="Use the filesystem tool to browse.")

    # Make the agent finish immediately so we can inspect the first history entry
    finish_call = FakeToolCall(
        "1",
        FakeFunction(
            "finish_task",
            json.dumps({"result": "done", "summary": "s"}),
        ),
    )
    completer = FakeCompleter([FakeMessage(tool_calls=[finish_call])])

    agent = make_test_agent(tools=[FinishTaskTool(), ShortenConversation()], mcp_servers=[server])

    await run_agent_loop(
        agent,
        NullCallbacks(),
        shorten_conversation_at_tokens=200_000,
        no_truncate_tools=set(),
        enable_user_feedback=False,
        completer=completer,
        ui=make_ui_mock(),
    )

    first = agent.history[0]
    assert first["role"] == "user"
    assert "## MCP server `fs` instructions" in first["content"]
    assert "Use the filesystem tool to browse." in first["content"]
