import json

import pytest

from coding_assistant.agents.tests.helpers import make_ui_mock
from coding_assistant.tools.tools import AskClientTool
from coding_assistant.ui import NullUI


@pytest.mark.asyncio
async def test_ask_client_tool_enabled():
    ui = make_ui_mock(ask_sequence=[("Do you want to proceed?", "yes")])
    tool = AskClientTool(enabled=True, ui=ui)
    result = await tool.execute({"question": "Do you want to proceed?", "default_answer": "no"})
    assert result.content == "yes"


@pytest.mark.asyncio
async def test_ask_client_tool_disabled():
    tool = AskClientTool(enabled=False, ui=NullUI())
    result = await tool.execute({"question": "Do you want to proceed?", "default_answer": "no"})
    assert (
        result.content
        == "Client input is disabled for this session. Please continue as if the client had given the most sensible answer to your question."
    )
