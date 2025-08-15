from unittest.mock import MagicMock
import pytest
from coding_assistant.agents.execution import handle_tool_call
from coding_assistant.agents.types import Agent, TextResult, Tool
from coding_assistant.llm.types import ToolCall

class MockTool(Tool):
    def __init__(self, name, content):
        self._name = name
        self._content = content

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "A mock tool for testing."

    def parameters(self) -> dict:
        return {}

    async def execute(self, parameters) -> TextResult:
        return TextResult(content=self._content)

@pytest.mark.asyncio
async def test_handle_tool_call_truncation():
    agent = Agent(
        name="Test Agent",
        model="test_model",
        description="A test agent.",
        parameters=[],
        feedback_function=lambda agent: None,
    )
    agent_callbacks = MagicMock()

    # Test case 1: No truncation
    tool_call = ToolCall(
        id="1",
        function=MagicMock(name="test_tool", arguments="{}"),
    )
    agent.tools = [MockTool("test_tool", "a" * 100)]
    agent.no_truncate_tools = ["test_tool"]
    await handle_tool_call(tool_call, agent, agent_callbacks)
    assert agent.history[-1]["content"] == "a" * 100

    # Test case 2: Truncation
    tool_call = ToolCall(
        id="2",
        function=MagicMock(name="test_tool", arguments="{}"),
    )
    agent.tools = [MockTool("test_tool", "a" * 60000)]
    agent.no_truncate_tools = []
    await handle_tool_call(tool_call, agent, agent_callbacks)
    assert agent.history[-1]["content"] == "System error: Tool call result too long."

    # Test case 3: No truncation with regex
    tool_call = ToolCall(
        id="3",
        function=MagicMock(name="test_tool_123", arguments="{}"),
    )
    agent.tools = [MockTool("test_tool_123", "a" * 60000)]
    agent.no_truncate_tools = ["test_tool.*"]
    await handle_tool_call(tool_call, agent, agent_callbacks)
    assert agent.history[-1]["content"] == "a" * 60000
