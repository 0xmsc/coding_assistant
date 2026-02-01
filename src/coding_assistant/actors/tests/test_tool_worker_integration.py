import pytest
from typing import Any
from unittest.mock import MagicMock

from coding_assistant.actors.system import ActorSystem
from coding_assistant.framework.agent import run_agent_loop
from coding_assistant.framework.types import AgentContext, AgentDescription, AgentState
from coding_assistant.llm.types import Tool, ToolResult, AssistantMessage, Usage, ToolCall, FunctionCall
from coding_assistant.framework.results import TextResult


class MockTool(Tool):
    def name(self) -> str:
        return "mock_tool"

    def description(self) -> str:
        return "A mock tool"

    def parameters(self) -> dict[str, Any]:
        return {}

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        return TextResult(content=f"Executed with {parameters}")


@pytest.mark.asyncio
async def test_run_agent_loop_with_actor_system() -> None:
    # Setup
    desc = AgentDescription(name="test_agent", model="test_model", parameters=[], tools=[MockTool()])
    state = AgentState()
    ctx = AgentContext(desc=desc, state=state)

    # Mock completer
    # First call: return a tool call
    # Second call: return a result (to break the loop)
    mock_completer = MagicMock()

    completion1 = MagicMock()
    completion1.message = AssistantMessage(
        role="assistant",
        content="I will use the tool.",
        tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="mock_tool", arguments='{"arg1": "val1"}'))],
    )
    completion1.usage = Usage(tokens=100, cost=0.01)

    completion2 = MagicMock()
    completion2.message = AssistantMessage(
        role="assistant",
        content="I am done.",
        tool_calls=[
            ToolCall(
                id="call_2",
                function=FunctionCall(name="finish_task", arguments='{"result": "Success", "summary": "Done"}'),
            )
        ],
    )
    completion2.usage = Usage(tokens=200, cost=0.02)

    # We need to make the completer return an awaitable
    async def side_effect(*args: Any, **kwargs: Any) -> Any:
        if not hasattr(side_effect, "already_called"):
            setattr(side_effect, "already_called", True)
            return completion1
        return completion2

    mock_completer.side_effect = side_effect

    ui = MagicMock()
    system = ActorSystem()

    # Run the loop
    await run_agent_loop(ctx, completer=mock_completer, ui=ui, actor_system=system)

    assert state.output is not None
    assert state.output.result == "Success"
    # Verify history contains tool results
    tool_msgs = [m for m in state.history if m.role == "tool"]
    assert len(tool_msgs) == 2

    # Use content accessor and cast or check types for mypy
    content0 = tool_msgs[0].content
    assert isinstance(content0, str)
    assert "Executed with {'arg1': 'val1'}" in content0

    content1 = tool_msgs[1].content
    assert isinstance(content1, str)
    assert "Agent output set." in content1
