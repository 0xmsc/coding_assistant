from __future__ import annotations

import json
from typing import Any

import pytest

from coding_assistant import AgentRunner, SessionOptions
from coding_assistant.llm.types import AssistantMessage, Completion, FunctionCall, Tool, ToolCall, Usage
from coding_assistant.runtime import AssistantMessageEvent, FinishedEvent, WaitingForUserEvent
from coding_assistant.tool_policy import ToolPolicy
from coding_assistant.tool_results import TextResult


class ScriptedCompleter:
    def __init__(self, script: list[AssistantMessage | Exception]) -> None:
        self.script = list(script)

    async def __call__(self, messages: Any, *, model: Any, tools: Any, callbacks: Any) -> Completion:
        if not self.script:
            raise AssertionError("Completer script exhausted")

        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action

        if isinstance(action.content, str) and action.content:
            callbacks.on_content_chunk(action.content)
        callbacks.on_chunks_end()
        return Completion(message=action, usage=Usage(tokens=10, cost=0.0))


class MockTool(Tool):
    def __init__(self, name: str = "mock_tool", result: str = "tool result") -> None:
        self._name = name
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Mock external tool"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, parameters: dict[str, Any]) -> TextResult:
        self.calls.append(parameters)
        return TextResult(content=self._result)


class DenyPolicy(ToolPolicy):
    async def before_tool_execution(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> TextResult | None:
        return TextResult(content="Tool execution denied.")


def make_options() -> SessionOptions:
    return SessionOptions(compact_conversation_at_tokens=200_000)


@pytest.mark.asyncio
async def test_runner_executes_tool_requests_internally() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments=json.dumps({})),
    )
    finish_call = ToolCall(
        id="finish-1",
        function=FunctionCall(name="finish_task", arguments=json.dumps({"result": "done", "summary": "all set"})),
    )
    tool = MockTool()
    runner = AgentRunner(
        instructions="# Instructions\n\nTest instructions",
        tools=[tool],
        model="test-model",
        expert_model="test-expert-model",
        runtime_options=make_options(),
        completer=ScriptedCompleter(
            [
                AssistantMessage(tool_calls=[external_call]),
                AssistantMessage(tool_calls=[finish_call]),
            ]
        ),
    )

    async with runner:
        await runner.start(mode="agent", task="Finish the task")
        event1 = await runner.next_event()
        event2 = await runner.next_event()
        event3 = await runner.next_event()

        assert isinstance(event1, AssistantMessageEvent)
        assert isinstance(event2, AssistantMessageEvent)
        assert isinstance(event3, FinishedEvent)
        assert tool.calls == [{}]
        tool_messages = [message for message in runner.history if message.role == "tool"]
        assert tool_messages[0].content == "tool result"


@pytest.mark.asyncio
async def test_runner_launch_agent_stays_internal() -> None:
    launch_call = ToolCall(
        id="launch-1",
        function=FunctionCall(
            name="launch_agent",
            arguments=json.dumps({"task": "Child task", "expert_knowledge": False}),
        ),
    )
    child_finish_call = ToolCall(
        id="child-finish",
        function=FunctionCall(name="finish_task", arguments=json.dumps({"result": "child result", "summary": "child"})),
    )
    parent_finish_call = ToolCall(
        id="parent-finish",
        function=FunctionCall(
            name="finish_task", arguments=json.dumps({"result": "parent result", "summary": "parent"})
        ),
    )
    runner = AgentRunner(
        instructions="# Instructions\n\nTest instructions",
        tools=[],
        model="test-model",
        expert_model="test-expert-model",
        runtime_options=make_options(),
        completer=ScriptedCompleter(
            [
                AssistantMessage(tool_calls=[launch_call]),
                AssistantMessage(tool_calls=[child_finish_call]),
                AssistantMessage(tool_calls=[parent_finish_call]),
            ]
        ),
    )

    async with runner:
        await runner.start(mode="agent", task="Parent task")
        event1 = await runner.next_event()
        event2 = await runner.next_event()
        event3 = await runner.next_event()

        assert isinstance(event1, AssistantMessageEvent)
        assert isinstance(event2, AssistantMessageEvent)
        assert isinstance(event3, FinishedEvent)
        tool_messages = [message for message in runner.history if message.role == "tool"]
        assert len(tool_messages) == 2
        assert tool_messages[0].content == "child result"


@pytest.mark.asyncio
async def test_runner_sub_agent_waiting_becomes_tool_result() -> None:
    launch_call = ToolCall(
        id="launch-1",
        function=FunctionCall(
            name="launch_agent",
            arguments=json.dumps({"task": "Child task", "expert_knowledge": False}),
        ),
    )
    runner = AgentRunner(
        instructions="# Instructions\n\nTest instructions",
        tools=[],
        model="test-model",
        expert_model="test-expert-model",
        runtime_options=make_options(),
        completer=ScriptedCompleter(
            [
                AssistantMessage(tool_calls=[launch_call]),
                AssistantMessage(content="I need one more detail"),
                AssistantMessage(content="What is the missing detail?"),
            ]
        ),
    )

    async with runner:
        await runner.start(mode="agent", task="Parent task")
        event1 = await runner.next_event()
        event2 = await runner.next_event()
        event3 = await runner.next_event()
        event4 = await runner.next_event()

        assert isinstance(event1, AssistantMessageEvent)
        assert event2.type == "assistant_delta"
        assert isinstance(event3, AssistantMessageEvent)
        assert isinstance(event4, WaitingForUserEvent)
        tool_messages = [message for message in runner.history if message.role == "tool"]
        assert tool_messages[0].content.startswith("Sub-agent needs more information:")


@pytest.mark.asyncio
async def test_runner_policy_can_deny_tool_execution() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments=json.dumps({})),
    )
    finish_call = ToolCall(
        id="finish-1",
        function=FunctionCall(name="finish_task", arguments=json.dumps({"result": "done", "summary": "all set"})),
    )
    tool = MockTool()
    runner = AgentRunner(
        instructions="# Instructions\n\nTest instructions",
        tools=[tool],
        model="test-model",
        expert_model="test-expert-model",
        runtime_options=make_options(),
        completer=ScriptedCompleter(
            [
                AssistantMessage(tool_calls=[external_call]),
                AssistantMessage(tool_calls=[finish_call]),
            ]
        ),
        tool_policy=DenyPolicy(),
    )

    async with runner:
        await runner.start(mode="agent", task="Parent task")
        await runner.next_event()
        await runner.next_event()
        await runner.next_event()

        assert tool.calls == []
        tool_messages = [message for message in runner.history if message.role == "tool"]
        assert tool_messages[0].content == "Tool execution denied."
