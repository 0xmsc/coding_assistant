from __future__ import annotations

import json
from typing import Any

import pytest

from coding_assistant import AssistantSession as RootAssistantSession
from coding_assistant.llm.types import (
    AssistantMessage,
    Completion,
    FunctionCall,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    Usage,
    UserMessage,
)
from coding_assistant.runtime import (
    AssistantDeltaEvent,
    AssistantMessageEvent,
    AssistantSession,
    CancelledEvent,
    FailedEvent,
    InputRequestedEvent,
    ToolCallRequestedEvent,
)


class ScriptedCompleter:
    def __init__(self, script: list[AssistantMessage | Exception]) -> None:
        self.script = list(script)
        self.before_completion: Any = None

    async def __call__(self, messages: Any, *, model: Any, tools: Any, callbacks: Any) -> Completion:
        if self.before_completion is not None:
            await self.before_completion()

        if not self.script:
            raise AssertionError("Completer script exhausted")

        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action

        if isinstance(action.content, str) and action.content:
            callbacks.on_content_chunk(action.content)
        callbacks.on_chunks_end()
        return Completion(message=action, usage=Usage(tokens=10, cost=0.0))


class MockToolDefinition(ToolDefinition):
    def __init__(self, name: str = "mock_tool") -> None:
        self._name = name

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Mock external tool"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}


def make_system_history() -> list[SystemMessage]:
    return [SystemMessage(content="# Instructions\n\nTest instructions")]


def test_root_package_exports_runtime_types() -> None:
    assert RootAssistantSession is AssistantSession


@pytest.mark.asyncio
async def test_start_without_pending_model_turn_emits_input_requested() -> None:
    session = AssistantSession(tools=[])

    async with session:
        await session.start(history=make_system_history(), model="test-model")
        event = await session.next_event()
        assert isinstance(event, InputRequestedEvent)
        assert session.history[-1].role == "system"


@pytest.mark.asyncio
async def test_user_reply_emits_delta_message_and_input_requested() -> None:
    completer = ScriptedCompleter([AssistantMessage(content="Hello from the assistant")])
    session = AssistantSession(
        tools=[],
        completer=completer,
    )

    async with session:
        await session.start(history=make_system_history(), model="test-model")
        waiting = await session.next_event()
        assert isinstance(waiting, InputRequestedEvent)

        await session.send_user_message("Hi")
        event1 = await session.next_event()
        event2 = await session.next_event()
        event3 = await session.next_event()

        assert isinstance(event1, AssistantDeltaEvent)
        assert event1.delta == "Hello from the assistant"
        assert isinstance(event2, AssistantMessageEvent)
        assert event2.message.content == "Hello from the assistant"
        assert isinstance(event3, InputRequestedEvent)
        assert [message.role for message in session.history] == ["system", "user", "assistant"]


@pytest.mark.asyncio
async def test_session_requires_non_empty_history() -> None:
    session = AssistantSession(tools=[])

    async with session:
        with pytest.raises(ValueError, match="non-empty history"):
            await session.start(history=[], model="test-model")


@pytest.mark.asyncio
async def test_external_tool_call_emits_request_and_accepts_result() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments=json.dumps({})),
    )
    completer = ScriptedCompleter(
        [
            AssistantMessage(tool_calls=[external_call]),
            AssistantMessage(content="Tool handled."),
        ]
    )
    session = AssistantSession(
        tools=[MockToolDefinition()],
        completer=completer,
    )

    async with session:
        await session.start(history=make_system_history(), model="test-model")
        waiting = await session.next_event()
        assert isinstance(waiting, InputRequestedEvent)

        await session.send_user_message("Finish the task")
        event1 = await session.next_event()
        event2 = await session.next_event()

        assert isinstance(event1, AssistantMessageEvent)
        assert isinstance(event2, ToolCallRequestedEvent)
        assert event2.tool_call.id == "call-1"
        assert event2.tool_call.function.name == "mock_tool"
        assert event2.arguments == {}

        await session.submit_tool_result("call-1", "external result")
        event3 = await session.next_event()
        event4 = await session.next_event()
        event5 = await session.next_event()

        assert isinstance(event3, AssistantDeltaEvent)
        assert event3.delta == "Tool handled."
        assert isinstance(event4, AssistantMessageEvent)
        assert event4.message.content == "Tool handled."
        assert isinstance(event5, InputRequestedEvent)
        tool_messages = [message for message in session.history if message.role == "tool"]
        assert tool_messages[0].content == "external result"


@pytest.mark.asyncio
async def test_submit_tool_error_resumes_model_loop() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments=json.dumps({})),
    )
    completer = ScriptedCompleter(
        [
            AssistantMessage(tool_calls=[external_call]),
            AssistantMessage(content="Need more input"),
        ]
    )
    session = AssistantSession(
        tools=[MockToolDefinition()],
        completer=completer,
    )

    async with session:
        await session.start(history=make_system_history(), model="test-model")
        await session.next_event()
        await session.send_user_message("Handle an error")
        await session.next_event()
        event2 = await session.next_event()
        assert isinstance(event2, ToolCallRequestedEvent)

        await session.submit_tool_error("call-1", "boom")
        event3 = await session.next_event()
        event4 = await session.next_event()
        event5 = await session.next_event()

        assert isinstance(event3, AssistantDeltaEvent)
        assert event3.delta == "Need more input"
        assert isinstance(event4, AssistantMessageEvent)
        assert event4.message.content == "Need more input"
        assert isinstance(event5, InputRequestedEvent)
        tool_messages = [message for message in session.history if message.role == "tool"]
        assert tool_messages[0].content == "Error executing tool: boom"


@pytest.mark.asyncio
async def test_replace_history_rewrites_transcript() -> None:
    session = AssistantSession(tools=[])

    async with session:
        await session.start(history=make_system_history(), model="test-model")
        await session.next_event()
        session.replace_history(
            [
                make_system_history()[0],
                UserMessage(content="A summary of your conversation until now:\n\nShort summary\n\nPlease continue."),
            ]
        )

        assert [message.role for message in session.history] == ["system", "user"]
        assert isinstance(session.history[-1].content, str)
        assert "Short summary" in session.history[-1].content


@pytest.mark.asyncio
async def test_cancel_pending_tool_request_emits_cancelled_event() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments=json.dumps({})),
    )
    completer = ScriptedCompleter([AssistantMessage(tool_calls=[external_call])])
    session = AssistantSession(
        tools=[MockToolDefinition()],
        completer=completer,
    )

    async with session:
        await session.start(history=make_system_history(), model="test-model")
        await session.next_event()
        await session.send_user_message("Wait on a tool")
        await session.next_event()
        event2 = await session.next_event()
        assert isinstance(event2, ToolCallRequestedEvent)

        await session.cancel()
        event3 = await session.next_event()
        assert isinstance(event3, CancelledEvent)


@pytest.mark.asyncio
async def test_failure_emits_failed_event() -> None:
    completer = ScriptedCompleter([RuntimeError("boom")])
    session = AssistantSession(
        tools=[],
        completer=completer,
    )

    async with session:
        await session.start(history=make_system_history(), model="test-model")
        await session.next_event()
        await session.send_user_message("Fail")
        event = await session.next_event()
        assert isinstance(event, FailedEvent)
        assert event.error == "boom"
