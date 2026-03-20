from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from coding_assistant import AssistantSession as RootAssistantSession, SessionOptions as RootSessionOptions
from coding_assistant.llm.types import AssistantMessage, Completion, FunctionCall, ToolCall, Usage
from coding_assistant.runtime import (
    AssistantDeltaEvent,
    AssistantMessageEvent,
    AssistantSession,
    CancelledEvent,
    FailedEvent,
    FileHistoryStore,
    FinishedEvent,
    SessionOptions,
    WaitingForUserEvent,
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


def make_options(tmp_path: Path, *, history_store: FileHistoryStore | None = None) -> SessionOptions:
    return SessionOptions(
        model="test-model",
        expert_model="test-expert-model",
        compact_conversation_at_tokens=200_000,
        working_directory=tmp_path,
        mcp_server_configs=(),
        history_store=history_store,
    )


@pytest.fixture
def mocked_runtime_services() -> Any:
    with (
        patch("coding_assistant.runtime.session.get_mcp_servers_from_config") as mock_get_mcp,
        patch("coding_assistant.runtime.session.get_mcp_wrapped_tools", new_callable=AsyncMock) as mock_get_tools,
        patch("coding_assistant.runtime.session.get_instructions") as mock_get_instructions,
    ):
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = ["mock-server"]
        mock_get_mcp.return_value = mock_cm
        mock_get_tools.return_value = []
        mock_get_instructions.return_value = "# Instructions\n\nTest instructions"
        yield {
            "get_mcp": mock_get_mcp,
            "get_tools": mock_get_tools,
            "get_instructions": mock_get_instructions,
            "cm": mock_cm,
        }


def test_get_default_mcp_server_config() -> None:
    config = AssistantSession.get_default_mcp_server_config(Path("/root"), ["skills"], env=["OPENAI_API_KEY"])
    assert config.name == "coding_assistant.mcp"
    assert "--skills-directories" in config.args
    assert "skills" in config.args
    assert config.env == ["OPENAI_API_KEY"]


def test_root_package_exports_runtime_types() -> None:
    assert RootAssistantSession is AssistantSession
    assert RootSessionOptions is SessionOptions


@pytest.mark.asyncio
async def test_session_context_manager_initializes_shared_state(tmp_path: Path, mocked_runtime_services: Any) -> None:
    session = AssistantSession(options=make_options(tmp_path))

    async with session:
        assert session.instructions == "# Instructions\n\nTest instructions"
        assert cast(list[Any], session.mcp_servers) == ["mock-server"]

    mocked_runtime_services["cm"].__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_chat_start_emits_waiting_for_user(tmp_path: Path, mocked_runtime_services: Any) -> None:
    session = AssistantSession(options=make_options(tmp_path))

    async with session:
        await session.start(mode="chat")
        event = await session.next_event()
        assert isinstance(event, WaitingForUserEvent)
        assert session.history[-1].role == "user"


@pytest.mark.asyncio
async def test_chat_reply_emits_delta_message_and_waiting(tmp_path: Path, mocked_runtime_services: Any) -> None:
    completer = ScriptedCompleter([AssistantMessage(content="Hello from the assistant")])
    session = AssistantSession(options=make_options(tmp_path), completer=completer)

    async with session:
        await session.start(mode="chat")
        waiting = await session.next_event()
        assert isinstance(waiting, WaitingForUserEvent)

        await session.send_user_message("Hi")
        event1 = await session.next_event()
        event2 = await session.next_event()
        event3 = await session.next_event()

        assert isinstance(event1, AssistantDeltaEvent)
        assert event1.delta == "Hello from the assistant"
        assert isinstance(event2, AssistantMessageEvent)
        assert event2.message.content == "Hello from the assistant"
        assert isinstance(event3, WaitingForUserEvent)
        assert [message.role for message in session.history] == ["user", "user", "assistant"]


@pytest.mark.asyncio
async def test_agent_finish_emits_finished_event(tmp_path: Path, mocked_runtime_services: Any) -> None:
    finish_call = ToolCall(
        id="finish-1",
        function=FunctionCall(name="finish_task", arguments=json.dumps({"result": "done", "summary": "all set"})),
    )
    completer = ScriptedCompleter([AssistantMessage(tool_calls=[finish_call])])
    session = AssistantSession(options=make_options(tmp_path), completer=completer)

    async with session:
        await session.start(mode="agent", task="Finish the task")
        event1 = await session.next_event()
        event2 = await session.next_event()

        assert isinstance(event1, AssistantMessageEvent)
        assert isinstance(event2, FinishedEvent)
        assert event2.result == "done"
        assert event2.summary == "all set"
        assert session.history[-1].role == "tool"


@pytest.mark.asyncio
async def test_sub_agent_completion_stays_internal_to_parent(tmp_path: Path, mocked_runtime_services: Any) -> None:
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
    completer = ScriptedCompleter(
        [
            AssistantMessage(tool_calls=[launch_call]),
            AssistantMessage(tool_calls=[child_finish_call]),
            AssistantMessage(tool_calls=[parent_finish_call]),
        ]
    )
    session = AssistantSession(options=make_options(tmp_path), completer=completer)

    async with session:
        await session.start(mode="agent", task="Parent task")
        event1 = await session.next_event()
        event2 = await session.next_event()
        event3 = await session.next_event()

        assert isinstance(event1, AssistantMessageEvent)
        assert isinstance(event2, AssistantMessageEvent)
        assert isinstance(event3, FinishedEvent)
        tool_messages = [message for message in session.history if message.role == "tool"]
        assert len(tool_messages) == 2
        assert tool_messages[0].content == "child result"


@pytest.mark.asyncio
async def test_sub_agent_waiting_is_converted_into_parent_tool_result(
    tmp_path: Path, mocked_runtime_services: Any
) -> None:
    launch_call = ToolCall(
        id="launch-1",
        function=FunctionCall(
            name="launch_agent",
            arguments=json.dumps({"task": "Child task", "expert_knowledge": False}),
        ),
    )
    parent_reply = AssistantMessage(content="What is the missing detail?")
    completer = ScriptedCompleter(
        [
            AssistantMessage(tool_calls=[launch_call]),
            AssistantMessage(content="I need one more detail"),
            parent_reply,
        ]
    )
    session = AssistantSession(options=make_options(tmp_path), completer=completer)

    async with session:
        await session.start(mode="agent", task="Parent task")
        await session.next_event()
        event2 = await session.next_event()
        event3 = await session.next_event()
        waiting = await session.next_event()

        assert isinstance(event2, AssistantDeltaEvent)
        assert event2.delta == "What is the missing detail?"
        assert isinstance(event3, AssistantMessageEvent)
        assert event3.message.content == "What is the missing detail?"
        assert isinstance(waiting, WaitingForUserEvent)
        tool_messages = [message for message in session.history if message.role == "tool"]
        assert isinstance(tool_messages[0].content, str)
        assert tool_messages[0].content.startswith("Sub-agent needs more information:")


@pytest.mark.asyncio
async def test_cancel_emits_cancelled_event(tmp_path: Path, mocked_runtime_services: Any) -> None:
    completer = ScriptedCompleter([AssistantMessage(content="never returned")])
    gate = asyncio.Event()

    async def before_completion() -> None:
        await gate.wait()

    completer.before_completion = before_completion
    session = AssistantSession(options=make_options(tmp_path), completer=completer)

    async with session:
        await session.start(mode="agent", task="Long running")
        await session.cancel()
        event = await session.next_event()
        assert isinstance(event, CancelledEvent)


@pytest.mark.asyncio
async def test_failure_emits_failed_event(tmp_path: Path, mocked_runtime_services: Any) -> None:
    completer = ScriptedCompleter([RuntimeError("boom")])
    session = AssistantSession(options=make_options(tmp_path), completer=completer)

    async with session:
        await session.start(mode="agent", task="Fail")
        event = await session.next_event()
        assert isinstance(event, FailedEvent)
        assert event.error == "boom"
