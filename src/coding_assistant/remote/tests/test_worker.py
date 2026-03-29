from __future__ import annotations

import asyncio
from argparse import Namespace
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from rich.markdown import Markdown

from coding_assistant.app.default_agent import DefaultAgentBundle, DefaultAgentConfig
from coding_assistant.core.history import build_system_prompt
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    FunctionCall,
    SystemMessage,
    ToolCall,
    Usage,
)
from coding_assistant.remote.server import WorkerServer
from coding_assistant.remote.worker import _run_worker_output, run_worker
from coding_assistant.remote.worker_session import (
    RunCancelledEvent,
    RunFinishedEvent,
    StateChangedEvent,
    ToolCallsEvent,
    WorkerSession,
    WorkerSessionEvent,
)


class ScriptedStreamer:
    def __init__(self, script: list[AssistantMessage]) -> None:
        self.script = list(script)

    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[object]:
        del messages, tools, model
        if not self.script:
            raise AssertionError("Streamer script exhausted.")

        action = self.script.pop(0)
        if isinstance(action.content, str) and action.content:
            yield ContentDeltaEvent(content=action.content)

        yield CompletionEvent(completion=Completion(message=action, usage=Usage(tokens=10, cost=0.0)))


class BlockingStreamer:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[object]:
        del messages, tools, model
        self.started.set()
        yield ContentDeltaEvent(content="Working...")
        await asyncio.Future()


def make_system_history() -> list[BaseMessage]:
    return [SystemMessage(content="# Instructions\n\nTest instructions")]


def make_worker_session(*, completion_streamer: Any) -> WorkerSession:
    return WorkerSession(
        history=make_system_history(),
        model="test-model",
        tools=[],
        completion_streamer=completion_streamer,
    )


async def wait_for_event(queue: asyncio.Queue[WorkerSessionEvent], event_type: type[object]) -> object:
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=1)
        if isinstance(event, event_type):
            return event


async def wait_for_matching_event(
    queue: asyncio.Queue[WorkerSessionEvent],
    predicate: Callable[[object], bool],
) -> object:
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=1)
        if predicate(event):
            return event


@pytest.mark.asyncio
async def test_worker_session_runs_prompt_and_updates_history() -> None:
    session = make_worker_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="Hello from the worker")]),
    )

    async with session.subscribe() as queue:
        initial_state = await wait_for_event(queue, StateChangedEvent)
        assert isinstance(initial_state, StateChangedEvent)

        assert await session.submit_prompt("Hi") is True

        finished_event = await wait_for_event(queue, RunFinishedEvent)

    assert isinstance(finished_event, RunFinishedEvent)
    assert finished_event.summary == "Hello from the worker"
    assert session.history[-1] == AssistantMessage(content="Hello from the worker")
    assert session.state.promptable is True
    assert session.state.running is False


@pytest.mark.asyncio
async def test_worker_session_rejects_new_prompt_while_run_is_in_flight() -> None:
    session = make_worker_session(completion_streamer=BlockingStreamer())

    assert await session.submit_prompt("Hi") is True
    assert await session.submit_prompt("Again") is False
    assert await session.cancel_current_run() is True
    assert session.state.promptable is True


@pytest.mark.asyncio
async def test_worker_session_cancel_current_run_publishes_cancellation_and_restores_state() -> None:
    streamer = BlockingStreamer()
    session = make_worker_session(completion_streamer=streamer)

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.submit_prompt("Do the task") is True

        await asyncio.wait_for(streamer.started.wait(), timeout=1)
        await session.cancel_current_run()

        cancelled_event = await wait_for_event(queue, RunCancelledEvent)
        assert isinstance(cancelled_event, RunCancelledEvent)

        state_event = await wait_for_matching_event(
            queue,
            lambda event: isinstance(event, StateChangedEvent) and event.state.promptable and not event.state.running,
        )
        assert isinstance(state_event, StateChangedEvent)
        assert state_event.state.promptable is True
        assert state_event.state.running is False


@pytest.mark.asyncio
async def test_worker_output_renders_system_message_and_streamed_content() -> None:
    session = make_worker_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="Hello from the worker")]),
    )
    system_message = SystemMessage(content="System")

    with (
        patch("coding_assistant.remote.worker.print_system_message") as mock_print_system,
        patch("coding_assistant.app.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(_run_worker_output(session=session, system_message=system_message))
        try:
            await asyncio.sleep(0)
            assert await session.submit_prompt("Hi") is True

            while session.state.running:
                await asyncio.sleep(0)

            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    mock_print_system.assert_called_once_with(system_message)
    markdown_blocks = [
        call.args[0] for call in mock_rich_print.call_args_list if call.args and isinstance(call.args[0], Markdown)
    ]
    assert [block.markup for block in markdown_blocks] == ["Hello from the worker"]


@pytest.mark.asyncio
async def test_worker_output_prints_tool_calls_without_extra_spacing() -> None:
    session = make_worker_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )
    system_message = SystemMessage(content="System")
    tool_call_message = AssistantMessage(
        tool_calls=[
            ToolCall(
                id="call-1",
                function=FunctionCall(
                    name="shell_execute",
                    arguments='{"command": "cat README.md"}',
                ),
            )
        ]
    )

    with (
        patch("coding_assistant.remote.worker.print_system_message"),
        patch("coding_assistant.app.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(_run_worker_output(session=session, system_message=system_message))
        try:
            await asyncio.sleep(0)
            session._publish_event(ContentDeltaEvent(content="Can you read README.md?"))
            session._publish_event(ToolCallsEvent(message=tool_call_message))
            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert len(mock_rich_print.call_args_list) == 4
    assert mock_rich_print.call_args_list[0].args == ()
    assert isinstance(mock_rich_print.call_args_list[1].args[0], Markdown)
    assert mock_rich_print.call_args_list[1].args[0].markup == "Can you read README.md?"
    assert mock_rich_print.call_args_list[2].args == ()
    assert mock_rich_print.call_args_list[3].args == (
        '[bold yellow]▶[/bold yellow] shell_execute(command="cat README.md")',
    )


@pytest.mark.asyncio
async def test_run_worker_starts_worker_server_without_worker_tools_and_with_local_output(tmp_path: Path) -> None:
    args = Namespace(
        instructions=[],
        mcp_servers=[],
        model="gpt-4",
        print_mcp_tools=False,
        skills_directories=[],
        trace=False,
        wait_for_debugger=False,
        worker=True,
    )
    config = DefaultAgentConfig(working_directory=tmp_path)
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def fake_create_default_agent(*, config: Any, include_worker_tools: bool = True) -> Any:
        captured["config"] = config
        captured["include_worker_tools"] = include_worker_tools
        yield DefaultAgentBundle(
            tools=[],
            instructions="Follow the repo instructions.",
            mcp_servers=[],
        )

    @asynccontextmanager
    async def fake_start_worker_server(*, session: Any) -> Any:
        captured["session"] = session
        yield WorkerServer(endpoint="ws://127.0.0.1:1234")

    async def fake_run_worker_output(*, session: Any, system_message: Any) -> None:
        captured["output_session"] = session
        captured["output_system_message"] = system_message
        await asyncio.Future()

    def resolved_future() -> asyncio.Future[None]:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        future.set_result(None)
        return future

    with (
        patch("coding_assistant.remote.worker.build_default_agent_config", return_value=config),
        patch("coding_assistant.remote.worker.create_default_agent", fake_create_default_agent),
        patch("coding_assistant.remote.worker.start_worker_server", fake_start_worker_server),
        patch("coding_assistant.remote.worker._run_worker_output", new=fake_run_worker_output),
        patch("coding_assistant.remote.worker.asyncio.Future", new=resolved_future),
    ):
        await run_worker(args)

    assert captured["config"] == config
    assert captured["include_worker_tools"] is False
    assert isinstance(captured["session"], WorkerSession)
    assert captured["session"].history == [
        SystemMessage(content=build_system_prompt(instructions="Follow the repo instructions.")),
    ]
    assert captured["output_session"] is captured["session"]
