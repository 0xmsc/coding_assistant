from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from coding_assistant.app.session_control import (
    RunCancelledEvent,
    RunFinishedEvent,
    SessionController,
    SessionEvent,
    StateChangedEvent,
)
from coding_assistant.app.session_runtime import SessionRuntime
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    SystemMessage,
    Usage,
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


class PassiveController:
    async def run(self, session: Any) -> None:
        del session
        await asyncio.Future()


def make_session_runtime(*, completion_streamer: Any) -> tuple[SessionRuntime, SessionController, SessionController]:
    cli_controller = PassiveController()
    remote_controller = PassiveController()
    return (
        SessionRuntime(
            history=make_system_history(),
            model="test-model",
            tools=[],
            default_controller=cli_controller,
            completion_streamer=completion_streamer,
        ),
        cli_controller,
        remote_controller,
    )


async def wait_for_event(queue: asyncio.Queue[SessionEvent], event_type: type[object]) -> object:
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=1)
        if isinstance(event, event_type):
            return event


async def wait_for_matching_event(
    queue: asyncio.Queue[SessionEvent],
    predicate: Callable[[object], bool],
) -> object:
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=1)
        if predicate(event):
            return event


@pytest.mark.asyncio
async def test_session_runtime_runs_local_prompt_and_updates_history() -> None:
    session_runtime, cli_controller, _remote_controller = make_session_runtime(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="Hello from the worker")]),
    )

    async with session_runtime.subscribe() as queue:
        initial_state = await wait_for_event(queue, StateChangedEvent)
        assert isinstance(initial_state, StateChangedEvent)

        result = await session_runtime.submit_prompt(controller=cli_controller, content="Hi")
        assert result.accepted is True

        finished_event = await wait_for_event(queue, RunFinishedEvent)

    assert isinstance(finished_event, RunFinishedEvent)
    assert finished_event.summary == "Hello from the worker"
    assert session_runtime.history[-1] == AssistantMessage(content="Hello from the worker")
    assert session_runtime.state.promptable is True
    assert session_runtime.state.running is False
    assert session_runtime.is_active_controller(cli_controller) is True


@pytest.mark.asyncio
async def test_session_runtime_rejects_local_prompt_when_remote_controller_is_connected() -> None:
    session_runtime, cli_controller, remote_controller = make_session_runtime(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )

    assert await session_runtime.activate_controller(remote_controller) is True

    result = await session_runtime.submit_prompt(controller=cli_controller, content="Hi")

    assert result.accepted is False
    assert result.reason == "inactive_controller"


@pytest.mark.asyncio
async def test_session_runtime_rejects_new_prompt_while_run_is_in_flight() -> None:
    session_runtime, cli_controller, _remote_controller = make_session_runtime(completion_streamer=BlockingStreamer())

    first_result = await session_runtime.submit_prompt(controller=cli_controller, content="Hi")
    assert first_result.accepted is True

    second_result = await session_runtime.submit_prompt(controller=cli_controller, content="Again")

    assert second_result.accepted is False
    assert second_result.reason == "not_ready"

    assert await session_runtime.cancel_current_run() is True
    assert session_runtime.state.promptable is True


@pytest.mark.asyncio
async def test_session_runtime_activate_default_controller_cancels_run_and_restores_local_input() -> None:
    streamer = BlockingStreamer()
    session_runtime, cli_controller, remote_controller = make_session_runtime(completion_streamer=streamer)

    async with session_runtime.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session_runtime.activate_controller(remote_controller) is True

        prompt_result = await session_runtime.submit_prompt(controller=remote_controller, content="Do the task")
        assert prompt_result.accepted is True

        await asyncio.wait_for(streamer.started.wait(), timeout=1)
        await session_runtime.activate_default_controller(cancel_current_run=True)

        cancelled_event = await wait_for_event(queue, RunCancelledEvent)
        assert isinstance(cancelled_event, RunCancelledEvent)

        state_event = await wait_for_matching_event(
            queue,
            lambda event: isinstance(event, StateChangedEvent) and event.state.promptable and not event.state.running,
        )
        assert isinstance(state_event, StateChangedEvent)
        assert state_event.state.promptable is True
        assert state_event.state.running is False
        assert session_runtime.is_active_controller(cli_controller) is True


@pytest.mark.asyncio
async def test_session_runtime_wait_for_controller_change_returns_new_controller() -> None:
    session_runtime, cli_controller, remote_controller = make_session_runtime(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )

    wait_task = asyncio.create_task(session_runtime.wait_for_controller_change(controller=cli_controller))

    assert await session_runtime.activate_controller(remote_controller) is True

    assert await asyncio.wait_for(wait_task, timeout=1) is remote_controller


class ShutdownController:
    async def run(self, session: Any) -> None:
        session.request_shutdown()


class RecordingOutput:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, session: Any) -> None:
        self.started.set()
        await asyncio.Future()


@pytest.mark.asyncio
async def test_session_runtime_run_starts_outputs_before_controllers_and_stops_on_shutdown() -> None:
    session_runtime, _cli_controller, _remote_controller = make_session_runtime(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )
    output = RecordingOutput()

    await session_runtime.run(
        controllers=[ShutdownController()],
        outputs=[output],
    )

    assert output.started.is_set() is True
