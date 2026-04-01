from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from coding_assistant.core.agent_session import AgentSession
from coding_assistant.llm.types import (
    AssistantMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    SystemMessage,
    Usage,
    UserMessage,
)
from coding_assistant.remote.server import start_worker_server
from coding_assistant.tools.workers import WorkerToolRuntime


@dataclass
class StreamStep:
    message: AssistantMessage
    started_event: asyncio.Event | None = None
    release_event: asyncio.Event | None = None


class ControlledStreamer:
    def __init__(self, steps: list[StreamStep]) -> None:
        self.steps = list(steps)
        self.prompts: list[str | list[dict[str, Any]]] = []

    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[object]:
        del tools, model
        if not self.steps:
            raise AssertionError("Streamer script exhausted.")

        self.prompts.append(_get_latest_user_content(messages))
        step = self.steps.pop(0)
        if step.started_event is not None:
            step.started_event.set()
        if isinstance(step.message.content, str) and step.message.content:
            yield ContentDeltaEvent(content=step.message.content)
        if step.release_event is not None:
            await step.release_event.wait()
        yield CompletionEvent(completion=Completion(message=step.message, usage=Usage(tokens=10, cost=0.0)))


def make_system_history() -> list[SystemMessage]:
    return [SystemMessage(content="# Instructions\n\nTest instructions")]


def make_agent_session(*, completion_streamer: Any) -> AgentSession:
    return AgentSession(
        history=make_system_history(),
        model="test-model",
        tools=[],
        completion_streamer=completion_streamer,
    )


def _get_latest_user_content(messages: list[object]) -> str | list[dict[str, Any]]:
    for message in reversed(messages):
        if isinstance(message, UserMessage):
            return message.content
    raise AssertionError("No user prompt found in streamer call.")


@pytest.mark.asyncio
async def test_worker_runtime_connects_prompts_and_waits_for_completion() -> None:
    session = make_agent_session(
        completion_streamer=ControlledStreamer(
            [StreamStep(message=AssistantMessage(content="Finished the delegated task"))]
        ),
    )
    worker_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        connect_result = await worker_runtime.connect(worker_server.endpoint)
        assert connect_result == f"Connected to remote {worker_id} at {worker_server.endpoint}."
        connected_workers = worker_runtime.format_connected_workers()
        assert f"remote {worker_id}" in connected_workers
        assert worker_server.endpoint in connected_workers

        prompt_result = await worker_runtime.prompt(worker_id, "Please finish the task.")
        assert prompt_result == f"Prompt accepted by remote {worker_id}.\nPlease finish the task."

        wait_result = await asyncio.wait_for(worker_runtime.wait(worker_id), timeout=1)
        assert wait_result == f"Remote {worker_id} finished:\nFinished the delegated task"

    await worker_runtime.close()
    await session.close()


@pytest.mark.asyncio
async def test_worker_runtime_queues_prompt_while_worker_is_busy() -> None:
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="First result"),
                started_event=first_started,
                release_event=first_release,
            ),
            StreamStep(message=AssistantMessage(content="Second result")),
        ]
    )
    session = make_agent_session(completion_streamer=streamer)
    worker_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == (
            f"Connected to remote {worker_id} at {worker_server.endpoint}."
        )
        assert await worker_runtime.prompt(worker_id, "Please start working.") == (
            f"Prompt accepted by remote {worker_id}.\nPlease start working."
        )

        await asyncio.wait_for(first_started.wait(), timeout=1)

        prompt_result = await worker_runtime.prompt(worker_id, "Please do something else.")
        assert prompt_result == f"Prompt accepted by remote {worker_id}.\nPlease do something else."

        first_release.set()

        first_wait = await asyncio.wait_for(worker_runtime.wait(worker_id), timeout=1)
        second_wait = await asyncio.wait_for(worker_runtime.wait(worker_id), timeout=1)

        assert first_wait == f"Remote {worker_id} finished:\nFirst result"
        assert second_wait == f"Remote {worker_id} finished:\nSecond result"

    await worker_runtime.close()
    await session.close()


@pytest.mark.asyncio
async def test_worker_runtime_priority_prompt_runs_before_existing_queue() -> None:
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="First result"),
                started_event=first_started,
                release_event=first_release,
            ),
            StreamStep(message=AssistantMessage(content="Priority result")),
            StreamStep(message=AssistantMessage(content="Second result")),
        ]
    )
    session = make_agent_session(completion_streamer=streamer)
    worker_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == (
            f"Connected to remote {worker_id} at {worker_server.endpoint}."
        )
        assert await worker_runtime.prompt(worker_id, "first") == f"Prompt accepted by remote {worker_id}.\nfirst"
        await asyncio.wait_for(first_started.wait(), timeout=1)

        assert await worker_runtime.prompt(worker_id, "second") == (f"Prompt accepted by remote {worker_id}.\nsecond")
        assert await worker_runtime.prompt(worker_id, "priority", mode="priority") == (
            f"Priority prompt accepted by remote {worker_id}.\npriority"
        )

        first_release.set()

        wait_results = [await asyncio.wait_for(worker_runtime.wait(worker_id), timeout=1) for _ in range(3)]

        assert wait_results == [
            f"Remote {worker_id} finished:\nFirst result",
            f"Remote {worker_id} finished:\nPriority result",
            f"Remote {worker_id} finished:\nSecond result",
        ]

    await worker_runtime.close()
    await session.close()


@pytest.mark.asyncio
async def test_worker_runtime_interrupt_prompt_cancels_current_run_and_runs_new_prompt_next() -> None:
    first_started = asyncio.Event()
    never_release = asyncio.Event()
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="Working..."),
                started_event=first_started,
                release_event=never_release,
            ),
            StreamStep(message=AssistantMessage(content="Interrupt result")),
        ]
    )
    session = make_agent_session(completion_streamer=streamer)
    worker_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == (
            f"Connected to remote {worker_id} at {worker_server.endpoint}."
        )
        assert await worker_runtime.prompt(worker_id, "first") == f"Prompt accepted by remote {worker_id}.\nfirst"
        await asyncio.wait_for(first_started.wait(), timeout=1)

        assert await worker_runtime.prompt(worker_id, "interrupt", mode="interrupt") == (
            f"Interrupt prompt accepted by remote {worker_id}.\ninterrupt"
        )

        cancel_wait = await asyncio.wait_for(worker_runtime.wait(worker_id), timeout=1)
        finish_wait = await asyncio.wait_for(worker_runtime.wait(worker_id), timeout=1)

        assert cancel_wait == f"Remote {worker_id} cancelled its current run."
        assert finish_wait == f"Remote {worker_id} finished:\nInterrupt result"

    await worker_runtime.close()
    await session.close()


@pytest.mark.asyncio
async def test_worker_runtime_second_connection_is_rejected_for_controlled_worker() -> None:
    session = make_agent_session(
        completion_streamer=ControlledStreamer([StreamStep(message=AssistantMessage(content="unused"))]),
    )
    first_runtime = WorkerToolRuntime()
    second_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        assert await first_runtime.connect(worker_server.endpoint) == (
            f"Connected to remote {worker_id} at {worker_server.endpoint}."
        )

        second_result = await second_runtime.connect(worker_server.endpoint)
        assert second_result == (
            f"Failed to connect to remote {worker_server.endpoint}: Session already has a remote client connected."
        )

    await first_runtime.close()
    await second_runtime.close()
    await session.close()


@pytest.mark.asyncio
async def test_worker_runtime_wait_returns_disconnect_once_then_reports_not_connected() -> None:
    session = make_agent_session(
        completion_streamer=ControlledStreamer([StreamStep(message=AssistantMessage(content="unused"))]),
    )
    worker_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == (
            f"Connected to remote {worker_id} at {worker_server.endpoint}."
        )
        assert await worker_runtime.disconnect(worker_id) == (f"Disconnected from remote {worker_id}.")

        wait_result = await asyncio.wait_for(worker_runtime.wait(worker_id), timeout=1)
        assert wait_result == f"Remote {worker_id} disconnected."

        second_wait_result = await asyncio.wait_for(worker_runtime.wait(worker_id), timeout=1)
        assert second_wait_result == f"Remote {worker_id} is not connected."

    await worker_runtime.close()
    await session.close()


@pytest.mark.asyncio
async def test_worker_runtime_wait_any_returns_pending_disconnect_after_last_connection_closes() -> None:
    session = make_agent_session(
        completion_streamer=ControlledStreamer([StreamStep(message=AssistantMessage(content="unused"))]),
    )
    worker_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == (
            f"Connected to remote {worker_id} at {worker_server.endpoint}."
        )
        assert await worker_runtime.disconnect(worker_id) == (f"Disconnected from remote {worker_id}.")

        wait_result = await asyncio.wait_for(worker_runtime.wait_any(), timeout=1)
        assert wait_result == f"Remote {worker_id} disconnected."

        second_wait_result = await asyncio.wait_for(worker_runtime.wait_any(), timeout=1)
        assert second_wait_result == "No connected remotes."

    await worker_runtime.close()
    await session.close()


@pytest.mark.asyncio
async def test_worker_runtime_wait_returns_idle_immediately_for_idle_connected_worker() -> None:
    session = make_agent_session(
        completion_streamer=ControlledStreamer([StreamStep(message=AssistantMessage(content="unused"))]),
    )
    worker_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == (
            f"Connected to remote {worker_id} at {worker_server.endpoint}."
        )

        wait_result = await asyncio.wait_for(worker_runtime.wait(worker_id), timeout=1)
        assert wait_result == f"Remote {worker_id} is idle."

    await worker_runtime.close()
    await session.close()


@pytest.mark.asyncio
async def test_worker_runtime_disconnect_does_not_stop_the_session() -> None:
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="First result"),
                started_event=first_started,
                release_event=first_release,
            ),
            StreamStep(message=AssistantMessage(content="Second result")),
        ]
    )
    session = make_agent_session(completion_streamer=streamer)
    worker_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == (
            f"Connected to remote {worker_id} at {worker_server.endpoint}."
        )
        assert await worker_runtime.prompt(worker_id, "first") == f"Prompt accepted by remote {worker_id}.\nfirst"
        await asyncio.wait_for(first_started.wait(), timeout=1)
        assert await worker_runtime.prompt(worker_id, "second") == (f"Prompt accepted by remote {worker_id}.\nsecond")
        assert await worker_runtime.disconnect(worker_id) == (f"Disconnected from remote {worker_id}.")

        first_release.set()
        await asyncio.wait_for(_wait_for_session_to_idle(session), timeout=1)

    assert streamer.prompts == ["first", "second"]

    await worker_runtime.close()
    await session.close()


@pytest.mark.asyncio
async def test_worker_runtime_wait_any_returns_idle_immediately_when_all_workers_are_idle() -> None:
    session = make_agent_session(
        completion_streamer=ControlledStreamer([StreamStep(message=AssistantMessage(content="unused"))]),
    )
    worker_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == (
            f"Connected to remote {worker_id} at {worker_server.endpoint}."
        )

        wait_result = await asyncio.wait_for(worker_runtime.wait_any(), timeout=1)
        assert wait_result == "All connected remotes are idle."

    await worker_runtime.close()
    await session.close()


async def _wait_for_session_to_idle(session: AgentSession) -> None:
    while session.state.running or session.state.queued_prompt_count:
        await asyncio.sleep(0)
