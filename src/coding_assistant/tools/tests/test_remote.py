from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
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
from coding_assistant.tools.remote import WorkerToolRuntime


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
            [StreamStep(message=AssistantMessage(content="Finished the delegated task"))],
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
        assert prompt_result == f"Prompt submitted to remote {worker_id}.\nPlease finish the task."

        wait_result = await asyncio.wait_for(worker_runtime.wait(worker_id), timeout=1)
        assert wait_result == f"Remote {worker_id} finished:\nFinished the delegated task"

    await worker_runtime.close()
    await session.close()


@pytest.mark.asyncio
async def test_worker_runtime_rejects_prompt_while_worker_is_busy() -> None:
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="First result"),
                started_event=first_started,
                release_event=first_release,
            ),
        ],
    )
    session = make_agent_session(completion_streamer=streamer)
    worker_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == (
            f"Connected to remote {worker_id} at {worker_server.endpoint}."
        )
        assert await worker_runtime.prompt(worker_id, "Please start working.") == (
            f"Prompt submitted to remote {worker_id}.\nPlease start working."
        )

        await asyncio.wait_for(first_started.wait(), timeout=1)

        prompt_result = await worker_runtime.prompt(worker_id, "Please do something else.")
        assert prompt_result == (
            f"Remote {worker_id} rejected the prompt: This remote connection already has an active prompt turn."
        )

        first_release.set()

        wait_result = await asyncio.wait_for(worker_runtime.wait(worker_id), timeout=1)

        assert wait_result == f"Remote {worker_id} finished:\nFirst result"

    await worker_runtime.close()
    await session.close()
    assert streamer.prompts == ["Please start working."]


@pytest.mark.asyncio
async def test_worker_runtime_can_cancel_current_run() -> None:
    first_started = asyncio.Event()
    never_release = asyncio.Event()
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="Working..."),
                started_event=first_started,
                release_event=never_release,
            ),
        ],
    )
    session = make_agent_session(completion_streamer=streamer)
    worker_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == (
            f"Connected to remote {worker_id} at {worker_server.endpoint}."
        )
        assert await worker_runtime.prompt(worker_id, "first") == f"Prompt submitted to remote {worker_id}.\nfirst"
        await asyncio.wait_for(first_started.wait(), timeout=1)

        assert await worker_runtime.cancel(worker_id) == f"Cancel requested for remote {worker_id}."

        cancel_wait = await asyncio.wait_for(worker_runtime.wait(worker_id), timeout=1)

        assert cancel_wait == f"Remote {worker_id} cancelled its current run."

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
        ],
    )
    session = make_agent_session(completion_streamer=streamer)
    worker_runtime = WorkerToolRuntime()
    worker_id = 1

    async with start_worker_server(session=session) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == (
            f"Connected to remote {worker_id} at {worker_server.endpoint}."
        )
        assert await worker_runtime.prompt(worker_id, "first") == f"Prompt submitted to remote {worker_id}.\nfirst"
        await asyncio.wait_for(first_started.wait(), timeout=1)
        assert await worker_runtime.disconnect(worker_id) == (f"Disconnected from remote {worker_id}.")

        first_release.set()
        await asyncio.wait_for(_wait_for_session_to_idle(session), timeout=1)

    assert streamer.prompts == ["first"]

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


def test_worker_runtime_discovers_other_registered_remotes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    registry_dir = tmp_path / "coding_assistant" / "remotes"
    registry_dir.mkdir(parents=True)

    self_pid = os.getpid()
    other_pid = self_pid + 1000
    (registry_dir / f"{self_pid}-4010.json").write_text(
        json.dumps(
            {
                "pid": self_pid,
                "port": 4010,
                "endpoint": "ws://127.0.0.1:4010",
                "cwd": "/self",
                "started_at": "2026-04-02T12:10:00+00:00",
            },
        ),
        encoding="utf-8",
    )
    (registry_dir / f"{other_pid}-4011.json").write_text(
        json.dumps(
            {
                "pid": other_pid,
                "port": 4011,
                "endpoint": "ws://127.0.0.1:4011",
                "cwd": "/other",
                "started_at": "2026-04-02T12:11:00+00:00",
            },
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "coding_assistant.remote.registry._pid_is_running",
        lambda pid: pid in {self_pid, other_pid},
    )

    worker_runtime = WorkerToolRuntime()

    assert worker_runtime.discover() == f"- pid {other_pid} at ws://127.0.0.1:4011 cwd=/other"


async def _wait_for_session_to_idle(session: AgentSession) -> None:
    while session.state.running or session.state.pending_prompts:
        await asyncio.sleep(0)
