from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from coding_assistant.llm.types import (
    AssistantMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    SystemMessage,
    Usage,
)
from coding_assistant.remote.server import start_worker_server
from coding_assistant.remote.worker_session import WorkerSession
from coding_assistant.tools.workers import WorkerToolRuntime


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


def make_system_history() -> list[SystemMessage]:
    return [SystemMessage(content="# Instructions\n\nTest instructions")]


def make_worker_session(*, completion_streamer: Any) -> WorkerSession:
    return WorkerSession(
        history=make_system_history(),
        model="test-model",
        tools=[],
        completion_streamer=completion_streamer,
    )


@pytest.mark.asyncio
async def test_worker_runtime_discovers_connects_prompts_and_waits_for_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    session = make_worker_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="Finished the delegated task")]),
    )
    worker_runtime = WorkerToolRuntime()

    async with start_worker_server(
        session=session,
        cwd=tmp_path,
    ) as worker_server:
        discovered = worker_runtime.discover_records()
        assert [record.endpoint for record in discovered] == [worker_server.endpoint]

        connect_result = await worker_runtime.connect(worker_server.endpoint)
        assert connect_result == f"Connected to worker {worker_server.endpoint}."
        assert worker_server.endpoint in worker_runtime.format_connected_workers()

        prompt_result = await worker_runtime.prompt(worker_server.endpoint, "Please finish the task.")
        assert prompt_result == f"Prompt sent to worker {worker_server.endpoint}."

        wait_result = await asyncio.wait_for(worker_runtime.wait(worker_server.endpoint), timeout=1)
        assert wait_result == f"Worker {worker_server.endpoint} finished:\nFinished the delegated task"

    await worker_runtime.close()


@pytest.mark.asyncio
async def test_worker_runtime_rejects_prompt_while_worker_is_busy_and_can_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    streamer = BlockingStreamer()
    session = make_worker_session(completion_streamer=streamer)
    worker_runtime = WorkerToolRuntime()

    async with start_worker_server(
        session=session,
        cwd=tmp_path,
    ) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == f"Connected to worker {worker_server.endpoint}."
        assert await worker_runtime.prompt(worker_server.endpoint, "Please start working.") == (
            f"Prompt sent to worker {worker_server.endpoint}."
        )

        await asyncio.wait_for(streamer.started.wait(), timeout=1)

        prompt_result = await worker_runtime.prompt(worker_server.endpoint, "Please do something else.")
        assert prompt_result == (
            f"Worker {worker_server.endpoint} is not ready for a prompt. promptable=False running=True"
        )

        cancel_result = await worker_runtime.cancel(worker_server.endpoint)
        assert cancel_result == f"Cancelled the current run on worker {worker_server.endpoint}."

        wait_result = await asyncio.wait_for(worker_runtime.wait(worker_server.endpoint), timeout=1)
        assert wait_result == f"Worker {worker_server.endpoint} cancelled its current run."

    await worker_runtime.close()


@pytest.mark.asyncio
async def test_worker_runtime_second_connection_is_rejected_for_controlled_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    session = make_worker_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )
    first_runtime = WorkerToolRuntime()
    second_runtime = WorkerToolRuntime()

    async with start_worker_server(
        session=session,
        cwd=tmp_path,
    ) as worker_server:
        assert await first_runtime.connect(worker_server.endpoint) == f"Connected to worker {worker_server.endpoint}."

        second_result = await second_runtime.connect(worker_server.endpoint)
        assert second_result == (
            f"Failed to connect to worker {worker_server.endpoint}: Worker is already remotely controlled."
        )

    await first_runtime.close()
    await second_runtime.close()


@pytest.mark.asyncio
async def test_worker_runtime_wait_returns_disconnect_once_then_reports_not_connected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    session = make_worker_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )
    worker_runtime = WorkerToolRuntime()

    async with start_worker_server(
        session=session,
        cwd=tmp_path,
    ) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == f"Connected to worker {worker_server.endpoint}."
        assert await worker_runtime.disconnect(worker_server.endpoint) == (
            f"Disconnected from worker {worker_server.endpoint}."
        )

        wait_result = await asyncio.wait_for(worker_runtime.wait(worker_server.endpoint), timeout=1)
        assert wait_result == f"Worker {worker_server.endpoint} disconnected."

        second_wait_result = await asyncio.wait_for(worker_runtime.wait(worker_server.endpoint), timeout=1)
        assert second_wait_result == f"Worker {worker_server.endpoint} is not connected."

    await worker_runtime.close()


@pytest.mark.asyncio
async def test_worker_runtime_wait_any_returns_pending_disconnect_after_last_connection_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    session = make_worker_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )
    worker_runtime = WorkerToolRuntime()

    async with start_worker_server(
        session=session,
        cwd=tmp_path,
    ) as worker_server:
        assert await worker_runtime.connect(worker_server.endpoint) == f"Connected to worker {worker_server.endpoint}."
        assert await worker_runtime.disconnect(worker_server.endpoint) == (
            f"Disconnected from worker {worker_server.endpoint}."
        )

        wait_result = await asyncio.wait_for(worker_runtime.wait_any(), timeout=1)
        assert wait_result == f"Worker {worker_server.endpoint} disconnected."

        second_wait_result = await asyncio.wait_for(worker_runtime.wait_any(), timeout=1)
        assert second_wait_result == "No connected workers."

    await worker_runtime.close()
