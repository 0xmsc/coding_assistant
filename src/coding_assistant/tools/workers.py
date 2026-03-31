from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from coding_assistant.llm.types import Tool
from coding_assistant.remote.client import RemoteWorkerConnection
from coding_assistant.remote.protocol import (
    CommandAcceptedMessage,
    ContentDeltaMessage,
    ErrorMessage,
    NotReadyMessage,
    PromptAcceptedMessage,
    RunCancelledMessage,
    RunFailedMessage,
    RunFinishedMessage,
    StateMessage,
    ToolCallsMessage,
    WorkerToSupervisorMessage,
)


@dataclass
class WorkerSnapshot:
    worker_id: int
    endpoint: str
    promptable: bool = False
    remote_connected: bool = False
    running: bool = False
    queued_prompt_count: int = 0
    outstanding_prompt_count: int = 0
    last_update: str = ""


@dataclass(frozen=True)
class WorkerMeaningfulEvent:
    worker_id: int
    endpoint: str
    kind: Literal["finished", "cancelled", "failed", "disconnected"]
    summary: str = ""


class _WorkerManager:
    """Track active supervisor-side connections to other workers."""

    def __init__(self) -> None:
        self._connections: dict[int, RemoteWorkerConnection] = {}
        self._snapshots: dict[int, WorkerSnapshot] = {}
        self._worker_queues: dict[int, asyncio.Queue[WorkerMeaningfulEvent]] = {}
        self._worker_ids_by_endpoint: dict[str, int] = {}
        self._connecting_worker_ids: set[int] = set()
        self._next_worker_id = 1
        self._any_queue: asyncio.Queue[WorkerMeaningfulEvent] = asyncio.Queue()

    def format_connected_workers(self) -> str:
        if not self._connections:
            return "No connected workers."

        lines = []
        for worker_id, snapshot in sorted(self._snapshots.items()):
            if worker_id not in self._connections:
                continue
            status = "running" if snapshot.running else "promptable" if snapshot.promptable else "idle"
            if snapshot.queued_prompt_count:
                status = f"{status}, {snapshot.queued_prompt_count} queued"
            suffix = f" -> {snapshot.last_update}" if snapshot.last_update else ""
            lines.append(f"- worker {worker_id} at {snapshot.endpoint} [{status}]{suffix}")
        return "\n".join(lines)

    async def connect(self, endpoint: str) -> str:
        existing_worker_id = self._worker_ids_by_endpoint.get(endpoint)
        if existing_worker_id is not None:
            return f"Already connected to worker {existing_worker_id} at {endpoint}."

        worker_id = self._next_worker_id
        self._next_worker_id += 1
        self._connecting_worker_ids.add(worker_id)
        self._snapshots[worker_id] = WorkerSnapshot(worker_id=worker_id, endpoint=endpoint)
        self._worker_queues[worker_id] = asyncio.Queue()
        try:
            connection = await RemoteWorkerConnection.open(
                endpoint=endpoint,
                on_event=lambda message: self._handle_message(worker_id, message),
                on_disconnect=lambda disconnected_endpoint: self._handle_disconnect(worker_id, disconnected_endpoint),
            )
        except Exception as exc:
            self._connecting_worker_ids.discard(worker_id)
            self._snapshots.pop(worker_id, None)
            self._worker_queues.pop(worker_id, None)
            return f"Failed to connect to worker {endpoint}: {exc}"

        self._connecting_worker_ids.discard(worker_id)
        self._connections[worker_id] = connection
        self._worker_ids_by_endpoint[endpoint] = worker_id
        return f"Connected to worker {worker_id} at {endpoint}."

    async def disconnect(self, worker_id: int) -> str:
        connection = self._connections.get(worker_id)
        if connection is None:
            return f"Worker {worker_id} is not connected."
        await connection.close()
        return f"Disconnected from worker {worker_id}."

    async def prompt(
        self, worker_id: int, prompt: str, *, mode: Literal["queue", "priority", "interrupt"] = "queue"
    ) -> str:
        connection = self._connections.get(worker_id)
        if connection is None:
            return f"Worker {worker_id} is not connected."

        response = await connection.prompt(prompt, mode=mode)
        snapshot = self._snapshot_for_worker(worker_id)
        if isinstance(response, CommandAcceptedMessage):
            self._mark_prompt_accepted(snapshot)
            prompt_text = f"\n{prompt}"
            if mode == "priority":
                return f"Priority prompt accepted by worker {worker_id}.{prompt_text}"
            if mode == "interrupt":
                return f"Interrupt prompt accepted by worker {worker_id}.{prompt_text}"
            return f"Prompt accepted by worker {worker_id}.{prompt_text}"
        if isinstance(response, NotReadyMessage):
            self._apply_state(
                snapshot,
                promptable=response.promptable,
                remote_connected=response.remote_connected,
                running=response.running,
                queued_prompt_count=response.queued_prompt_count,
            )
            return (
                f"Worker {worker_id} is not ready for a prompt. "
                f"promptable={response.promptable} running={response.running} "
                f"queued={response.queued_prompt_count}"
            )
        return f"Worker {worker_id} rejected the prompt: {response.message}"

    async def cancel(self, worker_id: int) -> str:
        connection = self._connections.get(worker_id)
        if connection is None:
            return f"Worker {worker_id} is not connected."

        response = await connection.cancel()
        if isinstance(response, CommandAcceptedMessage):
            return f"Cancelled the current run on worker {worker_id}."
        if isinstance(response, NotReadyMessage):
            snapshot = self._snapshot_for_worker(worker_id)
            self._apply_state(
                snapshot,
                promptable=response.promptable,
                remote_connected=response.remote_connected,
                running=response.running,
                queued_prompt_count=response.queued_prompt_count,
            )
            return f"Worker {worker_id} has no cancellable run."
        return f"Worker {worker_id} rejected the cancel request: {response.message}"

    async def wait(self, worker_id: int) -> str:
        queue = self._worker_queues.get(worker_id)
        if queue is None:
            return f"Worker {worker_id} is not connected."
        if queue.empty():
            if worker_id not in self._connections:
                self._worker_queues.pop(worker_id, None)
                return f"Worker {worker_id} is not connected."
            if self._worker_is_idle(worker_id):
                return f"Worker {worker_id} is idle."
        event = await queue.get()
        self._cleanup_idle_queue(worker_id)
        return _format_meaningful_event(event)

    async def wait_any(self) -> str:
        if not self._any_queue.empty():
            event = await self._any_queue.get()
            return _format_meaningful_event(event)
        if not self._connections:
            return "No connected workers."
        if all(self._worker_is_idle(worker_id) for worker_id in self._connections):
            return "All connected workers are idle."
        event = await self._any_queue.get()
        return _format_meaningful_event(event)

    async def close(self) -> None:
        for connection in list(self._connections.values()):
            await connection.close()

    async def _handle_message(self, worker_id: int, message: WorkerToSupervisorMessage) -> None:
        snapshot = self._snapshot_for_worker(worker_id)

        if isinstance(message, StateMessage):
            self._apply_state(
                snapshot,
                promptable=message.promptable,
                remote_connected=message.remote_connected,
                running=message.running,
                queued_prompt_count=message.queued_prompt_count,
            )
            return

        if isinstance(message, PromptAcceptedMessage):
            snapshot.last_update = f"Accepted prompt: {_format_prompt_preview(message.content)}"
            return

        if isinstance(message, ContentDeltaMessage):
            snapshot.last_update = _truncate_summary((snapshot.last_update + message.content).strip())
            return

        if isinstance(message, ToolCallsMessage):
            tool_names = ", ".join(tool_call.name for tool_call in message.tool_calls)
            snapshot.last_update = f"Tool calls: {tool_names}"
            return

        if isinstance(message, RunFinishedMessage):
            self._mark_terminal_event(snapshot)
            snapshot.last_update = _truncate_summary(message.summary)
            await self._push_meaningful_event(
                WorkerMeaningfulEvent(
                    worker_id=worker_id,
                    endpoint=snapshot.endpoint,
                    kind="finished",
                    summary=message.summary,
                )
            )
            return

        if isinstance(message, RunCancelledMessage):
            self._mark_terminal_event(snapshot)
            snapshot.last_update = "Run cancelled."
            await self._push_meaningful_event(
                WorkerMeaningfulEvent(worker_id=worker_id, endpoint=snapshot.endpoint, kind="cancelled")
            )
            return

        if isinstance(message, RunFailedMessage):
            self._mark_terminal_event(snapshot)
            snapshot.last_update = f"Run failed: {message.error}"
            await self._push_meaningful_event(
                WorkerMeaningfulEvent(
                    worker_id=worker_id,
                    endpoint=snapshot.endpoint,
                    kind="failed",
                    summary=message.error,
                )
            )
            return

        if isinstance(message, ErrorMessage):
            snapshot.last_update = f"Error: {message.message}"

    async def _handle_disconnect(self, worker_id: int, endpoint: str) -> None:
        was_connected = worker_id in self._connections
        snapshot = self._snapshots.pop(worker_id, None)
        self._connections.pop(worker_id, None)
        self._worker_ids_by_endpoint.pop(endpoint, None)
        if worker_id in self._connecting_worker_ids and not was_connected:
            self._connecting_worker_ids.discard(worker_id)
            self._worker_queues.pop(worker_id, None)
            return
        was_known = was_connected or snapshot is not None
        if not was_known:
            return
        await self._push_meaningful_event(
            WorkerMeaningfulEvent(worker_id=worker_id, endpoint=endpoint, kind="disconnected")
        )

    async def _push_meaningful_event(self, event: WorkerMeaningfulEvent) -> None:
        queue = self._worker_queues.setdefault(event.worker_id, asyncio.Queue())
        await queue.put(event)
        await self._any_queue.put(event)

    def _cleanup_idle_queue(self, worker_id: int) -> None:
        queue = self._worker_queues.get(worker_id)
        if queue is None:
            return
        if worker_id in self._connections or not queue.empty():
            return
        self._worker_queues.pop(worker_id, None)

    def _snapshot_for_worker(self, worker_id: int) -> WorkerSnapshot:
        snapshot = self._snapshots.get(worker_id)
        if snapshot is not None:
            return snapshot
        connection = self._connections.get(worker_id)
        endpoint = "" if connection is None else connection.endpoint
        snapshot = WorkerSnapshot(worker_id=worker_id, endpoint=endpoint)
        self._snapshots[worker_id] = snapshot
        return snapshot

    def _apply_state(
        self,
        snapshot: WorkerSnapshot,
        *,
        promptable: bool,
        remote_connected: bool,
        running: bool,
        queued_prompt_count: int,
    ) -> None:
        snapshot.promptable = promptable
        snapshot.remote_connected = remote_connected
        snapshot.running = running
        snapshot.queued_prompt_count = queued_prompt_count
        snapshot.outstanding_prompt_count = queued_prompt_count + int(running)

    def _mark_prompt_accepted(self, snapshot: WorkerSnapshot) -> None:
        snapshot.outstanding_prompt_count += 1
        if snapshot.running or snapshot.queued_prompt_count:
            snapshot.queued_prompt_count += 1
            return
        snapshot.running = True

    def _mark_terminal_event(self, snapshot: WorkerSnapshot) -> None:
        snapshot.outstanding_prompt_count = max(snapshot.outstanding_prompt_count - 1, 0)
        snapshot.running = False
        snapshot.promptable = True
        snapshot.queued_prompt_count = snapshot.outstanding_prompt_count

    def _worker_is_idle(self, worker_id: int) -> bool:
        snapshot = self._snapshots.get(worker_id)
        if snapshot is None:
            return True
        return snapshot.outstanding_prompt_count == 0


class WorkerToolRuntime:
    """Worker-tool runtime that hides supervisor-side connection management."""

    def __init__(self) -> None:
        self._manager = _WorkerManager()
        self.tools = _create_worker_tools(runtime=self)

    def format_connected_workers(self) -> str:
        return self._manager.format_connected_workers()

    async def connect(self, endpoint: str) -> str:
        return await self._manager.connect(endpoint)

    async def disconnect(self, worker_id: int) -> str:
        return await self._manager.disconnect(worker_id)

    async def prompt(
        self, worker_id: int, prompt: str, *, mode: Literal["queue", "priority", "interrupt"] = "queue"
    ) -> str:
        return await self._manager.prompt(worker_id, prompt, mode=mode)

    async def cancel(self, worker_id: int) -> str:
        return await self._manager.cancel(worker_id)

    async def wait(self, worker_id: int) -> str:
        return await self._manager.wait(worker_id)

    async def wait_any(self) -> str:
        return await self._manager.wait_any()

    async def close(self) -> None:
        await self._manager.close()


def _truncate_summary(text: str, *, limit: int = 120) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3]}..."


def _format_prompt_preview(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return _truncate_summary(content)
    return "structured prompt"


def _format_meaningful_event(event: WorkerMeaningfulEvent) -> str:
    if event.kind == "finished":
        if event.summary:
            return f"Worker {event.worker_id} finished:\n{event.summary}"
        return f"Worker {event.worker_id} finished."
    if event.kind == "cancelled":
        return f"Worker {event.worker_id} cancelled its current run."
    if event.kind == "failed":
        return f"Worker {event.worker_id} failed:\n{event.summary}"
    return f"Worker {event.worker_id} disconnected."


class EmptyInput(BaseModel):
    """Schema for tools that do not take any arguments."""


class WorkerConnectInput(BaseModel):
    endpoint: str = Field(description="The websocket endpoint of the worker to connect to.")


class WorkerPromptInput(BaseModel):
    worker_id: int = Field(description="The local worker id returned by worker_connect.")
    prompt: str = Field(description="The prompt to send to the worker.")
    mode: Literal["queue", "priority", "interrupt"] = Field(
        default="queue",
        description="How to schedule the prompt: queue normally, queue with priority, or interrupt the current run.",
    )


class WorkerIdInput(BaseModel):
    worker_id: int = Field(description="The local worker id returned by worker_connect.")


class WorkerConnectTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "worker_connect"

    def description(self) -> str:
        return "Connect to one worker websocket endpoint."

    def parameters(self) -> dict[str, Any]:
        return WorkerConnectInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerConnectInput.model_validate(parameters)
        return await self._runtime.connect(validated.endpoint)


class WorkersListTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "workers_list"

    def description(self) -> str:
        return "List currently connected workers and their latest state."

    def parameters(self) -> dict[str, Any]:
        return EmptyInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        EmptyInput.model_validate(parameters)
        return self._runtime.format_connected_workers()


class WorkerPromptTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "worker_prompt"

    def description(self) -> str:
        return "Send a prompt to one connected worker id, optionally as a priority or interrupting prompt."

    def parameters(self) -> dict[str, Any]:
        return WorkerPromptInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerPromptInput.model_validate(parameters)
        return await self._runtime.prompt(validated.worker_id, validated.prompt, mode=validated.mode)


class WorkerWaitTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "worker_wait"

    def description(self) -> str:
        return "Wait for the next meaningful update from one connected worker id."

    def parameters(self) -> dict[str, Any]:
        return WorkerIdInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerIdInput.model_validate(parameters)
        return await self._runtime.wait(validated.worker_id)


class WorkersWaitAnyTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "workers_wait_any"

    def description(self) -> str:
        return "Wait for the next meaningful update from any connected worker."

    def parameters(self) -> dict[str, Any]:
        return EmptyInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        EmptyInput.model_validate(parameters)
        return await self._runtime.wait_any()


class WorkerCancelTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "worker_cancel"

    def description(self) -> str:
        return "Cancel the current run on one connected worker id."

    def parameters(self) -> dict[str, Any]:
        return WorkerIdInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerIdInput.model_validate(parameters)
        return await self._runtime.cancel(validated.worker_id)


class WorkerDisconnectTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "worker_disconnect"

    def description(self) -> str:
        return "Disconnect from one connected worker id."

    def parameters(self) -> dict[str, Any]:
        return WorkerIdInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerIdInput.model_validate(parameters)
        return await self._runtime.disconnect(validated.worker_id)


def _create_worker_tools(*, runtime: WorkerToolRuntime) -> list[Tool]:
    return [
        WorkerConnectTool(runtime=runtime),
        WorkersListTool(runtime=runtime),
        WorkerPromptTool(runtime=runtime),
        WorkerWaitTool(runtime=runtime),
        WorkersWaitAnyTool(runtime=runtime),
        WorkerCancelTool(runtime=runtime),
        WorkerDisconnectTool(runtime=runtime),
    ]
