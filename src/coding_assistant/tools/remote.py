from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from coding_assistant.llm.types import TextToolResult, Tool
from coding_assistant.remote.client import (
    RemoteClientEvent,
    RemoteContentDeltaEvent,
    RemotePromptFailedEvent,
    RemotePromptFinishedEvent,
    RemoteToolCallEvent,
    RemoteToolCallUpdateEvent,
    RemoteWorkerConnection,
)
from coding_assistant.remote.registry import discover_remote_instances


@dataclass
class WorkerSnapshot:
    worker_id: int
    endpoint: str
    running: bool = False
    last_update: str = ""
    last_content: str = ""


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
            return "No connected remotes."

        lines = []
        for worker_id, snapshot in sorted(self._snapshots.items()):
            if worker_id not in self._connections:
                continue
            status = "running" if snapshot.running else "connected"
            suffix = f" -> {snapshot.last_update}" if snapshot.last_update else ""
            lines.append(f"- remote {worker_id} at {snapshot.endpoint} [{status}]{suffix}")
        return "\n".join(lines)

    def discover(self) -> str:
        discovered = discover_remote_instances(current_pid=os.getpid())
        if not discovered:
            return "No discoverable remotes."

        lines = []
        for entry in discovered:
            connected_suffix = ""
            existing_worker_id = self._worker_ids_by_endpoint.get(entry.endpoint)
            if existing_worker_id is not None:
                connected_suffix = f" [connected as remote {existing_worker_id}]"
            lines.append(f"- pid {entry.pid} at {entry.endpoint}{connected_suffix} cwd={entry.cwd}")
        return "\n".join(lines)

    async def connect(self, endpoint: str) -> str:
        existing_worker_id = self._worker_ids_by_endpoint.get(endpoint)
        if existing_worker_id is not None:
            return f"Already connected to remote {existing_worker_id} at {endpoint}."

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
            return f"Failed to connect to remote {endpoint}: {exc}"

        self._connecting_worker_ids.discard(worker_id)
        self._connections[worker_id] = connection
        self._worker_ids_by_endpoint[endpoint] = worker_id
        return f"Connected to remote {worker_id} at {endpoint}."

    async def disconnect(self, worker_id: int) -> str:
        connection = self._connections.get(worker_id)
        if connection is None:
            return f"Remote {worker_id} is not connected."
        await connection.close()
        return f"Disconnected from remote {worker_id}."

    async def prompt(self, worker_id: int, prompt: str) -> str:
        connection = self._connections.get(worker_id)
        if connection is None:
            return f"Remote {worker_id} is not connected."

        snapshot = self._snapshot_for_worker(worker_id)
        previous_running = snapshot.running
        previous_last_content = snapshot.last_content
        previous_last_update = snapshot.last_update
        snapshot.running = True
        snapshot.last_content = ""
        snapshot.last_update = f"Prompt submitted: {_format_prompt_preview(prompt)}"
        response_error = await connection.prompt(prompt)
        if response_error is not None:
            if response_error == "This remote connection already has an active prompt turn.":
                snapshot.running = previous_running
                snapshot.last_content = previous_last_content
                snapshot.last_update = previous_last_update
            else:
                snapshot.running = False
            return f"Remote {worker_id} rejected the prompt: {response_error}"
        return f"Prompt submitted to remote {worker_id}.\n{prompt}"

    async def cancel(self, worker_id: int) -> str:
        connection = self._connections.get(worker_id)
        if connection is None:
            return f"Remote {worker_id} is not connected."

        await connection.cancel()
        snapshot = self._snapshot_for_worker(worker_id)
        snapshot.last_update = "Cancellation requested."
        return f"Cancel requested for remote {worker_id}."

    async def wait(self, worker_id: int) -> str:
        queue = self._worker_queues.get(worker_id)
        if queue is None:
            return f"Remote {worker_id} is not connected."
        if queue.empty():
            if worker_id not in self._connections:
                self._worker_queues.pop(worker_id, None)
                return f"Remote {worker_id} is not connected."
            if self._worker_is_idle(worker_id):
                return f"Remote {worker_id} is idle."
        event = await queue.get()
        self._cleanup_idle_queue(worker_id)
        return _format_meaningful_event(event)

    async def wait_any(self) -> str:
        if not self._any_queue.empty():
            event = await self._any_queue.get()
            return _format_meaningful_event(event)
        if not self._connections:
            return "No connected remotes."
        if all(self._worker_is_idle(worker_id) for worker_id in self._connections):
            return "All connected remotes are idle."
        event = await self._any_queue.get()
        return _format_meaningful_event(event)

    async def close(self) -> None:
        for connection in list(self._connections.values()):
            await connection.close()

    async def _handle_message(self, worker_id: int, message: RemoteClientEvent) -> None:
        snapshot = self._snapshot_for_worker(worker_id)

        if isinstance(message, RemoteContentDeltaEvent):
            snapshot.running = True
            snapshot.last_content = (snapshot.last_content + message.content).strip()
            snapshot.last_update = snapshot.last_content
            return

        if isinstance(message, RemoteToolCallEvent):
            snapshot.running = True
            snapshot.last_update = f"Tool call: {message.title}"
            return

        if isinstance(message, RemoteToolCallUpdateEvent):
            snapshot.running = True
            if message.content:
                snapshot.last_update = message.content
            elif message.title:
                snapshot.last_update = f"Tool {message.status}: {message.title}"
            return

        if isinstance(message, RemotePromptFinishedEvent):
            snapshot.running = False
            if message.stop_reason == "cancelled":
                snapshot.last_content = ""
                snapshot.last_update = "Run cancelled."
                await self._push_meaningful_event(
                    WorkerMeaningfulEvent(worker_id=worker_id, endpoint=snapshot.endpoint, kind="cancelled"),
                )
                return
            summary = snapshot.last_content or snapshot.last_update or "Prompt finished."
            snapshot.last_update = summary
            await self._push_meaningful_event(
                WorkerMeaningfulEvent(
                    worker_id=worker_id,
                    endpoint=snapshot.endpoint,
                    kind="finished",
                    summary=summary,
                ),
            )
            return

        if isinstance(message, RemotePromptFailedEvent):
            snapshot.running = False
            snapshot.last_content = ""
            snapshot.last_update = f"Run failed: {message.message}"
            await self._push_meaningful_event(
                WorkerMeaningfulEvent(
                    worker_id=worker_id,
                    endpoint=snapshot.endpoint,
                    kind="failed",
                    summary=message.message,
                ),
            )
            return

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
            WorkerMeaningfulEvent(worker_id=worker_id, endpoint=endpoint, kind="disconnected"),
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

    def _worker_is_idle(self, worker_id: int) -> bool:
        snapshot = self._snapshots.get(worker_id)
        if snapshot is None:
            return True
        return not snapshot.running


class WorkerToolRuntime:
    """Remote-tool runtime that hides connection management for remote sessions."""

    def __init__(self) -> None:
        self._manager = _WorkerManager()
        self.tools = _create_worker_tools(runtime=self)

    def format_connected_workers(self) -> str:
        return self._manager.format_connected_workers()

    async def connect(self, endpoint: str) -> str:
        return await self._manager.connect(endpoint)

    def discover(self) -> str:
        return self._manager.discover()

    async def disconnect(self, worker_id: int) -> str:
        return await self._manager.disconnect(worker_id)

    async def prompt(self, worker_id: int, prompt: str) -> str:
        return await self._manager.prompt(worker_id, prompt)

    async def cancel(self, worker_id: int) -> str:
        return await self._manager.cancel(worker_id)

    async def wait(self, worker_id: int) -> str:
        return await self._manager.wait(worker_id)

    async def wait_any(self) -> str:
        return await self._manager.wait_any()

    async def close(self) -> None:
        await self._manager.close()


def _format_prompt_preview(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return _truncate_summary(content)
    return "structured prompt"


def _truncate_summary(text: str, *, limit: int = 120) -> str:
    """Truncate text for display purposes (prompt previews)."""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3]}..."


def _format_meaningful_event(event: WorkerMeaningfulEvent) -> str:
    if event.kind == "finished":
        if event.summary:
            return f"Remote {event.worker_id} finished:\n{event.summary}"
        return f"Remote {event.worker_id} finished."
    if event.kind == "cancelled":
        return f"Remote {event.worker_id} cancelled its current run."
    if event.kind == "failed":
        return f"Remote {event.worker_id} failed:\n{event.summary}"
    return f"Remote {event.worker_id} disconnected."


class EmptyInput(BaseModel):
    """Schema for tools that do not take any arguments."""


class RemoteConnectInput(BaseModel):
    endpoint: str = Field(description="The websocket endpoint of the remote session to connect to.")


class RemotePromptInput(BaseModel):
    remote_id: int = Field(description="The local remote id returned by remote_connect.")
    prompt: str = Field(description="The prompt to send to the remote session.")


class RemoteIdInput(BaseModel):
    remote_id: int = Field(description="The local remote id returned by remote_connect.")


class RemoteConnectTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "remote_connect"

    def description(self) -> str:
        return "Connect to one remote session websocket endpoint."

    def parameters(self) -> dict[str, Any]:
        return RemoteConnectInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        validated = RemoteConnectInput.model_validate(parameters)
        return TextToolResult(content=await self._runtime.connect(validated.endpoint))


class RemotesListTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "remotes_list"

    def description(self) -> str:
        return "List currently connected remote sessions and their latest state."

    def parameters(self) -> dict[str, Any]:
        return EmptyInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        EmptyInput.model_validate(parameters)
        return TextToolResult(content=self._runtime.format_connected_workers())


class RemotesDiscoverTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "remotes_discover"

    def description(self) -> str:
        return "List locally advertised remote sessions discovered on this machine."

    def parameters(self) -> dict[str, Any]:
        return EmptyInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        EmptyInput.model_validate(parameters)
        return TextToolResult(content=self._runtime.discover())


class RemotePromptTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "remote_prompt"

    def description(self) -> str:
        return "Send a prompt to one connected remote id when that remote is idle."

    def parameters(self) -> dict[str, Any]:
        return RemotePromptInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        validated = RemotePromptInput.model_validate(parameters)
        return TextToolResult(content=await self._runtime.prompt(validated.remote_id, validated.prompt))


class RemoteWaitTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "remote_wait"

    def description(self) -> str:
        return "Wait for the next meaningful update from one connected remote id."

    def parameters(self) -> dict[str, Any]:
        return RemoteIdInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        validated = RemoteIdInput.model_validate(parameters)
        return TextToolResult(content=await self._runtime.wait(validated.remote_id))


class RemotesWaitAnyTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "remotes_wait_any"

    def description(self) -> str:
        return "Wait for the next meaningful update from any connected remote session."

    def parameters(self) -> dict[str, Any]:
        return EmptyInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        EmptyInput.model_validate(parameters)
        return TextToolResult(content=await self._runtime.wait_any())


class RemoteCancelTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "remote_cancel"

    def description(self) -> str:
        return "Cancel the current run on one connected remote id."

    def parameters(self) -> dict[str, Any]:
        return RemoteIdInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        validated = RemoteIdInput.model_validate(parameters)
        return TextToolResult(content=await self._runtime.cancel(validated.remote_id))


class RemoteDisconnectTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "remote_disconnect"

    def description(self) -> str:
        return "Disconnect from one connected remote id."

    def parameters(self) -> dict[str, Any]:
        return RemoteIdInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        validated = RemoteIdInput.model_validate(parameters)
        return TextToolResult(content=await self._runtime.disconnect(validated.remote_id))


def _create_worker_tools(*, runtime: WorkerToolRuntime) -> list[Tool]:
    return [
        RemoteConnectTool(runtime=runtime),
        RemotesDiscoverTool(runtime=runtime),
        RemotesListTool(runtime=runtime),
        RemotePromptTool(runtime=runtime),
        RemoteWaitTool(runtime=runtime),
        RemotesWaitAnyTool(runtime=runtime),
        RemoteCancelTool(runtime=runtime),
        RemoteDisconnectTool(runtime=runtime),
    ]
