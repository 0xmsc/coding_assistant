from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from coding_assistant.llm.types import Tool
from coding_assistant.remote.client import RemoteWorkerConnection
from coding_assistant.remote.discovery import WorkerRecord, list_worker_records
from coding_assistant.remote.protocol import (
    CommandAcceptedMessage,
    ContentDeltaMessage,
    ErrorMessage,
    NotReadyMessage,
    RunCancelledMessage,
    RunFailedMessage,
    RunFinishedMessage,
    StateMessage,
    ToolCallsMessage,
    WorkerToSupervisorMessage,
)


@dataclass
class WorkerSnapshot:
    endpoint: str
    promptable: bool = False
    remote_connected: bool = False
    running: bool = False
    last_update: str = ""


@dataclass(frozen=True)
class WorkerMeaningfulEvent:
    endpoint: str
    kind: Literal["finished", "cancelled", "failed", "disconnected"]
    summary: str = ""


class WorkerManager:
    """Track discovered workers plus active supervisor-side connections."""

    def __init__(self) -> None:
        self._local_endpoint: str | None = None
        self._connections: dict[str, RemoteWorkerConnection] = {}
        self._snapshots: dict[str, WorkerSnapshot] = {}
        self._worker_queues: dict[str, asyncio.Queue[WorkerMeaningfulEvent]] = {}
        self._any_queue: asyncio.Queue[WorkerMeaningfulEvent] = asyncio.Queue()

    def set_local_endpoint(self, endpoint: str) -> None:
        self._local_endpoint = endpoint

    def format_discovered_workers(self) -> str:
        records = self.discover_records()
        if not records:
            return "No workers discovered."
        return "\n".join(f"- {record.endpoint} (pid={record.pid}, cwd={record.cwd})" for record in records)

    def discover_records(self) -> list[WorkerRecord]:
        return list_worker_records(exclude_endpoint=self._local_endpoint)

    def format_connected_workers(self) -> str:
        if not self._connections:
            return "No connected workers."

        lines = []
        for endpoint, snapshot in sorted(self._snapshots.items()):
            status = "running" if snapshot.running else "promptable" if snapshot.promptable else "idle"
            suffix = f" -> {snapshot.last_update}" if snapshot.last_update else ""
            lines.append(f"- {endpoint} [{status}]{suffix}")
        return "\n".join(lines)

    async def connect(self, endpoint: str) -> str:
        if self._local_endpoint is not None and endpoint == self._local_endpoint:
            return "Cannot connect to the current worker endpoint."
        if endpoint in self._connections:
            return f"Already connected to worker {endpoint}."
        try:
            connection = await RemoteWorkerConnection.open(
                endpoint=endpoint,
                on_event=lambda message: self._handle_message(endpoint, message),
                on_disconnect=self._handle_disconnect,
            )
        except Exception as exc:
            self._snapshots.pop(endpoint, None)
            self._worker_queues.pop(endpoint, None)
            return f"Failed to connect to worker {endpoint}: {exc}"

        self._snapshots.setdefault(endpoint, WorkerSnapshot(endpoint=endpoint))
        self._worker_queues.setdefault(endpoint, asyncio.Queue())
        self._connections[endpoint] = connection
        return f"Connected to worker {endpoint}."

    async def disconnect(self, endpoint: str) -> str:
        connection = self._connections.get(endpoint)
        if connection is None:
            return f"Worker {endpoint} is not connected."
        await connection.close()
        return f"Disconnected from worker {endpoint}."

    async def prompt(self, endpoint: str, prompt: str) -> str:
        connection = self._connections.get(endpoint)
        if connection is None:
            return f"Worker {endpoint} is not connected."

        response = await connection.prompt(prompt)
        if isinstance(response, CommandAcceptedMessage):
            return f"Prompt sent to worker {endpoint}."
        if isinstance(response, NotReadyMessage):
            return (
                f"Worker {endpoint} is not ready for a prompt. "
                f"promptable={response.promptable} running={response.running}"
            )
        return f"Worker {endpoint} rejected the prompt: {response.message}"

    async def cancel(self, endpoint: str) -> str:
        connection = self._connections.get(endpoint)
        if connection is None:
            return f"Worker {endpoint} is not connected."

        response = await connection.cancel()
        if isinstance(response, CommandAcceptedMessage):
            return f"Cancelled the current run on worker {endpoint}."
        if isinstance(response, NotReadyMessage):
            return f"Worker {endpoint} has no cancellable run."
        return f"Worker {endpoint} rejected the cancel request: {response.message}"

    async def wait(self, endpoint: str) -> str:
        queue = self._worker_queues.get(endpoint)
        if queue is None:
            return f"Worker {endpoint} is not connected."
        event = await queue.get()
        return _format_meaningful_event(event)

    async def wait_any(self) -> str:
        if not self._connections:
            return "No connected workers."
        event = await self._any_queue.get()
        return _format_meaningful_event(event)

    async def close(self) -> None:
        for connection in list(self._connections.values()):
            await connection.close()

    async def _handle_message(self, endpoint: str, message: WorkerToSupervisorMessage) -> None:
        snapshot = self._snapshots.setdefault(endpoint, WorkerSnapshot(endpoint=endpoint))

        if isinstance(message, StateMessage):
            snapshot.promptable = message.promptable
            snapshot.remote_connected = message.remote_connected
            snapshot.running = message.running
            return

        if isinstance(message, ContentDeltaMessage):
            snapshot.last_update = _truncate_summary((snapshot.last_update + message.content).strip())
            return

        if isinstance(message, ToolCallsMessage):
            tool_names = ", ".join(tool_call.name for tool_call in message.tool_calls)
            snapshot.last_update = f"Tool calls: {tool_names}"
            return

        if isinstance(message, RunFinishedMessage):
            snapshot.running = False
            snapshot.promptable = True
            snapshot.last_update = _truncate_summary(message.summary)
            await self._push_meaningful_event(
                WorkerMeaningfulEvent(
                    endpoint=endpoint,
                    kind="finished",
                    summary=message.summary,
                )
            )
            return

        if isinstance(message, RunCancelledMessage):
            snapshot.running = False
            snapshot.promptable = True
            snapshot.last_update = "Run cancelled."
            await self._push_meaningful_event(WorkerMeaningfulEvent(endpoint=endpoint, kind="cancelled"))
            return

        if isinstance(message, RunFailedMessage):
            snapshot.running = False
            snapshot.promptable = True
            snapshot.last_update = f"Run failed: {message.error}"
            await self._push_meaningful_event(
                WorkerMeaningfulEvent(
                    endpoint=endpoint,
                    kind="failed",
                    summary=message.error,
                )
            )
            return

        if isinstance(message, ErrorMessage):
            snapshot.last_update = f"Error: {message.message}"

    async def _handle_disconnect(self, endpoint: str) -> None:
        was_known = endpoint in self._connections or endpoint in self._snapshots
        self._connections.pop(endpoint, None)
        self._snapshots.pop(endpoint, None)
        if not was_known:
            return
        await self._push_meaningful_event(WorkerMeaningfulEvent(endpoint=endpoint, kind="disconnected"))

    async def _push_meaningful_event(self, event: WorkerMeaningfulEvent) -> None:
        queue = self._worker_queues.setdefault(event.endpoint, asyncio.Queue())
        await queue.put(event)
        await self._any_queue.put(event)


def _truncate_summary(text: str, *, limit: int = 120) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3]}..."


def _format_meaningful_event(event: WorkerMeaningfulEvent) -> str:
    if event.kind == "finished":
        if event.summary:
            return f"Worker {event.endpoint} finished:\n{event.summary}"
        return f"Worker {event.endpoint} finished."
    if event.kind == "cancelled":
        return f"Worker {event.endpoint} cancelled its current run."
    if event.kind == "failed":
        return f"Worker {event.endpoint} failed:\n{event.summary}"
    return f"Worker {event.endpoint} disconnected."


class EmptyInput(BaseModel):
    """Schema for tools that do not take any arguments."""


class WorkerConnectInput(BaseModel):
    endpoint: str = Field(description="The websocket endpoint of the worker to connect to.")


class WorkerPromptInput(BaseModel):
    endpoint: str = Field(description="The websocket endpoint of the worker to prompt.")
    prompt: str = Field(description="The prompt to send to the worker.")


class WorkerWaitInput(BaseModel):
    endpoint: str = Field(description="The websocket endpoint of the worker to wait on.")


class WorkerDisconnectInput(BaseModel):
    endpoint: str = Field(description="The websocket endpoint of the worker to disconnect from.")


class WorkerDiscoverTool(Tool):
    def __init__(self, *, manager: WorkerManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "worker_discover"

    def description(self) -> str:
        return "List local worker endpoints that can be connected to."

    def parameters(self) -> dict[str, Any]:
        return EmptyInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        EmptyInput.model_validate(parameters)
        return self._manager.format_discovered_workers()


class WorkerConnectTool(Tool):
    def __init__(self, *, manager: WorkerManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "worker_connect"

    def description(self) -> str:
        return "Connect to one discovered worker endpoint."

    def parameters(self) -> dict[str, Any]:
        return WorkerConnectInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerConnectInput.model_validate(parameters)
        return await self._manager.connect(validated.endpoint)


class WorkersListTool(Tool):
    def __init__(self, *, manager: WorkerManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "workers_list"

    def description(self) -> str:
        return "List currently connected workers and their latest state."

    def parameters(self) -> dict[str, Any]:
        return EmptyInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        EmptyInput.model_validate(parameters)
        return self._manager.format_connected_workers()


class WorkerPromptTool(Tool):
    def __init__(self, *, manager: WorkerManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "worker_prompt"

    def description(self) -> str:
        return "Send a prompt to one connected worker when it is ready."

    def parameters(self) -> dict[str, Any]:
        return WorkerPromptInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerPromptInput.model_validate(parameters)
        return await self._manager.prompt(validated.endpoint, validated.prompt)


class WorkerWaitTool(Tool):
    def __init__(self, *, manager: WorkerManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "worker_wait"

    def description(self) -> str:
        return "Wait for the next meaningful update from one connected worker."

    def parameters(self) -> dict[str, Any]:
        return WorkerWaitInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerWaitInput.model_validate(parameters)
        return await self._manager.wait(validated.endpoint)


class WorkersWaitAnyTool(Tool):
    def __init__(self, *, manager: WorkerManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "workers_wait_any"

    def description(self) -> str:
        return "Wait for the next meaningful update from any connected worker."

    def parameters(self) -> dict[str, Any]:
        return EmptyInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        EmptyInput.model_validate(parameters)
        return await self._manager.wait_any()


class WorkerCancelTool(Tool):
    def __init__(self, *, manager: WorkerManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "worker_cancel"

    def description(self) -> str:
        return "Cancel the current run on one connected worker."

    def parameters(self) -> dict[str, Any]:
        return WorkerDisconnectInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerDisconnectInput.model_validate(parameters)
        return await self._manager.cancel(validated.endpoint)


class WorkerDisconnectTool(Tool):
    def __init__(self, *, manager: WorkerManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "worker_disconnect"

    def description(self) -> str:
        return "Disconnect from one connected worker."

    def parameters(self) -> dict[str, Any]:
        return WorkerDisconnectInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerDisconnectInput.model_validate(parameters)
        return await self._manager.disconnect(validated.endpoint)


def create_worker_tools(*, manager: WorkerManager) -> list[Tool]:
    return [
        WorkerDiscoverTool(manager=manager),
        WorkerConnectTool(manager=manager),
        WorkersListTool(manager=manager),
        WorkerPromptTool(manager=manager),
        WorkerWaitTool(manager=manager),
        WorkersWaitAnyTool(manager=manager),
        WorkerCancelTool(manager=manager),
        WorkerDisconnectTool(manager=manager),
    ]
