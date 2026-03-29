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
    queued_prompt_count: int = 0
    last_update: str = ""


@dataclass(frozen=True)
class WorkerMeaningfulEvent:
    endpoint: str
    kind: Literal["finished", "cancelled", "failed", "disconnected"]
    summary: str = ""


class _WorkerManager:
    """Track active supervisor-side connections to other workers."""

    def __init__(self) -> None:
        self._connections: dict[str, RemoteWorkerConnection] = {}
        self._snapshots: dict[str, WorkerSnapshot] = {}
        self._worker_queues: dict[str, asyncio.Queue[WorkerMeaningfulEvent]] = {}
        self._any_queue: asyncio.Queue[WorkerMeaningfulEvent] = asyncio.Queue()

    def format_connected_workers(self) -> str:
        if not self._connections:
            return "No connected workers."

        lines = []
        for endpoint, snapshot in sorted(self._snapshots.items()):
            status = "running" if snapshot.running else "promptable" if snapshot.promptable else "idle"
            if snapshot.queued_prompt_count:
                status = f"{status}, {snapshot.queued_prompt_count} queued"
            suffix = f" -> {snapshot.last_update}" if snapshot.last_update else ""
            lines.append(f"- {endpoint} [{status}]{suffix}")
        return "\n".join(lines)

    async def connect(self, endpoint: str) -> str:
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

    async def prompt(
        self, endpoint: str, prompt: str, *, mode: Literal["queue", "priority", "interrupt"] = "queue"
    ) -> str:
        connection = self._connections.get(endpoint)
        if connection is None:
            return f"Worker {endpoint} is not connected."

        response = await connection.prompt(prompt, mode=mode)
        if isinstance(response, CommandAcceptedMessage):
            if mode == "priority":
                return f"Priority prompt sent to worker {endpoint}."
            if mode == "interrupt":
                return f"Interrupt prompt sent to worker {endpoint}."
            return f"Prompt sent to worker {endpoint}."
        if isinstance(response, NotReadyMessage):
            return (
                f"Worker {endpoint} is not ready for a prompt. "
                f"promptable={response.promptable} running={response.running} "
                f"queued={response.queued_prompt_count}"
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
        if endpoint not in self._connections and queue.empty():
            self._worker_queues.pop(endpoint, None)
            return f"Worker {endpoint} is not connected."
        event = await queue.get()
        self._cleanup_idle_queue(endpoint)
        return _format_meaningful_event(event)

    async def wait_any(self) -> str:
        if self._any_queue.empty() and not self._connections:
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
            snapshot.queued_prompt_count = message.queued_prompt_count
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

    def _cleanup_idle_queue(self, endpoint: str) -> None:
        queue = self._worker_queues.get(endpoint)
        if queue is None:
            return
        if endpoint in self._connections or not queue.empty():
            return
        self._worker_queues.pop(endpoint, None)


class WorkerToolRuntime:
    """Worker-tool runtime that hides supervisor-side connection management."""

    def __init__(self) -> None:
        self._manager = _WorkerManager()
        self.tools = _create_worker_tools(runtime=self)

    def format_connected_workers(self) -> str:
        return self._manager.format_connected_workers()

    async def connect(self, endpoint: str) -> str:
        return await self._manager.connect(endpoint)

    async def disconnect(self, endpoint: str) -> str:
        return await self._manager.disconnect(endpoint)

    async def prompt(
        self, endpoint: str, prompt: str, *, mode: Literal["queue", "priority", "interrupt"] = "queue"
    ) -> str:
        return await self._manager.prompt(endpoint, prompt, mode=mode)

    async def cancel(self, endpoint: str) -> str:
        return await self._manager.cancel(endpoint)

    async def wait(self, endpoint: str) -> str:
        return await self._manager.wait(endpoint)

    async def wait_any(self) -> str:
        return await self._manager.wait_any()

    async def close(self) -> None:
        await self._manager.close()


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
    mode: Literal["queue", "priority", "interrupt"] = Field(
        default="queue",
        description="How to schedule the prompt: queue normally, queue with priority, or interrupt the current run.",
    )


class WorkerWaitInput(BaseModel):
    endpoint: str = Field(description="The websocket endpoint of the worker to wait on.")


class WorkerDisconnectInput(BaseModel):
    endpoint: str = Field(description="The websocket endpoint of the worker to disconnect from.")


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
        return "Send a prompt to one connected worker, optionally as a priority or interrupting prompt."

    def parameters(self) -> dict[str, Any]:
        return WorkerPromptInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerPromptInput.model_validate(parameters)
        return await self._runtime.prompt(validated.endpoint, validated.prompt, mode=validated.mode)


class WorkerWaitTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "worker_wait"

    def description(self) -> str:
        return "Wait for the next meaningful update from one connected worker."

    def parameters(self) -> dict[str, Any]:
        return WorkerWaitInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerWaitInput.model_validate(parameters)
        return await self._runtime.wait(validated.endpoint)


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
        return "Cancel the current run on one connected worker."

    def parameters(self) -> dict[str, Any]:
        return WorkerDisconnectInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerDisconnectInput.model_validate(parameters)
        return await self._runtime.cancel(validated.endpoint)


class WorkerDisconnectTool(Tool):
    def __init__(self, *, runtime: WorkerToolRuntime) -> None:
        self._runtime = runtime

    def name(self) -> str:
        return "worker_disconnect"

    def description(self) -> str:
        return "Disconnect from one connected worker."

    def parameters(self) -> dict[str, Any]:
        return WorkerDisconnectInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = WorkerDisconnectInput.model_validate(parameters)
        return await self._runtime.disconnect(validated.endpoint)


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
