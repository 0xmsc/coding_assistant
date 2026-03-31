from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel


class ToolCallPayload(BaseModel):
    """Serialized form of one tool call emitted by a worker."""

    id: str
    name: str
    arguments: str


class WorkerStatePayload(BaseModel):
    """Authoritative worker state snapshot carried by protocol messages."""

    promptable: bool
    remote_connected: bool
    running: bool
    queued_prompt_count: int = 0


class StateMessage(WorkerStatePayload):
    """Current promptability/running snapshot for one worker."""

    type: Literal["state"] = "state"


class PromptAcceptedMessage(WorkerStatePayload):
    """Notification that the worker accepted one prompt into its queue."""

    type: Literal["prompt_accepted"] = "prompt_accepted"
    content: str | list[dict[str, Any]]


class PromptStartedMessage(WorkerStatePayload):
    """Notification that one accepted prompt has started running."""

    type: Literal["prompt_started"] = "prompt_started"
    content: str | list[dict[str, Any]]


class ContentDeltaMessage(BaseModel):
    """Incremental assistant text streamed from a worker run."""

    type: Literal["content_delta"] = "content_delta"
    content: str


class ToolCallsMessage(BaseModel):
    """Notification that the worker emitted tool calls."""

    type: Literal["tool_calls"] = "tool_calls"
    tool_calls: list[ToolCallPayload]


class RunFinishedMessage(WorkerStatePayload):
    """Terminal message for a completed worker run."""

    type: Literal["run_finished"] = "run_finished"
    summary: str


class RunCancelledMessage(WorkerStatePayload):
    """Terminal message for a cancelled worker run."""

    type: Literal["run_cancelled"] = "run_cancelled"


class RunFailedMessage(WorkerStatePayload):
    """Terminal message for a worker run that failed with an exception."""

    type: Literal["run_failed"] = "run_failed"
    error: str


class CommandAcceptedMessage(WorkerStatePayload):
    """Acknowledgement for a prompt or cancel command."""

    type: Literal["command_accepted"] = "command_accepted"
    request_id: str


class NotReadyMessage(WorkerStatePayload):
    """Command rejection carrying the worker state that caused the rejection."""

    type: Literal["not_ready"] = "not_ready"
    request_id: str


class ErrorMessage(BaseModel):
    """Protocol-level error response that is not tied to a state transition."""

    type: Literal["error"] = "error"
    message: str
    request_id: str | None = None


class PromptCommand(BaseModel):
    """Supervisor request to start a worker run from a prompt."""

    type: Literal["prompt"] = "prompt"
    request_id: str
    prompt: str
    mode: Literal["queue", "priority", "interrupt"] = "queue"


class CancelCommand(BaseModel):
    """Supervisor request to cancel the active worker run."""

    type: Literal["cancel"] = "cancel"
    request_id: str


WorkerToSupervisorMessage = (
    StateMessage
    | PromptAcceptedMessage
    | PromptStartedMessage
    | ContentDeltaMessage
    | ToolCallsMessage
    | RunFinishedMessage
    | RunCancelledMessage
    | RunFailedMessage
    | CommandAcceptedMessage
    | NotReadyMessage
    | ErrorMessage
)

SupervisorToWorkerMessage = PromptCommand | CancelCommand


def parse_worker_message(data: str | bytes) -> WorkerToSupervisorMessage:
    """Parse one inbound worker-to-supervisor JSON message."""
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    payload = json.loads(data)
    message_type = payload.get("type")
    if message_type == "state":
        return StateMessage.model_validate(payload)
    if message_type == "prompt_accepted":
        return PromptAcceptedMessage.model_validate(payload)
    if message_type == "prompt_started":
        return PromptStartedMessage.model_validate(payload)
    if message_type == "content_delta":
        return ContentDeltaMessage.model_validate(payload)
    if message_type == "tool_calls":
        return ToolCallsMessage.model_validate(payload)
    if message_type == "run_finished":
        return RunFinishedMessage.model_validate(payload)
    if message_type == "run_cancelled":
        return RunCancelledMessage.model_validate(payload)
    if message_type == "run_failed":
        return RunFailedMessage.model_validate(payload)
    if message_type == "command_accepted":
        return CommandAcceptedMessage.model_validate(payload)
    if message_type == "not_ready":
        return NotReadyMessage.model_validate(payload)
    if message_type == "error":
        return ErrorMessage.model_validate(payload)
    raise ValueError(f"Unknown worker message type: {message_type}")


def parse_supervisor_message(data: str | bytes) -> SupervisorToWorkerMessage:
    """Parse one inbound supervisor-to-worker JSON message."""
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    payload = json.loads(data)
    message_type = payload.get("type")
    if message_type == "prompt":
        return PromptCommand.model_validate(payload)
    if message_type == "cancel":
        return CancelCommand.model_validate(payload)
    raise ValueError(f"Unknown supervisor message type: {message_type}")


def message_to_json(message: WorkerToSupervisorMessage | SupervisorToWorkerMessage) -> str:
    """Serialize a protocol model to the websocket payload format."""
    return message.model_dump_json()
