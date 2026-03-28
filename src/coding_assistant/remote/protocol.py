from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel


class ToolCallPayload(BaseModel):
    id: str
    name: str
    arguments: str


class StateMessage(BaseModel):
    type: Literal["state"] = "state"
    promptable: bool
    remote_connected: bool
    running: bool


class ContentDeltaMessage(BaseModel):
    type: Literal["content_delta"] = "content_delta"
    content: str


class ToolCallsMessage(BaseModel):
    type: Literal["tool_calls"] = "tool_calls"
    tool_calls: list[ToolCallPayload]


class RunFinishedMessage(BaseModel):
    type: Literal["run_finished"] = "run_finished"
    summary: str


class RunCancelledMessage(BaseModel):
    type: Literal["run_cancelled"] = "run_cancelled"


class RunFailedMessage(BaseModel):
    type: Literal["run_failed"] = "run_failed"
    error: str


class CommandAcceptedMessage(BaseModel):
    type: Literal["command_accepted"] = "command_accepted"
    request_id: str


class NotReadyMessage(BaseModel):
    type: Literal["not_ready"] = "not_ready"
    request_id: str
    promptable: bool
    remote_connected: bool
    running: bool


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str
    request_id: str | None = None


class PromptCommand(BaseModel):
    type: Literal["prompt"] = "prompt"
    request_id: str
    prompt: str


class CancelCommand(BaseModel):
    type: Literal["cancel"] = "cancel"
    request_id: str


WorkerToSupervisorMessage = (
    StateMessage
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
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    payload = json.loads(data)
    message_type = payload.get("type")
    if message_type == "state":
        return StateMessage.model_validate(payload)
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
    return message.model_dump_json()
