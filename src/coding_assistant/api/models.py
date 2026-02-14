from typing import Any, Literal, Optional, Union
from pydantic import BaseModel


class StatusMessage(BaseModel):
    type: Literal["status"] = "status"
    level: Literal["info", "success", "warning", "error"]
    message: str


class ContentChunk(BaseModel):
    type: Literal["chunk"] = "chunk"
    content: str


class ToolStart(BaseModel):
    type: Literal["tool_start"] = "tool_start"
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    type: Literal["tool_message"] = "tool_message"
    id: str
    name: str
    content: str


class AskRequest(BaseModel):
    type: Literal["ask"] = "ask"
    request_id: str
    prompt: str
    default: Optional[str] = None


class ConfirmRequest(BaseModel):
    type: Literal["confirm"] = "confirm"
    request_id: str
    prompt: str


class StartCommand(BaseModel):
    type: Literal["start"] = "start"
    task: str


class AnswerResponse(BaseModel):
    type: Literal["answer"] = "answer"
    request_id: str
    text: str


class ConfirmationResponse(BaseModel):
    type: Literal["confirmation"] = "confirmation"
    request_id: str
    value: bool


class InterruptCommand(BaseModel):
    type: Literal["interrupt"] = "interrupt"


AgentMessage = Union[
    StatusMessage,
    ContentChunk,
    ToolStart,
    ToolResult,
    AskRequest,
    ConfirmRequest,
]

ClientMessage = Union[
    StartCommand,
    AnswerResponse,
    ConfirmationResponse,
    InterruptCommand,
]


class Envelope(BaseModel):
    session_id: Optional[str] = None
    payload: Union[AgentMessage, ClientMessage]
