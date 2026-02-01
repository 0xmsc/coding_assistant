from typing import Union, Literal, Any, Optional
from pydantic import BaseModel, ConfigDict
from coding_assistant.llm.types import BaseMessage, Tool, Completion
from coding_assistant.messaging.ui_messages import UserInputRequested, UserInputReceived


class Message(BaseModel):
    """Base class for all Actor messages."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class StartTask(Message):
    type: Literal["start_task"] = "start_task"
    prompt: str


class LLMRequest(Message):
    type: Literal["llm_request"] = "llm_request"
    messages: list[BaseMessage]
    tools: list[Tool]
    model: str


class LLMResponse(Message):
    type: Literal["llm_response"] = "llm_response"
    completion: Completion


class ExecuteTool(Message):
    type: Literal["execute_tool"] = "execute_tool"
    tool_name: str
    arguments: dict[str, Any]


class ToolResult(Message):
    type: Literal["tool_result"] = "tool_result"
    result: Any
    is_error: bool = False


class DisplayMessage(Message):
    type: Literal["display_message"] = "display_message"
    content: str
    role: Literal["user", "assistant", "system"] = "assistant"


class TaskCompleted(Message):
    type: Literal["task_completed"] = "task_completed"
    result: str


class Error(Message):
    type: Literal["error"] = "error"
    message: str
    code: Optional[str] = None


# Type alias for any actor message
ActorMessage = Union[
    StartTask,
    LLMRequest,
    LLMResponse,
    ExecuteTool,
    ToolResult,
    DisplayMessage,
    TaskCompleted,
    Error,
    UserInputRequested,
    UserInputReceived,
]
