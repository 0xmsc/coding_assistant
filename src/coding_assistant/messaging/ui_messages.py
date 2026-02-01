from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    """Base class for all Actor messages."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class UserInputRequested(Message):
    type: Literal["user_input_requested"] = "user_input_requested"
    prompt: str
    default: Optional[str] = None
    input_type: Literal["ask", "confirm", "prompt"] = "ask"


class UserInputReceived(Message):
    type: Literal["user_input_received"] = "user_input_received"
    content: str
    confirmed: Optional[bool] = None
