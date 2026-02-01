import uuid
from datetime import datetime, timezone
from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, Field, ConfigDict

# Define a generic Payload type for the Envelope
T = TypeVar("T")


class Envelope(BaseModel, Generic[T]):
    """
    The standard wrapper for all messages in the Actor system.
    Ensures traceability and routing.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    sender: str
    recipient: str

    # correlation_id: links related messages (e.g., request and response)
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # trace_id: tracks a single user request across the entire system
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # parent_id: optional reference to the message that triggered this one
    parent_id: Optional[str] = None

    payload: T
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
