from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from coding_assistant.llm.types import BaseMessage


AgentStatus = Literal["awaiting_user", "failed"]


@dataclass(slots=True)
class AgentRunResult:
    history: list[BaseMessage]
    status: AgentStatus
    error: str | None = None
