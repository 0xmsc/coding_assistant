from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from coding_assistant.llm.types import AssistantMessage, BaseMessage


@dataclass(frozen=True)
class AwaitingUser:
    """Boundary returned when the caller should provide the next user input."""

    history: list[BaseMessage]


@dataclass(frozen=True)
class AwaitingToolCalls:
    """Boundary returned when the last history entry is an assistant tool-call turn."""

    history: list[BaseMessage]

    @property
    def message(self) -> AssistantMessage:
        """Return the pending assistant message that requested the tool calls."""
        pending_message = get_pending_tool_call_message(self.history)
        if pending_message is None:
            raise RuntimeError(
                "AwaitingToolCalls requires the last history entry to be an assistant tool-call message.",
            )
        return pending_message


AgentBoundary = AwaitingUser | AwaitingToolCalls


def get_pending_tool_call_message(history: Sequence[BaseMessage]) -> AssistantMessage | None:
    """Return the last assistant message when it still has unhandled tool calls."""
    if not history:
        return None

    last_message = history[-1]
    if isinstance(last_message, AssistantMessage) and last_message.tool_calls:
        return last_message
    return None
