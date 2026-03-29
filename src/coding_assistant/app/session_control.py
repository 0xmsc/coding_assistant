from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    CompletionEvent,
    ContentDeltaEvent,
    ReasoningDeltaEvent,
    StatusEvent,
)


@dataclass(frozen=True)
class SessionState:
    promptable: bool
    running: bool


@dataclass(frozen=True)
class StateChangedEvent:
    state: SessionState


@dataclass(frozen=True)
class ToolCallsEvent:
    message: AssistantMessage


@dataclass(frozen=True)
class RunFinishedEvent:
    summary: str


@dataclass(frozen=True)
class RunCancelledEvent:
    """The current run was cancelled."""


@dataclass(frozen=True)
class RunFailedEvent:
    error: str


SessionEvent = (
    ContentDeltaEvent
    | ReasoningDeltaEvent
    | StatusEvent
    | CompletionEvent
    | StateChangedEvent
    | ToolCallsEvent
    | RunFinishedEvent
    | RunCancelledEvent
    | RunFailedEvent
)


@dataclass(frozen=True)
class PromptSubmissionResult:
    accepted: bool
    reason: Literal["accepted", "inactive_controller", "not_ready"]


class SessionController(Protocol):
    async def run(self, session: SessionControlSurface) -> None:
        """Drive one input source against the shared session runtime."""

        ...


class SessionOutput(Protocol):
    async def run(self, session: SessionControlSurface) -> None:
        """Observe and render or forward session events."""

        ...


class SessionControlSurface(Protocol):
    @property
    def state(self) -> SessionState:
        """Return the current promptability and run status."""

        ...

    @property
    def history(self) -> list[BaseMessage]:
        """Return a snapshot of the current conversation history."""

        ...

    def subscribe(self) -> AbstractAsyncContextManager[asyncio.Queue[SessionEvent]]:
        """Subscribe to session events; new subscribers receive the current state first."""

        ...

    def is_active_controller(self, controller: SessionController) -> bool:
        """Return whether `controller` currently owns input."""

        ...

    async def wait_for_controller_change(self, *, controller: SessionController) -> SessionController:
        """Wait until the active controller differs from `controller` and return the new owner."""

        ...

    async def activate_controller(self, controller: SessionController) -> bool:
        """Switch input ownership to `controller`; return `False` if it already owns input."""

        ...

    async def activate_default_controller(self, *, cancel_current_run: bool = False) -> None:
        """Restore the default controller, optionally cancelling the current run first."""

        ...

    async def submit_prompt(
        self,
        *,
        controller: SessionController,
        content: str | list[dict[str, Any]],
    ) -> PromptSubmissionResult:
        """Submit a prompt if `controller` still owns input, reporting why rejection happened."""

        ...

    async def cancel_current_run(self) -> bool:
        """Cancel the active run and return whether any run was actually cancelled."""

        ...

    def request_shutdown(self) -> None:
        """Request shutdown of the surrounding session runtime."""

        ...
