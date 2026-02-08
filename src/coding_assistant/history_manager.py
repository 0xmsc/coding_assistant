from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Sequence

from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.history import save_orchestrator_history
from coding_assistant.llm.types import BaseMessage


@dataclass(slots=True)
class _SaveOrchestratorHistory:
    working_directory: Path
    history: Sequence[BaseMessage]


_Message = _SaveOrchestratorHistory


class HistoryManager:
    def __init__(self, *, context_name: str = "history") -> None:
        self._actor: Actor[_Message, None] = Actor(name=f"{context_name}.history", handler=self._handle_message)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._actor.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self._actor.stop()
        self._started = False

    async def save_orchestrator_history(self, *, working_directory: Path, history: Sequence[BaseMessage]) -> None:
        self.start()
        await self._actor.ask(_SaveOrchestratorHistory(working_directory=working_directory, history=history))

    async def _handle_message(self, message: _Message) -> None:
        if isinstance(message, _SaveOrchestratorHistory):
            save_orchestrator_history(message.working_directory, message.history)
            return None
        raise RuntimeError(f"Unknown history manager message: {message!r}")


@asynccontextmanager
async def history_manager_scope(*, context_name: str) -> AsyncIterator[HistoryManager]:
    manager = HistoryManager(context_name=context_name)
    manager.start()
    try:
        yield manager
    finally:
        await manager.stop()
