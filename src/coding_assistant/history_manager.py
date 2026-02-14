from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Sequence

from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.history import save_orchestrator_history
from coding_assistant.llm.types import BaseMessage


class HistoryManager:
    async def save_orchestrator_history(self, *, working_directory: Path, history: Sequence[BaseMessage]) -> None:
        save_orchestrator_history(working_directory, history)


@dataclass(slots=True)
class _SaveOrchestratorHistoryRequest:
    working_directory: Path
    history: Sequence[BaseMessage]
    future: asyncio.Future[None]


_Message = _SaveOrchestratorHistoryRequest


class HistoryManagerActor:
    def __init__(self, *, context_name: str = "history") -> None:
        self._actor: Actor[_Message] = Actor(name=f"{context_name}.history", handler=self._handle_message)
        self._manager = HistoryManager()
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
        future = asyncio.get_running_loop().create_future()
        await self._actor.send(
            _SaveOrchestratorHistoryRequest(
                working_directory=working_directory,
                history=history,
                future=future,
            )
        )
        await future

    async def _handle_message(self, message: _Message) -> None:
        if isinstance(message, _SaveOrchestratorHistoryRequest):
            try:
                await self._manager.save_orchestrator_history(
                    working_directory=message.working_directory,
                    history=message.history,
                )
            except BaseException as exc:
                if not message.future.done():
                    message.future.set_exception(exc)
                return None
            if not message.future.done():
                message.future.set_result(None)
            return None
        raise RuntimeError(f"Unknown history manager actor message: {message!r}")


@asynccontextmanager
async def history_manager_scope(*, context_name: str) -> AsyncIterator[HistoryManagerActor]:
    manager = HistoryManagerActor(context_name=context_name)
    manager.start()
    try:
        yield manager
    finally:
        await manager.stop()
