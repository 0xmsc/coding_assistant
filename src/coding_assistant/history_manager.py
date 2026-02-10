from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Sequence
from uuid import uuid4

from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.history import save_orchestrator_history
from coding_assistant.llm.types import BaseMessage


@dataclass(slots=True)
class _SaveOrchestratorHistoryRequest:
    request_id: str
    working_directory: Path
    history: Sequence[BaseMessage]


@dataclass(slots=True)
class _SaveOrchestratorHistoryDone:
    request_id: str
    error: BaseException | None = None


_Message = _SaveOrchestratorHistoryRequest | _SaveOrchestratorHistoryDone


class HistoryManager:
    def __init__(self, *, context_name: str = "history") -> None:
        self._actor: Actor[_Message] = Actor(name=f"{context_name}.history", handler=self._handle_message)
        self._started = False
        self._pending: dict[str, asyncio.Future[None]] = {}

    def start(self) -> None:
        if self._started:
            return
        self._actor.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self._actor.stop()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("HistoryManager stopped before reply."))
        self._pending.clear()
        self._started = False

    async def save_orchestrator_history(self, *, working_directory: Path, history: Sequence[BaseMessage]) -> None:
        self.start()
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        self._pending[request_id] = future
        await self._actor.send(
            _SaveOrchestratorHistoryRequest(
                request_id=request_id,
                working_directory=working_directory,
                history=history,
            )
        )
        await future

    async def _handle_message(self, message: _Message) -> None:
        if isinstance(message, _SaveOrchestratorHistoryRequest):
            error: BaseException | None = None
            try:
                save_orchestrator_history(message.working_directory, message.history)
            except BaseException as exc:
                error = exc
            await self._actor.send(_SaveOrchestratorHistoryDone(request_id=message.request_id, error=error))
            return None
        if isinstance(message, _SaveOrchestratorHistoryDone):
            future = self._pending.pop(message.request_id, None)
            if future is None:
                raise RuntimeError(f"Unknown history response id: {message.request_id}")
            if message.error is not None:
                future.set_exception(message.error)
            else:
                future.set_result(None)
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
