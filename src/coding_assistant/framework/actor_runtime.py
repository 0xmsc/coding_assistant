from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

MessageT = TypeVar("MessageT")
ResponseT = TypeVar("ResponseT")


class ActorStoppedError(RuntimeError):
    pass


class _Stop:
    pass


@dataclass(slots=True)
class _Envelope(Generic[MessageT, ResponseT]):
    message: MessageT | _Stop
    reply_future: asyncio.Future[ResponseT] | None


class Actor(Generic[MessageT, ResponseT]):
    def __init__(self, *, name: str, handler: Callable[[MessageT], Awaitable[ResponseT]]) -> None:
        self._name = name
        self._handler = handler
        self._queue: asyncio.Queue[_Envelope[MessageT, ResponseT]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name=f"actor:{self._name}")

    async def stop(self) -> None:
        if self._task is None or self._task.done():
            return
        await self._queue.put(_Envelope(message=_Stop(), reply_future=None))
        await self._task

    async def send(self, message: MessageT) -> None:
        self._ensure_running()
        await self._queue.put(_Envelope(message=message, reply_future=None))

    async def ask(self, message: MessageT) -> ResponseT:
        self._ensure_running()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ResponseT] = loop.create_future()
        await self._queue.put(_Envelope(message=message, reply_future=future))
        return await future

    def _ensure_running(self) -> None:
        if self._task is None or self._task.done():
            raise ActorStoppedError(f"Actor {self._name} is not running.")

    async def _run(self) -> None:
        while True:
            envelope = await self._queue.get()
            if isinstance(envelope.message, _Stop):
                break
            try:
                result = await self._handler(envelope.message)
            except Exception as exc:
                if envelope.reply_future is not None and not envelope.reply_future.done():
                    envelope.reply_future.set_exception(exc)
                else:
                    logger.exception("Actor %s handler error", self._name)
                continue
            if envelope.reply_future is not None and not envelope.reply_future.done():
                envelope.reply_future.set_result(result)
