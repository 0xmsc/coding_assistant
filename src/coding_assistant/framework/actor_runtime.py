from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, fields, is_dataclass
from typing import Mapping
from time import monotonic
from typing import Any, Awaitable, Callable, Generic, TypeVar

from coding_assistant.trace import trace_enabled, trace_json

logger = logging.getLogger(__name__)

MessageT = TypeVar("MessageT")


class ActorStoppedError(RuntimeError):
    pass


class _Stop:
    pass


@dataclass(slots=True)
class _Envelope(Generic[MessageT]):
    message: MessageT | _Stop
    enqueued_at: float


_TRACE_REPR_MAX_CHARS = 2000


def _truncate_text(text: str, *, max_chars: int = _TRACE_REPR_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...<truncated>"


def _serialize_dataclass_trace_value(value: object) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if is_dataclass(value):
        return {field.name: _serialize_dataclass_trace_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _serialize_dataclass_trace_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_serialize_dataclass_trace_value(item) for item in value]
    return _truncate_text(repr(value))


def _serialize_trace_message(message: object) -> Any | None:
    if not is_dataclass(message):
        return None
    try:
        return _serialize_dataclass_trace_value(message)
    except BaseException as exc:
        return {
            "serialization_error": _truncate_text(repr(exc)),
            "repr": _truncate_text(repr(message)),
        }


def _trace_actor_event(event: str, *, actor_name: str, data: dict[str, Any] | None = None) -> None:
    if not trace_enabled():
        return
    payload = {"event": event, "actor": actor_name}
    if data:
        payload.update(data)
    trace_json("actor_event.json5", payload)


class Actor(Generic[MessageT]):
    def __init__(self, *, name: str, handler: Callable[[MessageT], Awaitable[None]]) -> None:
        self._name = name
        self._handler = handler
        self._queue: asyncio.Queue[_Envelope[MessageT]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        _trace_actor_event("actor.start", actor_name=self._name)
        self._task = asyncio.create_task(self._run(), name=f"actor:{self._name}")

    async def stop(self) -> None:
        if self._task is None or self._task.done():
            return
        await self._queue.put(_Envelope(message=_Stop(), enqueued_at=monotonic()))
        await self._task
        _trace_actor_event("actor.stop", actor_name=self._name)

    async def send(self, message: MessageT) -> None:
        self._ensure_running()
        await self._queue.put(_Envelope(message=message, enqueued_at=monotonic()))

    def _ensure_running(self) -> None:
        if self._task is None or self._task.done():
            raise ActorStoppedError(f"Actor {self._name} is not running.")

    async def _run(self) -> None:
        while True:
            envelope = await self._queue.get()
            if isinstance(envelope.message, _Stop):
                break
            serialized_message = _serialize_trace_message(envelope.message)
            queue_wait_ms = (monotonic() - envelope.enqueued_at) * 1000
            handler_start = monotonic()
            try:
                await self._handler(envelope.message)
                handler_ms = (monotonic() - handler_start) * 1000
                _trace_actor_event(
                    "actor.message",
                    actor_name=self._name,
                    data={
                        "message_type": type(envelope.message).__name__,
                        "message": serialized_message,
                        "queue_wait_ms": queue_wait_ms,
                        "handler_ms": handler_ms,
                        "status": "ok",
                    },
                )
            except Exception as exc:
                handler_ms = (monotonic() - handler_start) * 1000
                _trace_actor_event(
                    "actor.message",
                    actor_name=self._name,
                    data={
                        "message_type": type(envelope.message).__name__,
                        "message": serialized_message,
                        "queue_wait_ms": queue_wait_ms,
                        "handler_ms": handler_ms,
                        "status": "error",
                        "error": repr(exc),
                    },
                )
                logger.exception("Actor %s handler error", self._name)
                continue
            except BaseException as exc:
                handler_ms = (monotonic() - handler_start) * 1000
                _trace_actor_event(
                    "actor.message",
                    actor_name=self._name,
                    data={
                        "message_type": type(envelope.message).__name__,
                        "message": serialized_message,
                        "queue_wait_ms": queue_wait_ms,
                        "handler_ms": handler_ms,
                        "status": "fatal",
                        "error": repr(exc),
                    },
                )
                if isinstance(exc, asyncio.CancelledError):
                    raise
                break
