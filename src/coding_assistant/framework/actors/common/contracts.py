from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

MessageT = TypeVar("MessageT", contravariant=True)


@runtime_checkable
class MessageSink(Protocol[MessageT]):
    async def send_message(self, message: MessageT) -> None: ...
