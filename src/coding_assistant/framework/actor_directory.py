from __future__ import annotations

from typing import Any, Protocol, TypeVar


MessageT = TypeVar("MessageT", contravariant=True)


class _MessageSink(Protocol[MessageT]):
    async def send_message(self, message: MessageT) -> None: ...


class ActorDirectory:
    def __init__(self) -> None:
        self._actors: dict[str, _MessageSink[Any]] = {}

    def register(self, *, uri: str, actor: _MessageSink[Any]) -> None:
        if uri in self._actors:
            raise RuntimeError(f"Actor URI already registered: {uri}")
        self._actors[uri] = actor

    def unregister(self, *, uri: str) -> None:
        self._actors.pop(uri, None)

    def resolve(self, *, uri: str) -> _MessageSink[Any]:
        actor = self._actors.get(uri)
        if actor is None:
            raise RuntimeError(f"Unknown actor URI: {uri}")
        return actor

    async def send_message(self, *, uri: str, message: object) -> None:
        actor = self.resolve(uri=uri)
        await actor.send_message(message)
