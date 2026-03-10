from __future__ import annotations

import asyncio
from dataclasses import dataclass

from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actors.common.messages import RunCompleted, RunFailed


@dataclass(slots=True)
class RunReplyWaiter:
    request_id: str
    future: asyncio.Future[None]

    async def send_message(self, message: object) -> None:
        if isinstance(message, RunCompleted):
            if message.request_id != self.request_id:
                self.future.set_exception(RuntimeError(f"Mismatched run response id: {message.request_id}"))
                return
            self.future.set_result(None)
            return
        if isinstance(message, RunFailed):
            if message.request_id != self.request_id:
                self.future.set_exception(RuntimeError(f"Mismatched run response id: {message.request_id}"))
                return
            self.future.set_exception(message.error)
            return
        self.future.set_exception(RuntimeError(f"Unexpected run response type: {type(message).__name__}"))


@dataclass(slots=True)
class RunPayloadReplyWaiter:
    request_id: str
    future: asyncio.Future[RunCompleted]

    async def send_message(self, message: object) -> None:
        if isinstance(message, RunCompleted):
            if message.request_id != self.request_id:
                self.future.set_exception(RuntimeError(f"Mismatched run response id: {message.request_id}"))
                return
            self.future.set_result(message)
            return
        if isinstance(message, RunFailed):
            if message.request_id != self.request_id:
                self.future.set_exception(RuntimeError(f"Mismatched run response id: {message.request_id}"))
                return
            self.future.set_exception(message.error)
            return
        self.future.set_exception(RuntimeError(f"Unexpected run response type: {type(message).__name__}"))


def register_run_reply_waiter(
    actor_directory: ActorDirectory,
    *,
    request_id: str,
    reply_uri: str,
) -> asyncio.Future[None]:
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    actor_directory.register(uri=reply_uri, actor=RunReplyWaiter(request_id=request_id, future=future))
    return future


def register_run_payload_reply_waiter(
    actor_directory: ActorDirectory,
    *,
    request_id: str,
    reply_uri: str,
) -> asyncio.Future[RunCompleted]:
    future: asyncio.Future[RunCompleted] = asyncio.get_running_loop().create_future()
    actor_directory.register(uri=reply_uri, actor=RunPayloadReplyWaiter(request_id=request_id, future=future))
    return future
