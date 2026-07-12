from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from coding_assistant.core.session_updates import SessionUpdate
from coding_assistant.manager.service import WorkerPrompt, WorkerRunFinished
from coding_assistant.remote.client import (
    RemoteClientEvent,
    RemotePromptFailedEvent,
    RemoteRunFinished,
    RemoteSessionClient,
)
from coding_assistant.remote.protocol import messages_to_jsonrpc


class RemoteWorkerError(RuntimeError):
    pass


class RemoteWorkerRunner:
    """Run manager prompts through a remote worker session endpoint."""

    def __init__(self, *, endpoint: str) -> None:
        self._endpoint = endpoint
        self._active_clients: dict[str, RemoteSessionClient] = {}
        self._active_lock = asyncio.Lock()

    async def run_prompt(
        self,
        *,
        prompt: WorkerPrompt,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> WorkerRunFinished:
        finish_future: asyncio.Future[RemoteRunFinished] = asyncio.get_running_loop().create_future()

        async def handle_event(event: RemoteClientEvent) -> None:
            if isinstance(event, RemotePromptFailedEvent) and not finish_future.done():
                finish_future.set_exception(RemoteWorkerError(event.message))

        async def handle_disconnect(endpoint: str) -> None:
            if not finish_future.done():
                finish_future.set_exception(RemoteWorkerError(f"Remote worker {endpoint} disconnected."))

        async def handle_run_finished(finished: RemoteRunFinished) -> None:
            if finished.session_id != prompt.session_id:
                return
            if finished.base_version != prompt.base_version:
                if not finish_future.done():
                    finish_future.set_exception(
                        RemoteWorkerError(
                            f"Worker finish for {prompt.session_id} used base version "
                            f"{finished.base_version}, not {prompt.base_version}.",
                        ),
                    )
                return
            if not finish_future.done():
                finish_future.set_result(finished)

        client = await RemoteSessionClient.connect(
            endpoint=self._endpoint,
            on_event=handle_event,
            on_disconnect=handle_disconnect,
            on_session_update=on_update,
            on_run_finished=handle_run_finished,
        )
        await self._register_client(session_id=prompt.session_id, client=client)
        try:
            await client.initialize()
            await client.start_session(
                {
                    "sessionId": prompt.session_id,
                    "baseVersion": prompt.base_version,
                    "messages": messages_to_jsonrpc(prompt.history),
                    "workspace": prompt.workspace,
                },
            )
            prompt_error = await client.prompt_blocks(prompt.prompt, session_id=prompt.session_id)
            if prompt_error is not None:
                raise RemoteWorkerError(prompt_error)

            finished = await finish_future
            return WorkerRunFinished(
                stop_reason=finished.stop_reason,
                messages=finished.messages,
                title=finished.title,
            )
        finally:
            await self._unregister_client(session_id=prompt.session_id, client=client)
            await client.close()

    async def cancel(self, *, session_id: str) -> None:
        async with self._active_lock:
            client = self._active_clients.get(session_id)
        if client is None:
            return
        await client.cancel(session_id=session_id)

    async def _register_client(self, *, session_id: str, client: RemoteSessionClient) -> None:
        async with self._active_lock:
            self._active_clients[session_id] = client

    async def _unregister_client(self, *, session_id: str, client: RemoteSessionClient) -> None:
        async with self._active_lock:
            if self._active_clients.get(session_id) is client:
                self._active_clients.pop(session_id, None)
