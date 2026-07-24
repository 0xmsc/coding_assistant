from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from coding_assistant.core.session_updates import SessionUpdate
from coding_assistant.manager.service import WorkerPrompt, WorkerRunResult
from coding_assistant.remote.client import WorkerClient
from coding_assistant.remote.protocol import messages_to_jsonrpc


class RemoteWorkerError(RuntimeError):
    pass


class RemoteWorkerRunner:
    """Run manager prompts through a remote worker session endpoint."""

    def __init__(self, *, endpoint: str) -> None:
        self._endpoint = endpoint
        self._active_clients: dict[str, WorkerClient] = {}
        self._active_lock = asyncio.Lock()

    async def run_prompt(
        self,
        *,
        prompt: WorkerPrompt,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> WorkerRunResult:
        client = await WorkerClient.connect(
            endpoint=self._endpoint,
            on_update=on_update,
        )
        await self._register_client(session_id=prompt.session_id, client=client)
        try:
            try:
                if prompt.cancel_requested.is_set():
                    await client.cancel(session_id=prompt.session_id)
                finished = await client.run(
                    {
                        "sessionId": prompt.session_id,
                        "messages": messages_to_jsonrpc(prompt.history),
                        "prompt": prompt.prompt,
                    },
                )
            except RuntimeError as exc:
                raise RemoteWorkerError(str(exc)) from exc
            return WorkerRunResult(
                stop_reason=finished.stop_reason,
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

    async def _register_client(self, *, session_id: str, client: WorkerClient) -> None:
        async with self._active_lock:
            self._active_clients[session_id] = client

    async def _unregister_client(self, *, session_id: str, client: WorkerClient) -> None:
        async with self._active_lock:
            if self._active_clients.get(session_id) is client:
                self._active_clients.pop(session_id, None)
