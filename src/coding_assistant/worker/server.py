from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field

from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from coding_assistant.core.agent_session import AgentSession, AgentSessionEvent, CompletionStreamer, RunOutcome
from coding_assistant.llm.types import BaseMessage, Tool, UserMessage
from coding_assistant.remote.control import (
    send_session_update,
    session_updates_from_agent_event,
    wait_for_session_events,
    worker_run_result,
)
from coding_assistant.remote.jsonrpc import (
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_SERVER,
    JsonObject,
    jsonrpc_error,
    jsonrpc_result,
    params_from_payload,
    prompt_content_from_acp,
    response_id_from_payload,
    session_id_from_params,
)
from coding_assistant.remote.protocol import messages_from_jsonrpc
from coding_assistant.remote.websocket_server import receive_jsonrpc_messages, serve_jsonrpc_websocket


@dataclass(frozen=True)
class WorkerServer:
    endpoint: str
    _finished: asyncio.Event

    async def wait_finished(self) -> None:
        await self._finished.wait()


@dataclass(frozen=True)
class WorkerRuntimeConfig:
    model: str
    tools: list[Tool]
    completion_streamer: CompletionStreamer | None = None
    finish_metadata_provider: Callable[[], JsonObject | None] | None = None


@dataclass
class _WorkerConnectionState:
    session_id: str | None = None
    session: AgentSession | None = None
    run_task: asyncio.Task[None] | None = None
    cancel_requested_session_id: str | None = None
    run_ready: asyncio.Event = field(default_factory=asyncio.Event)


def _messages_from_params(params: JsonObject) -> list[BaseMessage]:
    messages = params.get("messages")
    if not isinstance(messages, list):
        raise ValueError("_worker/run requires messages array.")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("_worker/run messages must be objects.")
    return messages_from_jsonrpc(messages)


def _prompt_from_params(params: JsonObject) -> str | list[JsonObject]:
    prompt = params.get("prompt")
    if not isinstance(prompt, list) or not all(isinstance(block, dict) for block in prompt):
        raise ValueError("_worker/run requires a prompt array.")
    return prompt_content_from_acp(prompt)


async def _publish_session_events(
    *,
    websocket: ServerConnection,
    session_id: str,
    queue: asyncio.Queue[AgentSessionEvent],
) -> None:
    while True:
        event = await queue.get()
        try:
            for update in session_updates_from_agent_event(event):
                await send_session_update(websocket=websocket, session_id=session_id, update=update)
        finally:
            queue.task_done()


async def _execute_run(
    *,
    websocket: ServerConnection,
    response_id: int | str,
    session_id: str,
    prompt: str | list[JsonObject],
    session: AgentSession,
    state: _WorkerConnectionState,
    runtime: WorkerRuntimeConfig,
    finished: asyncio.Event,
) -> None:
    sender_task: asyncio.Task[None] | None = None
    try:
        async with session.subscribe() as queue:
            sender_task = asyncio.create_task(
                _publish_session_events(websocket=websocket, session_id=session_id, queue=queue),
            )
            scheduled_run = await session.enqueue_prompt(prompt)
            if scheduled_run is None:
                raise RuntimeError("Worker session closed before its run could start.")
            state.run_ready.set()
            if state.cancel_requested_session_id == session_id:
                pending_prompt = await session.pop_last_queued_prompt()
                if pending_prompt is not None:
                    outcome = RunOutcome(
                        stop_reason="cancelled",
                        messages=[UserMessage(content=pending_prompt)],
                    )
                else:
                    await session.cancel_current_run()
                    outcome = await scheduled_run.completion
            else:
                outcome = await scheduled_run.completion
            await wait_for_session_events(queue=queue, publisher_task=sender_task)
            if outcome.stop_reason == "failed":
                await websocket.send(jsonrpc_error(response_id, ERROR_SERVER, outcome.error or "Worker run failed."))
                return
            metadata = runtime.finish_metadata_provider() if runtime.finish_metadata_provider is not None else None
            await websocket.send(
                jsonrpc_result(
                    response_id,
                    worker_run_result(outcome=outcome, metadata=metadata),
                ),
            )
    except ConnectionClosed:
        return
    finally:
        state.run_ready.set()
        if sender_task is not None:
            sender_task.cancel()
            await asyncio.gather(sender_task, return_exceptions=True)
        await session.close()
        finished.set()


async def _handle_run(
    *,
    websocket: ServerConnection,
    response_id: int | str | None,
    params: JsonObject,
    state: _WorkerConnectionState,
    runtime: WorkerRuntimeConfig,
    finished: asyncio.Event,
    claim_run: Callable[[], Awaitable[bool]],
) -> None:
    if response_id is None:
        return
    try:
        session_id = session_id_from_params(params)
        messages = _messages_from_params(params)
        prompt = _prompt_from_params(params)
    except ValueError as exc:
        await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, str(exc)))
        return
    if not await claim_run():
        await websocket.send(jsonrpc_error(response_id, ERROR_SERVER, "Worker already has a run."))
        return

    session = AgentSession(
        history=messages,
        model=runtime.model,
        tools=runtime.tools,
        completion_streamer=runtime.completion_streamer,
    )
    state.session_id = session_id
    state.session = session
    state.run_task = asyncio.create_task(
        _execute_run(
            websocket=websocket,
            response_id=response_id,
            session_id=session_id,
            prompt=prompt,
            session=session,
            state=state,
            runtime=runtime,
            finished=finished,
        ),
    )


async def _handle_cancel(
    *,
    websocket: ServerConnection,
    response_id: int | str | None,
    params: JsonObject,
    state: _WorkerConnectionState,
) -> None:
    session_id = session_id_from_params(params)
    if state.session is None or state.session_id is None:
        state.cancel_requested_session_id = session_id
        if response_id is not None:
            await websocket.send(jsonrpc_result(response_id, None))
        return
    if session_id != state.session_id:
        if response_id is not None:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "Unknown sessionId."))
        return
    state.cancel_requested_session_id = session_id
    await state.run_ready.wait()
    await state.session.cancel_current_run()
    if response_id is not None:
        await websocket.send(jsonrpc_result(response_id, None))


@asynccontextmanager
async def start_session_worker_server(
    *,
    runtime: WorkerRuntimeConfig,
    host: str = "127.0.0.1",
    port: int = 0,
) -> AsyncIterator[WorkerServer]:
    finished = asyncio.Event()
    run_claim_lock = asyncio.Lock()
    run_claimed = False

    async def claim_run() -> bool:
        nonlocal run_claimed
        async with run_claim_lock:
            if run_claimed:
                return False
            run_claimed = True
            return True

    async def handle_connection(websocket: ServerConnection) -> None:
        state = _WorkerConnectionState()

        async def handle_message(payload: JsonObject) -> None:
            response_id = response_id_from_payload(payload)
            method = payload.get("method")
            if payload.get("jsonrpc") != "2.0" or not isinstance(method, str):
                await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "Invalid JSON-RPC request."))
                return
            try:
                params = params_from_payload(payload)
                if method == "_worker/run":
                    if response_id is None:
                        return
                    await _handle_run(
                        websocket=websocket,
                        response_id=response_id,
                        params=params,
                        state=state,
                        runtime=runtime,
                        finished=finished,
                        claim_run=claim_run,
                    )
                    return
                if method == "session/cancel":
                    await _handle_cancel(
                        websocket=websocket,
                        response_id=response_id,
                        params=params,
                        state=state,
                    )
                    return
                if response_id is not None:
                    await websocket.send(
                        jsonrpc_error(response_id, ERROR_METHOD_NOT_FOUND, f"Unsupported method: {method}")
                    )
            except ValueError as exc:
                if response_id is not None:
                    await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, str(exc)))

        try:
            await receive_jsonrpc_messages(websocket=websocket, on_message=handle_message)
        finally:
            if state.session is not None:
                await state.session.close()
            if state.run_task is not None:
                with suppress(asyncio.CancelledError):
                    await state.run_task

    async with serve_jsonrpc_websocket(handler=handle_connection, host=host, port=port, close_when=finished) as server:
        yield WorkerServer(endpoint=server.endpoint, _finished=finished)
