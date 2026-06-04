from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from coding_assistant.core.session_updates import (
    SessionUpdate,
    committed_message_from_history_message,
    replay_updates_from_committed_message,
)
from coding_assistant.llm.types import BaseMessage
from coding_assistant.manager.store import LoadedSession, SessionRecord, SessionStore
from coding_assistant.remote.acp import JsonObject, prompt_content_from_acp


class ManagerError(RuntimeError):
    pass


class SessionBusyError(ManagerError):
    pass


@dataclass(frozen=True)
class PromptResult:
    stop_reason: str


@dataclass(frozen=True)
class WorkerPrompt:
    session_id: str
    base_version: int
    history: list[BaseMessage]
    workspace: str
    prompt: list[JsonObject]


@dataclass(frozen=True)
class WorkerCommit:
    messages: list[BaseMessage]
    stop_reason: str
    title: str | None = None
    metadata: JsonObject | None = None


class WorkerRunner(Protocol):
    async def run_prompt(
        self,
        *,
        prompt: WorkerPrompt,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> WorkerCommit: ...

    async def cancel(self, *, session_id: str) -> None: ...


def _scope_id_from_params(params: JsonObject) -> str:
    metadata = params.get("_meta")
    if not isinstance(metadata, dict):
        raise ManagerError("Request params must include _meta.scopeId.")
    scope_id = metadata.get("scopeId")
    if not isinstance(scope_id, str) or not scope_id:
        raise ManagerError("Request params must include _meta.scopeId.")
    return scope_id


def _session_metadata(session: LoadedSession) -> JsonObject:
    payload: JsonObject = {
        "sessionId": session.record.session_id,
        "updatedAt": session.record.updated_at,
        "_meta": {
            "version": session.record.version,
            **session.record.metadata,
        },
    }
    if session.record.title is not None:
        payload["title"] = session.record.title
    return payload


def _record_metadata(record: SessionRecord) -> JsonObject:
    payload: JsonObject = {
        "sessionId": record.session_id,
        "updatedAt": record.updated_at,
        "_meta": {
            "version": record.version,
            **record.metadata,
        },
    }
    if record.title is not None:
        payload["title"] = record.title
    return payload


class ManagerService:
    def __init__(self, *, store: SessionStore, worker_runner: WorkerRunner) -> None:
        self._store = store
        self._worker_runner = worker_runner
        self._active_prompts: set[str] = set()
        self._active_lock = asyncio.Lock()

    def list_sessions(self, *, params: JsonObject) -> JsonObject:
        scope_id = _scope_id_from_params(params)
        return {
            "sessions": [_record_metadata(record) for record in self._store.list_sessions(scope_id=scope_id)],
            "nextCursor": None,
        }

    def new_session(self, *, params: JsonObject, initial_messages: list[BaseMessage]) -> JsonObject:
        scope_id = _scope_id_from_params(params)
        session = self._store.create_session(scope_id=scope_id, messages=initial_messages)
        return {"sessionId": session.record.session_id}

    async def load_session(
        self, *, params: JsonObject, on_update: Callable[[SessionUpdate], Awaitable[None]]
    ) -> JsonObject:
        scope_id = _scope_id_from_params(params)
        session_id = self._session_id_from_params(params)
        session = self._store.load_session(scope_id=scope_id, session_id=session_id)
        for message in session.messages:
            committed = committed_message_from_history_message(message)
            if committed is None:
                continue
            for update in replay_updates_from_committed_message(committed):
                await on_update(update)
        return _session_metadata(session)

    async def prompt(
        self,
        *,
        params: JsonObject,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> PromptResult:
        scope_id = _scope_id_from_params(params)
        session_id = self._session_id_from_params(params)
        prompt_blocks = params.get("prompt")
        if not isinstance(prompt_blocks, list) or not all(isinstance(block, dict) for block in prompt_blocks):
            raise ManagerError("session/prompt requires a prompt array.")
        try:
            prompt_content_from_acp(prompt_blocks)
        except ValueError as exc:
            raise ManagerError(str(exc)) from exc

        await self._mark_prompt_active(session_id)
        try:
            session = self._store.load_session(scope_id=scope_id, session_id=session_id)
            worker_commit = await self._worker_runner.run_prompt(
                prompt=WorkerPrompt(
                    session_id=session_id,
                    base_version=session.record.version,
                    history=session.messages,
                    workspace=str(session.workspace),
                    prompt=prompt_blocks,
                ),
                on_update=on_update,
            )
            self._store.commit_messages(
                scope_id=scope_id,
                session_id=session_id,
                base_version=session.record.version,
                messages=worker_commit.messages,
                title=worker_commit.title,
                metadata=worker_commit.metadata,
            )
            return PromptResult(stop_reason=worker_commit.stop_reason)
        finally:
            await self._mark_prompt_idle(session_id)

    async def cancel(self, *, params: JsonObject) -> None:
        scope_id = _scope_id_from_params(params)
        session_id = self._session_id_from_params(params)
        self._store.load_session(scope_id=scope_id, session_id=session_id)
        await self._worker_runner.cancel(session_id=session_id)

    async def _mark_prompt_active(self, session_id: str) -> None:
        async with self._active_lock:
            if session_id in self._active_prompts:
                raise SessionBusyError("Session already has an active prompt.")
            self._active_prompts.add(session_id)

    async def _mark_prompt_idle(self, session_id: str) -> None:
        async with self._active_lock:
            self._active_prompts.discard(session_id)

    def _session_id_from_params(self, params: JsonObject) -> str:
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise ManagerError("Request params must include sessionId.")
        return session_id
