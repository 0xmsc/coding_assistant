from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from coding_assistant.core.session_updates import (
    SessionUpdate,
    committed_message_from_history_message,
    replay_updates_from_committed_message,
)
from coding_assistant.llm.openai import list_models as list_provider_models
from coding_assistant.llm.types import BaseMessage
from coding_assistant.manager.store import LoadedSession, SessionRecord, SessionStore
from coding_assistant.remote.acp import JsonObject, prompt_content_from_acp, session_id_from_params

MODEL_METADATA_KEY = "model"
MODEL_CACHE_TTL_SECONDS = 300.0
ModelLister = Callable[[], Awaitable[list[str]]]


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
    model: str
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


def _metadata_with_model(metadata: JsonObject, *, default_model: str) -> JsonObject:
    model = metadata.get(MODEL_METADATA_KEY)
    return {
        **metadata,
        MODEL_METADATA_KEY: model if isinstance(model, str) and model else default_model,
    }


def _model_from_record(record: SessionRecord, *, default_model: str) -> str:
    model = record.metadata.get(MODEL_METADATA_KEY)
    return model if isinstance(model, str) and model else default_model


def _models_with_default(*, default_model: str, provider_models: list[str]) -> list[str]:
    result: list[str] = []
    for model in [default_model, *provider_models]:
        if model and model not in result:
            result.append(model)
    return result


def _model_entries(models: list[str]) -> list[JsonObject]:
    return [{"id": model} for model in models]


def _model_param(params: JsonObject) -> str:
    model = params.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ManagerError("session/set_model requires a non-empty model string.")
    return model.strip()


def _session_metadata(session: LoadedSession, *, default_model: str) -> JsonObject:
    payload: JsonObject = {
        "sessionId": session.record.session_id,
        "updatedAt": session.record.updated_at,
        "_meta": {
            "version": session.record.version,
            **_metadata_with_model(session.record.metadata, default_model=default_model),
        },
    }
    if session.record.title is not None:
        payload["title"] = session.record.title
    return payload


def _record_metadata(record: SessionRecord, *, default_model: str) -> JsonObject:
    payload: JsonObject = {
        "sessionId": record.session_id,
        "updatedAt": record.updated_at,
        "_meta": {
            "version": record.version,
            **_metadata_with_model(record.metadata, default_model=default_model),
        },
    }
    if record.title is not None:
        payload["title"] = record.title
    return payload


class ManagerService:
    def __init__(
        self,
        *,
        store: SessionStore,
        worker_runner: WorkerRunner,
        default_model: str,
        model_lister: ModelLister = list_provider_models,
        model_cache_ttl_seconds: float = MODEL_CACHE_TTL_SECONDS,
    ) -> None:
        if not default_model.strip():
            raise ValueError("ManagerService requires a non-empty default model.")
        self._store = store
        self._worker_runner = worker_runner
        self._default_model = default_model.strip()
        self._model_lister = model_lister
        self._model_cache_ttl_seconds = model_cache_ttl_seconds
        self._model_cache: tuple[float, list[str]] | None = None
        self._active_prompts: set[str] = set()
        self._active_lock = asyncio.Lock()

    async def list_models(self) -> JsonObject:
        models = await self._available_models()
        return {
            "defaultModel": self._default_model,
            "models": _model_entries(models),
        }

    def list_sessions(self, *, params: JsonObject) -> JsonObject:
        scope_id = _scope_id_from_params(params)
        return {
            "sessions": [
                _record_metadata(record, default_model=self._default_model)
                for record in self._store.list_sessions(scope_id=scope_id)
            ],
            "nextCursor": None,
        }

    def new_session(self, *, params: JsonObject, initial_messages: list[BaseMessage]) -> JsonObject:
        scope_id = _scope_id_from_params(params)
        session = self._store.create_session(
            scope_id=scope_id,
            messages=initial_messages,
            metadata={MODEL_METADATA_KEY: self._default_model},
        )
        return {"sessionId": session.record.session_id}

    def rename_session(self, *, params: JsonObject) -> JsonObject:
        scope_id = _scope_id_from_params(params)
        session_id = session_id_from_params(params)
        title = params.get("title")
        if title is None:
            next_title = None
        elif isinstance(title, str):
            stripped_title = title.strip()
            next_title = stripped_title or None
        else:
            raise ManagerError("session/rename requires a string or null title.")
        record = self._store.rename_session(scope_id=scope_id, session_id=session_id, title=next_title)
        return _record_metadata(record, default_model=self._default_model)

    async def set_session_model(self, *, params: JsonObject) -> JsonObject:
        scope_id = _scope_id_from_params(params)
        session_id = session_id_from_params(params)
        model = _model_param(params)
        self._store.load_session(scope_id=scope_id, session_id=session_id)

        async with self._active_lock:
            if session_id in self._active_prompts:
                raise SessionBusyError("Cannot change model while session has an active prompt.")

        if model not in await self._available_models():
            raise ManagerError(f"Model {model} is not available.")

        record = self._store.update_session_metadata(
            scope_id=scope_id,
            session_id=session_id,
            metadata={MODEL_METADATA_KEY: model},
        )
        return _record_metadata(record, default_model=self._default_model)

    async def load_session(
        self, *, params: JsonObject, on_update: Callable[[SessionUpdate], Awaitable[None]]
    ) -> JsonObject:
        scope_id = _scope_id_from_params(params)
        session_id = session_id_from_params(params)
        session = self._store.load_session(scope_id=scope_id, session_id=session_id)
        for message in session.messages:
            committed = committed_message_from_history_message(message)
            if committed is None:
                continue
            for update in replay_updates_from_committed_message(committed):
                await on_update(update)
        return _session_metadata(session, default_model=self._default_model)

    async def prompt(
        self,
        *,
        params: JsonObject,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> PromptResult:
        scope_id = _scope_id_from_params(params)
        session_id = session_id_from_params(params)
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
                    model=_model_from_record(session.record, default_model=self._default_model),
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
        session_id = session_id_from_params(params)
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

    async def _available_models(self) -> list[str]:
        now = monotonic()
        if self._model_cache is not None:
            cached_at, cached_models = self._model_cache
            if now - cached_at < self._model_cache_ttl_seconds:
                return cached_models

        try:
            provider_models = await self._model_lister()
        except Exception:
            if self._model_cache is not None:
                return self._model_cache[1]
            return [self._default_model]

        models = _models_with_default(default_model=self._default_model, provider_models=provider_models)
        self._model_cache = (now, models)
        return models
