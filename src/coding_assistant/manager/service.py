from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Protocol

from coding_assistant.core.session_updates import (
    SessionUpdate,
    UserMessageChunkUpdate,
    committed_message_from_history_message,
    replay_updates_from_committed_message,
)
from coding_assistant.llm.openai import list_models as list_provider_models
from coding_assistant.llm.types import BaseMessage, UserMessage
from coding_assistant.manager.store import LoadedSession, SessionRecord, SessionStore
from coding_assistant.remote.acp import JsonObject, prompt_content_from_acp, session_id_from_params

MODEL_METADATA_KEY = "model"
MODEL_CACHE_TTL_SECONDS = 300.0
ModelLister = Callable[[], Awaitable[list[str]]]
ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_WORKER_ENV_VARS = 32
MAX_WORKER_ENV_VALUE_BYTES = 8192
MAX_SKILL_BUNDLES = 8
MAX_SKILL_FILES_PER_BUNDLE = 32
MAX_SKILL_FILE_BYTES = 64 * 1024
MAX_SKILL_BUNDLE_BYTES = 256 * 1024
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
SAFE_ATTACHMENT_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
SUPPORTED_ATTACHMENT_TYPES = {
    "application/csv",
    "application/json",
    "application/markdown",
    "application/x-ndjson",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
    "text/csv",
    "text/markdown",
    "text/plain",
}


class ManagerError(RuntimeError):
    pass


class SessionBusyError(ManagerError):
    pass


@dataclass(frozen=True)
class PromptResult:
    stop_reason: str


@dataclass(frozen=True)
class SkillBundle:
    name: str
    description: str
    files: dict[str, str]


@dataclass(frozen=True)
class PromptCapabilities:
    worker_env: dict[str, str] = field(default_factory=dict)
    skills: tuple[SkillBundle, ...] = ()


@dataclass(frozen=True)
class SessionAttachment:
    attachment_id: str
    name: str
    mime_type: str
    size: int
    path: str
    sha256: str

    def to_json(self) -> JsonObject:
        return {
            "id": self.attachment_id,
            "name": self.name,
            "mimeType": self.mime_type,
            "size": self.size,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class WorkerPrompt:
    session_id: str
    base_version: int
    history: list[BaseMessage]
    model: str
    workspace: str
    prompt: list[JsonObject]
    capabilities: PromptCapabilities = field(default_factory=PromptCapabilities)


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


def _scope_id_from_params(params: JsonObject, *, allow_capabilities: bool = False) -> str:
    metadata = params.get("_meta")
    if not isinstance(metadata, dict):
        raise ManagerError("Request params must include _meta.scopeId.")
    if not allow_capabilities and "capabilities" in metadata:
        raise ManagerError("_meta.capabilities is only accepted on session/prompt.")
    scope_id = metadata.get("scopeId")
    if not isinstance(scope_id, str) or not scope_id:
        raise ManagerError("Request params must include _meta.scopeId.")
    return scope_id


def _prompt_capabilities_from_params(params: JsonObject) -> PromptCapabilities:
    metadata = params.get("_meta")
    if not isinstance(metadata, dict):
        return PromptCapabilities()
    raw_capabilities = metadata.get("capabilities")
    if raw_capabilities is None:
        return PromptCapabilities()
    if not isinstance(raw_capabilities, dict):
        raise ManagerError("_meta.capabilities must be an object.")
    return PromptCapabilities(
        worker_env=_worker_env_from_capabilities(raw_capabilities),
        skills=tuple(_skill_bundles_from_capabilities(raw_capabilities)),
    )


def _worker_env_from_capabilities(raw_capabilities: JsonObject) -> dict[str, str]:
    worker_env = raw_capabilities.get("workerEnv", {})
    if worker_env is None:
        return {}
    if not isinstance(worker_env, dict):
        raise ManagerError("_meta.capabilities.workerEnv must be an object.")
    if len(worker_env) > MAX_WORKER_ENV_VARS:
        raise ManagerError("_meta.capabilities.workerEnv has too many entries.")

    result: dict[str, str] = {}
    for key, value in worker_env.items():
        if not isinstance(key, str) or not ENV_NAME_RE.fullmatch(key):
            raise ManagerError(f"Invalid worker environment variable name: {key!r}.")
        if not isinstance(value, str):
            raise ManagerError(f"Worker environment variable {key} must be a string.")
        if len(value.encode("utf-8")) > MAX_WORKER_ENV_VALUE_BYTES:
            raise ManagerError(f"Worker environment variable {key} is too large.")
        result[key] = value
    return result


def _skill_bundles_from_capabilities(raw_capabilities: JsonObject) -> list[SkillBundle]:
    raw_skills = raw_capabilities.get("skills", [])
    if raw_skills is None:
        return []
    if not isinstance(raw_skills, list):
        raise ManagerError("_meta.capabilities.skills must be an array.")
    if len(raw_skills) > MAX_SKILL_BUNDLES:
        raise ManagerError("_meta.capabilities.skills has too many bundles.")

    bundles: list[SkillBundle] = []
    seen_names: set[str] = set()
    for raw_skill in raw_skills:
        if not isinstance(raw_skill, dict):
            raise ManagerError("Each injected skill must be an object.")
        name = raw_skill.get("name")
        description = raw_skill.get("description")
        raw_files = raw_skill.get("files")
        if not isinstance(name, str) or not SKILL_NAME_RE.fullmatch(name):
            raise ManagerError(f"Invalid injected skill name: {name!r}.")
        if name in seen_names:
            raise ManagerError(f"Duplicate injected skill name: {name}.")
        if not isinstance(description, str) or not description.strip():
            raise ManagerError(f"Injected skill {name} requires a description.")
        if not isinstance(raw_files, dict):
            raise ManagerError(f"Injected skill {name} files must be an object.")
        files = _skill_files_from_payload(name=name, raw_files=raw_files)
        if "SKILL.md" not in files:
            raise ManagerError(f"Injected skill {name} requires SKILL.md.")
        bundles.append(SkillBundle(name=name, description=description, files=files))
        seen_names.add(name)
    return bundles


def _skill_files_from_payload(*, name: str, raw_files: dict[object, object]) -> dict[str, str]:
    if len(raw_files) > MAX_SKILL_FILES_PER_BUNDLE:
        raise ManagerError(f"Injected skill {name} has too many files.")

    total_size = 0
    files: dict[str, str] = {}
    for raw_path, raw_content in raw_files.items():
        if not isinstance(raw_path, str) or not _safe_skill_file_path(raw_path):
            raise ManagerError(f"Injected skill {name} has an invalid file path: {raw_path!r}.")
        if not isinstance(raw_content, str):
            raise ManagerError(f"Injected skill {name} file {raw_path} must be a string.")
        content_size = len(raw_content.encode("utf-8"))
        if content_size > MAX_SKILL_FILE_BYTES:
            raise ManagerError(f"Injected skill {name} file {raw_path} is too large.")
        total_size += content_size
        if total_size > MAX_SKILL_BUNDLE_BYTES:
            raise ManagerError(f"Injected skill {name} bundle is too large.")
        files[raw_path] = raw_content
    return files


def _safe_skill_file_path(path: str) -> bool:
    parsed = PurePosixPath(path)
    return path == parsed.as_posix() and not parsed.is_absolute() and bool(parsed.parts) and ".." not in parsed.parts


def _safe_attachment_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ManagerError("session/upload_file requires a non-empty file name.")
    candidate = PurePosixPath(name).name.strip()
    candidate = SAFE_ATTACHMENT_NAME_RE.sub("-", candidate).strip(".-")
    if not candidate:
        candidate = "attachment"
    return candidate[:120]


def _attachment_mime_type(*, raw_mime_type: object, name: str) -> str:
    if isinstance(raw_mime_type, str) and raw_mime_type.strip():
        mime_type = raw_mime_type.strip().lower()
    elif name.lower().endswith(".json"):
        mime_type = "application/json"
    elif name.lower().endswith(".md"):
        mime_type = "text/markdown"
    elif name.lower().endswith(".csv"):
        mime_type = "text/csv"
    elif name.lower().endswith(".txt"):
        mime_type = "text/plain"
    else:
        raise ManagerError("session/upload_file requires a supported MIME type.")

    if mime_type.startswith("text/") or mime_type in SUPPORTED_ATTACHMENT_TYPES:
        return mime_type
    raise ManagerError(f"Unsupported attachment MIME type: {mime_type}.")


def _upload_bytes(raw_data: object) -> bytes:
    if not isinstance(raw_data, str) or not raw_data:
        raise ManagerError("session/upload_file requires base64 file data.")
    if len(raw_data) > MAX_ATTACHMENT_BYTES * 2:
        raise ManagerError("Attachment upload is too large.")
    try:
        data = base64.b64decode(raw_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ManagerError("session/upload_file data must be valid base64.") from exc
    if not data:
        raise ManagerError("Attachment upload cannot be empty.")
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise ManagerError("Attachment upload is too large.")
    return data


def _write_attachment(*, workspace: Path, params: JsonObject) -> SessionAttachment:
    name = _safe_attachment_name(params.get("name"))
    mime_type = _attachment_mime_type(raw_mime_type=params.get("mimeType"), name=name)
    data = _upload_bytes(params.get("data"))
    content_hash = hashlib.sha256(data).hexdigest()
    attachment_id = f"att_{content_hash[:16]}"
    relative_path = f"attachments/{attachment_id}-{name}"
    target = workspace / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return SessionAttachment(
        attachment_id=attachment_id,
        name=name,
        mime_type=mime_type,
        size=len(data),
        path=relative_path,
        sha256=content_hash,
    )


def _attachment_message(attachment: SessionAttachment) -> str:
    return (
        f"Attached file `{attachment.name}` as `{attachment.path}` "
        f"({attachment.mime_type}, {attachment.size} bytes). "
        f'Use `load_file("{attachment.path}")` before reasoning from this file.'
    )


def _model_from_record(record: SessionRecord) -> str:
    model = record.metadata.get(MODEL_METADATA_KEY)
    if isinstance(model, str) and model.strip():
        return model.strip()
    raise ManagerError("Session has no model selected.")


def _model_entries(models: list[str]) -> list[JsonObject]:
    return [{"id": model} for model in models]


def _model_param(params: JsonObject) -> str:
    model = params.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ManagerError("session/set_model requires a non-empty model string.")
    return model.strip()


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
    def __init__(
        self,
        *,
        store: SessionStore,
        worker_runner: WorkerRunner,
        model_lister: ModelLister = list_provider_models,
        model_cache_ttl_seconds: float = MODEL_CACHE_TTL_SECONDS,
    ) -> None:
        self._store = store
        self._worker_runner = worker_runner
        self._model_lister = model_lister
        self._model_cache_ttl_seconds = model_cache_ttl_seconds
        self._model_cache: tuple[float, list[str]] | None = None
        self._active_prompts: set[str] = set()
        self._active_lock = asyncio.Lock()

    async def list_models(self) -> JsonObject:
        models = await self._available_models()
        return {
            "models": _model_entries(models),
        }

    def list_sessions(self, *, params: JsonObject) -> JsonObject:
        scope_id = _scope_id_from_params(params)
        return {
            "sessions": [_record_metadata(record) for record in self._store.list_sessions(scope_id=scope_id)],
            "nextCursor": None,
        }

    def new_session(self, *, params: JsonObject, initial_messages: list[BaseMessage]) -> JsonObject:
        scope_id = _scope_id_from_params(params)
        session = self._store.create_session(
            scope_id=scope_id,
            messages=initial_messages,
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
        return _record_metadata(record)

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
        return _record_metadata(record)

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
        return _session_metadata(session)

    async def upload_file(
        self,
        *,
        params: JsonObject,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> JsonObject:
        scope_id = _scope_id_from_params(params)
        session_id = session_id_from_params(params)

        async with self._active_lock:
            if session_id in self._active_prompts:
                raise SessionBusyError("Cannot upload a file while session has an active prompt.")
            session = self._store.load_session(scope_id=scope_id, session_id=session_id)
            attachment = _write_attachment(workspace=session.workspace, params=params)
            message_text = _attachment_message(attachment)
            self._store.commit_messages(
                scope_id=scope_id,
                session_id=session_id,
                base_version=session.record.version,
                messages=[UserMessage(content=message_text)],
            )

        await on_update(UserMessageChunkUpdate(content=message_text))
        updated = self._store.load_session(scope_id=scope_id, session_id=session_id)
        return {"attachment": attachment.to_json(), "session": _record_metadata(updated.record)}

    async def prompt(
        self,
        *,
        params: JsonObject,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> PromptResult:
        scope_id = _scope_id_from_params(params, allow_capabilities=True)
        session_id = session_id_from_params(params)
        capabilities = _prompt_capabilities_from_params(params)
        prompt_blocks = params.get("prompt")
        if not isinstance(prompt_blocks, list) or not all(isinstance(block, dict) for block in prompt_blocks):
            raise ManagerError("session/prompt requires a prompt array.")
        try:
            prompt_content_from_acp(prompt_blocks)
        except ValueError as exc:
            raise ManagerError(str(exc)) from exc

        session = self._store.load_session(scope_id=scope_id, session_id=session_id)
        model = _model_from_record(session.record)
        await self._mark_prompt_active(session_id)
        try:
            worker_commit = await self._worker_runner.run_prompt(
                prompt=WorkerPrompt(
                    session_id=session_id,
                    base_version=session.record.version,
                    history=session.messages,
                    model=model,
                    workspace=str(session.workspace),
                    prompt=prompt_blocks,
                    capabilities=capabilities,
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
            return []

        models = list(dict.fromkeys(model for model in provider_models if model))
        self._model_cache = (now, models)
        return models
