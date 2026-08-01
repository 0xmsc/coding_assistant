from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import re
import shutil
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Protocol
from uuid import uuid4

from coding_assistant.core.runtime import build_initial_system_message
from coding_assistant.core.session_updates import (
    AttachmentAddedUpdate,
    HistoryCompleteUpdate,
    HistoryResetUpdate,
    MessageAddedUpdate,
    MessageDeltaUpdate,
    RunUpdatedUpdate,
    SessionUpdate,
    SessionAttachment,
    SessionUpdatedUpdate,
    content_text,
)
from coding_assistant.llm.openai import (
    ProviderModel,
    list_models as list_provider_models,
    parse_model_and_reasoning,
)
from coding_assistant.llm.types import BaseMessage, UserMessage
from coding_assistant.manager.store import LoadedSession, SessionRecord, SessionStore
from coding_assistant.remote.jsonrpc import JsonObject, prompt_content_from_acp, session_id_from_params
from coding_assistant.worker.agent import WorkerAgentConfig, build_worker_instructions

MODEL_METADATA_KEY = "model"
REASONING_EFFORT_METADATA_KEY = "reasoningEffort"
MODEL_CACHE_TTL_SECONDS = 300.0
ModelLister = Callable[[], Awaitable[list[ProviderModel]]]
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
    "application/pdf",
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
class WorkerPrompt:
    session_id: str
    history: list[BaseMessage]
    model: str
    workspace: str
    attachments: str
    prompt: list[JsonObject]
    worker_env: dict[str, str] = field(default_factory=dict)
    cancel_requested: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)


@dataclass(frozen=True)
class WorkerRunResult:
    stop_reason: str
    title: str | None = None


@dataclass(frozen=True)
class PromptRunRecord:
    run_id: str
    session_id: str
    status: str
    started_at: str
    updated_at: str
    ended_at: str | None = None
    stop_reason: str | None = None
    error: str | None = None


@dataclass
class ActivePrompt:
    run: PromptRunRecord
    cancel_requested: asyncio.Event
    draft: MessageDeltaUpdate | None = None


class WorkerRunner(Protocol):
    async def run_prompt(
        self,
        *,
        prompt: WorkerPrompt,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> WorkerRunResult: ...

    async def cancel(self, *, session_id: str) -> None: ...


def scope_id_from_params(params: JsonObject) -> str:
    metadata = params.get("_meta")
    if not isinstance(metadata, dict):
        raise ManagerError("Request params must include _meta.scopeId.")
    scope_id = metadata.get("scopeId")
    if not isinstance(scope_id, str) or not scope_id:
        raise ManagerError("Request params must include _meta.scopeId.")
    return scope_id


def _skill_bundles_from_params(params: JsonObject) -> tuple[SkillBundle, ...]:
    metadata = params.get("_meta")
    if not isinstance(metadata, dict):
        return ()
    return tuple(_skill_bundles_from_value(metadata.get("skills", []), field_name="_meta.skills"))


def _worker_env_from_params(params: JsonObject) -> dict[str, str]:
    metadata = params.get("_meta")
    if not isinstance(metadata, dict):
        return {}
    return _worker_env_from_value(metadata.get("workerEnv", {}), field_name="_meta.workerEnv")


def _reject_prompt_skills(params: JsonObject) -> None:
    metadata = params.get("_meta")
    if isinstance(metadata, dict) and "skills" in metadata:
        raise ManagerError("session/prompt does not accept _meta.skills; pass skills to session/new.")


def _worker_env_from_value(raw_worker_env: object, *, field_name: str) -> dict[str, str]:
    worker_env = raw_worker_env
    if worker_env is None:
        return {}
    if not isinstance(worker_env, dict):
        raise ManagerError(f"{field_name} must be an object.")
    if len(worker_env) > MAX_WORKER_ENV_VARS:
        raise ManagerError(f"{field_name} has too many entries.")

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


def _skill_bundles_from_value(raw_skills: object, *, field_name: str) -> list[SkillBundle]:
    if raw_skills is None:
        return []
    if not isinstance(raw_skills, list):
        raise ManagerError(f"{field_name} must be an array.")
    if len(raw_skills) > MAX_SKILL_BUNDLES:
        raise ManagerError(f"{field_name} has too many bundles.")

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


def _write_session_skill_bundles(*, workspace: Path, skills: tuple[SkillBundle, ...]) -> Path:
    skills_root = workspace / ".agents" / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    for skill in skills:
        skill_root = skills_root / skill.name
        shutil.rmtree(skill_root, ignore_errors=True)
        skill_root.mkdir(parents=True, exist_ok=True)
        for relative_path, content in skill.files.items():
            target = skill_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    return skills_root


def _safe_attachment_name(name: object) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    candidate = PurePosixPath(name).name.strip()
    candidate = SAFE_ATTACHMENT_NAME_RE.sub("-", candidate).strip(".-")
    return candidate[:120] if candidate else None


def _attachment_mime_type(*, raw_mime_type: object, name: str | None) -> str:
    if isinstance(raw_mime_type, str) and raw_mime_type.strip():
        mime_type = raw_mime_type.strip().lower()
    elif name and name.lower().endswith(".json"):
        mime_type = "application/json"
    elif name and name.lower().endswith(".md"):
        mime_type = "text/markdown"
    elif name and name.lower().endswith(".csv"):
        mime_type = "text/csv"
    elif name and name.lower().endswith(".pdf"):
        mime_type = "application/pdf"
    elif name and name.lower().endswith(".txt"):
        mime_type = "text/plain"
    else:
        raise ManagerError("session/upload_file requires a supported MIME type.")

    if mime_type.startswith("text/") or mime_type in SUPPORTED_ATTACHMENT_TYPES:
        return mime_type
    raise ManagerError(f"Unsupported attachment MIME type: {mime_type}.")


def _default_attachment_name(mime_type: str) -> str:
    if mime_type == "image/gif":
        return "attachment.gif"
    if mime_type == "image/jpeg":
        return "attachment.jpg"
    if mime_type == "image/png":
        return "attachment.png"
    if mime_type == "image/webp":
        return "attachment.webp"
    if mime_type == "application/json":
        return "attachment.json"
    if mime_type == "application/pdf":
        return "attachment.pdf"
    if mime_type in {"application/markdown", "text/markdown"}:
        return "attachment.md"
    if mime_type in {"application/csv", "text/csv"}:
        return "attachment.csv"
    if mime_type == "application/x-ndjson":
        return "attachment.ndjson"
    return "attachment.txt"


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


def _write_attachment(*, attachments: Path, params: JsonObject) -> SessionAttachment:
    name = _safe_attachment_name(params.get("name"))
    mime_type = _attachment_mime_type(raw_mime_type=params.get("mimeType"), name=name)
    if name is None:
        name = _default_attachment_name(mime_type)
    data = _upload_bytes(params.get("data"))
    content_hash = hashlib.sha256(data).hexdigest()
    attachment_id = f"att_{uuid4().hex}"
    filename = f"{attachment_id}-{name}"
    worker_path = f"/attachments/{filename}"
    target = attachments / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return SessionAttachment(
        attachment_id=attachment_id,
        name=name,
        mime_type=mime_type,
        size=len(data),
        path=worker_path,
        sha256=content_hash,
    )


def _attachment_message(attachment: SessionAttachment) -> str:
    return (
        f"Attached file `{attachment.name}` as `{attachment.path}` "
        f"({attachment.mime_type}, {attachment.size} bytes). "
        f"{_attachment_instruction(attachment)}"
    )


def _attachment_instruction(attachment: SessionAttachment) -> str:
    if attachment.mime_type.startswith("image/"):
        return f'Use `load_image("{attachment.path}")` before reasoning from this image.'
    if attachment.mime_type == "application/pdf":
        return (
            "Use the `pdf-text-extraction` skill and extract text with "
            f'`mkdir -p extracted && pdftotext -layout -enc UTF-8 "{attachment.path}" '
            f'"{_pdf_text_output_path(attachment)}"` '
            "before reasoning from this PDF."
        )
    return "Inspect this text-like file with shell or Python before reasoning from it."


def _pdf_text_output_path(attachment: SessionAttachment) -> str:
    stem = Path(attachment.name).stem or "attachment"
    return f"extracted/{attachment.attachment_id}-{stem}.txt"


def _new_run_id() -> str:
    return f"run_{uuid4().hex}"


def _stored_message_update_id(message_id: int) -> str:
    return f"msg_{message_id}"


def _history_updates(session: LoadedSession) -> list[SessionUpdate]:
    updates: list[SessionUpdate] = []
    attachments_by_message_id: dict[int, list[AttachmentAddedUpdate]] = {}
    for attachment_record in session.attachment_records:
        attachments_by_message_id.setdefault(attachment_record.message_id, []).append(
            AttachmentAddedUpdate(
                attachment=attachment_record.attachment,
                created_at=attachment_record.created_at,
            )
        )

    for record in session.message_records:
        updates.append(
            MessageAddedUpdate(
                message_id=_stored_message_update_id(record.message_id),
                message=record.message,
                created_at=record.created_at,
            )
        )
        updates.extend(attachments_by_message_id.get(record.message_id, []))
    return updates


def _attachment_file_path(*, attachments: Path, attachment: SessionAttachment) -> Path:
    parsed = PurePosixPath(attachment.path)
    if not parsed.is_absolute() or parsed.parts[:2] != ("/", "attachments") or len(parsed.parts) != 3:
        raise ManagerError(f"Attachment {attachment.attachment_id} has invalid stored path.")
    return attachments / parsed.name


def _verified_attachment_bytes(*, attachments: Path, attachment: SessionAttachment) -> bytes:
    target = _attachment_file_path(attachments=attachments, attachment=attachment)
    data = target.read_bytes()
    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != attachment.sha256:
        raise ManagerError(f"Attachment {attachment.attachment_id} failed hash verification.")
    return data


def _model_from_record(record: SessionRecord) -> str:
    model = record.metadata.get(MODEL_METADATA_KEY)
    if isinstance(model, str) and model.strip():
        model = model.strip()
        reasoning_effort = record.metadata.get(REASONING_EFFORT_METADATA_KEY)
        return f"{model} ({reasoning_effort})" if isinstance(reasoning_effort, str) else model
    raise ManagerError("Session has no model selected.")


def _model_entries(models: Sequence[ProviderModel]) -> list[JsonObject]:
    entries: list[JsonObject] = []
    for model in models:
        entry: JsonObject = {"id": model.id}
        if model.reasoning_efforts:
            entry["reasoning"] = {"supportedEfforts": list(model.reasoning_efforts)}
        entries.append(entry)
    return entries


def _model_param(params: JsonObject) -> str:
    model = params.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ManagerError("session/set_model requires a non-empty model string.")
    return model.strip()


def _record_metadata(record: SessionRecord) -> JsonObject:
    payload: JsonObject = {
        "sessionId": record.session_id,
        "updatedAt": record.updated_at,
        "_meta": dict(record.metadata),
    }
    if record.title is not None:
        payload["title"] = record.title
    return payload


def _session_updated(record: SessionRecord) -> SessionUpdatedUpdate:
    return SessionUpdatedUpdate(session=_record_metadata(record))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_metadata(run: PromptRunRecord) -> JsonObject:
    payload: JsonObject = {
        "runId": run.run_id,
        "sessionId": run.session_id,
        "status": run.status,
        "startedAt": run.started_at,
        "updatedAt": run.updated_at,
    }
    if run.ended_at is not None:
        payload["endedAt"] = run.ended_at
    if run.stop_reason is not None:
        payload["stopReason"] = run.stop_reason
    if run.error is not None:
        payload["error"] = run.error
    return payload


def _run_updated(run: PromptRunRecord) -> RunUpdatedUpdate:
    return RunUpdatedUpdate(run=_run_metadata(run))


class ManagerService:
    def __init__(
        self,
        *,
        store: SessionStore,
        worker_runner: WorkerRunner,
        model_lister: ModelLister = list_provider_models,
        model_cache_ttl_seconds: float = MODEL_CACHE_TTL_SECONDS,
        worker_workspace: Path = Path("/workspace"),
        user_instructions: Sequence[str] = (),
    ) -> None:
        self._store = store
        self._worker_runner = worker_runner
        self._model_lister = model_lister
        self._model_cache_ttl_seconds = model_cache_ttl_seconds
        self._worker_workspace = worker_workspace
        self._user_instructions = tuple(user_instructions)
        self._model_cache: tuple[float, list[ProviderModel]] | None = None
        self._active_prompts: dict[str, ActivePrompt] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

    async def list_models(self) -> JsonObject:
        models = await self._available_models()
        return {
            "models": _model_entries(models),
        }

    def list_sessions(self, *, params: JsonObject) -> JsonObject:
        scope_id = scope_id_from_params(params)
        return {
            "sessions": [_record_metadata(record) for record in self._store.list_sessions(scope_id=scope_id)],
            "nextCursor": None,
        }

    def new_session(self, *, params: JsonObject) -> JsonObject:
        scope_id = scope_id_from_params(params)
        worker_env = _worker_env_from_params(params)
        skills = _skill_bundles_from_params(params)
        reservation = self._store.reserve_session_workspace()
        try:
            skills_root = _write_session_skill_bundles(workspace=reservation.workspace, skills=skills)
            initial_messages = self._initial_messages(skills_root=skills_root)
            session = self._store.create_session(
                scope_id=scope_id,
                messages=initial_messages,
                worker_env=worker_env,
                reserved_workspace=reservation,
            )
        except Exception:
            shutil.rmtree(reservation.root, ignore_errors=True)
            raise
        return {"sessionId": session.record.session_id}

    async def rename_session(
        self,
        *,
        params: JsonObject,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> JsonObject:
        scope_id = scope_id_from_params(params)
        session_id = session_id_from_params(params)
        title = params.get("title")
        if title is None:
            next_title = None
        elif isinstance(title, str):
            stripped_title = title.strip()
            next_title = stripped_title or None
        else:
            raise ManagerError("session/rename requires a string or null title.")
        async with self._session_lock(session_id):
            self._require_idle_session(session_id)
            record = self._store.rename_session(scope_id=scope_id, session_id=session_id, title=next_title)
            await on_update(_session_updated(record))
            return _record_metadata(record)

    async def delete_session(self, *, params: JsonObject) -> None:
        scope_id = scope_id_from_params(params)
        session_id = session_id_from_params(params)
        async with self._session_lock(session_id):
            self._require_idle_session(
                session_id, "Cannot delete session while it has an active prompt. Cancel it first."
            )
            self._store.delete_session(scope_id=scope_id, session_id=session_id)

    async def set_session_model(
        self,
        *,
        params: JsonObject,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> JsonObject:
        scope_id = scope_id_from_params(params)
        session_id = session_id_from_params(params)
        model = _model_param(params)
        try:
            base_model, reasoning_effort = parse_model_and_reasoning(model)
        except ValueError as exc:
            raise ManagerError(str(exc)) from exc

        available_models = await self._available_models()
        selected_model = next((item for item in available_models if item.id == base_model), None)
        if selected_model is None:
            raise ManagerError(f"Model {model} is not available.")
        if reasoning_effort is not None and reasoning_effort not in selected_model.reasoning_efforts:
            raise ManagerError(f"Reasoning effort {reasoning_effort} is not available for model {base_model}.")

        async with self._session_lock(session_id):
            self._require_idle_session(session_id, "Cannot change model while session has an active prompt.")
            record = self._store.update_session_metadata(
                scope_id=scope_id,
                session_id=session_id,
                metadata={
                    MODEL_METADATA_KEY: base_model,
                    REASONING_EFFORT_METADATA_KEY: reasoning_effort,
                },
            )
            await on_update(_session_updated(record))
            return _record_metadata(record)

    async def load_session(
        self, *, params: JsonObject, on_update: Callable[[SessionUpdate], Awaitable[None]]
    ) -> JsonObject:
        scope_id = scope_id_from_params(params)
        session_id = session_id_from_params(params)
        async with self._session_lock(session_id):
            session = self._store.load_session(scope_id=scope_id, session_id=session_id)
            await on_update(HistoryResetUpdate())
            for update in _history_updates(session):
                await on_update(update)
            active_prompt = self._active_prompts.get(session_id)
            if active_prompt is not None:
                if active_prompt.draft is not None:
                    await on_update(active_prompt.draft)
                await on_update(_run_updated(active_prompt.run))
            await on_update(HistoryCompleteUpdate())
            return _record_metadata(session.record)

    async def upload_file(
        self,
        *,
        params: JsonObject,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> JsonObject:
        scope_id = scope_id_from_params(params)
        session_id = session_id_from_params(params)

        async with self._session_lock(session_id):
            self._require_idle_session(session_id, "Cannot upload a file while session has an active prompt.")
            session = self._store.load_session(scope_id=scope_id, session_id=session_id)
            attachment = _write_attachment(attachments=session.attachments, params=params)
            message_text = _attachment_message(attachment)
            stored_messages = self._store.append_messages(
                scope_id=scope_id,
                session_id=session_id,
                messages=[UserMessage(content=message_text)],
                attachments=[attachment],
            )

            await on_update(
                MessageAddedUpdate(
                    message_id=_stored_message_update_id(stored_messages[0].message_id),
                    message=UserMessage(content=message_text),
                )
            )
            await on_update(AttachmentAddedUpdate(attachment=attachment))
            updated = self._store.load_session(scope_id=scope_id, session_id=session_id)
            await on_update(_session_updated(updated.record))
            return {"attachment": attachment.to_json(), "session": _record_metadata(updated.record)}

    async def download_attachment(self, *, params: JsonObject) -> JsonObject:
        scope_id = scope_id_from_params(params)
        session_id = session_id_from_params(params)
        attachment_id = params.get("attachmentId")
        if not isinstance(attachment_id, str) or not attachment_id:
            raise ManagerError("session/download_attachment requires an attachmentId.")

        session = self._store.load_session(scope_id=scope_id, session_id=session_id)
        attachment = next(
            (
                record.attachment
                for record in session.attachment_records
                if record.attachment.attachment_id == attachment_id
            ),
            None,
        )
        if attachment is None:
            raise ManagerError(f"Attachment {attachment_id} was not found.")

        data = _verified_attachment_bytes(attachments=session.attachments, attachment=attachment)
        return {
            "attachment": attachment.to_json(),
            "encoding": "base64",
            "data": base64.b64encode(data).decode("ascii"),
        }

    async def prompt(
        self,
        *,
        params: JsonObject,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> PromptResult:
        scope_id = scope_id_from_params(params)
        session_id = session_id_from_params(params)
        prompt_blocks = params.get("prompt")
        if not isinstance(prompt_blocks, list) or not all(isinstance(block, dict) for block in prompt_blocks):
            raise ManagerError("session/prompt requires a prompt array.")
        try:
            prompt_content = prompt_content_from_acp(prompt_blocks)
        except ValueError as exc:
            raise ManagerError(str(exc)) from exc
        prompt_worker_env = _worker_env_from_params(params)
        _reject_prompt_skills(params)

        lock = self._session_lock(session_id)
        async with lock:
            self._require_idle_session(session_id)
            session = self._store.load_session(scope_id=scope_id, session_id=session_id)
            model = _model_from_record(session.record)
            active_prompt = self._start_prompt_run(session_id)
            run = active_prompt.run

        async def publish_worker_update(update: SessionUpdate) -> None:
            async with lock:
                if self._active_prompts.get(session_id) is not active_prompt:
                    raise RuntimeError(f"Session {session_id} is no longer owned by run {run.run_id}.")
                if isinstance(update, MessageAddedUpdate):
                    self._store.append_messages(
                        scope_id=scope_id,
                        session_id=session_id,
                        messages=[update.message],
                    )
                    if active_prompt.draft is not None and active_prompt.draft.message_id == update.message_id:
                        active_prompt.draft = None
                elif isinstance(update, MessageDeltaUpdate):
                    draft = active_prompt.draft
                    if draft is None or draft.message_id != update.message_id:
                        active_prompt.draft = update
                    else:
                        active_prompt.draft = replace(
                            draft,
                            append_text=f"{draft.append_text}{update.append_text}",
                        )
                await on_update(update)

        try:
            async with lock:
                await on_update(_run_updated(run))
                if content_text(prompt_content) is not None:
                    user_message = UserMessage(content=prompt_content)
                    stored_user_message = self._store.append_messages(
                        scope_id=scope_id,
                        session_id=session_id,
                        messages=[user_message],
                    )[0]
                    await on_update(
                        MessageAddedUpdate(
                            message_id=_stored_message_update_id(stored_user_message.message_id),
                            message=user_message,
                        )
                    )
            worker_result = await self._worker_runner.run_prompt(
                prompt=WorkerPrompt(
                    session_id=session_id,
                    history=session.messages,
                    model=model,
                    workspace=str(session.workspace),
                    attachments=str(session.attachments),
                    prompt=prompt_blocks,
                    worker_env={**session.worker_env, **prompt_worker_env},
                    cancel_requested=active_prompt.cancel_requested,
                ),
                on_update=publish_worker_update,
            )
            status = "cancelled" if worker_result.stop_reason == "cancelled" else "completed"
            async with lock:
                if worker_result.title is None:
                    updated_record = self._store.load_session(scope_id=scope_id, session_id=session_id).record
                else:
                    updated_record = self._store.rename_session(
                        scope_id=scope_id,
                        session_id=session_id,
                        title=worker_result.title,
                    )
                finished_run = self._finish_prompt_run(
                    active_prompt,
                    status=status,
                    stop_reason=worker_result.stop_reason,
                )
                await on_update(_run_updated(finished_run))
                await on_update(_session_updated(updated_record))
            return PromptResult(stop_reason=worker_result.stop_reason)
        except Exception as exc:
            async with lock:
                failed_run = self._finish_prompt_run(
                    active_prompt,
                    status="failed",
                    error=str(exc),
                )
                await on_update(_run_updated(failed_run))
            raise

    async def cancel(self, *, params: JsonObject) -> None:
        scope_id = scope_id_from_params(params)
        session_id = session_id_from_params(params)
        self._store.load_session(scope_id=scope_id, session_id=session_id)
        active_prompt = self._active_prompts.get(session_id)
        if active_prompt is None:
            return
        active_prompt.cancel_requested.set()
        await self._worker_runner.cancel(session_id=session_id)

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        return self._session_locks.setdefault(session_id, asyncio.Lock())

    def _require_idle_session(self, session_id: str, message: str = "Session already has an active prompt.") -> None:
        if session_id in self._active_prompts:
            raise SessionBusyError(message)

    def _start_prompt_run(self, session_id: str) -> ActivePrompt:
        now = _now_iso()
        run = PromptRunRecord(
            run_id=_new_run_id(),
            session_id=session_id,
            status="running",
            started_at=now,
            updated_at=now,
        )
        active_prompt = ActivePrompt(run=run, cancel_requested=asyncio.Event())
        self._active_prompts[session_id] = active_prompt
        return active_prompt

    def _finish_prompt_run(
        self,
        active_prompt: ActivePrompt,
        *,
        status: str,
        stop_reason: str | None = None,
        error: str | None = None,
    ) -> PromptRunRecord:
        run = active_prompt.run
        now = _now_iso()
        finished_run = replace(
            run,
            status=status,
            updated_at=now,
            ended_at=now,
            stop_reason=stop_reason,
            error=error,
        )
        if self._active_prompts.get(run.session_id) is active_prompt:
            self._active_prompts.pop(run.session_id)
        return finished_run

    def _initial_messages(self, *, skills_root: Path) -> list[BaseMessage]:
        instructions = build_worker_instructions(
            config=WorkerAgentConfig(
                working_directory=self._worker_workspace,
                skills_directories=(str(skills_root),),
                user_instructions=self._user_instructions,
            ),
        )
        return [build_initial_system_message(instructions=instructions)]

    async def _available_models(self) -> list[ProviderModel]:
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

        self._model_cache = (now, provider_models)
        return provider_models
