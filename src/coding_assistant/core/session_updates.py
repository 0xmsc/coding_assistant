from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coding_assistant.llm.types import (
    BaseMessage,
)

JsonObject = dict[str, Any]


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

    @classmethod
    def from_json(cls, value: JsonObject) -> SessionAttachment | None:
        attachment_id = value.get("id")
        name = value.get("name")
        mime_type = value.get("mimeType")
        size = value.get("size")
        path = value.get("path")
        sha256 = value.get("sha256")
        if not (
            isinstance(attachment_id, str)
            and isinstance(name, str)
            and isinstance(mime_type, str)
            and isinstance(size, int)
            and isinstance(path, str)
            and isinstance(sha256, str)
        ):
            return None
        return cls(
            attachment_id=attachment_id,
            name=name,
            mime_type=mime_type,
            size=size,
            path=path,
            sha256=sha256,
        )


@dataclass(frozen=True)
class HistoryResetUpdate:
    pass


@dataclass(frozen=True)
class HistoryCompleteUpdate:
    version: int


@dataclass(frozen=True)
class MessageAddedUpdate:
    message_id: str
    message: BaseMessage
    created_at: str | None = None


@dataclass(frozen=True)
class MessageDeltaUpdate:
    message_id: str
    append_text: str


@dataclass(frozen=True)
class AttachmentAddedUpdate:
    attachment: SessionAttachment
    created_at: str | None = None


@dataclass(frozen=True)
class SessionUpdatedUpdate:
    session: JsonObject


@dataclass(frozen=True)
class RunUpdatedUpdate:
    run: JsonObject


SessionUpdate = (
    HistoryResetUpdate
    | HistoryCompleteUpdate
    | MessageAddedUpdate
    | MessageDeltaUpdate
    | AttachmentAddedUpdate
    | SessionUpdatedUpdate
    | RunUpdatedUpdate
)


def content_text(content: str | list[JsonObject] | None) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    for block in content:
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
    if not text_parts:
        return None
    return "\n".join(text_parts)
