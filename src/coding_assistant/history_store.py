from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from coding_assistant.llm.types import AssistantMessage, BaseMessage, message_from_dict, message_to_dict


def sanitize_history(history: Sequence[BaseMessage]) -> list[BaseMessage]:
    if not history:
        return []

    fixed_history = list(history)
    while fixed_history:
        last_message = fixed_history[-1]
        has_tool_calls = isinstance(last_message, AssistantMessage) and bool(last_message.tool_calls)
        if not has_tool_calls:
            break
        fixed_history.pop()
    return fixed_history


class HistoryStore(Protocol):
    def load(self) -> list[BaseMessage] | None: ...

    def save(self, history: Sequence[BaseMessage]) -> None: ...


def get_project_cache_dir(working_directory: Path) -> Path:
    cache_dir = working_directory / ".coding_assistant"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_history_path(working_directory: Path) -> Path:
    return get_project_cache_dir(working_directory) / "history.json"


class FileHistoryStore:
    def __init__(self, working_directory: Path, *, path: Path | None = None) -> None:
        self._path = path or get_history_path(working_directory)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[BaseMessage] | None:
        if not self._path.exists():
            return None
        data = json.loads(self._path.read_text())
        return [message_from_dict(item) for item in data]

    def save(self, history: Sequence[BaseMessage]) -> None:
        payload = [message_to_dict(item) for item in sanitize_history(history)]
        self._path.write_text(json.dumps(payload, indent=2))
