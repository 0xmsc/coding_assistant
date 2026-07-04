from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from coding_assistant.llm.types import TextToolResult, Tool


@dataclass
class SessionTitleState:
    title: str | None = None

    def finish_metadata(self) -> dict[str, Any] | None:
        if self.title is None:
            return None
        return {"title": self.title}


class SetSessionTitleInput(BaseModel):
    title: str = Field(description="Short descriptive title for the current session.")


class SetSessionTitleTool(Tool):
    """Set the title that the worker returns for the current manager session."""

    def __init__(self, *, state: SessionTitleState) -> None:
        self._state = state

    def name(self) -> str:
        return "set_session_title"

    def description(self) -> str:
        return "Set the short descriptive title for the current session."

    def parameters(self) -> dict[str, Any]:
        return SetSessionTitleInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        validated = SetSessionTitleInput.model_validate(parameters)
        title = validated.title.strip()
        if not title:
            raise ValueError("Title must not be empty.")
        self._state.title = title
        return TextToolResult(content=f"Session title set to: {title}")


def create_session_title_tools(*, state: SessionTitleState) -> list[Tool]:
    return [SetSessionTitleTool(state=state)]
