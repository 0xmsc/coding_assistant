from __future__ import annotations

from copy import deepcopy
from typing import Any

from coding_assistant.llm.types import ToolDefinition


class ToolSpec(ToolDefinition):
    def __init__(self, *, name: str, description: str, parameters: dict[str, Any]) -> None:
        self._name = name
        self._description = description
        self._parameters = deepcopy(parameters)

    @classmethod
    def from_definition(cls, tool: ToolDefinition) -> "ToolSpec":
        return cls(
            name=tool.name(),
            description=tool.description(),
            parameters=tool.parameters(),
        )

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def parameters(self) -> dict[str, Any]:
        return deepcopy(self._parameters)
