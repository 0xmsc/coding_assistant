from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from coding_assistant.llm.types import Tool

ValidationModelT = TypeVar("ValidationModelT", bound=BaseModel)


class StructuredTool(Tool, Generic[ValidationModelT]):
    """Tool wrapper backed by a Pydantic schema and one async handler."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        schema_model: type[ValidationModelT],
        handler: Callable[[ValidationModelT], Awaitable[str]],
    ) -> None:
        self._name = name
        self._description = description
        self._schema_model = schema_model
        self._handler = handler

    def name(self) -> str:
        """Return the stable tool name exposed to the model."""
        return self._name

    def description(self) -> str:
        """Describe when the tool should be used."""
        return self._description

    def parameters(self) -> dict[str, Any]:
        """Return the JSON schema for the tool arguments."""
        return self._schema_model.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        """Validate the arguments and run the handler."""
        validated = self._schema_model.model_validate(parameters)
        return await self._handler(validated)
