from abc import ABC, abstractmethod
from typing import Any, Optional
from coding_assistant.llm.types import ToolResult


class ToolCallbacks(ABC):
    @abstractmethod
    async def before_tool_execution(
        self,
        context_name: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        ui: Any,
    ) -> Optional[ToolResult]:
        pass


class NullToolCallbacks(ToolCallbacks):
    async def before_tool_execution(
        self,
        context_name: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        ui: Any,
    ) -> Optional[ToolResult]:
        return None
