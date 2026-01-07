from abc import ABC, abstractmethod
from typing import Any, Optional
from coding_assistant.framework.results import ToolResult
from coding_assistant.llm.types import UserMessage, AssistantMessage, ToolMessage, ToolCall


class ProgressCallbacks(ABC):
    """Abstract interface for agent callbacks."""

    @abstractmethod
    def on_user_message(self, context_name: str, message: UserMessage, force: bool = False) -> None:
        """Handle messages with role: user."""
        pass

    @abstractmethod
    def on_assistant_message(self, context_name: str, message: AssistantMessage, force: bool = False) -> None:
        """Handle messages with role: assistant."""
        pass

    @abstractmethod
    def on_tool_start(self, context_name: str, tool_call: ToolCall, arguments: dict[str, Any]) -> None:
        """Handle tool start events."""
        pass

    @abstractmethod
    def on_tool_message(
        self, context_name: str, message: ToolMessage, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        """Handle messages with role: tool."""
        pass

    @abstractmethod
    def on_content_chunk(self, chunk: str) -> None:
        """Handle LLM content chunks."""
        pass

    @abstractmethod
    def on_reasoning_chunk(self, chunk: str) -> None:
        """Handle LLM reasoning chunks."""
        pass

    @abstractmethod
    def on_chunks_end(self) -> None:
        """Handle end of LLM chunks."""
        pass


class NullProgressCallbacks(ProgressCallbacks):
    """Null object implementation that does nothing."""

    def on_user_message(self, context_name: str, message: UserMessage, force: bool = False) -> None:
        pass

    def on_assistant_message(self, context_name: str, message: AssistantMessage, force: bool = False) -> None:
        pass

    def on_tool_start(self, context_name: str, tool_call: ToolCall, arguments: dict[str, Any]) -> None:
        pass

    def on_tool_message(
        self, context_name: str, message: ToolMessage, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        pass

    def on_content_chunk(self, chunk: str) -> None:
        pass

    def on_reasoning_chunk(self, chunk: str) -> None:
        pass

    def on_chunks_end(self) -> None:
        pass


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
