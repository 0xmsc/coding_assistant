from abc import ABC, abstractmethod
from typing import Optional
from coding_assistant.framework.results import ToolResult


class ProgressCallbacks(ABC):
    """Abstract interface for agent callbacks."""

    @abstractmethod
    def on_user_message(self, context_name: str, content: str, force: bool = False):
        """Handle messages with role: user."""
        pass

    @abstractmethod
    def on_assistant_message(self, context_name: str, content: str, force: bool = False):
        """Handle messages with role: assistant."""
        pass

    @abstractmethod
    def on_assistant_reasoning(self, context_name: str, content: str):
        """Handle reasoning content from assistant."""
        pass

    @abstractmethod
    def on_tool_start(self, context_name: str, tool_call_id: str, tool_name: str, arguments: dict):
        """Handle tool start events."""
        pass

    @abstractmethod
    def on_tool_message(self, context_name: str, tool_call_id: str, tool_name: str, arguments: dict, result: str):
        """Handle messages with role: tool."""
        pass

    @abstractmethod
    def on_content_chunk(self, chunk: str):
        """Handle LLM content chunks."""
        pass

    @abstractmethod
    def on_reasoning_chunk(self, chunk: str):
        """Handle LLM reasoning chunks."""
        pass

    @abstractmethod
    def on_chunks_end(self):
        """Handle end of LLM chunks."""
        pass


class NullProgressCallbacks(ProgressCallbacks):
    """Null object implementation that does nothing."""

    def on_user_message(self, context_name: str, content: str, force: bool = False):
        pass

    def on_assistant_message(self, context_name: str, content: str, force: bool = False):
        pass

    def on_assistant_reasoning(self, context_name: str, content: str):
        pass

    def on_tool_start(self, context_name: str, tool_call_id: str, tool_name: str, arguments: dict):
        pass

    def on_tool_message(self, context_name: str, tool_call_id: str, tool_name: str, arguments: dict, result: str):
        pass

    def on_content_chunk(self, chunk: str):
        pass

    def on_reasoning_chunk(self, chunk: str):
        pass

    def on_chunks_end(self):
        pass


class ToolCallbacks(ABC):
    @abstractmethod
    async def before_tool_execution(
        self,
        context_name: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict,
        *,
        ui,
    ) -> Optional[ToolResult]:
        pass


class NullToolCallbacks(ToolCallbacks):
    async def before_tool_execution(
        self,
        context_name: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict,
        *,
        ui,
    ) -> Optional[ToolResult]:
        return None
