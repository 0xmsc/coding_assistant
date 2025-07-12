"""Callback interfaces for agent interactions."""

from abc import ABC, abstractmethod
from pprint import pformat
import json

from rich import print
from rich.panel import Panel
from rich.pretty import Pretty


class AgentCallbacks(ABC):
    """Abstract interface for agent callbacks."""

    @abstractmethod
    def on_agent_start(self, agent_name: str, model: str, start_message: str, is_resuming: bool = False):
        """Handle agent start events."""
        pass

    @abstractmethod
    def on_agent_end(self, agent_name: str, result: str, summary: str):
        """Handle agent end events."""
        pass

    @abstractmethod
    def on_user_message(self, agent_name: str, content: str):
        """Handle messages with role: user."""
        pass

    @abstractmethod
    def on_assistant_message(self, agent_name: str, content: str):
        """Handle messages with role: assistant."""
        pass

    @abstractmethod
    def on_tool_message(self, agent_name: str, tool_name: str, arguments: dict, result: str):
        """Handle messages with role: tool."""
        pass


class NullCallbacks(AgentCallbacks):
    """Null object implementation that does nothing."""

    def on_agent_start(self, agent_name: str, model: str, start_message: str, is_resuming: bool = False):
        pass

    def on_agent_end(self, agent_name: str, result: str, summary: str):
        pass

    def on_user_message(self, agent_name: str, content: str):
        pass

    def on_assistant_message(self, agent_name: str, content: str):
        pass

    def on_tool_message(self, agent_name: str, tool_name: str, arguments: dict, result: str):
        pass


class RichCallbacks(AgentCallbacks):
    def on_agent_start(self, agent_name: str, model: str, start_message: str, is_resuming: bool = False):
        status = "resuming" if is_resuming else "starting"
        print(
            Panel(
                start_message,
                title=f"Agent {agent_name} ({model}) {status}",
                border_style="red",
            ),
        )

    def on_agent_end(self, agent_name: str, result: str, summary: str):
        print(
            Panel(
                f"Result: {result}\n\nSummary: {summary}",
                title=f"Agent {agent_name} result",
                border_style="red",
            ),
        )

    def on_user_message(self, agent_name: str, content: str):
        print(
            Panel(
                content,
                title=f"Agent {agent_name} user",
                border_style="blue",
            ),
        )

    def on_assistant_message(self, agent_name: str, content: str):
        print(
            Panel(
                content,
                title=f"Agent {agent_name} assistant",
                border_style="green",
            ),
        )

    def on_tool_message(self, agent_name: str, tool_name: str, arguments: dict, result: str):
        message = f"Name: {tool_name}\n\nArguments: {json.dumps(arguments, indent=2)}\n\nResult:\n\n{result}"
        print(
            Panel(
                message,
                title=f"Agent {agent_name} TC",
                border_style="yellow",
            ),
        )
