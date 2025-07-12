"""Callback interfaces for agent interactions."""

from abc import ABC, abstractmethod

from rich import print
from rich.panel import Panel
from rich.pretty import Pretty


class AgentCallbacks(ABC):
    """Abstract interface for agent callbacks."""
    
    @abstractmethod
    def add_agent_start(self, agent_name: str, model: str, start_message: str, is_resuming: bool = False):
        """Add an agent start message."""
        pass
    
    @abstractmethod
    def add_tool_call(self, agent_name: str, tool_call):
        """Add a tool call message."""
        pass
    
    @abstractmethod
    def add_tool_result(self, tool_name: str, result: str):
        """Add a tool result message."""
        pass
    
    @abstractmethod
    def add_agent_response(self, agent_name: str, content: str):
        """Add an agent response."""
        pass
    
    @abstractmethod
    def add_agent_result(self, agent_name: str, result: str, summary: str):
        """Add an agent result."""
        pass
    
    @abstractmethod
    def add_agent_feedback(self, agent_name: str, feedback: str):
        """Add agent feedback."""
        pass


class NullCallbacks(AgentCallbacks):
    """Null object implementation that does nothing."""
    
    def add_agent_start(self, agent_name: str, model: str, start_message: str, is_resuming: bool = False):
        pass
    
    def add_tool_call(self, agent_name: str, tool_call):
        pass
    
    def add_tool_result(self, tool_name: str, result: str):
        pass
    
    def add_agent_response(self, agent_name: str, content: str):
        pass
    
    def add_agent_result(self, agent_name: str, result: str, summary: str):
        pass
    
    def add_agent_feedback(self, agent_name: str, feedback: str):
        pass


class RichCallbacks(AgentCallbacks):
    """Rich implementation that displays agent progress using Rich panels (like master branch)."""
    
    def add_agent_start(self, agent_name: str, model: str, start_message: str, is_resuming: bool = False):
        """Add an agent start message."""
        status = "resuming" if is_resuming else "starting"
        print(
            Panel(
                start_message,
                title=f"Agent {agent_name} ({model}) {status}",
                border_style="red",
            ),
        )
    
    def add_tool_call(self, agent_name: str, tool_call):
        """Add a tool call message."""
        print(
            Panel(
                Pretty(tool_call.function),
                title=f"Agent {agent_name} tool call",
                border_style="green",
            ),
        )
    
    def add_tool_result(self, tool_name: str, result: str):
        """Add a tool result message."""
        print(
            Panel(
                result,
                title=f"Tool {tool_name} result",
                border_style="yellow",
            ),
        )
    
    def add_agent_response(self, agent_name: str, content: str):
        """Add an agent response."""
        print(
            Panel(
                content,
                title=f"Agent {agent_name} response",
                border_style="green",
            ),
        )
    
    def add_agent_result(self, agent_name: str, result: str, summary: str):
        """Add an agent result."""
        print(
            Panel(
                f"Result: {result}\n\nSummary: {summary}",
                title=f"Agent {agent_name} result",
                border_style="red",
            ),
        )
    
    def add_agent_feedback(self, agent_name: str, feedback: str):
        """Add agent feedback."""
        print(
            Panel(
                feedback,
                title=f"Agent {agent_name} feedback",
                border_style="red",
            ),
        )
