"""Callback interfaces for agent interactions."""

import json
import textwrap
from abc import ABC, abstractmethod
from pprint import pformat

from rich import print
from rich.console import Group
from rich.json import JSON
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text


class AgentCallbacks(ABC):
    """Abstract interface for agent callbacks."""

    @abstractmethod
    def on_agent_start(self, agent_name: str, model: str, is_resuming: bool = False):
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
    def on_assistant_reasoning(self, agent_name: str, content: str):
        """Handle reasoning content from assistant."""
        pass

    @abstractmethod
    def on_tool_message(self, agent_name: str, tool_name: str, arguments: dict, result: str):
        """Handle messages with role: tool."""
        pass

    @abstractmethod
    def on_chunk(self, chunk: str):
        """Handle LLM chunks."""
        pass

    @abstractmethod
    def on_chunks_end(self):
        """Handle end of LLM chunks."""
        pass


class NullCallbacks(AgentCallbacks):
    """Null object implementation that does nothing."""

    def on_agent_start(self, agent_name: str, model: str, is_resuming: bool = False):
        pass

    def on_agent_end(self, agent_name: str, result: str, summary: str):
        pass

    def on_user_message(self, agent_name: str, content: str):
        pass

    def on_assistant_message(self, agent_name: str, content: str):
        pass

    def on_assistant_reasoning(self, agent_name: str, content: str):
        pass

    def on_tool_message(self, agent_name: str, tool_name: str, arguments: dict, result: str):
        pass

    def on_chunk(self, chunk: str):
        pass

    def on_chunks_end(self):
        pass


class RichCallbacks(AgentCallbacks):
    def __init__(self, print_chunks: bool = True, print_reasoning: bool = True):
        self._print_chunks = print_chunks
        self._print_reasoning = print_reasoning

    def on_agent_start(self, agent_name: str, model: str, is_resuming: bool = False):
        status = "resuming" if is_resuming else "starting"
        print(
            Panel(
                "",
                title=f"Agent {agent_name} ({model}) {status}",
                border_style="red",
            ),
        )

    def on_agent_end(self, agent_name: str, result: str, summary: str):
        quoted_result = textwrap.indent(result, "> ", lambda _: True)
        quoted_summary = textwrap.indent(summary, "> ", lambda _: True)
        print(
            Panel(
                Markdown(f"Result\n\n{quoted_result}\n\nSummary\n\n{quoted_summary}"),
                title=f"Agent {agent_name} result",
                border_style="red",
            ),
        )

    def on_user_message(self, agent_name: str, content: str):
        print(
            Panel(
                Markdown(content),
                title=f"Agent {agent_name} user",
                border_style="blue",
            ),
        )

    def on_assistant_message(self, agent_name: str, content: str):
        print(
            Panel(
                Markdown(content),
                title=f"Agent {agent_name} assistant",
                border_style="green",
            ),
        )

    def on_assistant_reasoning(self, agent_name: str, content: str):
        if self._print_reasoning:
            print(
                Panel(
                    Markdown(content),
                    title=f"Agent {agent_name} reasoning",
                    border_style="cyan",
                ),
            )

    def _try_parse_json(self, content: str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def _format_tool_result(self, result: str, tool_name: str):
        if data := self._try_parse_json(result):
            if tool_name == "execute_shell_command":
                result = f"RC: {data['returncode']}"

                if data["stderr"]:
                    result += f"\n\nstderr\n\n```\n{data['stderr']}\n```"

                if data["stdout"]:
                    result += f"\n\nstdout\n\n```\n{data['stdout']}\n```"

                return Markdown(result)

            return Pretty(data, expand_all=True, indent_size=2)

        else:
            return Markdown(f"```\n{result}\n```")

    def on_tool_message(self, agent_name: str, tool_name: str, arguments: dict, result: str):
        render_group = Group(
            Markdown(f"Name: `{tool_name}`"),
            Padding(Pretty(arguments, expand_all=True, indent_size=2), (1, 0, 0, 0)),
            Padding(self._format_tool_result(result, tool_name), (1, 0, 0, 0)),
        )
        print(
            Panel(
                render_group,
                title=f"Agent {agent_name} tool call",
                border_style="yellow",
            ),
        )

    def on_chunk(self, chunk: str):
        if self._print_chunks:
            print(chunk, end="", flush=True)

    def on_chunks_end(self):
        if self._print_chunks:
            print()
