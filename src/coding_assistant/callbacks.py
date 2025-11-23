from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any, Optional

from rich import print
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty

from coding_assistant.agents.callbacks import AgentProgressCallbacks, AgentToolCallbacks
from coding_assistant.agents.types import TextResult, ToolResult

logger = logging.getLogger(__name__)


async def confirm_tool_if_needed(*, tool_name: str, arguments: dict, patterns: list[str], ui) -> Optional[TextResult]:
    for pat in patterns:
        if re.search(pat, tool_name):
            question = f"Execute tool `{tool_name}` with arguments `{arguments}`?"
            allowed = await ui.confirm(question)
            if not allowed:
                return TextResult(content="Tool execution denied.")
            break
    return None


async def confirm_shell_if_needed(*, tool_name: str, arguments: dict, patterns: list[str], ui) -> Optional[TextResult]:
    if tool_name != "mcp_coding_assistant_mcp_shell_execute":
        return None

    command = arguments.get("command")
    if not isinstance(command, str):
        return None

    for pat in patterns:
        if re.search(pat, command):
            question = f"Execute shell command `{command}` for tool `{tool_name}`?"
            allowed = await ui.confirm(question)
            if not allowed:
                return TextResult(content="Shell command execution denied.")
            break
    return None


class RichAgentProgressCallbacks(AgentProgressCallbacks):
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
        Console().bell()

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

    def _format_tool_result(self, tool_name: str, result: str):
        if data := self._try_parse_json(result):
            return Pretty(data, expand_all=True, indent_size=2)
        # TODO: Avoid hard-coding tool-name prefixes to decide how to render tool results.
        elif tool_name.startswith("mcp_coding_assistant_mcp_todo_"):
            return Markdown(result)
        else:
            return Markdown(f"```\n{result}\n```")

    def _format_arguments(self, arguments: dict, tool_name: str):
        parts = []

        if tool_name == "mcp_coding_assistant_mcp_python_execute":
            arguments_without_code = {k: v for k, v in arguments.items() if k != "code"}

            if arguments_without_code:
                parts.append(Padding(Pretty(arguments_without_code, expand_all=True, indent_size=2), (1, 0, 0, 0)))

            code = arguments["code"]
            parts.append(Padding(Markdown(f"```python\n{code}\n```"), (1, 0, 0, 0)))

        elif tool_name == "mcp_coding_assistant_mcp_shell_execute":
            arguments_without_command = {k: v for k, v in arguments.items() if k != "command"}

            if arguments_without_command:
                parts.append(Padding(Pretty(arguments_without_command, expand_all=True, indent_size=2), (1, 0, 0, 0)))

            command = arguments["command"]
            parts.append(Padding(Markdown(f"```bash\n{command}\n```"), (1, 0, 0, 0)))

        else:
            parts.append(Padding(Pretty(arguments, expand_all=True, indent_size=2), (1, 0, 0, 0)))

        return Group(*parts)

    def on_tool_message(self, agent_name: str, tool_call_id: str, tool_name: str, arguments: dict | None, result: str):
        parts: list[Any] = [Markdown(f"Name: `{tool_name}`")]

        if arguments is not None:
            parts.append(self._format_arguments(arguments, tool_name))

        parts.append(Padding(self._format_tool_result(tool_name, result), (1, 0, 0, 0)))

        render_group = Group(*parts)
        print(
            Panel(
                render_group,
                title=f"Agent {agent_name} tool call",
                border_style="yellow",
            ),
        )

    def on_tool_start(self, agent_name: str, tool_call_id: str, tool_name: str, arguments: dict | None):
        pass  # Default implementation does nothing

    def on_chunk(self, chunk: str):
        if self._print_chunks:
            print(chunk, end="", flush=True)

    def on_chunks_end(self):
        if self._print_chunks:
            print()


class DenseProgressCallbacks(AgentProgressCallbacks):
    """Dense progress callbacks with minimal formatting."""

    def __init__(self):
        self._last_tool_info: tuple[str, str, str] | None = None  # (tool_call_id, tool_name, args_str)
        self._chunk_buffer = ""
        self._console = Console()
        self._live: Live | None = None

    def on_agent_start(self, agent_name: str, model: str, is_resuming: bool = False):
        status = "resuming" if is_resuming else "starting"
        print()
        print(f"[bold red]▸[/bold red] Agent {agent_name} ({model}) {status}")
        self._last_tool_info = None

    def on_agent_end(self, agent_name: str, result: str, summary: str):
        print()
        print(f"[bold red]◂[/bold red] Agent {agent_name} complete")
        print(f"[dim]Summary: {summary}[/dim]")
        self._last_tool_info = None

    def on_user_message(self, agent_name: str, content: str):
        print()
        print(f"[bold blue]◉[/bold blue] User: {content}")
        self._last_tool_info = None

    def on_assistant_message(self, agent_name: str, content: str):
        # Don't print - content is already printed via chunks
        pass

    def on_assistant_reasoning(self, agent_name: str, content: str):
        print()
        print(f"[dim cyan]💭 {content}[/dim cyan]")
        self._last_tool_info = None

    def _count_lines(self, text: str) -> int:
        """Count number of lines in text."""
        return len(text.splitlines())

    def _format_arguments(self, arguments: dict) -> str:
        """Format arguments with each on a separate line."""
        # For large arguments, show count instead of full content
        formatted = {}
        for key, value in arguments.items():
            if isinstance(value, str) and len(value) > 100:
                formatted[key] = f"<{len(value)} chars>"
            else:
                formatted[key] = value

        # Format with each argument on a separate line
        if not formatted:
            return ""

        lines = []
        for key, value in formatted.items():
            value_json = json.dumps(value)
            lines.append(f"\n    {key}={value_json}")

        return "".join(lines)

    def on_tool_start(self, agent_name: str, tool_call_id: str, tool_name: str, arguments: dict | None):
        print()
        # Print tool name and arguments with call ID
        args_str = self._format_arguments(arguments) if arguments else ""
        if args_str:
            print(f"[bold yellow]▸[/bold yellow] {tool_name} [dim]({tool_call_id})[/dim]{args_str}")
        else:
            print(f"[bold yellow]▸[/bold yellow] {tool_name} [dim]({tool_call_id})[/dim]")

        # Remember what we printed
        self._last_tool_info = (tool_call_id, tool_name, args_str)

    def on_tool_message(self, agent_name: str, tool_call_id: str, tool_name: str, arguments: dict | None, result: str):
        # If we printed something between start and end, reprint the tool info
        if self._last_tool_info is None:
            # Something was printed, need to show tool info again
            print()
            args_str = self._format_arguments(arguments) if arguments else ""
            if args_str:
                print(f"[bold yellow]▸[/bold yellow] {tool_name} [dim]({tool_call_id})[/dim]{args_str}")
            else:
                print(f"[bold yellow]▸[/bold yellow] {tool_name} [dim]({tool_call_id})[/dim]")

        # Print result summary (line count only, no call ID if nothing printed between)
        line_count = self._count_lines(result)
        if line_count > 0:
            if self._last_tool_info is None:
                print(f" [dim]({tool_call_id}) → {line_count} lines[/dim]")
            else:
                print(f" [dim]→ {line_count} lines[/dim]")
        elif self._last_tool_info is None:
            print(f" [dim]({tool_call_id})[/dim]")

        # Reset state
        self._last_tool_info = None

    def on_chunk(self, chunk: str):
        # Start live display on first non-empty chunk
        if not self._live and chunk:
            print()
            print("[bold green]◉[/bold green] ", end="")
            self._last_tool_info = None
            self._live = Live("", console=self._console, refresh_per_second=10, auto_refresh=True)
            self._live.start()

        # Buffer and render markdown in real-time
        self._chunk_buffer += chunk
        if self._live:
            self._live.update(Markdown(self._chunk_buffer))

    def on_chunks_end(self):
        if self._live:
            self._live.stop()
            self._live = None
        print()  # Newline after chunks


class ConfirmationToolCallbacks(AgentToolCallbacks):
    def __init__(
        self,
        *,
        tool_confirmation_patterns: list[str] | None = None,
        shell_confirmation_patterns: list[str] | None = None,
    ):
        self._tool_patterns = tool_confirmation_patterns or []
        self._shell_patterns = shell_confirmation_patterns or []

    async def before_tool_execution(
        self,
        agent_name: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict,
        *,
        ui,
    ) -> Optional[ToolResult]:
        if result := await confirm_tool_if_needed(
            tool_name=tool_name,
            arguments=arguments,
            patterns=self._tool_patterns,
            ui=ui,
        ):
            return result

        if result := await confirm_shell_if_needed(
            tool_name=tool_name,
            arguments=arguments,
            patterns=self._shell_patterns,
            ui=ui,
        ):
            return result

        return None
