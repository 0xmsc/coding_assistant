import json
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage, AIMessage
from rich.panel import Panel
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
import logging

from pprint import pformat

from coding_assistant.config import get_global_config

console = Console()
logger = logging.getLogger(__name__)


def print_to_console(content, do_print):
    if do_print:
        console.print(content)


def trace_to_file(data, tracing_file):
    if tracing_file:
        tracing_file.write(json.dumps(data) + "\n")


def print_agent_progress(chunk, name, do_print=False):
    tracing_file = get_global_config().tracing_file

    if "tools" in chunk and "messages" in chunk["tools"]:
        for message in chunk["tools"]["messages"]:
            assert isinstance(message, ToolMessage)
            print_to_console(
                Panel(
                    message.content,
                    title=f"Tool ({message.name})",
                    border_style="yellow",
                ),
                do_print,
            )

            trace_to_file(
                {
                    "type": "tool",
                    "name": message.name,
                    "content": message.content,
                },
                tracing_file,
            )
    elif "agent" in chunk and "messages" in chunk["agent"]:
        for message in chunk["agent"]["messages"]:
            assert isinstance(message, AIMessage)
            title = f"Agent progress: {name}"
            if message.content:
                print_to_console(
                    Panel(Markdown(message.content), title=title, border_style="red"),
                    print_to_console,
                )

                trace_to_file(
                    {
                        "type": "agent_output",
                        "agent": name,
                        "content": message.content,
                    },
                    tracing_file,
                )

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    print_to_console(
                        Panel(
                            f"{tool_call.get('name')}\n{pformat(tool_call.get('args'))}\n",
                            title=title,
                            border_style="red",
                        ),
                        print_to_console,
                    )

                    trace_to_file(
                        {
                            "type": "agent_tool_call",
                            "agent": name,
                            "tool": tool_call.get("name"),
                            "content": message.content,
                        },
                        tracing_file,
                    )
    else:
        logger.warning(f"Unhandled chunk: {pformat(chunk)}")
