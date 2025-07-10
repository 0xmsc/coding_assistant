from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage, AIMessage
from rich.panel import Panel
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
import logging

from pprint import pformat

console = Console()
logger = logging.getLogger(__name__)


def print_agent_progress(chunk, name, print_to_console=False):
    if "tools" in chunk and "messages" in chunk["tools"]:
        for message in chunk["tools"]["messages"]:
            assert isinstance(message, ToolMessage)
            console.print(Panel(message.content, title=f"Tool ({message.name})", border_style="yellow"))
    elif "agent" in chunk and "messages" in chunk["agent"]:
        for message in chunk["agent"]["messages"]:
            assert isinstance(message, AIMessage)
            if message.content:
                console.print(Panel(Markdown(message.content), title=f"{name} progress", border_style="red"))
            for tc in message.tool_calls if message.tool_calls else []:
                pretty_args = pformat(tc.get("args"))
                content = f"{pretty_args}"
                console.print(Panel(content, title=f"{name} TC: {tc.get('name')}", border_style="red"))
    else:
        logger.warning(f"Unhandled chunk: {pformat(chunk)}")
