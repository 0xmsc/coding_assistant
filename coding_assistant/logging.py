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


def print_agent_progress(chunk, name=None):
    if "tools" in chunk and "messages" in chunk["tools"]:
        for message in chunk["tools"]["messages"]:
            assert isinstance(message, ToolMessage)
            content = f"{message.name}\n\n{message.content}"
            console.print(Panel(Markdown(content), title="Tool", border_style="yellow"))
    elif "agent" in chunk and "messages" in chunk["agent"]:
        for message in chunk["agent"]["messages"]:
            assert isinstance(message, AIMessage)
            content = ""
            if message.content:
                content += f"{message.content}\n\n"
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    pretty_args = pformat(tool_call.get("args"))
                    content += f"{tool_call.get('name')}\n{pretty_args}\n"
            content = content.strip()
            title = f"Agent: {name}" if name else "Agent"
            console.print(Panel(Markdown(content), title=title, border_style="red"))
    else:
        logger.warning(f"Unhandled chunk: {pformat(chunk)}")
