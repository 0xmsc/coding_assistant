from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage, AIMessage
from rich.panel import Panel
from rich.console import Console
from rich.text import Text
import logging

from beeprint import pp

console = Console()
logger = logging.getLogger(__name__)


def print_agent_progress(chunk):
    # logger.debug(f"Chunk: {pp(chunk, output=False)}")
    if "tools" in chunk and "messages" in chunk["tools"]:
        for message in chunk["tools"]["messages"]:
            assert isinstance(message, ToolMessage)
            console.print(Panel(f"{message.name}\n\n{message.content}", title="Tool", border_style="yellow"))
    elif "agent" in chunk and "messages" in chunk["agent"]:
        for message in chunk["agent"]["messages"]:
            assert isinstance(message, AIMessage)
            content = ""
            if message.content:
                content += f"{message.content}\n\n"
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    content += f"{tool_call.get('name')}\n{pp(tool_call.get('args'), output=False)}\n"
            content = content.strip()
            console.print(Panel(f"{content}", title="Agent", border_style="red"))
    else:
        logger.warning(f"Unhandled chunk: {pp(chunk, output=False)}")
