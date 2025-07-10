from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Import Tool
from agents import Tool
from agents.mcp import MCPServer, MCPServerStdio

from coding_assistant.config import Config


@dataclass
class Tools:
    mcp_servers: list[MCPServer] = field(default_factory=list)
    file_tools: list[Tool] = field(default_factory=list)
    read_only_file_tools: list[Tool] = field(default_factory=list)
    notebook_tools: list[Tool] = field(default_factory=list)
    sequential_thinking_tools: list[Tool] = field(default_factory=list)


def get_tools(config: Config) -> Tools:
    assert config.working_directory.exists()

    tools = Tools()

    tools.mcp_servers.append(
        MCPServerStdio(
            params={
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    str(config.working_directory),
                ],
            }
        )
    )

    for mcp in tools.mcp_servers:
        mcp.connect()

    return tools
