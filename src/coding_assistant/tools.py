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

    # TODO: Populate the tool lists properly based on requirements
    # For now, they remain empty lists initialized by the dataclass factory.

    return tools
