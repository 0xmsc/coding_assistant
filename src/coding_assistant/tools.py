from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from agents import Tool
from agents.mcp import MCPServer, MCPServerStdio

from coding_assistant.config import Config


@dataclass
class Tools:
    mcp_servers: list[MCPServer] = field(default_factory=list)


def get_filesystem_server(config: Config) -> MCPServer:
    assert config.working_directory.exists()

    return MCPServerStdio(
        params={
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(config.working_directory),
            ],
        }
    )
