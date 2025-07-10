from dataclasses import dataclass

from mcp import StdioServerParameters
from smolagents import Tool, ToolCollection

from coding_assistant.config import Config


@dataclass
class Tools:
    file_tools: list[Tool]


def get_file_tool_collection(config: Config) -> ToolCollection:
    assert config.working_directory.exists()

    return ToolCollection.from_mcp(
        StdioServerParameters(
            command="docker",
            args=[
                "run",
                "-i",
                "--rm",
                f"--mount=type=bind,src={config.working_directory},dst=/project",
                "mcp/filesystem",
                "/project",
            ],
        ),
        trust_remote_code=True,
    )
