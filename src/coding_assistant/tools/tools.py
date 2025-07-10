from mcp import StdioServerParameters
from smolagents import Tool, ToolCollection

from coding_assistant.config import Config


def get_file_tools(config: Config) -> list[Tool]:
    assert config.working_directory.exists()

    tc = ToolCollection.from_mcp(
        StdioServerParameters(
            command="docker",
            args=[
                "run",
                "-it",
                "--rm",
                "--mount",
                f"type=bind,src={config},dst=/project",
            ],
        )
    )

    return tc.tools
