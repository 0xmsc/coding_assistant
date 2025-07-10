import subprocess
from pydantic import BaseModel, Field
from coding_assistant.config import get_global_config
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import BaseTool, ToolException
from typing import Any, Dict, Optional, Type
from pathlib import Path

FILE_TYPE_WHITELIST = ["py", "cpp"]


class RipgrepToolInput(BaseModel):
    pattern: str = Field(description="The pattern to search for in the files.")
    case_insensitive: bool = Field(default=False, description="Whether the search should be case-insensitive.")


class RipgrepTool(BaseTool):
    name: str = "ripgrep"
    description: str = "A tool for searching files using the `ripgrep` command-line utility."
    args_schema: Type[BaseModel] = RipgrepToolInput

    def _run(self, pattern: str, case_insensitive: bool = False) -> str:
        cmd = ["rg"]

        if case_insensitive:
            cmd.append("-i")

        for extension in FILE_TYPE_WHITELIST:
            cmd.append(f"-t{extension}")

        cmd.append(pattern)

        result = subprocess.check_output(cmd)
        return result.decode("utf-8")


def read_only_file_tools():
    working_directory = get_global_config().working_directory
    tools = []
    tools.extend(
        FileManagementToolkit(
            root_dir=str(working_directory),
            selected_tools=["read_file", "list_directory"],
        ).get_tools()
    )
    tools.append(RipgrepTool())
    return tools


def all_file_tools():
    working_directory = get_global_config().working_directory
    tools = []
    tools.extend(read_only_file_tools())
    tools.extend(
        FileManagementToolkit(
            root_dir=str(working_directory),
            selected_tools=["write_file"],
        ).get_tools()
    )
    return tools
