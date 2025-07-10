import subprocess
from typing import Any, Dict, Optional, Type
from langchain_core.tools import tool
from pathlib import Path
from pydantic import BaseModel, Field
from coding_assistant.config import get_global_config
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import BaseTool, ToolException

from coding_assistant.config import get_global_config

FILE_TYPE_WHITELIST = ["py", "cpp"]


@tool
def ripgrep(pattern: str, case_insensitive: bool = False, max_output_lines=100) -> str:
    """
    A tool for searching files using the ripgrep command-line utility.
    """
    cmd = ["rg"]

    if case_insensitive:
        cmd.append("-i")

    for extension in FILE_TYPE_WHITELIST:
        cmd.append(f"-t{extension}")

    cmd.append(pattern)

    result = subprocess.check_output(cmd, text=True)

    if len(result.splitlines()) > max_output_lines:
        return f"Result was more than {max_output_lines} lines"

    return result


@tool
def fdfind(pattern: str, file_type: Optional[str] = None, max_output_lines: int = 100) -> str:
    """
    A tool for searching files using the fd command-line utility.
    """
    cmd = ["fd"]

    if file_type:
        cmd.extend(["--extension", file_type])

    cmd.append(pattern)

    result = subprocess.check_output(cmd, text=True)

    if len(result.splitlines()) > max_output_lines:
        return f"Result was more than {max_output_lines} lines"

    return result


def read_only_file_tools():
    working_directory = get_global_config().working_directory
    tools = []
    tools.extend(
        FileManagementToolkit(
            root_dir=str(working_directory),
            selected_tools=["read_file", "list_directory"],
        ).get_tools()
    )
    tools.append(ripgrep)
    tools.append(fdfind)
    return tools


def all_file_tools():
    working_directory = get_global_config().working_directory
    tools = []
    tools.extend(read_only_file_tools())
    tools.extend(
        FileManagementToolkit(
            root_dir=str(working_directory),
            selected_tools=["write_file", "copy_file", "file_delete", "move_file"],
        ).get_tools()
    )
    return tools
