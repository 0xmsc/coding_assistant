import tempfile
from pathlib import Path
from typing import Any

import pytest

from coding_assistant.llm.types import Tool
from coding_assistant.tool_policy import ToolApproved, ToolDenied
from coding_assistant.tools.builtin import RedirectToolCallTool


class MockTextTool(Tool):
    def name(self) -> str:
        return "mock_text"

    def description(self) -> str:
        return "returns text"

    def parameters(self) -> dict[str, Any]:
        return {}

    async def execute(self, parameters: dict[str, Any]) -> str:
        return "pure text output"


class MockStructuredTool(Tool):
    def name(self) -> str:
        return "mock_struct"

    def description(self) -> str:
        return "returns json"

    def parameters(self) -> dict[str, Any]:
        return {}

    async def execute(self, parameters: dict[str, Any]) -> Any:
        return {"status": "ok", "value": 42}


class MockErrorTool(Tool):
    def name(self) -> str:
        return "mock_error"

    def description(self) -> str:
        return "raises exception"

    def parameters(self) -> dict[str, Any]:
        return {}

    async def execute(self, parameters: dict[str, Any]) -> str:
        raise ValueError("Something went wrong")


async def execute_tool(tool_name: str, tool_args: dict[str, Any], tools: list[Tool]) -> ToolApproved | ToolDenied:
    tool = next(tool for tool in tools if tool.name() == tool_name)
    result = await tool.execute(tool_args)
    if not isinstance(result, str):
        return ToolDenied(content=f"Error: Tool '{tool_name}' did not return text.")
    return ToolApproved(content=result)


@pytest.mark.asyncio
async def test_redirect_to_specific_file() -> None:
    mock = MockTextTool()
    redirect = RedirectToolCallTool(tools=[mock], execute_tool=lambda name, args: execute_tool(name, args, [mock]))

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "output.txt"
        result = await redirect.execute({"tool_name": "mock_text", "tool_args": {}, "output_file": str(output_file)})

        assert f"Output redirected to {output_file}" in result
        assert output_file.exists()
        assert output_file.read_text() == "pure text output"


@pytest.mark.asyncio
async def test_redirect_to_nested_file() -> None:
    mock = MockTextTool()
    redirect = RedirectToolCallTool(tools=[mock], execute_tool=lambda name, args: execute_tool(name, args, [mock]))

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "subdir" / "nested" / "output.txt"
        result = await redirect.execute({"tool_name": "mock_text", "tool_args": {}, "output_file": str(output_file)})

        assert f"Output redirected to {output_file}" in result
        assert output_file.exists()
        assert output_file.read_text() == "pure text output"
        assert output_file.parent.exists()
        assert output_file.parent.parent.exists()


@pytest.mark.asyncio
async def test_redirect_structured_result_error() -> None:
    mock = MockStructuredTool()

    async def execute_struct(tool_name: str, tool_args: dict[str, Any]) -> ToolApproved | ToolDenied:
        result = await mock.execute(tool_args)
        if not isinstance(result, str):
            return ToolDenied(content="Error: Tool 'mock_struct' did not return text.")
        return ToolApproved(content=result)

    redirect = RedirectToolCallTool(tools=[mock], execute_tool=execute_struct)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "output.json"
        result = await redirect.execute({"tool_name": "mock_struct", "tool_args": {}, "output_file": str(output_file)})

        assert "Error: Tool 'mock_struct' did not return text." in result
        assert not output_file.exists()


@pytest.mark.asyncio
async def test_redirect_to_temp_file() -> None:
    mock = MockTextTool()
    redirect = RedirectToolCallTool(tools=[mock], execute_tool=lambda name, args: execute_tool(name, args, [mock]))

    result = await redirect.execute({"tool_name": "mock_text", "tool_args": {}})

    assert "Output redirected to temporary file:" in result
    tmp_path = result.split("temporary file: ")[1]
    path = Path(tmp_path)

    try:
        assert path.exists()
        assert path.read_text() == "pure text output"
    finally:
        if path.exists():
            path.unlink()


@pytest.mark.asyncio
async def test_recursion_protection() -> None:
    redirect = RedirectToolCallTool(tools=[], execute_tool=lambda name, args: execute_tool(name, args, []))
    # It adds itself to the internal list in actual usage, here we just test the name check
    result = await redirect.execute({"tool_name": "redirect_tool_call", "tool_args": {}})
    assert "Error: Cannot call redirect_tool_call recursively." in result


@pytest.mark.asyncio
async def test_tool_not_found() -> None:
    redirect = RedirectToolCallTool(tools=[], execute_tool=lambda name, args: execute_tool(name, args, []))
    result = await redirect.execute({"tool_name": "non_existent", "tool_args": {}})
    assert "Error: Tool 'non_existent' not found or cannot be redirected." in result


@pytest.mark.asyncio
async def test_redirect_tool_exception() -> None:
    mock = MockErrorTool()

    async def execute_error(tool_name: str, tool_args: dict[str, Any]) -> ToolApproved | ToolDenied:
        del tool_name
        result = await mock.execute(tool_args)
        return ToolApproved(content=result)

    redirect = RedirectToolCallTool(tools=[mock], execute_tool=execute_error)
    with pytest.raises(ValueError, match="Something went wrong"):
        await redirect.execute({"tool_name": "mock_error", "tool_args": {}})
