import pytest
import tempfile
from pathlib import Path
from typing import Any
from coding_assistant.framework.results import TextResult
from coding_assistant.llm.types import Tool, ToolResult
from coding_assistant.tools.tools import RedirectToolCallTool
from dataclasses import dataclass


class MockTextTool(Tool):
    def name(self) -> str:
        return "mock_text"

    def description(self) -> str:
        return "returns text"

    def parameters(self) -> dict[str, Any]:
        return {}

    async def execute(self, parameters: dict[str, Any]) -> TextResult:
        return TextResult(content="pure text output")


@dataclass
class StructuredResult(ToolResult):
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.data


class MockStructuredTool(Tool):
    def name(self) -> str:
        return "mock_struct"

    def description(self) -> str:
        return "returns json"

    def parameters(self) -> dict[str, Any]:
        return {}

    async def execute(self, parameters: dict[str, Any]) -> StructuredResult:
        return StructuredResult(data={"status": "ok", "value": 42})


class MockErrorTool(Tool):
    def name(self) -> str:
        return "mock_error"

    def description(self) -> str:
        return "raises exception"

    def parameters(self) -> dict[str, Any]:
        return {}

    async def execute(self, parameters: dict[str, Any]) -> TextResult:
        raise ValueError("Something went wrong")


@pytest.mark.asyncio
async def test_redirect_to_specific_file() -> None:
    mock = MockTextTool()
    redirect = RedirectToolCallTool([mock])

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "output.txt"
        result = await redirect.execute({"tool_name": "mock_text", "tool_args": {}, "output_file": str(output_file)})

        assert f"Output redirected to {output_file}" in result.content
        assert output_file.exists()
        assert output_file.read_text() == "pure text output"


@pytest.mark.asyncio
async def test_redirect_to_nested_file() -> None:
    mock = MockTextTool()
    redirect = RedirectToolCallTool([mock])

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "subdir" / "nested" / "output.txt"
        result = await redirect.execute({"tool_name": "mock_text", "tool_args": {}, "output_file": str(output_file)})

        assert f"Output redirected to {output_file}" in result.content
        assert output_file.exists()
        assert output_file.read_text() == "pure text output"
        assert output_file.parent.exists()
        assert output_file.parent.parent.exists()


@pytest.mark.asyncio
async def test_redirect_structured_result_error() -> None:
    mock = MockStructuredTool()
    redirect = RedirectToolCallTool([mock])

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "output.json"
        result = await redirect.execute({"tool_name": "mock_struct", "tool_args": {}, "output_file": str(output_file)})

        assert "Error: Tool 'mock_struct' did not return a TextResult." in result.content
        assert not output_file.exists()


@pytest.mark.asyncio
async def test_redirect_to_temp_file() -> None:
    mock = MockTextTool()
    redirect = RedirectToolCallTool([mock])

    result = await redirect.execute({"tool_name": "mock_text", "tool_args": {}})

    assert "Output redirected to temporary file:" in result.content
    tmp_path = result.content.split("temporary file: ")[1]
    path = Path(tmp_path)

    try:
        assert path.exists()
        assert path.read_text() == "pure text output"
    finally:
        if path.exists():
            path.unlink()


@pytest.mark.asyncio
async def test_recursion_protection() -> None:
    redirect = RedirectToolCallTool([])
    # It adds itself to the internal list in actual usage, here we just test the name check
    result = await redirect.execute({"tool_name": "redirect_tool_call", "tool_args": {}})
    assert "Error: Cannot call redirect_tool_call recursively." in result.content


@pytest.mark.asyncio
async def test_tool_not_found() -> None:
    redirect = RedirectToolCallTool([])
    result = await redirect.execute({"tool_name": "non_existent", "tool_args": {}})
    assert "Error: Tool 'non_existent' not found or cannot be redirected." in result.content


@pytest.mark.asyncio
async def test_redirect_tool_exception() -> None:
    mock = MockErrorTool()
    redirect = RedirectToolCallTool([mock])
    result = await redirect.execute({"tool_name": "mock_error", "tool_args": {}})
    assert "Error executing tool 'mock_error': Something went wrong" in result.content
