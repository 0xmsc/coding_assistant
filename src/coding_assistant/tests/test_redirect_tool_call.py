import pytest
import tempfile
from pathlib import Path
from typing import Any

from coding_assistant.llm.types import Tool
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


@pytest.mark.asyncio
async def test_redirect_to_specific_file() -> None:
    mock = MockTextTool()
    redirect = RedirectToolCallTool(tools=[mock])

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "output.txt"
        result = await redirect.execute({"tool_name": "mock_text", "tool_args": {}, "output_file": str(output_file)})

        assert f"Output redirected to {output_file}" in result
        assert output_file.exists()
        assert output_file.read_text() == "pure text output"


@pytest.mark.asyncio
async def test_redirect_to_nested_file() -> None:
    mock = MockTextTool()
    redirect = RedirectToolCallTool(tools=[mock])

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
    redirect = RedirectToolCallTool(tools=[mock])

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "output.json"
        result = await redirect.execute({"tool_name": "mock_struct", "tool_args": {}, "output_file": str(output_file)})

        assert "Error: Tool 'mock_struct' did not return text." in result
        assert not output_file.exists()


@pytest.mark.asyncio
async def test_redirect_to_temp_file() -> None:
    mock = MockTextTool()
    redirect = RedirectToolCallTool(tools=[mock])

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
    redirect = RedirectToolCallTool(tools=[])
    # It adds itself to the internal list in actual usage, here we just test the name check
    result = await redirect.execute({"tool_name": "redirect_tool_call", "tool_args": {}})
    assert "Error: Cannot call redirect_tool_call recursively." in result


@pytest.mark.asyncio
async def test_tool_not_found() -> None:
    redirect = RedirectToolCallTool(tools=[])
    result = await redirect.execute({"tool_name": "non_existent", "tool_args": {}})
    assert "Error: Tool 'non_existent' not found or cannot be redirected." in result


@pytest.mark.asyncio
async def test_redirect_tool_exception() -> None:
    mock = MockErrorTool()
    redirect = RedirectToolCallTool(tools=[mock])
    result = await redirect.execute({"tool_name": "mock_error", "tool_args": {}})
    assert "Error executing tool 'mock_error': Something went wrong" in result
