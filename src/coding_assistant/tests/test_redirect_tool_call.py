import json
import pytest
import tempfile
from pathlib import Path
from coding_assistant.framework.types import TextResult, Tool
from coding_assistant.framework.results import ToolResult
from coding_assistant.tools.tools import RedirectToolCallTool
from dataclasses import dataclass

class MockTextTool(Tool):
    def name(self) -> str:
        return "mock_text"
    def description(self) -> str:
        return "returns text"
    def parameters(self) -> dict:
        return {}
    async def execute(self, parameters: dict) -> TextResult:
        return TextResult(content="pure text output")

@dataclass
class StructuredResult(ToolResult):
    data: dict
    def to_dict(self) -> dict:
        return self.data

class MockStructuredTool(Tool):
    def name(self) -> str:
        return "mock_struct"
    def description(self) -> str:
        return "returns json"
    def parameters(self) -> dict:
        return {}
    async def execute(self, parameters: dict) -> StructuredResult:
        return StructuredResult(data={"status": "ok", "value": 42})

@pytest.mark.asyncio
async def test_redirect_to_specific_file():
    mock = MockTextTool()
    redirect = RedirectToolCallTool([mock])
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "output.txt"
        result = await redirect.execute({
            "tool_name": "mock_text",
            "tool_args": {},
            "output_file": str(output_file)
        })
        
        assert f"Output redirected to {output_file}" in result.content
        assert output_file.exists()
        assert output_file.read_text() == "pure text output"

@pytest.mark.asyncio
async def test_redirect_structured_result_to_json():
    mock = MockStructuredTool()
    redirect = RedirectToolCallTool([mock])
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "output.json"
        await redirect.execute({
            "tool_name": "mock_struct",
            "tool_args": {},
            "output_file": str(output_file)
        })
        
        content = output_file.read_text()
        parsed = json.loads(content)
        assert parsed == {"status": "ok", "value": 42}

@pytest.mark.asyncio
async def test_redirect_to_temp_file():
    mock = MockTextTool()
    redirect = RedirectToolCallTool([mock])
    
    result = await redirect.execute({
        "tool_name": "mock_text",
        "tool_args": {}
    })
    
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
async def test_recursion_protection():
    redirect = RedirectToolCallTool([])
    # It adds itself to the internal list in actual usage, here we just test the name check
    result = await redirect.execute({
        "tool_name": "redirect_tool_call",
        "tool_args": {}
    })
    assert "Error: Cannot call redirect_tool_call recursively." in result.content

@pytest.mark.asyncio
async def test_tool_not_found():
    redirect = RedirectToolCallTool([])
    result = await redirect.execute({
        "tool_name": "non_existent",
        "tool_args": {}
    })
    assert "Error: Tool 'non_existent' not found." in result.content
