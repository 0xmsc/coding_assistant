from typing import Any
from coding_assistant.llm import openai


def test_fix_input_schema_removes_uri_format() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "format": "uri"},
            "other": {"type": "string"},
        },
    }

    openai.fix_input_schema(schema)

    assert "format" not in schema["properties"]["url"]
    assert "format" not in schema["properties"]["other"]  # unchanged
