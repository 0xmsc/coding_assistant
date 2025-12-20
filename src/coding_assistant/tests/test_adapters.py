from coding_assistant.llm import adapters


def test_fix_input_schema_removes_uri_format():
    schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "format": "uri"},
            "other": {"type": "string"},
        },
    }

    adapters.fix_input_schema(schema)

    assert "format" not in schema["properties"]["url"]
    assert "format" not in schema["properties"]["other"]  # unchanged
