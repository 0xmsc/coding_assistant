import pytest
from coding_assistant.agents.parameters import fill_parameters, _extract_type_from_schema, _format_value_by_type


class TestFillParameters:
    def test_basic_string_parameter(self):
        schema = {"properties": {"name": {"type": "string", "description": "A person's name"}}}
        values = {"name": "Alice"}

        result = fill_parameters(schema, values)

        assert len(result) == 1
        assert result[0].name == "name"
        assert result[0].description == "A person's name"
        assert result[0].value == "Alice"

    def test_array_parameter(self):
        schema = {"properties": {"items": {"type": "array", "description": "List of items"}}}
        values = {"items": ["apple", "banana", "cherry"]}

        result = fill_parameters(schema, values)

        assert len(result) == 1
        assert result[0].name == "items"
        assert result[0].value == "- apple\n- banana\n- cherry"

    def test_boolean_parameter(self):
        schema = {"properties": {"enabled": {"type": "boolean", "description": "Whether feature is enabled"}}}
        values = {"enabled": True}

        result = fill_parameters(schema, values)

        assert len(result) == 1
        assert result[0].value == "True"

    def test_integer_parameter(self):
        schema = {"properties": {"count": {"type": "integer", "description": "Number of items"}}}
        values = {"count": 42}

        result = fill_parameters(schema, values)

        assert len(result) == 1
        assert result[0].value == "42"

    def test_number_parameter(self):
        schema = {"properties": {"price": {"type": "number", "description": "Price in dollars"}}}
        values = {"price": 19.99}

        result = fill_parameters(schema, values)

        assert len(result) == 1
        assert result[0].value == "19.99"

    def test_anyof_parameter_with_string_or_null(self):
        schema = {
            "properties": {
                "optional_name": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Optional name field"}
            }
        }
        values = {"optional_name": "Bob"}

        result = fill_parameters(schema, values)

        assert len(result) == 1
        assert result[0].value == "Bob"

    def test_required_parameter_missing(self):
        schema = {
            "properties": {"required_field": {"type": "string", "description": "This field is required"}},
            "required": ["required_field"],
        }
        values = {}

        with pytest.raises(RuntimeError, match="Required parameter 'required_field' is missing"):
            fill_parameters(schema, values)

    def test_optional_parameter_missing(self):
        schema = {
            "properties": {
                "optional_field": {"type": "string", "description": "This field is optional"},
                "present_field": {"type": "string", "description": "This field is present"},
            }
        }
        values = {"present_field": "value"}

        result = fill_parameters(schema, values)

        assert len(result) == 1
        assert result[0].name == "present_field"

    def test_parameter_with_none_value_optional(self):
        schema = {"properties": {"optional_field": {"type": "string", "description": "Optional field"}}}
        values = {"optional_field": None}

        result = fill_parameters(schema, values)

        assert len(result) == 0

    def test_parameter_with_none_value_required(self):
        schema = {
            "properties": {"required_field": {"type": "string", "description": "Required field"}},
            "required": ["required_field"],
        }
        values = {"required_field": None}

        with pytest.raises(RuntimeError, match="Required parameter 'required_field' is missing"):
            fill_parameters(schema, values)

    def test_unsupported_parameter_type(self):
        schema = {"properties": {"weird_field": {"type": "object", "description": "Unsupported type"}}}
        values = {"weird_field": {"key": "value"}}

        with pytest.raises(RuntimeError, match="Unsupported parameter type for parameter 'weird_field'"):
            fill_parameters(schema, values)

    def test_no_type_determinable(self):
        schema = {"properties": {"mysterious_field": {"description": "Field with no type info"}}}
        values = {"mysterious_field": "value"}

        with pytest.raises(RuntimeError, match="Could not determine type for parameter 'mysterious_field'"):
            fill_parameters(schema, values)

    def test_multiple_parameters(self):
        schema = {
            "properties": {
                "name": {"type": "string", "description": "Name"},
                "age": {"type": "integer", "description": "Age"},
                "hobbies": {"type": "array", "description": "Hobbies"},
                "active": {"type": "boolean", "description": "Active status"},
            },
            "required": ["name", "age"],
        }
        values = {"name": "Alice", "age": 30, "hobbies": ["reading", "swimming"], "active": True}

        result = fill_parameters(schema, values)

        assert len(result) == 4
        names = [p.name for p in result]
        assert "name" in names
        assert "age" in names
        assert "hobbies" in names
        assert "active" in names

    def test_missing_description(self):
        schema = {
            "properties": {
                "field": {
                    "type": "string"
                    # No description
                }
            }
        }
        values = {"field": "value"}

        result = fill_parameters(schema, values)

        assert len(result) == 1
        assert result[0].description == ""


class TestExtractTypeFromSchema:
    def test_direct_type(self):
        schema = {"type": "string"}
        assert _extract_type_from_schema(schema) == "string"

    def test_anyof_with_string_and_null(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        assert _extract_type_from_schema(schema) == "string"

    def test_anyof_with_null_first(self):
        schema = {"anyOf": [{"type": "null"}, {"type": "integer"}]}
        assert _extract_type_from_schema(schema) == "integer"

    def test_no_type_info(self):
        schema = {"description": "No type info"}
        assert _extract_type_from_schema(schema) is None

    def test_anyof_only_null(self):
        schema = {"anyOf": [{"type": "null"}]}
        assert _extract_type_from_schema(schema) is None


class TestFormatValueByType:
    def test_string_type(self):
        assert _format_value_by_type("hello", "test") == "hello"
        assert _format_value_by_type("123", "test") == "123"

    def test_array_type(self):
        assert _format_value_by_type(["a", "b"], "test") == "- a\n- b"
        assert _format_value_by_type([1, 2, 3], "test") == "- 1\n- 2\n- 3"
        assert _format_value_by_type(["- a", "b"], "test") == "- a\n- b"

    def test_boolean_type(self):
        assert _format_value_by_type(True, "test") == "True"
        assert _format_value_by_type(False, "test") == "False"

    def test_integer_type(self):
        assert _format_value_by_type(42, "test") == "42"

    def test_number_type(self):
        assert _format_value_by_type(3.14, "test") == "3.14"

    def test_unsupported_type(self):
        with pytest.raises(RuntimeError, match="Unsupported parameter type for parameter 'test'"):
            _format_value_by_type({}, "test")
