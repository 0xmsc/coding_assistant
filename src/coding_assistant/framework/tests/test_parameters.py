import pytest
from pydantic import BaseModel, Field, ValidationError

from coding_assistant.framework.parameters import parameters_from_model, format_parameters


class ExampleSchema(BaseModel):
    name: str = Field(description="Name")
    age: int | None = Field(default=None, description="Age")
    hobbies: list[str] = Field(default_factory=list, description="Hobbies")
    active: bool = Field(description="Active status")


def test_parameters_from_model_basic_and_optional_skip() -> None:
    model = ExampleSchema(name="Alice", active=True)  # age omitted, hobbies default empty
    params = parameters_from_model(model)
    names = [p.name for p in params]
    assert names == ["name", "hobbies", "active"]  # age skipped; hobbies present (empty list -> should it appear?)
    # Empty list should render to empty string list? We currently render it as "" (join of empty). Accept that.
    hobbies_param = next(p for p in params if p.name == "hobbies")
    assert hobbies_param.value == ""  # join of []


def test_parameters_from_model_list_rendering() -> None:
    model = ExampleSchema(name="Alice", active=False, hobbies=["reading", "- preformatted"])
    params = parameters_from_model(model)
    hobbies_value = next(p for p in params if p.name == "hobbies").value
    assert hobbies_value.splitlines() == ["- reading", "- preformatted"]


def test_parameters_from_model_unsupported_type() -> None:
    class Bad(BaseModel):
        data: dict = Field(description="Unsupported")

    bad = Bad(data={"a": 1})
    with pytest.raises(RuntimeError, match="Unsupported parameter type for parameter 'data'"):
        parameters_from_model(bad)


def test_parameters_from_model_validation_error() -> None:
    # Missing required field 'name'
    with pytest.raises(ValidationError):
        ExampleSchema(active=True)  # type: ignore


def test_format_parameters_multiline_list() -> None:
    model = ExampleSchema(name="Bob", active=True, hobbies=["one", "two"])
    params = parameters_from_model(model)
    output = format_parameters(params)
    assert "- Name: hobbies" in output
    assert "- one" in output
    assert "- two" in output


def test_format_parameters_list_item_with_multiline_string_indentation() -> None:
    multi = "first line of item\nsecond line continues"
    model = ExampleSchema(name="Eve", active=True, hobbies=[multi])
    params = parameters_from_model(model)
    output = format_parameters(params)

    expected_snippet = "\n  - Value: \n    - first line of item\n      second line continues"
    assert expected_snippet in output


def test_format_parameters_list_item_preserves_prefixed_bullet_and_indents_continuation() -> None:
    pre_bulleted = "- already bulleted first line\ncontinuation line"
    model = ExampleSchema(name="Zoe", active=False, hobbies=[pre_bulleted])
    params = parameters_from_model(model)
    output = format_parameters(params)

    expected_snippet = "\n  - Value: \n    - already bulleted first line\n      continuation line"
    assert expected_snippet in output
