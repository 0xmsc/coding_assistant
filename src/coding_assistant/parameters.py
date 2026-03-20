import textwrap
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class Parameter:
    name: str
    description: str
    value: str


def parameters_from_model(model: BaseModel) -> list[Parameter]:
    params: list[Parameter] = []
    data = model.model_dump()
    for name, field in model.__class__.model_fields.items():
        value: Any | None = data.get(name)

        if value is None:
            continue

        if isinstance(value, list):
            rendered_items: list[str] = []
            for item in value:
                item_str = str(item)
                if "\n" in item_str:
                    lines = item_str.splitlines()
                    first = lines[0]
                    first_bulleted = first if first.startswith("- ") else f"- {first}"
                    continuation = [f"  {line}" for line in lines[1:]]
                    rendered_items.append(
                        "\n".join([first_bulleted, *continuation]) if continuation else first_bulleted
                    )
                else:
                    rendered_items.append(item_str if item_str.startswith("- ") else f"- {item_str}")
            value_str = "\n".join(rendered_items)
        elif isinstance(value, (str, int, float, bool)):
            value_str = str(value)
        else:
            raise RuntimeError(f"Unsupported parameter type for parameter '{name}'")

        if not field.description:
            raise RuntimeError(f"Parameter '{name}' is missing a description.")

        params.append(
            Parameter(
                name=name,
                description=field.description,
                value=value_str,
            )
        )

    return params


def format_parameters(parameters: list[Parameter]) -> str:
    parameter_template = """
- Name: {name}
  - Description: {description}
  - Value: {value}
""".strip()
    parts: list[str] = []
    for parameter in parameters:
        value_str = parameter.value
        if "\n" in value_str:
            value_str = "\n" + textwrap.indent(value_str, "    ")
        else:
            value_str = " " + value_str
        parts.append(
            parameter_template.format(
                name=parameter.name,
                description=parameter.description,
                value=value_str,
            )
        )
    return "\n\n".join(parts)
