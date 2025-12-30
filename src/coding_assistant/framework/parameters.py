import textwrap
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class Parameter:
    """Simple serialisable representation of a validated parameter.

    We intentionally keep this a lightweight dataclass rather than re-using the
    underlying Pydantic model instance so that the rest of the agent code stays
    decoupled from Pydantic specifics (and to keep prompts clean / explicit).
    """

    name: str
    description: str
    value: str


def parameters_from_model(model: BaseModel) -> list[Parameter]:
    """Create a list of Parameter objects from a validated Pydantic model instance.

    Rules:
    - Skip fields whose value is ``None``
    - Lists are rendered as bullet lists (``- item``) preserving existing ``- `` prefix.
    - Primitive values (str / int / float / bool) are stringified.
    - Any other value types raise a RuntimeError
    """

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
                    continuation = [f"  {ln}" for ln in lines[1:]]
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
    PARAMETER_TEMPLATE = """
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
            PARAMETER_TEMPLATE.format(
                name=parameter.name,
                description=parameter.description,
                value=value_str,
            )
        )
    return "\n\n".join(parts)
