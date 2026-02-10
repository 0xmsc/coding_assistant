import textwrap

from coding_assistant.framework.parameters import Parameter


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
