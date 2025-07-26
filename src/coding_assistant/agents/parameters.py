import textwrap
from dataclasses import dataclass


@dataclass
class Parameter:
    name: str
    description: str
    value: str


def fill_parameters(
    parameter_description: dict,
    parameter_values: dict,
) -> list[Parameter]:
    parameters = []

    required = set(parameter_description.get("required", []))

    for name, parameter in parameter_description["properties"].items():
        # Check if required parameters are provided
        if name not in parameter_values or parameter_values[name] is None:
            if name in required:
                raise RuntimeError(f"Parameter {name} is required but not provided.")
            else:
                continue

        # Extract parameter type
        parameter_type = None
        if "type" in parameter:
            parameter_type = parameter["type"]
        elif "anyOf" in parameter:
            types_except_null = [t for t in parameter["anyOf"] if t.get("type") != "null"]
            if len(types_except_null) == 1:
                parameter_type = types_except_null[0].get("type")

        if not parameter_type:
            raise RuntimeError(f"Could not determine type for parameter {name}, schema {parameter}")

        if parameter_type == "string":
            value = str(parameter_values[name])
        elif parameter_type == "array":
            value = textwrap.indent("\n".join(parameter_values[name]), "- ")
        elif parameter_type == "boolean":
            value = str(parameter_values[name])
        else:
            raise RuntimeError(f"Unsupported parameter type: {parameter_type} for parameter {name}")

        parameters.append(
            Parameter(
                name=name,
                description=parameter["description"],
                value=value,
            )
        )

    return parameters


def format_parameters(parameters: list[Parameter]) -> str:
    PARAMETER_TEMPLATE = """
Name: {name}
Description: {description}
Value: {value}
""".strip()
    parameter_descriptions = []

    for parameter in parameters:
        value_str = parameter.value

        if "\n" in value_str:
            value_str = "\n" + textwrap.indent(value_str, "  ")

        parameter_descriptions.append(
            PARAMETER_TEMPLATE.format(
                name=parameter.name,
                description=parameter.description,
                value=value_str,
            )
        )

    return "\n\n".join(parameter_descriptions)
