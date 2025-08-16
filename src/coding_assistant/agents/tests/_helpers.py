from dataclasses import dataclass


@dataclass
class FakeFunction:
    name: str
    arguments: str = "{}"


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
