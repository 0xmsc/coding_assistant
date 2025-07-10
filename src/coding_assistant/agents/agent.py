from dataclasses import dataclass


@dataclass
class Agent:
    name: str
    model: str
    instructions: str
    tools: list
    history = list
