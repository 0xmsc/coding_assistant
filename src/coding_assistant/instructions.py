INSTRUCTIONS = [
    "Do not initialize a new git repository, unless your client explicitly requests it.",
    "Do not commit any changes to the git repository, unless your client explicitly requests it.",
]


def get_instructions() -> str:
    return " ".join(INSTRUCTIONS)
