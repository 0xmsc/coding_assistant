INSTRUCTIONS = """
- Do not initialize a new git repository, unless your client explicitly requests it.
- Do not commit any changes to the git repository, unless your client explicitly requests it.
- When you have made a change to a project, ask the user if you should commit the changes.
- Do not use the 'mcp_filesystem_search_files' tool, use the 'rg' shelll command instead.
""".strip()


def get_instructions() -> str:
    return INSTRUCTIONS
