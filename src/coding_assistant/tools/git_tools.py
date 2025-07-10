from langchain_core.tools import tool
import subprocess
from typing import Annotated


@tool
def git_add(file_path: str) -> str:
    """
    Add a file to git staging.
    """
    return subprocess.check_output(["git", "add", file_path], text=True)


@tool
def git_commit(message: str) -> str:
    """
    Commit changes to the repository.
    """
    return subprocess.check_output(["git", "commit", "-m", message], text=True)


@tool
def git_diff() -> str:
    """
    Get a git diff of unstaged changes.
    """
    return subprocess.check_output(["git", "diff"], text=True)


@tool
def git_status() -> str:
    """
    Get the current git status.
    """
    return subprocess.check_output(["git", "status"], text=True)


@tool
def fetch_pr_diff(pr_number: int) -> str:
    """
    Fetches the diff of a Pull Request using the GitHub CLI.

    Args:
        pr_number (int): The number of the PR to fetch the diff for.

    Returns:
        str: The diff of the PR.
    """
    return subprocess.check_output(["gh", "pr", "diff", str(pr_number)], text=True)


def get_git_tools():
    return [git_add, git_commit, git_diff, git_status, fetch_pr_diff]
