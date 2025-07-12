import logging
import os
import json
from pathlib import Path

logger = logging.getLogger("coding_assistant.cache")


def get_project_cache_dir(working_directory: Path) -> Path:
    """Get the project-specific .coding_assistant cache directory."""
    cache_dir = working_directory / ".coding_assistant"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_conversation_history_file(working_directory: Path) -> Path:
    """Get the conversation history file path for the specific project."""
    conversations_file = get_project_cache_dir(working_directory) / "conversations_summary.json"
    return conversations_file


def get_conversation_history(working_directory: Path) -> list[str]:
    conversations_file = get_conversation_history_file(working_directory)

    if not conversations_file.exists():
        logger.info("No conversations summary file found.")
        return []

    logger.info(f"Loading conversations summary from {conversations_file}.")
    conversations = json.loads(conversations_file.read_text())
    return conversations.get("summaries", [])


def save_conversation_history(working_directory: Path, summary: str):
    conversations_file = get_conversation_history_file(working_directory)

    if conversations_file.exists():
        conversations = json.loads(conversations_file.read_text())
    else:
        conversations = {"summaries": []}

    conversations["summaries"].append(summary)
    conversations_file.write_text(json.dumps(conversations, indent=2))

    logger.info(f"Saved conversations summary for {working_directory} to {conversations_file}.")
