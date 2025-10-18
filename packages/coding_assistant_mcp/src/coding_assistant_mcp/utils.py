from __future__ import annotations


def truncate_output(result: str, truncate_at: int) -> str:
    """Truncate a string to ``truncate_at`` characters and append a note."""

    if len(result) > truncate_at:
        note = f"\n\n[truncated output at: {truncate_at}, full length: {len(result)}]"
        truncated = result[: max(0, truncate_at - len(note))]
        return truncated + note

    return result
