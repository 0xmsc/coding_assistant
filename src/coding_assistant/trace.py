import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

def trace_enabled() -> bool:
    """Check if tracing is enabled via environment variable."""
    return bool(os.getenv("CODING_ASSISTANT_TRACE"))

def trace_data(name: str, content: str) -> None:
    """
    Write content to a trace file if tracing is enabled.

    Args:
        name: The base name for the trace file.
        content: The content to trace.
    """
    if not trace_enabled():
        return

    trace_path = Path("/tmp/coding_assistant_trace")
    trace_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    trace_file = trace_path / f"{timestamp}_{name}.txt"
    with open(trace_file, "w") as f:
        f.write(content)
